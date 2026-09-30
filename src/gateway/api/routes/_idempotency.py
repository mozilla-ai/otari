"""The ``Idempotency-Key`` header on the completion routes.

A non-streaming completion that carries the header claims it before the budget
is reserved. A retry with the same key and body is answered with the stored
response (and its original request ID and cost) without calling the provider or
billing again, or is answered 409 while the request holding the key still runs.
Streaming requests and hybrid mode ignore the header: a stream the client
dropped is already refunded, and a hybrid gateway has no database to hold the
key in. So does a deployment without ``OTARI_SECRET_KEY``, since the stored
response is encrypted with it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Annotated, Any

from fastapi import Depends, Header, Request, Response
from fastapi.encoders import jsonable_encoder

from gateway.api.deps import build_idempotency_service, get_config, get_unit_of_work_if_needed
from gateway.api.routes._helpers import GUARDRAILS_RESULT_HEADER
from gateway.api.routes._tools import CODE_EXECUTION_HEADER, WEB_SEARCH_HEADER
from gateway.core.config import (
    CONVERSATION_HEADER,
    REQUEST_ID_HEADER,
    ROUTER_HEADER,
    ROUTER_TASK_HEADER,
    GatewayConfig,
)
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.services.inference import (
    Admission,
    Claimed,
    IdempotencyService,
    IdempotentRequest,
    InvalidKey,
    Replay,
    keep_claim_alive,
)

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
IDEMPOTENT_REPLAYED_HEADER = "Otari-Idempotent-Replayed"
# The response headers a replay repeats: the ones that describe this request
# rather than the moment it was answered, which rate-limit headers do.
_REPLAYED_HEADERS = (REQUEST_ID_HEADER, "Otari-Container-Id", "Otari-Container-Expires-At", GUARDRAILS_RESULT_HEADER)
# The request headers that change what a request does, so they count toward
# whether a retry is the same request.
_REQUEST_SHAPING_HEADERS = (
    CODE_EXECUTION_HEADER,
    WEB_SEARCH_HEADER,
    ROUTER_HEADER,
    "anthropic-beta",
    ROUTER_TASK_HEADER,
    CONVERSATION_HEADER,
)

INVALID_IDEMPOTENCY_KEY_DETAIL = (
    f"{IDEMPOTENCY_KEY_HEADER} must be 1 to {IdempotentRequest.MAX_KEY_LENGTH} printable ASCII characters."
)
IDEMPOTENCY_KEY_REUSED_DETAIL = (
    f"This {IDEMPOTENCY_KEY_HEADER} was already used for a different request. Use a new key for a new request."
)
IDEMPOTENCY_KEY_IN_FLIGHT_DETAIL = (
    f"A request with this {IDEMPOTENCY_KEY_HEADER} is still in progress. Retry with the same key to get its result."
)


class IdempotentReplay(Exception):
    """Raised from the request preamble when the key already holds this request's response."""

    def __init__(self, replay: Replay) -> None:
        super().__init__("idempotent replay")
        self.replay = replay

    def response(self) -> Response:
        """The stored response, marked as a replay."""
        headers = {**self.replay.headers, IDEMPOTENT_REPLAYED_HEADER: "true"}
        return Response(
            content=self.replay.body,
            status_code=self.replay.status_code,
            headers=headers,
            media_type="application/json",
        )


KeepAlive = Callable[[IdempotentRequest, Claimed], Coroutine[Any, Any, None]]


def _header_values(raw_request: Request, name: str) -> str | None:
    """Every value a request sent for ``name``, since a header such as ``anthropic-beta`` may repeat."""
    values = raw_request.headers.getlist(name)
    return ",".join(values) if values else None


class IdempotencyGuard:
    """One request's hold on its ``Idempotency-Key``, released unless the request completes.

    While the request runs, ``keep_alive`` holds the claim in the background.
    """

    def __init__(
        self,
        raw_request: Request,
        service: IdempotencyService | None,
        key: str | None,
        *,
        keep_alive: KeepAlive | None = None,
    ) -> None:
        self._raw_request = raw_request
        self._service = service
        self._key = key
        self._keep_alive = keep_alive
        self._request: IdempotentRequest | None = None
        self._claimed: Claimed | None = None
        self._heartbeat: asyncio.Task[None] | None = None

    async def admit(self, *, endpoint: str, user_id: str, api_key_id: str | None) -> Admission | InvalidKey | None:
        """Claim the key for this caller, or say what the request already holding it produced.

        None when the request carries no key or the deployment ignores it.
        """
        if self._service is None or self._key is None:
            return None
        request = IdempotentRequest.of(
            self._key,
            endpoint=endpoint,
            body=await self._raw_request.body(),
            options=[_header_values(self._raw_request, name) for name in _REQUEST_SHAPING_HEADERS],
            user_id=user_id,
            api_key_id=api_key_id,
        )
        if isinstance(request, InvalidKey):
            return request
        self._request = request
        outcome = await self._service.admit(request)
        if isinstance(outcome, Claimed):
            self._claimed = outcome
            if self._keep_alive is not None:
                self._heartbeat = asyncio.create_task(self._keep_alive(request, outcome))
        return outcome

    async def _stop_heartbeat(self) -> None:
        heartbeat, self._heartbeat = self._heartbeat, None
        if heartbeat is not None:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            except Exception:
                # The response is already paid for, so a failed renewal must not lose it.
                logger.warning("Idempotency claim renewal failed", exc_info=True)

    async def complete(self, body: Any, response: Response, *, status_code: int = 200) -> None:
        """Store the response this request is about to return, for a retry to be given.

        The claim stays renewed until the response is stored, so a retry cannot take it over meanwhile.
        """
        try:
            if self._service is None or self._request is None or self._claimed is None:
                return
            encoded = json.dumps(jsonable_encoder(body), separators=(",", ":"))
            headers = {name: response.headers[name] for name in _REPLAYED_HEADERS if name in response.headers}
            claimed, self._claimed = self._claimed, None
            await self._service.complete(self._request, claimed, status_code=status_code, body=encoded, headers=headers)
        except Exception:
            # The response is already paid for. A failed store leaves the claim to lapse at its lease.
            logger.exception("Could not store the response for idempotent replay")
        finally:
            await self._stop_heartbeat()

    async def release(self) -> None:
        """Give the key back so a retry runs the request again. A no-op once completed."""
        try:
            if self._service is None or self._request is None or self._claimed is None:
                return
            claimed, self._claimed = self._claimed, None
            await self._service.release(self._request, claimed)
        except Exception:
            # The request already has its outcome. A failed release leaves the claim to lapse at its lease.
            logger.exception("Could not release an idempotency claim")
        finally:
            await self._stop_heartbeat()


async def get_idempotency_guard(
    raw_request: Request,
    config: Annotated[GatewayConfig, Depends(get_config)],
    uow: Annotated[UnitOfWork | None, Depends(get_unit_of_work_if_needed)],
    idempotency_key: Annotated[
        str | None,
        Header(
            alias=IDEMPOTENCY_KEY_HEADER,
            description=(
                "A unique value, such as a UUID, that makes a non-streaming request safe to retry. "
                "A retry with the same key and body returns the original response, request ID and "
                "cost without calling the provider or billing again. A retry while the original is still "
                "running is answered 409 with Retry-After. Reusing a key for a different body is refused with 422. "
                "Ignored for streaming requests, in hybrid mode, and on a deployment without OTARI_SECRET_KEY, "
                "which encrypts the stored response."
            ),
        ),
    ] = None,
) -> AsyncIterator[IdempotencyGuard]:
    """Yield the request's guard, and release its claim if the request did not complete."""
    service = None
    if uow is not None and idempotency_key is not None and IdempotencyService.is_enabled(config):
        service = build_idempotency_service(uow, config)

    async def keep_alive(request: IdempotentRequest, claimed: Claimed) -> None:
        await keep_claim_alive(
            request,
            claimed,
            config.idempotency_lease_sec,
            lambda worker_uow: build_idempotency_service(worker_uow, config),
        )

    guard = IdempotencyGuard(raw_request, service, idempotency_key, keep_alive=keep_alive)
    try:
        yield guard
    finally:
        await guard.release()


IdempotencyGuardDep = Annotated[IdempotencyGuard, Depends(get_idempotency_guard)]
