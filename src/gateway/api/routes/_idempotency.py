"""The ``Idempotency-Key`` header on the completion routes.

A non-streaming completion that carries the header claims it before the budget
is reserved. A retry with the same key and body is answered with the stored
response (and its original request ID and cost) without calling the provider or
billing again, or waits for the request still holding the key. Streaming
requests and hybrid mode ignore the header: a stream the client dropped is
already refunded, and a hybrid gateway has no database to hold the key in. So
does a deployment without ``OTARI_SECRET_KEY``, since the stored response is
encrypted with it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, Header, Request, Response
from fastapi.encoders import jsonable_encoder

from gateway.api.deps import build_idempotency_service, get_config, get_unit_of_work_if_needed
from gateway.api.routes._tools import CODE_EXECUTION_HEADER, WEB_SEARCH_HEADER
from gateway.core.config import REQUEST_ID_HEADER, ROUTER_HEADER, GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.services.inference import (
    Admission,
    Claimed,
    IdempotencyService,
    IdempotentRequest,
    Replay,
    storage_available,
)
from gateway.services.secret_box import SecretBoxUnavailableError

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
IDEMPOTENT_REPLAYED_HEADER = "Otari-Idempotent-Replayed"
_MAX_KEY_LENGTH = 255
# A response larger than this is not stored, so a retry of it runs again.
_MAX_STORED_BODY_BYTES = 8 * 1024 * 1024
# The response headers a replay repeats: the ones that describe this request
# rather than the moment it was answered, which rate-limit headers do.
_REPLAYED_HEADERS = (REQUEST_ID_HEADER, "Otari-Container-Id", "Otari-Container-Expires-At")
# The request headers that change what a request does, so they count toward
# whether a retry is the same request.
_REQUEST_SHAPING_HEADERS = (CODE_EXECUTION_HEADER, WEB_SEARCH_HEADER, ROUTER_HEADER, "anthropic-beta")

INVALID_IDEMPOTENCY_KEY_DETAIL = f"{IDEMPOTENCY_KEY_HEADER} must be 1 to {_MAX_KEY_LENGTH} printable ASCII characters."
IDEMPOTENCY_KEY_REUSED_DETAIL = (
    f"This {IDEMPOTENCY_KEY_HEADER} was already used for a different request. Use a new key for a new request."
)
IDEMPOTENCY_KEY_IN_FLIGHT_DETAIL = (
    f"A request with this {IDEMPOTENCY_KEY_HEADER} is still in progress. Retry with the same key to get its result."
)


@dataclass(frozen=True)
class InvalidKey:
    """The header is not a usable key."""


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


def _request_hash(endpoint: str, body: bytes, headers: Mapping[str, str]) -> str:
    """SHA-256 of what the request asks for: the endpoint, the body and the headers that change the result.

    The JSON is canonicalized so key order and spacing do not matter.
    """
    try:
        canonical = json.dumps(json.loads(body), sort_keys=True, separators=(",", ":")).encode()
    except ValueError:
        canonical = body
    options = json.dumps([headers.get(name) for name in _REQUEST_SHAPING_HEADERS]).encode()
    return hashlib.sha256(endpoint.encode() + b"\n" + options + b"\n" + canonical).hexdigest()


def _usable(key: str) -> bool:
    return 0 < len(key) <= _MAX_KEY_LENGTH and key.isascii() and key.isprintable() and key.strip() == key


class IdempotencyGuard:
    """One request's hold on its ``Idempotency-Key``, released unless the request completes."""

    def __init__(self, raw_request: Request, service: IdempotencyService | None, key: str | None) -> None:
        self._raw_request = raw_request
        self._service = service
        self._key = key
        self._request: IdempotentRequest | None = None
        self._claimed: Claimed | None = None

    @property
    def active(self) -> bool:
        """Whether this request carries a key this deployment honors."""
        return self._service is not None and self._key is not None

    async def admit(self, *, endpoint: str, user_id: str, api_key_id: str | None) -> Admission | InvalidKey | None:
        """Claim the key for this caller, or say what the request already holding it produced.

        None when the request carries no key or the deployment ignores it.
        """
        if self._service is None or self._key is None:
            return None
        if not _usable(self._key):
            return InvalidKey()
        self._request = IdempotentRequest(
            scope=f"key:{api_key_id}" if api_key_id is not None else f"master:{user_id}",
            key=self._key,
            request_hash=_request_hash(endpoint, await self._raw_request.body(), self._raw_request.headers),
            user_id=user_id,
            api_key_id=api_key_id,
        )
        outcome = await self._service.admit(self._request)
        if isinstance(outcome, Claimed):
            self._claimed = outcome
        return outcome

    async def complete(self, body: Any, response: Response, *, status_code: int = 200) -> None:
        """Store the response this request is about to return, for a retry to be given."""
        if self._service is None or self._request is None or self._claimed is None:
            return
        encoded = json.dumps(jsonable_encoder(body), separators=(",", ":"))
        if len(encoded.encode()) > _MAX_STORED_BODY_BYTES:
            logger.info("Response too large to store for idempotent replay; releasing the key")
            await self.release()
            return
        headers = {name: response.headers[name] for name in _REPLAYED_HEADERS if name in response.headers}
        claimed, self._claimed = self._claimed, None
        try:
            stored = await self._service.complete(
                self._request, claimed, status_code=status_code, body=encoded, headers=headers
            )
        except SecretBoxUnavailableError:
            logger.warning("Could not encrypt the response for idempotent replay; releasing the key")
            self._claimed = claimed
            await self.release()
            return
        except DATABASE_ERRORS:
            # The response is already paid for and about to be returned; failing it
            # now would lose it. The claim lapses at its lease instead.
            logger.warning("Could not store the response for idempotent replay", exc_info=True)
            return
        if not stored:
            logger.warning("Idempotency claim was taken over before the request completed")

    async def release(self) -> None:
        """Give the key back so a retry runs the request again. A no-op once completed."""
        if self._service is None or self._request is None or self._claimed is None:
            return
        claimed, self._claimed = self._claimed, None
        try:
            await self._service.release(self._request, claimed)
        except DATABASE_ERRORS:
            logger.warning("Could not release an idempotency claim; it lapses at its lease", exc_info=True)


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
                "cost without calling the provider or billing again, and waits for the original "
                "while it is still running. Reusing a key for a different body is refused with 422. "
                "Ignored for streaming requests, in hybrid mode, and on a deployment without OTARI_SECRET_KEY, "
                "which encrypts the stored response."
            ),
        ),
    ] = None,
) -> AsyncIterator[IdempotencyGuard]:
    """Yield the request's guard, and release its claim if the request did not complete."""
    service = None
    if uow is not None and idempotency_key is not None and config.idempotency_retention_sec > 0 and storage_available():
        service = build_idempotency_service(uow, config)
    guard = IdempotencyGuard(raw_request, service, idempotency_key)
    try:
        yield guard
    finally:
        await guard.release()


IdempotencyGuardDep = Annotated[IdempotencyGuard, Depends(get_idempotency_guard)]
