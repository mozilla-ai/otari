"""Deduplicate completion requests that carry an ``Idempotency-Key``.

A retry of a request whose response was lost (a dropped connection, a client
timeout) would otherwise call the provider again and be billed again. The first
request claims the key before it is dispatched; a retry with the same key and
body gets the stored response, or waits for the claim holder to finish. Only a
success is stored: a failed request is refunded, so a retry of it runs again.

The stored body is the generated content, so it is encrypted with
``OTARI_SECRET_KEY``. A deployment without that key stores nothing, and a body
that no configured key can decrypt is treated as missing and runs again.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.inference import IdempotencyRecord, IdempotencyState
from gateway.repositories.inference import InferenceRepositories
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    secret_box_configured,
)

_POLL_INTERVAL_SEC = 0.25


@dataclass(frozen=True)
class IdempotentRequest:
    """One request's claim on its key: who sent it, and what it asked for."""

    scope: str
    key: str
    request_hash: str
    user_id: str
    api_key_id: str | None


@dataclass(frozen=True)
class Claimed:
    """This request holds the key and runs; ``token`` completes or releases the claim."""

    token: str


@dataclass(frozen=True)
class Replay:
    """The key already holds a response to this same request."""

    status_code: int
    body: str
    headers: dict[str, str]


@dataclass(frozen=True)
class KeyReused:
    """The key holds, or is running, a different request."""


@dataclass(frozen=True)
class StillInFlight:
    """The request holding the key did not finish within the wait."""


Admission = Claimed | Replay | KeyReused | StillInFlight


class _Retry:
    """The claim changed under this attempt: look again straight away."""


_RETRY = _Retry()


def _as_utc(value: datetime) -> datetime:
    """Read a naive stored timestamp (SQLite) as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class IdempotencyService:
    """Claim, replay and settle idempotency keys, one Unit of Work block per step."""

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: InferenceRepositories,
        config: GatewayConfig,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._uow = uow
        self._keys = repositories.idempotency
        self._config = config
        self._sleep = sleep

    async def admit(self, request: IdempotentRequest) -> Admission:
        """Claim the key, or answer with what the request already holding it produced.

        While another request with the same key and body is in flight this waits
        for it, for at most ``idempotency_wait_sec``, rather than running the
        request a second time.
        """
        deadline = time.monotonic() + self._config.idempotency_wait_sec
        while True:
            async with self._uow:
                outcome = await self._try_admit(request)
            if isinstance(outcome, _Retry):
                continue
            if outcome is not None:
                return outcome
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return StillInFlight()
            await self._sleep(min(_POLL_INTERVAL_SEC, remaining))

    async def _try_admit(self, request: IdempotentRequest) -> Admission | _Retry | None:
        now = datetime.now(UTC)
        token = str(uuid.uuid4())
        claim = self._claim_values(request, token, now)
        if await self._keys.insert_claim({"scope": request.scope, "idempotency_key": request.key, **claim}):
            return Claimed(token)
        record = await self._keys.find(request.scope, request.key)
        if record is None:
            return _RETRY
        # A running claim is stale only once its lease lapses, since the request
        # renews the lease and not the retention; a stored response once its
        # retention does.
        stale_at = record.locked_until if record.state == IdempotencyState.IN_PROGRESS else record.expires_at
        if _as_utc(stale_at) <= now:
            taken = await self._keys.take_over(
                request.scope, request.key, previous_token=record.claim_token, values=claim
            )
            return Claimed(token) if taken else _RETRY
        if record.request_hash != request.request_hash:
            return KeyReused()
        if record.state == IdempotencyState.COMPLETED:
            replay = _replay(record)
            if replay is not None:
                return replay
            taken = await self._keys.take_over(
                request.scope, request.key, previous_token=record.claim_token, values=claim
            )
            return Claimed(token) if taken else _RETRY
        return None

    def _claim_values(self, request: IdempotentRequest, token: str, now: datetime) -> dict[str, Any]:
        lease = timedelta(seconds=self._config.idempotency_lease_sec)
        retention = timedelta(seconds=self._config.idempotency_retention_sec)
        return {
            "request_hash": request.request_hash,
            "claim_token": token,
            "state": IdempotencyState.IN_PROGRESS,
            "user_id": request.user_id,
            "api_key_id": request.api_key_id,
            "status_code": None,
            "response_body": None,
            "response_headers": None,
            "created_at": now,
            "locked_until": now + lease,
            # A claim is kept at least as long as its lease, so the sweep never
            # deletes one that is still live.
            "expires_at": now + max(lease, retention),
        }

    async def complete(
        self,
        request: IdempotentRequest,
        claimed: Claimed,
        *,
        status_code: int,
        body: str,
        headers: dict[str, str],
    ) -> bool:
        """Store the response, encrypted, for a retry to be given; False when the claim was lost meanwhile."""
        expires_at = datetime.now(UTC) + timedelta(seconds=self._config.idempotency_retention_sec)
        ciphertext = encrypt_secret(body)
        async with self._uow:
            return await self._keys.complete(
                request.scope,
                request.key,
                claim_token=claimed.token,
                status_code=status_code,
                response_body=ciphertext,
                response_headers=headers,
                expires_at=expires_at,
            )

    async def renew(self, request: IdempotentRequest, claimed: Claimed) -> bool:
        """Extend the claim's lease while its request runs; False once the claim is no longer this request's."""
        locked_until = datetime.now(UTC) + timedelta(seconds=self._config.idempotency_lease_sec)
        async with self._uow:
            return await self._keys.extend_lease(
                request.scope, request.key, claim_token=claimed.token, locked_until=locked_until
            )

    async def release(self, request: IdempotentRequest, claimed: Claimed) -> None:
        """Give the key back so a retry runs the request again."""
        async with self._uow:
            await self._keys.release(request.scope, request.key, claim_token=claimed.token)

    async def sweep(self) -> int:
        """Delete the records whose retention has passed, returning how many went."""
        async with self._uow:
            return await self._keys.delete_expired(datetime.now(UTC))


def storage_available() -> bool:
    """Whether a response can be stored, which needs a key to encrypt it with."""
    return secret_box_configured()


def _replay(record: IdempotencyRecord) -> Replay | None:
    """The stored response, or None when no configured key can decrypt it."""
    if record.response_body is None:
        return None
    try:
        body = decrypt_secret(record.response_body)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        return None
    return Replay(status_code=record.status_code or 200, body=body, headers=dict(record.response_headers or {}))
