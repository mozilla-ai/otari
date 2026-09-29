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
import hashlib
import json
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.models.inference import IdempotencyRecord, IdempotencyState
from gateway.repositories.inference import InferenceRepositories
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    secret_box_configured,
)

_FIRST_POLL_SEC = 0.1
_MAX_POLL_SEC = 2.0
# A claim can change between the insert and the read. After this many changes in a row, the retry waits.
_MAX_IMMEDIATE_RETRIES = 3
# A response larger than this is not stored, so a retry of it runs again.
_MAX_STORED_BODY_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class InvalidKey:
    """The key is not one this service accepts."""


@dataclass(frozen=True)
class IdempotentRequest:
    """One request's claim on its key: who sent it, and what it asked for."""

    MAX_KEY_LENGTH: ClassVar[int] = 255

    scope: str
    key: str
    request_hash: str
    user_id: str
    api_key_id: str | None

    @classmethod
    def of(
        cls,
        key: str,
        *,
        endpoint: str,
        body: bytes,
        options: Sequence[str | None],
        user_id: str,
        api_key_id: str | None,
    ) -> IdempotentRequest | InvalidKey:
        """The claim a caller makes with ``key``, or InvalidKey when the key is unusable.

        A key belongs to the API key that sent it, or to the billed user for the master key,
        so two callers never share one.
        ``options`` are the request settings outside the body that change its result.
        """
        if not _usable(key):
            return InvalidKey()
        return cls(
            scope=f"key:{api_key_id}" if api_key_id is not None else f"master:{user_id}",
            key=key,
            request_hash=_request_hash(endpoint, body, options),
            user_id=user_id,
            api_key_id=api_key_id,
        )


def _usable(key: str) -> bool:
    return (
        0 < len(key) <= IdempotentRequest.MAX_KEY_LENGTH and key.isascii() and key.isprintable() and key.strip() == key
    )


def _request_hash(endpoint: str, body: bytes, options: Sequence[str | None]) -> str:
    """SHA-256 of what the request asks for: the endpoint, the body and the options that change the result.

    The JSON is canonicalized so key order and spacing do not matter.
    """
    try:
        canonical = json.dumps(json.loads(body), sort_keys=True, separators=(",", ":")).encode()
    except ValueError:
        canonical = body
    encoded_options = json.dumps(list(options)).encode()
    return hashlib.sha256(endpoint.encode() + b"\n" + encoded_options + b"\n" + canonical).hexdigest()


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


@dataclass(frozen=True)
class UnknownCaller:
    """The user the key would be claimed for does not exist."""


@dataclass(frozen=True)
class BlockedCaller:
    """The user the key would be claimed for is blocked, so is given nothing, not even a stored response."""


Admission = Claimed | Replay | KeyReused | StillInFlight | UnknownCaller | BlockedCaller


class _Retry:
    """The claim changed under this attempt: look again straight away."""


_RETRY = _Retry()


def _poll_delays(rng: random.Random) -> Iterator[float]:
    """Yield the delays between looks at a claim, which double up to a cap and carry jitter so retries spread out."""
    base = _FIRST_POLL_SEC
    while True:
        yield base / 2 + rng.uniform(0, base / 2)
        base = min(base * 2, _MAX_POLL_SEC)


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

    @staticmethod
    def is_enabled(config: GatewayConfig) -> bool:
        """Whether this deployment honors idempotency keys, which needs a retention and a key to encrypt with."""
        return config.idempotency_retention_sec > 0 and secret_box_configured()

    async def admit(self, request: IdempotentRequest) -> Admission:
        """Claim the key, or answer with what the request already holding it produced.

        While another request with the same key and body is in flight this waits
        for it, for at most ``idempotency_wait_sec``, rather than running the
        request a second time.
        An unknown or blocked user is refused before any key is read or claimed.
        """
        async with self._uow:
            caller = await self._keys.get_caller(request.user_id)
        if caller is None:
            return UnknownCaller()
        if caller.blocked:
            return BlockedCaller()
        deadline = time.monotonic() + self._config.idempotency_wait_sec
        delays = _poll_delays(random.Random())
        changes = 0
        while True:
            async with self._uow:
                outcome = await self._try_admit(request)
            if isinstance(outcome, _Retry):
                changes += 1
                if changes < _MAX_IMMEDIATE_RETRIES:
                    continue
            elif outcome is not None:
                return outcome
            changes = 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return StillInFlight()
            await self._sleep(min(next(delays), remaining))

    async def _try_admit(self, request: IdempotentRequest) -> Admission | _Retry | None:
        now = await self._keys.get_database_time()
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
        """Store the response, encrypted, for a retry to be given, returning whether it was stored.

        A response that cannot be stored gives the key back, so a retry runs again.
        A failed write leaves the claim to lapse at its lease, since the response is already paid for.
        """
        if len(body.encode()) > _MAX_STORED_BODY_BYTES:
            logger.info("Response too large to store for idempotent replay; releasing the key")
            await self.release(request, claimed)
            return False
        try:
            ciphertext = encrypt_secret(body)
        except SecretBoxUnavailableError:
            logger.warning("Could not encrypt the response for idempotent replay; releasing the key")
            await self.release(request, claimed)
            return False
        retention = timedelta(seconds=self._config.idempotency_retention_sec)
        try:
            async with self._uow:
                expires_at = await self._keys.get_database_time() + retention
                stored = await self._keys.complete(
                    request.scope,
                    request.key,
                    claim_token=claimed.token,
                    status_code=status_code,
                    response_body=ciphertext,
                    response_headers=headers,
                    expires_at=expires_at,
                )
        except DATABASE_ERRORS:
            logger.warning("Could not store the response for idempotent replay", exc_info=True)
            return False
        if not stored:
            logger.warning("Idempotency claim was taken over before the request completed")
        return stored

    async def renew(self, request: IdempotentRequest, claimed: Claimed) -> bool:
        """Extend the claim's lease while its request runs; False once the claim is no longer this request's."""
        async with self._uow:
            locked_until = await self._keys.get_database_time() + timedelta(seconds=self._config.idempotency_lease_sec)
            return await self._keys.extend_lease(
                request.scope, request.key, claim_token=claimed.token, locked_until=locked_until
            )

    async def release(self, request: IdempotentRequest, claimed: Claimed) -> None:
        """Give the key back so a retry runs the request again.

        A failed delete leaves the claim to lapse at its lease.
        """
        try:
            async with self._uow:
                await self._keys.release(request.scope, request.key, claim_token=claimed.token)
        except DATABASE_ERRORS:
            logger.warning("Could not release an idempotency claim; it lapses at its lease", exc_info=True)

    async def sweep(self, *, batch_size: int = 500) -> int:
        """Delete the records whose retention has passed, returning how many went.

        Each batch commits on its own, so a large backlog never holds many rows locked at once.
        """
        total = 0
        while True:
            async with self._uow:
                deleted = await self._keys.delete_expired(await self._keys.get_database_time(), limit=batch_size)
            total += deleted
            if deleted < batch_size:
                return total


def _replay(record: IdempotencyRecord) -> Replay | None:
    """The stored response, or None when no configured key can decrypt it."""
    if record.response_body is None:
        return None
    try:
        body = decrypt_secret(record.response_body)
    except (SecretBoxUnavailableError, SecretDecryptionError):
        return None
    return Replay(status_code=record.status_code or 200, body=body, headers=dict(record.response_headers or {}))
