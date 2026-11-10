"""Connect this deployment to otari.ai: ask for a code, wait for approval, store the key.

A device authorization (RFC 8628) against the two endpoints
``docs/platform-connect-protocol.md`` specifies. The deployment holds one
attempt at a time, in one row, so it survives a restart and every replica reads
the same one. Nothing polls in the background: a status read asks otari.ai at
most once, and only when the interval otari.ai gave has passed.

Nothing is sent to otari.ai before an operator starts an attempt, and the
device code never leaves this process except to otari.ai.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from gateway.core.config import API_ROOT, API_VERSION
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions import TenancyError
from gateway.exceptions.providers_exceptions import (
    HostedProviderAlreadyExistsError,
    HostedProviderNotFoundError,
    HostedProviderReadOnlyError,
    PlatformUnreachableError,
)
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.log_config import logger
from gateway.models.providers import (
    PLATFORM_CONNECT_CONNECTED,
    PLATFORM_CONNECT_DENIED,
    PLATFORM_CONNECT_EXPIRED,
    PLATFORM_CONNECT_PENDING,
    PlatformConnectAttempt,
)
from gateway.rate_limit import RateLimiter
from gateway.repositories.providers import PlatformConnectRepository
from gateway.schemas.providers import (
    HostedProviderCreateRequest,
    HostedProviderUpdateRequest,
    PlatformConnectStartPublic,
    PlatformConnectStatusPublic,
)
from gateway.services.providers._hosted_provider_service import HostedProviderService
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
    secret_box_configured,
)
from gateway.services.url_safety import UnsafeURLError, validate_credential_address

# The hosted provider a connection stores, and the any-llm implementation that serves it.
PLATFORM_PROVIDER_NAME = "otari-ai"
PLATFORM_PROVIDER_IMPLEMENTATION = "otari"

MAX_GATEWAY_NAME_LENGTH = 100
# RFC 8628 section 3.5: a ``slow_down`` adds this to the interval.
SLOW_DOWN_SECONDS = 5
_TIMEOUT_SECONDS = 10
_MAX_INTERVAL_SECONDS = 60
# The longest a ``Retry-After`` on a poll is honored for.
_MAX_RETRY_AFTER_SECONDS = 300
_MAX_EXPIRES_SECONDS = 24 * 60 * 60

STARTS_PER_WINDOW = 10
START_WINDOW_SEC = 10 * 60

PollError = Literal["authorization_pending", "slow_down", "expired_token", "access_denied"]


def new_platform_connect_rate_limiter() -> RateLimiter:
    """A fresh per-caller limit on starting a connection, kept on ``app.state.platform_connect_rate_limiter``."""
    return RateLimiter(STARTS_PER_WINDOW, window_sec=START_WINDOW_SEC)


class _DeviceGrant(BaseModel):
    """otari.ai's answer to a request for a code."""

    device_code: str = Field(min_length=1, max_length=4096)
    user_code: str = Field(min_length=1, max_length=64)
    verification_uri_complete: str = Field(min_length=1, max_length=2048)
    expires_in: int = Field(gt=0)
    interval: int = Field(ge=0)


class _IssuedKey(BaseModel):
    """otari.ai's answer once the operator approved: the key, and the inference root it is for."""

    api_key: str = Field(min_length=1, max_length=1024, repr=False)
    api_base: str = Field(min_length=1, max_length=1024)


class _PollRefusal(BaseModel):
    error: PollError


@dataclass(frozen=True)
class PollDelay:
    """otari.ai asked for fewer polls: a ``slow_down``, or a 429 with the seconds its ``Retry-After`` named."""

    retry_after: int | None = None


class PlatformProtocolError(Exception):
    """otari.ai could not be reached, or answered outside the contract."""


@dataclass(frozen=True)
class PlatformClient:
    """The two otari.ai endpoints a connection uses. Sends no credential and follows no redirect."""

    base_url: str
    transport: httpx.AsyncBaseTransport | None = None

    async def request_code(self, gateway_name: str) -> _DeviceGrant:
        """Ask otari.ai for a device code and the code the operator approves.

        Raises:
            PlatformProtocolError: no answer, or one outside the contract.
        """
        status, body, _ = await self._post("/connect/device", {"gateway_name": gateway_name})
        if status != 200:
            raise PlatformProtocolError
        return _parsed(_DeviceGrant, body)

    async def exchange(self, device_code: str) -> _IssuedKey | PollError | PollDelay:
        """Ask otari.ai for the key, or learn why there is none yet.

        Raises:
            PlatformProtocolError: no answer, or one outside the contract.
        """
        status, body, retry_after = await self._post("/connect/token", {"device_code": device_code})
        if status == 200:
            return _parsed(_IssuedKey, body)
        if status == 400:
            error = _parsed(_PollRefusal, body).error
            return PollDelay() if error == "slow_down" else error
        if status == 429:
            return PollDelay(retry_after)
        raise PlatformProtocolError

    async def _post(self, path: str, payload: dict[str, str]) -> tuple[int, bytes, int | None]:
        try:
            async with (
                asyncio.timeout(_TIMEOUT_SECONDS),
                httpx.AsyncClient(
                    timeout=_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False, transport=self.transport
                ) as client,
            ):
                response = await client.post(f"{self.base_url}{API_ROOT}{path}", json=payload)
        except (httpx.HTTPError, TimeoutError):
            logger.warning("otari.ai could not be reached for %s", path)
            raise PlatformProtocolError from None
        if response.status_code not in (200, 400):
            # The status only: the body is otari.ai's and may echo the request.
            logger.warning("otari.ai answered %s to %s", response.status_code, path)
        retry = response.headers.get("Retry-After", "")
        retry_after = int(retry) if retry.isascii() and retry.isdecimal() and len(retry) <= 6 else None
        return response.status_code, response.content, retry_after


def _parsed[T: BaseModel](shape: type[T], body: bytes) -> T:
    try:
        return shape.model_validate_json(body)
    except ValidationError:
        # Not the error itself: it carries the input, which may be the key.
        logger.warning("otari.ai sent a %s this gateway cannot read", shape.__name__.lstrip("_"))
        raise PlatformProtocolError from None


def _now() -> datetime:
    return datetime.now(UTC)


class PlatformConnectService:
    """Start, follow and cancel the deployment's connection to otari.ai."""

    def __init__(
        self,
        uow: UnitOfWork,
        *,
        attempts: PlatformConnectRepository,
        hosted: HostedProviderService,
        client: PlatformClient,
    ) -> None:
        self.uow = uow
        self.attempts = attempts
        self.hosted = hosted
        self.client = client

    async def start(self, *, gateway_name: str, organization_id: uuid.UUID | None) -> PlatformConnectStartPublic:
        """Ask otari.ai for a code, and keep the attempt. A pending attempt is answered as it stands.

        ``organization_id`` is the organization the stored provider will be kept to.

        Raises:
            SecretBoxUnavailableTenancyError: no usable ``OTARI_SECRET_KEY``, so nothing could be stored.
            PlatformUnreachableError: otari.ai gave no usable answer.
        """
        if not secret_box_configured():
            raise SecretBoxUnavailableTenancyError("the otari.ai connection")
        now = _now()
        async with self.uow:
            pending = await self.attempts.current()
            if pending is not None and pending.status == PLATFORM_CONNECT_PENDING and pending.expires_at > now:
                return _start_public(pending, now)
        try:
            grant = await self.client.request_code(gateway_name[:MAX_GATEWAY_NAME_LENGTH])
            validate_credential_address(grant.verification_uri_complete, label="otari.ai approval page")
        except (PlatformProtocolError, UnsafeURLError):
            raise PlatformUnreachableError from None
        try:
            encrypted = encrypt_secret(grant.device_code)
        except SecretBoxUnavailableError:
            raise SecretBoxUnavailableTenancyError("the otari.ai connection") from None
        interval = min(max(grant.interval, 1), _MAX_INTERVAL_SECONDS)
        now = _now()
        async with self.uow:
            row = await self.attempts.replace(
                attempt_id=uuid.uuid4(),
                status=PLATFORM_CONNECT_PENDING,
                encrypted_device_code=encrypted,
                user_code=grant.user_code,
                verification_uri_complete=grant.verification_uri_complete,
                expires_at=now + timedelta(seconds=min(grant.expires_in, _MAX_EXPIRES_SECONDS)),
                interval_seconds=interval,
                next_poll_at=now + timedelta(seconds=interval),
                organization_id=organization_id,
            )
            return _start_public(row, now)

    async def status(self) -> PlatformConnectStatusPublic:
        """Where the connection stands, asking otari.ai at most once, and only when a poll is due.

        Raises:
            TenancyError: otari.ai issued a key and it could not be stored; the attempt ends as expired.
        """
        now = _now()
        async with self.uow:
            row = await self.attempts.current()
            if row is None:
                return PlatformConnectStatusPublic(status="idle")
            if row.status != PLATFORM_CONNECT_PENDING:
                ended = row.status
            elif row.expires_at <= now:
                await self.attempts.finish(row.attempt_id, PLATFORM_CONNECT_EXPIRED)
                return PlatformConnectStatusPublic(status=PLATFORM_CONNECT_EXPIRED)
            else:
                ended = None
                pending = _status_public(row, now)
                attempt_id, organization_id = row.attempt_id, row.organization_id
                interval, encrypted = row.interval_seconds, row.encrypted_device_code
                claimed = await self.attempts.claim_poll(attempt_id, now, now + timedelta(seconds=interval))
        if ended is not None:
            return await self._ended(ended)
        if not claimed or encrypted is None:
            return pending

        try:
            answer = await self.client.exchange(decrypt_secret(encrypted))
        except (PlatformProtocolError, SecretBoxUnavailableError, SecretDecryptionError):
            # Still pending: the next due read asks again, until the code expires.
            return pending
        if answer == "authorization_pending":
            return pending
        if isinstance(answer, PollDelay):
            slower = min(interval + SLOW_DOWN_SECONDS, _MAX_INTERVAL_SECONDS)
            wait = max(slower, min(answer.retry_after or 0, _MAX_RETRY_AFTER_SECONDS))
            async with self.uow:
                await self.attempts.slow_down(attempt_id, slower, _now() + timedelta(seconds=wait))
            return pending
        if not isinstance(answer, _IssuedKey):
            outcome = PLATFORM_CONNECT_DENIED if answer == "access_denied" else PLATFORM_CONNECT_EXPIRED
            async with self.uow:
                await self.attempts.finish(attempt_id, outcome)
            return PlatformConnectStatusPublic(status=outcome)

        async with self.uow:
            current = await self.attempts.current()
            if current is None or current.attempt_id != attempt_id or current.status != PLATFORM_CONNECT_PENDING:
                # Canceled or replaced while otari.ai answered, so the key is dropped.
                return PlatformConnectStatusPublic(status="idle") if current is None else _status_public(current, now)
        try:
            await self._store(answer, organization_id)
        except TenancyError:
            # The key is issued once, so this attempt cannot be finished later.
            async with self.uow:
                await self.attempts.finish(attempt_id, PLATFORM_CONNECT_EXPIRED)
            raise
        async with self.uow:
            await self.attempts.finish(attempt_id, PLATFORM_CONNECT_CONNECTED)
        return PlatformConnectStatusPublic(status=PLATFORM_CONNECT_CONNECTED)

    async def cancel(self) -> None:
        """Forget the attempt. otari.ai is not told: an unapproved code expires there on its own."""
        async with self.uow:
            await self.attempts.clear()

    async def _ended(self, status: str) -> PlatformConnectStatusPublic:
        """An ended attempt's status, read as idle once the provider it connected is gone."""
        if status == PLATFORM_CONNECT_CONNECTED:
            stored = (await self.hosted.list_providers(limit=1000)).data
            if not any(provider.name == PLATFORM_PROVIDER_NAME for provider in stored):
                return PlatformConnectStatusPublic(status="idle")
        return PlatformConnectStatusPublic.model_validate({"status": status})

    async def _store(self, key: _IssuedKey, organization_id: uuid.UUID | None) -> None:
        """Keep the issued key on the ``otari-ai`` hosted provider, replacing the credential of one already there."""
        try:
            validate_credential_address(key.api_base, label="otari.ai inference")
        except UnsafeURLError:
            raise PlatformUnreachableError from None
        api_base = _provider_api_base(key.api_base)
        try:
            stored = await self.hosted.update_provider(
                PLATFORM_PROVIDER_NAME,
                HostedProviderUpdateRequest(
                    api_key=key.api_key,
                    api_base=api_base,
                    enabled=True,
                    restricted_to_organization_id=organization_id,
                ),
            )
        except (HostedProviderNotFoundError, HostedProviderReadOnlyError):
            await self._create(key.api_key, api_base, organization_id)
            return
        if stored.provider != PLATFORM_PROVIDER_IMPLEMENTATION:
            # The name was held by another implementation, which cannot serve this key.
            await self.hosted.delete_provider(PLATFORM_PROVIDER_NAME)
            await self._create(key.api_key, api_base, organization_id)
            return
        await self.hosted.refresh_models(PLATFORM_PROVIDER_NAME)

    async def _create(self, api_key: str, api_base: str, organization_id: uuid.UUID | None) -> None:
        request = HostedProviderCreateRequest(
            provider=PLATFORM_PROVIDER_IMPLEMENTATION,
            name=PLATFORM_PROVIDER_NAME,
            api_key=api_key,
            api_base=api_base,
            restricted_to_organization_id=organization_id,
        )
        try:
            await self.hosted.create_provider(request)
        except HostedProviderAlreadyExistsError:
            # Another replica finished the same attempt first.
            logger.info("The %s hosted provider was already stored", PLATFORM_PROVIDER_NAME)


def _provider_api_base(inference_root: str) -> str:
    """The address any-llm's ``otari`` implementation takes: the origin, without the API root it appends itself.

    otari.ai issues the inference root (``https://api.otari.ai/api/v1``), which that implementation refuses.
    """
    trimmed = inference_root.rstrip("/")
    for suffix in (API_ROOT, f"/{API_VERSION}"):
        if trimmed.endswith(suffix):
            return trimmed.removesuffix(suffix).rstrip("/")
    return trimmed


def _seconds_left(row: PlatformConnectAttempt, now: datetime) -> int:
    return max(0, int((row.expires_at - now).total_seconds()))


def _start_public(row: PlatformConnectAttempt, now: datetime) -> PlatformConnectStartPublic:
    return PlatformConnectStartPublic(
        user_code=row.user_code,
        verification_uri_complete=row.verification_uri_complete,
        expires_in=_seconds_left(row, now),
    )


def _status_public(row: PlatformConnectAttempt, now: datetime) -> PlatformConnectStatusPublic:
    if row.status != PLATFORM_CONNECT_PENDING:
        return PlatformConnectStatusPublic.model_validate({"status": row.status})
    return PlatformConnectStatusPublic(
        status=PLATFORM_CONNECT_PENDING,
        user_code=row.user_code,
        verification_uri_complete=row.verification_uri_complete,
        expires_in=_seconds_left(row, now),
    )
