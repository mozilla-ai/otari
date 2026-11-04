"""The key a hosted sandbox provider's sandboxes are leased with, stored or from the environment.

A key stored through the dashboard wins over the provider's own environment
variable, so rotating it is a write rather than a redeploy. Sandboxes are opened
with no session in hand (``api/deps.get_code_execution_port``), so stored keys
are held decrypted in a process-wide cache, refreshed on a TTL and right after a
write, the arrangement ``services/providers/_provider_endpoint_cache`` uses: a
write reaches its own worker at once and every other worker within
``SANDBOX_CREDENTIAL_CACHE_TTL_SECONDS``.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Literal

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.shared_exceptions import SecretBoxUnavailableTenancyError
from gateway.exceptions.tools_exceptions import SandboxCredentialRefusedError
from gateway.log_config import logger
from gateway.models.tools import SandboxProvider
from gateway.repositories.tools import SandboxCredentialRepository
from gateway.services.secret_box import (
    SecretBoxUnavailableError,
    SecretDecryptionError,
    decrypt_secret,
    encrypt_secret,
)

SANDBOX_CREDENTIAL_CACHE_TTL_SECONDS = 30.0

# The environment variable each hosted provider's SDK reads its key from, which
# is also where a deployment that stores none gets it.
SANDBOX_KEY_ENV: dict[SandboxProvider, str] = {SandboxProvider.E2B: "E2B_API_KEY"}


@dataclass(frozen=True)
class _StoredKey:
    api_key: str
    last4: str


@dataclass(frozen=True)
class SandboxCredentialStatus:
    """Where the key a provider's sandboxes are leased with comes from, never the key."""

    source: Literal["stored", "environment"] | None
    last4: str | None = None


_stored: dict[SandboxProvider, _StoredKey] = {}
# Bumped by every write this worker makes, so a refresh that read the table
# before the write does not publish its older answer over it.
_writes = 0


def _env_key(provider: SandboxProvider) -> str | None:
    name = SANDBOX_KEY_ENV.get(provider)
    return (os.getenv(name) or None) if name else None


def sandbox_api_key(provider: SandboxProvider) -> str | None:
    """The key to lease ``provider``'s sandboxes with: the stored one, else the environment's."""
    stored = _stored.get(provider)
    return stored.api_key if stored is not None else _env_key(provider)


def sandbox_credential_status(provider: SandboxProvider) -> SandboxCredentialStatus:
    """Which key ``provider``'s sandboxes are leased with, as this worker last loaded it."""
    stored = _stored.get(provider)
    if stored is not None:
        return SandboxCredentialStatus(source="stored", last4=stored.last4)
    env_key = _env_key(provider)
    if env_key:
        return SandboxCredentialStatus(source="environment", last4=env_key[-4:])
    return SandboxCredentialStatus(source=None)


def _publish_write(provider: SandboxProvider, stored: _StoredKey | None) -> None:
    global _stored, _writes  # noqa: PLW0603

    updated = {key: value for key, value in _stored.items() if key != provider}
    if stored is not None:
        updated[provider] = stored
    _stored = updated
    _writes += 1


def _hosted(provider: SandboxProvider) -> None:
    if provider not in SANDBOX_KEY_ENV:
        raise SandboxCredentialRefusedError(
            f"sandbox_provider '{provider}' leases no sandboxes with a key of its own; "
            "a protocol backend authenticates each request instead."
        )


async def store_sandbox_api_key(uow: UnitOfWork, provider: SandboxProvider, api_key: str) -> SandboxCredentialStatus:
    """Store ``provider``'s key, which this worker leases with at once and every other within the TTL."""
    _hosted(provider)
    api_key = api_key.strip()
    if not api_key:
        raise SandboxCredentialRefusedError("api_key must not be blank.")
    try:
        ciphertext = encrypt_secret(api_key)
    except SecretBoxUnavailableError:
        raise SecretBoxUnavailableTenancyError("sandbox credentials") from None
    async with uow:
        await SandboxCredentialRepository(uow).put(provider.value, encrypted_api_key=ciphertext, last4=api_key[-4:])
    _publish_write(provider, _StoredKey(api_key=api_key, last4=api_key[-4:]))
    logger.info("Sandbox credential for %s stored via dashboard", provider.value)
    return sandbox_credential_status(provider)


async def clear_sandbox_api_key(uow: UnitOfWork, provider: SandboxProvider) -> SandboxCredentialStatus:
    """Delete ``provider``'s stored key, so its sandboxes are leased with the environment's again."""
    _hosted(provider)
    async with uow:
        removed = await SandboxCredentialRepository(uow).remove(provider.value)
    _publish_write(provider, None)
    if removed:
        logger.info("Sandbox credential for %s cleared via dashboard", provider.value)
    return sandbox_credential_status(provider)


async def refresh_sandbox_credential_cache(uow: UnitOfWork) -> None:
    """Reload every hosted provider's stored key and swap the cache in one step.

    A key that will not decrypt is skipped, so the provider falls back to the
    environment's key rather than leasing with a garbled one.
    """
    global _stored  # noqa: PLW0603

    writes_before = _writes
    loaded: dict[SandboxProvider, _StoredKey] = {}
    async with uow:
        repository = SandboxCredentialRepository(uow)
        for provider in SANDBOX_KEY_ENV:
            row = await repository.get_committed(provider.value)
            if row is None:
                continue
            try:
                loaded[provider] = _StoredKey(api_key=decrypt_secret(row.encrypted_api_key), last4=row.last4)
            except (SecretBoxUnavailableError, SecretDecryptionError):
                logger.warning(
                    "Ignoring the stored %s sandbox credential: it could not be decrypted (check OTARI_SECRET_KEY).",
                    provider.value,
                )
    if _writes == writes_before:
        _stored = loaded


def reset_sandbox_credential_cache() -> None:
    """Drop the cache so the next load starts clean (startup and tests)."""
    global _stored  # noqa: PLW0603

    _stored = {}


async def _refresh_on_a_session_of_its_own() -> None:
    async with create_unit_of_work() as uow:
        await refresh_sandbox_credential_cache(uow)


async def load_sandbox_credentials_at_startup(provider: SandboxProvider) -> None:
    """Prime the cache before the first request, and say when ``provider`` has no key at all.

    A failure is logged, not raised: the environment's key still serves.
    """
    reset_sandbox_credential_cache()
    try:
        await _refresh_on_a_session_of_its_own()
    except DATABASE_ERRORS as exc:
        logger.error(
            "Failed to load stored sandbox credentials; continuing with the environment's: %s",
            describe_database_error(exc),
        )
    except Exception as exc:  # noqa: BLE001 - the environment's key still serves
        logger.error(
            "Failed to load stored sandbox credentials; continuing with the environment's: %s", type(exc).__name__
        )
    if provider in SANDBOX_KEY_ENV and sandbox_api_key(provider) is None:
        logger.error(
            "sandbox_provider '%s' has no key: code execution fails until one is set on the Tools settings page "
            "or in %s.",
            provider.value,
            SANDBOX_KEY_ENV[provider],
        )


async def run_sandbox_credential_refresher(interval: float = SANDBOX_CREDENTIAL_CACHE_TTL_SECONDS) -> None:
    """Reload the cache forever, so another worker's writes arrive. Every error is retried next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            await _refresh_on_a_session_of_its_own()
        except asyncio.CancelledError:
            raise
        except DATABASE_ERRORS as exc:
            logger.warning(
                "Sandbox credential refresh failed; retrying in %ss: %s", interval, describe_database_error(exc)
            )
        except Exception as exc:  # noqa: BLE001 - retried next tick
            logger.warning("Sandbox credential refresh failed; retrying in %ss: %s", interval, type(exc).__name__)
