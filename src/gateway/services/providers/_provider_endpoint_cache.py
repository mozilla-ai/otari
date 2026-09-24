"""The dispatch-path copy of the provider endpoints workspaces and users own.

Selector resolution is synchronous and holds no session
(``services/provider_kwargs``), so endpoints are held in a process-wide cache
with their keys decrypted, refreshed on a TTL and right after a write, the same
arrangement ``services/alias_service`` uses and for the same reason. A write
reaches its own worker at once and every other worker within
``PROVIDER_ENDPOINT_CACHE_TTL_SECONDS``.

The layers mirror aliases: a workspace-wide layer per workspace, and a user
layer per workspace and user that shadows it.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.models.providers import ProviderEndpoint
from gateway.repositories.providers import ProviderEndpointRepository
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError, decrypt_secret

PROVIDER_ENDPOINT_CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class OwnedEndpoint:
    """What a dispatch needs from one endpoint, its key decrypted."""

    id: uuid.UUID
    provider: str
    api_base: str
    api_key: str | None
    default_params: dict[str, Any] = field(default_factory=dict)


# workspace_id -> name -> endpoint
_shared: dict[uuid.UUID, dict[str, OwnedEndpoint]] = {}
# workspace_id -> user_id -> name -> endpoint
_per_user: dict[uuid.UUID, dict[str, dict[str, OwnedEndpoint]]] = {}


def cached_owned_endpoint(name: str, *, workspace_id: uuid.UUID, user_id: str | None) -> OwnedEndpoint | None:
    """The endpoint ``name`` reaches for this caller: their own first, then the workspace's."""
    if user_id is not None:
        own = _per_user.get(workspace_id, {}).get(user_id, {}).get(name)
        if own is not None:
            return own
    return _shared.get(workspace_id, {}).get(name)


def _to_entry(row: ProviderEndpoint) -> OwnedEndpoint:
    """Raises ``SecretBoxUnavailableError`` / ``SecretDecryptionError`` when the key will not decrypt."""
    return OwnedEndpoint(
        id=row.id,
        provider=row.provider,
        api_base=row.api_base,
        api_key=decrypt_secret(row.encrypted_api_key) if row.encrypted_api_key else None,
        default_params=dict(row.default_params or {}),
    )


async def refresh_provider_endpoint_cache(uow: UnitOfWork) -> None:
    """Reload every endpoint in one query and swap the cache in one step.

    A row whose key will not decrypt is skipped, so its name stops resolving
    rather than dispatching without the key.
    """
    global _shared, _per_user  # noqa: PLW0603

    async with uow:
        rows = await ProviderEndpointRepository(uow).list_all()

    shared: dict[uuid.UUID, dict[str, OwnedEndpoint]] = {}
    per_user: dict[uuid.UUID, dict[str, dict[str, OwnedEndpoint]]] = {}
    for row in rows:
        try:
            entry = _to_entry(row)
        except (SecretBoxUnavailableError, SecretDecryptionError):
            logger.warning(
                "Skipping provider endpoint %s: its API key could not be decrypted (check OTARI_SECRET_KEY).",
                row.id,
            )
            continue
        if row.user_id is None:
            shared.setdefault(row.workspace_id, {})[row.name] = entry
        else:
            per_user.setdefault(row.workspace_id, {}).setdefault(row.user_id, {})[row.name] = entry

    _shared, _per_user = shared, per_user


def reset_provider_endpoint_cache() -> None:
    """Drop the cache so the next load starts clean (startup and tests)."""
    global _shared, _per_user  # noqa: PLW0603

    _shared, _per_user = {}, {}


async def _refresh_on_a_session_of_its_own() -> None:
    async with create_unit_of_work() as uow:
        await refresh_provider_endpoint_cache(uow)


async def load_provider_endpoints_at_startup() -> None:
    """Prime the cache before the first request. A failure is logged, not raised."""
    reset_provider_endpoint_cache()
    try:
        await _refresh_on_a_session_of_its_own()
    except Exception:
        logger.exception("Failed to load provider endpoints; continuing with none cached")


async def run_provider_endpoint_refresher(interval: float = PROVIDER_ENDPOINT_CACHE_TTL_SECONDS) -> None:
    """Reload the cache forever, so another worker's writes arrive. Every error is retried next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            await _refresh_on_a_session_of_its_own()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Provider endpoint refresh failed; retrying in %ss", interval, exc_info=True)
