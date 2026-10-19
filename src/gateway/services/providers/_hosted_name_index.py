"""The dispatch-path copy of which hosted provider names are not an implementation's own.

A request reaches a hosted provider named ``openai-eu`` as ``openai-eu:gpt-4o``,
and selector resolution is synchronous and holds no session
(``services/provider_kwargs``), so the names are held in a process-wide map to
the implementation serving each, refreshed on a TTL and right after a write, as
``_provider_endpoint_cache`` is. A write reaches its own worker at once and
every other worker within ``HOSTED_NAME_INDEX_TTL_SECONDS``. A name that is its
implementation's own needs no entry: any-llm already reads it.

Only the names live here, never a credential: the credential is resolved per
request through ``ModelProviderPort``, as for any hosted provider.
"""

import asyncio

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.providers import HostedProviderRepository

HOSTED_NAME_INDEX_TTL_SECONDS = 30.0

# name -> any-llm implementation, for each hosted provider named otherwise than its implementation
_names: dict[str, str] = {}


def hosted_implementation(name: str) -> str | None:
    """The implementation serving the hosted provider ``name``, or None when ``name`` is no such provider."""
    return _names.get(name)


def remember_hosted_name(name: str, implementation: str) -> None:
    """Record a hosted provider this worker just created."""
    if name != implementation:
        _names[name] = implementation


def forget_hosted_name(name: str) -> None:
    """Drop a hosted provider this worker just removed."""
    _names.pop(name, None)


async def refresh_hosted_name_index(uow: UnitOfWork, config: GatewayConfig) -> None:
    """Reload the names from the rows and the ``hosted_providers:`` block, and swap the map in one step.

    A row wins over an entry of the same name, as everywhere else.
    """
    global _names  # noqa: PLW0603

    async with uow:
        rows = await HostedProviderRepository(uow).list_all()
    names = {name: entry.provider for name, entry in config.hosted_provider_entries().items()}
    names.update({row.name: row.provider for row in rows})
    _names = {name: implementation for name, implementation in names.items() if name != implementation}


def reset_hosted_name_index() -> None:
    """Drop the map so the next load starts clean (startup and tests)."""
    global _names  # noqa: PLW0603

    _names = {}


async def _refresh_on_a_session_of_its_own(config: GatewayConfig) -> None:
    async with create_unit_of_work() as uow:
        await refresh_hosted_name_index(uow, config)


async def load_hosted_names_at_startup(config: GatewayConfig) -> None:
    """Prime the map before the first request. A failure is logged, not raised."""
    reset_hosted_name_index()
    try:
        await _refresh_on_a_session_of_its_own(config)
    except Exception:
        logger.exception("Failed to load hosted provider names; a named hosted provider is unreachable until then")


async def run_hosted_name_refresher(config: GatewayConfig, interval: float = HOSTED_NAME_INDEX_TTL_SECONDS) -> None:
    """Reload the map forever, so another worker's writes arrive. Every error is retried next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            await _refresh_on_a_session_of_its_own(config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Hosted provider name refresh failed; retrying in %ss", interval, exc_info=True)
