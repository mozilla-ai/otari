"""The dispatch-path copy of which hosted provider names are not an implementation's own.

A request reaches a hosted provider named ``openai-eu`` as ``openai-eu:gpt-4o``,
and selector resolution is synchronous and holds no session
(``services/provider_kwargs``), so the names are held in a process-wide map to
the implementation serving each, refreshed on a TTL and right after a write, as
``_provider_endpoint_cache`` is. A write reaches its own worker at once and
every other worker within ``HOSTED_NAME_INDEX_TTL_SECONDS``. A name that is its
implementation's own needs no entry: any-llm already reads it.

Only the names live here, never a credential: the credential is resolved per
request through ``ModelProviderPort``, as for any hosted provider. Beside them,
the names declared zero data retention, which a workspace requiring it is held
to at dispatch; a change to that declaration reaches every worker on the next
refresh.
"""

import asyncio

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.providers import HostedProviderRepository

HOSTED_NAME_INDEX_TTL_SECONDS = 30.0

# name -> any-llm implementation, for each hosted provider named otherwise than its implementation
_names: dict[str, str] = {}
# Every hosted provider name, own or not, whose upstream is declared to keep no data.
_zero_retention: frozenset[str] = frozenset()
# Every hosted provider name with a ``supported_countries`` list of its own.
_country_limited: frozenset[str] = frozenset()
# Writes this worker made, numbered, so a refresh whose read began before one
# replays it rather than overwriting it with an older snapshot. name ->
# (write number, implementation, or None for a name removed), and for each
# flag, name -> (write number, whether the name is in the set).
_write_count = 0
_recent_writes: dict[str, tuple[int, str | None]] = {}
_recent_retention: dict[str, tuple[int, bool]] = {}
_recent_country_limits: dict[str, tuple[int, bool]] = {}


def hosted_implementation(name: str) -> str | None:
    """The implementation serving the hosted provider ``name``, or None when ``name`` is no such provider."""
    return _names.get(name)


def hosted_zero_data_retention(name: str) -> bool:
    """Whether the hosted provider ``name`` is declared zero data retention."""
    return name in _zero_retention


def remember_hosted_zero_data_retention(name: str, declared: bool) -> None:
    """Record what this worker just stored as ``name``'s declaration, ahead of the next refresh."""
    global _zero_retention  # noqa: PLW0603

    _recent_retention[name] = (_next_write(), declared)
    _zero_retention = _zero_retention | {name} if declared else _zero_retention - {name}


def hosted_country_lists_exist() -> bool:
    """Whether any hosted provider serves only some countries, so a request's account country can matter."""
    return bool(_country_limited)


def remember_hosted_country_limit(name: str, limited: bool) -> None:
    """Record whether ``name``, as this worker just stored it, has a country list, ahead of the next refresh."""
    global _country_limited  # noqa: PLW0603

    _recent_country_limits[name] = (_next_write(), limited)
    _country_limited = _country_limited | {name} if limited else _country_limited - {name}


def _next_write() -> int:
    global _write_count  # noqa: PLW0603

    _write_count += 1
    return _write_count


def _record_write(name: str, implementation: str | None) -> None:
    _recent_writes[name] = (_next_write(), implementation)
    if implementation is None:
        _names.pop(name, None)
    else:
        _names[name] = implementation


def remember_hosted_name(name: str, implementation: str) -> None:
    """Record a hosted provider this worker just created."""
    _record_write(name, implementation if name != implementation else None)


def forget_hosted_name(name: str) -> None:
    """Drop a hosted provider this worker just removed."""
    _record_write(name, None)


async def refresh_hosted_name_index(uow: UnitOfWork, config: GatewayConfig) -> None:
    """Reload the names from the rows and the ``hosted_providers:`` block, and swap the map in one step.

    A row wins over an entry of the same name, as everywhere else. A write
    this worker made while the read was in flight is newer than what the read
    saw, so it is replayed over the snapshot rather than lost.
    """
    global _country_limited, _names, _recent_country_limits, _recent_retention, _recent_writes, _zero_retention  # noqa: PLW0603

    read_from = _write_count
    async with uow:
        rows = await HostedProviderRepository(uow).list_all()
    names = {name: entry.provider for name, entry in config.hosted_provider_entries().items()}
    names.update({row.name: row.provider for row in rows})
    entries = config.hosted_provider_entries()
    declared = {name for name, entry in entries.items() if entry.zero_data_retention}
    limited = {name for name, entry in entries.items() if entry.supported_countries is not None}
    for row in rows:
        if row.zero_data_retention:
            declared.add(row.name)
        else:
            declared.discard(row.name)
        if row.supported_countries is not None:
            limited.add(row.name)
        else:
            limited.discard(row.name)
    fresh = {name: implementation for name, implementation in names.items() if name != implementation}
    newer = {name: write for name, write in _recent_writes.items() if write[0] > read_from}
    for name, (_, implementation) in newer.items():
        if implementation is None:
            fresh.pop(name, None)
        else:
            fresh[name] = implementation
    retention = _replay(_recent_retention, read_from, declared)
    limits = _replay(_recent_country_limits, read_from, limited)
    _names, _recent_writes, _recent_retention, _recent_country_limits = fresh, newer, retention, limits
    _zero_retention, _country_limited = frozenset(declared), frozenset(limited)


def _replay(journal: dict[str, tuple[int, bool]], read_from: int, members: set[str]) -> dict[str, tuple[int, bool]]:
    """Apply the flag writes made after ``read_from`` to ``members``, and keep only those for the next refresh."""
    newer = {name: write for name, write in journal.items() if write[0] > read_from}
    for name, (_, member) in newer.items():
        if member:
            members.add(name)
        else:
            members.discard(name)
    return newer


def reset_hosted_name_index() -> None:
    """Drop the map so the next load starts clean (startup and tests)."""
    global _country_limited, _names, _recent_country_limits, _recent_retention, _recent_writes, _zero_retention  # noqa: PLW0603

    _names, _recent_writes, _recent_retention, _recent_country_limits = {}, {}, {}, {}
    _zero_retention, _country_limited = frozenset(), frozenset()


async def _refresh_on_a_session_of_its_own(config: GatewayConfig) -> None:
    async with create_unit_of_work() as uow:
        await refresh_hosted_name_index(uow, config)


async def load_hosted_names_at_startup(config: GatewayConfig) -> None:
    """Prime the map before the first request. A failure is logged, not raised."""
    reset_hosted_name_index()
    try:
        await _refresh_on_a_session_of_its_own(config)
    except DATABASE_ERRORS as exc:
        # By class only: a database error's message can carry the row it read.
        logger.error(
            "Failed to load hosted provider names (%s); a named hosted provider is unreachable until the next refresh",
            describe_database_error(exc),
        )
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
        except DATABASE_ERRORS as exc:
            logger.warning(
                "Hosted provider name refresh failed (%s); retrying in %ss", describe_database_error(exc), interval
            )
        except Exception:
            logger.warning("Hosted provider name refresh failed; retrying in %ss", interval, exc_info=True)
