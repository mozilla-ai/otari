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
from collections.abc import Collection, Sequence
from contextvars import ContextVar

from gateway.core.config import REGION_SEPARATOR, GatewayConfig, region_instance
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.models.providers import HostedProviderRegion
from gateway.repositories.providers import HostedProviderRegionRepository, HostedProviderRepository

HOSTED_NAME_INDEX_TTL_SECONDS = 30.0

# name -> any-llm implementation, for each hosted provider named otherwise than its implementation
_names: dict[str, str] = {}
# Every hosted provider name, own or not, whose upstream is declared to keep no data.
_zero_retention: frozenset[str] = frozenset()
# Where each hosted provider name keeps request data: for a provider, the
# declared location of each region it may serve from (None where undeclared);
# for a name pinning one region (``tokenhub@sg``), that region's alone.
_locations: dict[str, tuple[str | None, ...]] = {}
# Each region's name (``tokenhub@sg``) -> the hosted provider it is a region of.
_region_parents: dict[str, str] = {}
# Every hosted provider's own name, a region's never.
_providers: frozenset[str] = frozenset()
# Refreshes started, and the number of the newest one published, so a refresh
# whose read began earlier never replaces what a later one already published.
_refresh_count = 0
_published_refresh = 0
# Writes this worker made, numbered, so a refresh whose read began before one
# replays it rather than overwriting it with an older snapshot. name ->
# (write number, implementation, or None for a name removed), and for the
# retention flag, name -> (write number, whether the name is declared).
_write_count = 0
_recent_writes: dict[str, tuple[int, str | None]] = {}
_recent_retention: dict[str, tuple[int, bool]] = {}


def hosted_implementation(name: str) -> str | None:
    """The implementation serving the hosted provider ``name``, or None when ``name`` is no such provider."""
    return _names.get(name)


def hosted_region_parent(name: str) -> str | None:
    """The hosted provider ``name`` is a region of, or None when ``name`` is no region.

    A provider's own name is never a region, even one holding the separator,
    which names made before regions could. A region this worker has not loaded
    yet, one another worker just added, is still recognized by its provider's
    name, so its price falls back to the provider's from the first request.
    """
    if (parent := _region_parents.get(name)) is not None:
        return parent
    if name in _providers:
        return None
    base, separator, slug = name.rpartition(REGION_SEPARATOR)
    return base if separator and slug and base in _providers else None


# The data location of each hosted credential this request was dispatched on,
# by the name it settles under, as the credential declared it when chosen.
_served_locations: ContextVar[dict[str, str | None] | None] = ContextVar("hosted_served_locations", default=None)


def remember_served_location(name: str, location: str | None) -> None:
    """Record where the credential this request is dispatched on keeps data, for the usage row it writes."""
    _served_locations.set({**(_served_locations.get() or {}), name: location})


def hosted_data_location(name: str) -> str | None:
    """Where the hosted provider or region ``name`` served this request from.

    The location the serving credential declared when the request chose it;
    for a row written outside such a request (a job settled later), the one
    declared now, when that is one place.
    """
    served = _served_locations.get()
    if served is not None and name in served:
        return served[name]
    locations = _locations.get(name, ())
    return locations[0] if len(locations) == 1 else None


def hosted_provider_named(name: str) -> bool:
    """Whether a hosted provider goes by exactly ``name``, rather than a region of one."""
    return name in _providers


def serving_data_locations(
    name: str,
    provider_location: str | None,
    regions: Sequence[HostedProviderRegion],
    provider_names: Collection[str],
) -> tuple[str | None, ...]:
    """Where the hosted provider ``name`` may keep request data: one entry per place it may serve from.

    A provider with regions gives each enabled region's own declaration, never
    the provider's, leaving out a region a provider of its whole name shadows
    (it does not serve); one with none gives its own. ``None`` stands for a
    connection that declares nothing. The one place this is decided, so the
    catalog and the workspace narrowing cannot disagree.
    """
    if not regions:
        return (provider_location,)
    return tuple(location for _, location in _serving_regions(name, regions, provider_names))


def serving_region_data_locations(
    name: str, regions: Sequence[HostedProviderRegion], provider_names: Collection[str]
) -> dict[str, str | None]:
    """The declared location of each region of ``name`` a request may be pinned to, by its name (``tokenhub@sg``).

    The regions :func:`serving_data_locations` counts: enabled, and not shadowed by a provider of that whole name.
    """
    return dict(_serving_regions(name, regions, provider_names))


def _serving_regions(
    name: str, regions: Sequence[HostedProviderRegion], provider_names: Collection[str]
) -> list[tuple[str, str | None]]:
    return [
        (region_instance(name, region.slug), region.data_location)
        for region in regions
        if region.enabled and region_instance(name, region.slug) not in provider_names
    ]


class HostedRegionIndex:
    """The region lookups the composition root registers where a name is priced, re-keyed or recorded."""

    parent_of = staticmethod(hosted_region_parent)
    data_location_of = staticmethod(hosted_data_location)


def hosted_zero_data_retention(name: str) -> bool:
    """Whether the hosted provider ``name`` is declared zero data retention; a region shares its provider's."""
    if name in _zero_retention:
        return True
    parent = hosted_region_parent(name)
    return parent is not None and parent in _zero_retention


def hosted_data_locations(name: str) -> tuple[str | None, ...]:
    """Where the hosted provider (or region) ``name`` may keep request data; empty when it is unknown here."""
    return _locations.get(name, ())


def remember_hosted_zero_data_retention(name: str, declared: bool) -> None:
    """Record what this worker just stored as ``name``'s declaration, ahead of the next refresh."""
    global _zero_retention  # noqa: PLW0603

    _recent_retention[name] = (_next_write(), declared)
    _zero_retention = _zero_retention | {name} if declared else _zero_retention - {name}


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
    global _names, _recent_retention, _recent_writes, _zero_retention, _locations, _region_parents, _providers  # noqa: PLW0603
    global _refresh_count, _published_refresh  # noqa: PLW0603

    _refresh_count += 1
    number = _refresh_count
    read_from = _write_count
    async with uow:
        rows = await HostedProviderRepository(uow).list_all()
        regions = await HostedProviderRegionRepository(uow).list_for_providers([row.id for row in rows])
    entries = config.hosted_provider_entries()
    names = {name: entry.provider for name, entry in entries.items()}
    names.update({row.name: row.provider for row in rows})
    declared = {name for name, entry in entries.items() if entry.zero_data_retention}
    locations: dict[str, tuple[str | None, ...]] = {name: (entry.data_location,) for name, entry in entries.items()}
    parents: dict[str, str] = {}
    providers = frozenset(entries) | frozenset(row.name for row in rows)
    for row in rows:
        if row.zero_data_retention:
            declared.add(row.name)
        else:
            declared.discard(row.name)
        # A region a provider of its whole name shadows does not serve, so it is not where data may go.
        locations[row.name] = serving_data_locations(row.name, row.data_location, regions.get(row.id, []), providers)
    for row in rows:
        for region in regions.get(row.id, []):
            instance = region_instance(row.name, region.slug)
            if instance in providers:
                # A provider of that whole name wins, as it does at dispatch.
                continue
            # A region name is never its implementation's, so it always needs an entry.
            names[instance] = row.provider
            locations[instance] = (region.data_location,)
            parents[instance] = row.name
    if number < _published_refresh:
        return
    fresh = {name: implementation for name, implementation in names.items() if name != implementation}
    newer = {name: write for name, write in _recent_writes.items() if write[0] > read_from}
    for name, (_, implementation) in newer.items():
        if implementation is None:
            fresh.pop(name, None)
        else:
            fresh[name] = implementation
    retention = _replay(_recent_retention, read_from, declared)
    _names, _recent_writes, _recent_retention, _zero_retention = fresh, newer, retention, frozenset(declared)
    _locations, _region_parents, _providers, _published_refresh = locations, parents, providers, number


def _replay(journal: dict[str, tuple[int, bool]], read_from: int, members: set[str]) -> dict[str, tuple[int, bool]]:
    """Apply the retention writes made after ``read_from`` to ``members``, and keep only those for the next refresh."""
    newer = {name: write for name, write in journal.items() if write[0] > read_from}
    for name, (_, member) in newer.items():
        if member:
            members.add(name)
        else:
            members.discard(name)
    return newer


def reset_hosted_name_index() -> None:
    """Drop the map so the next load starts clean (startup and tests)."""
    global _names, _recent_retention, _recent_writes, _zero_retention, _locations, _region_parents, _providers  # noqa: PLW0603
    global _published_refresh  # noqa: PLW0603

    _names, _recent_writes, _recent_retention, _zero_retention, _locations = {}, {}, {}, frozenset(), {}
    _region_parents, _providers, _published_refresh = {}, frozenset(), 0


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
