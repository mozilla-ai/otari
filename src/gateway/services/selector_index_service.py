"""Building the spelling index from the deployment's catalog view and each organization's offerings.

``services.catalog`` holds the index and answers from it; this is
what fills it. The two are split because the answer is on the dispatch path and
must stay synchronous, while the build reads the database, models.dev and
discovery.

The offering seed helpers live here too: a seed is what the merged catalog's
selectors are folded by, and the index build and the grouped catalog fold them
the same way.
"""

import asyncio
import uuid
from collections.abc import Callable, Iterable, Mapping, Set
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work_with_session
from gateway.log_config import logger
from gateway.ports.model_provider_port import ModelProviderPort
from gateway.services.catalog import (
    Identities,
    OfferingRow,
    OrganizationSelectors,
    build_organization_selectors,
    build_selector_index,
    metadata_entry,
    resolve_catalog_offerings,
    set_selector_index,
)
from gateway.services.merged_catalog_service import (
    ALIAS_OWNED_BY,
    MergedCatalog,
    ModelObject,
    build_merged_catalog,
)
from gateway.services.model_catalog_service import ModelCatalogEntry
from gateway.services.model_identity import ModelIdentity, OfferingSeed, group_offerings, vendor_slug
from gateway.services.pricing_service import (
    OverridePeriod,
    default_model_pricing,
    default_pricing_enabled,
    load_organization_override_index,
    normalize_effective_at,
    pricing_key_forms,
    resolve_organization_override,
)
from gateway.services.provider_kwargs import normalize_pricing_key
from gateway.services.tenancy.organization_model_access import (
    resolve_all_organizations_offered_keys,
    workspace_organizations,
)


def split_offering_selector(obj: ModelObject) -> tuple[str, str]:
    """The instance and the provider's own model id a merged selector names."""
    instance, separator, model_id = obj.id.partition(":")
    if not separator:
        return obj.owned_by, obj.id
    return instance, model_id


def real_offerings(merged: MergedCatalog) -> list[ModelObject]:
    """The selectors themselves, sorted so grouping is order-stable.

    Aliases and policies are names over selectors and live on Routing.
    """
    return sorted(
        (obj for obj in merged.models.values() if obj.owned_by != ALIAS_OWNED_BY and obj.pricing_source != "dynamic"),
        key=lambda obj: obj.id,
    )


def seed_for(
    config: GatewayConfig, catalog: dict[str, Any] | None, selector: str, instance: str, model_id: str
) -> tuple[OfferingSeed, ModelCatalogEntry | None, str]:
    """One selector's identity seed, its models.dev entry, and the provider type behind the instance."""
    provider_type = config.provider_instance_type(instance)
    metadata = metadata_entry(catalog, provider_type, model_id)
    seed = OfferingSeed(
        selector=selector,
        provider_type=provider_type,
        model_id=model_id,
        name=metadata.name if metadata else None,
        canonical_id=metadata.canonical_model_id if metadata else None,
    )
    return seed, metadata, provider_type


def offering_seed(
    config: GatewayConfig, catalog: dict[str, Any] | None, obj: ModelObject
) -> tuple[OfferingSeed, ModelCatalogEntry | None, str, str, str]:
    """One offering's identity seed, with the facts the seed was read from."""
    instance, model_id = split_offering_selector(obj)
    seed, metadata, provider_type = seed_for(config, catalog, obj.id, instance, model_id)
    return seed, metadata, instance, model_id, provider_type


def answered_by_listing(model_id: str, aliases: Mapping[str, str]) -> bool:
    """Whether a single-model read answers ``model_id`` from the per-model listing.

    A catalog id or a routing policy is a row only the folded listing holds. A
    selector names one offering and an alias resolves to its own entry, so
    neither needs the whole catalog built.
    """
    return ":" not in model_id and model_id not in aliases


def models_by_identity(
    config: GatewayConfig,
    catalog: dict[str, Any] | None,
    listed: Iterable[ModelObject],
    *,
    organization_id: uuid.UUID | None = None,
    alias_targets: Set[str] = frozenset(),
) -> list[ModelObject]:
    """Fold a flat listing into one row per model, keyed by catalog id, sorted by id.

    Each model's row describes the offering a request sent by that id tries
    first (:func:`resolve_catalog_offerings`), among the offerings in
    ``listed``; while the index does not know the model yet, the cheapest
    priced one. Aliases and policies are names rather than offerings and pass
    through, and a model whose id one of them already takes is left to it.

    An offering whose canonical key is in ``alias_targets`` still ranks, since
    the catalog id reaches it, but a model offered only by alias targets gets
    no row: the alias stands in for it.
    """
    names: dict[str, ModelObject] = {}
    offerings: dict[str, ModelObject] = {}
    for obj in listed:
        if obj.owned_by == ALIAS_OWNED_BY or obj.pricing_source == "dynamic":
            names[obj.id] = obj
        else:
            offerings[obj.id] = obj
    seeds = [offering_seed(config, catalog, offerings[selector])[0] for selector in sorted(offerings)]
    rows = dict(names)
    for identity in group_offerings(seeds).values():
        if identity.id in rows:
            continue
        members = set(identity.selectors)
        if all(normalize_pricing_key(config, selector) in alias_targets for selector in members):
            continue
        ranked = [
            selector
            for selector in resolve_catalog_offerings(identity.id, organization_id=organization_id)
            if selector in members
        ] or sorted(members, key=lambda selector: _cheapest_first(offerings[selector]))
        first = offerings[ranked[0]]
        rows[identity.id] = first.model_copy(
            update={
                "id": identity.id,
                "owned_by": vendor_slug(identity.vendor) if identity.vendor else first.owned_by,
            }
        )
    return [rows[model_id] for model_id in sorted(rows)]


def _cheapest_first(obj: ModelObject) -> tuple[bool, float, str]:
    rate = obj.pricing.input_price_per_million if obj.pricing is not None else None
    return (rate is None, 0.0 if rate is None else rate, obj.id)


def _row(seed: OfferingSeed, instance: str, input_rate: float | None) -> OfferingRow:
    return OfferingRow(
        selector=seed.selector,
        instance=instance,
        provider_type=seed.provider_type,
        input_rate=input_rate,
    )


def _identities(grouped: Iterable[ModelIdentity]) -> Identities:
    return {identity.id: (identity.key, identity.selectors) for identity in grouped}


def _names(grouped: Iterable[ModelIdentity]) -> dict[str, str]:
    return {identity.id: identity.name for identity in grouped}


def _organization_rate(
    overrides: dict[str, list[OverridePeriod]],
    selector: str,
    deployment_rate: float | None,
    *,
    provider: str,
    model_id: str,
    as_of: datetime,
) -> float | None:
    """What this organization pays for one selector: its override, then the deployment's rate, then the default."""
    override = resolve_organization_override(overrides, pricing_key_forms(selector), as_of)
    if override is not None:
        return float(override.input_price_per_million)
    if deployment_rate is not None:
        return deployment_rate
    if not default_pricing_enabled():
        return None
    default = default_model_pricing(provider, model_id, as_of)
    return float(default.input_price_per_million) if default is not None else None


async def _organization_views(
    db: AsyncSession,
    config: GatewayConfig,
    catalog: dict[str, Any] | None,
    rows: list[OfferingRow],
    seeds: list[OfferingSeed],
) -> dict[uuid.UUID, OrganizationSelectors]:
    """One view per organization that offers a model on its own key.

    The organization's selectors join the deployment's before grouping, so a
    model it reaches on its own key and a model the deployment serves fold
    into one identity, and every row in a touched identity is re-rated at the
    organization's price before the cheapest is picked.
    """
    offered = await resolve_all_organizations_offered_keys(db)
    if not offered:
        return {}
    as_of = normalize_effective_at(None)
    deployment_rates = {row.selector: row.input_rate for row in rows}
    deployment_full = frozenset(deployment_rates)
    views: dict[uuid.UUID, OrganizationSelectors] = {}
    for organization_id, keys in offered.items():
        own_seeds: list[tuple[OfferingSeed, str]] = []
        for selector in sorted(keys - deployment_full):
            instance, _separator, model_id = selector.partition(":")
            seed, _metadata, _provider_type = seed_for(config, catalog, selector, instance, model_id)
            own_seeds.append((seed, instance))
        union = group_offerings([*seeds, *(seed for seed, _instance in own_seeds)])
        touched = {identity for identity in union.values() if any(selector in keys for selector in identity.selectors)}
        touched_selectors = {selector for identity in touched for selector in identity.selectors}
        overrides = await load_organization_override_index(
            db, organization_id, (key for selector in touched_selectors for key in pricing_key_forms(selector))
        )

        def rate(selector: str, instance: str, model_id: str) -> float | None:
            if selector not in touched_selectors:
                return deployment_rates.get(selector)
            return _organization_rate(
                overrides,
                selector,
                deployment_rates.get(selector),
                provider=instance,
                model_id=model_id,
                as_of=as_of,
            )

        deployment_rows = [
            OfferingRow(
                selector=row.selector,
                instance=row.instance,
                provider_type=row.provider_type,
                input_rate=rate(row.selector, row.instance, row.selector.partition(":")[2]),
            )
            for row in rows
        ]
        organization_rows = [
            _row(seed, instance, rate(seed.selector, instance, seed.model_id)) for seed, instance in own_seeds
        ]
        views[organization_id] = build_organization_selectors(
            deployment_rows, organization_rows, _identities(union.values()), _names(union.values())
        )
    return views


async def rebuild_selector_index(
    db: AsyncSession, config: GatewayConfig, *, model_provider: ModelProviderPort | None = None, fetch: bool = False
) -> None:
    """Index the spellings from the deployment's catalog view and each organization's offerings.

    The deployment's view is the master key's, which is every configured
    instance priced from the deployment's list and the defaults, with no
    organization's own offerings in it: what a catalog id resolves to for
    callers with no key of their own must not depend on who asks. Each
    organization that offers models on its own keys then gets a view of its
    own, so its callers reach those models by the same spellings and nobody
    else's do.

    ``model_provider`` is the hosted port, so a hosted model the deployment no
    longer advertises is not indexed as the offering a short spelling lands on.

    ``fetch`` lets a request-time rebuild pull models.dev the way a page load
    does. The scheduled rebuild reads the cache as it stands instead: fetching
    from a background task would bind the catalog's fetch lock to that task's
    loop, and the cache is warm within a tick of any page load anyway. Discovery
    is read the same way, and for a second reason: dialing here would fan out to
    every configured provider on a timer the operator never asked for, and would
    dial even while ``model_cache_ttl_seconds`` is 0, whose whole meaning is that
    the reads do their own dialing.
    """
    merged = await build_merged_catalog(
        db,
        config,
        auth=(None, True),
        session_identity=None,
        cached_only=not fetch,
        model_provider=model_provider,
        include_offered=False,
        # Indexed by real offering, so an aliased model keeps its selectors.
        withhold_alias_targets=False,
    )
    catalog = merged.models_dev
    rows: list[OfferingRow] = []
    seeds: list[OfferingSeed] = []
    for obj in real_offerings(merged):
        seed, _metadata, instance, _model_id, _provider_type = offering_seed(config, catalog, obj)
        seeds.append(seed)
        rows.append(_row(seed, instance, obj.pricing.input_price_per_million if obj.pricing is not None else None))
    grouped = group_offerings(seeds).values()
    identities = _identities(grouped)
    organizations = await _organization_views(db, config, catalog, rows, seeds)
    set_selector_index(
        build_selector_index(
            rows,
            identities,
            organizations=organizations,
            workspace_organization=await workspace_organizations(db) if organizations else {},
            names=_names(grouped),
        )
    )


SELECTOR_INDEX_INTERVAL_SECONDS = 60.0


async def run_selector_index_refresher(
    config: GatewayConfig,
    resolve_model_provider: Callable[[AsyncSession, UnitOfWork], ModelProviderPort] | None = None,
    interval: float | None = None,
) -> None:
    """Keep the selector index current with discovery, pricing and providers.

    Rebuilt on a short fixed interval rather than hooked into every writer that
    could change it (a price set, a provider added, a discovery tick): the
    build is one catalog read, and a spelling that lags a minute behind a
    change is a far smaller hazard than one hook missed.

    ``resolve_model_provider`` hands each tick the hosted port for its session
    and Unit of Work, the way a request resolves it; the composition root is
    not named here.
    """
    while True:
        try:
            async with create_unit_of_work_with_session() as (session, uow):
                model_provider = None if resolve_model_provider is None else resolve_model_provider(session, uow)
                await rebuild_selector_index(session, config, model_provider=model_provider)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("catalog selector index rebuild failed; retrying on the next tick", exc_info=True)
        await asyncio.sleep(interval if interval is not None else SELECTOR_INDEX_INTERVAL_SECONDS)
