"""Building the short-spelling index from the deployment's own catalog view.

``services.catalog_selectors`` holds the index and answers from it; this is
what fills it. The two are split because the answer is on the dispatch path and
must stay synchronous, while the build reads the database, models.dev and
discovery.

The offering seed helpers live here too: a seed is what the merged catalog's
selectors are folded by, and the index build and the grouped catalog fold them
the same way.
"""

import asyncio
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.services.catalog_selectors import OfferingRow, build_selector_index, set_selector_index
from gateway.services.merged_catalog_service import (
    ALIAS_OWNED_BY,
    MergedCatalog,
    ModelObject,
    build_merged_catalog,
)
from gateway.services.model_catalog_service import (
    ModelCatalogEntry,
    background_catalog_enabled,
    cached_models_dev_catalog,
    load_models_dev_catalog,
    models_dev_provider_id,
    parse_entry,
)
from gateway.services.model_identity import OfferingSeed, clean_model_id, group_offerings, slugify


def split_offering_selector(obj: ModelObject) -> tuple[str, str]:
    """The instance and the provider's own model id a merged selector names."""
    instance, separator, model_id = obj.id.partition(":")
    if not separator:
        return obj.owned_by, obj.id
    return instance, model_id


def metadata_entry(catalog: dict[str, Any] | None, provider_type: str, model_id: str) -> ModelCatalogEntry | None:
    """models.dev's entry for a model under a provider type, or None.

    Looked up directly rather than through ``build_metadata_map`` so an
    organization's key, which is not a ``providers:`` instance, is enriched too.
    """
    if not catalog:
        return None
    provider = catalog.get(models_dev_provider_id(provider_type))
    if not isinstance(provider, dict):
        return None
    models = provider.get("models")
    if not isinstance(models, dict):
        return None
    model = models.get(model_id)
    return parse_entry(model) if isinstance(model, dict) else None


def real_offerings(merged: MergedCatalog) -> list[ModelObject]:
    """The selectors themselves, sorted so grouping is order-stable.

    Aliases and policies are names over selectors and live on Routing.
    """
    return sorted(
        (obj for obj in merged.models.values() if obj.owned_by != ALIAS_OWNED_BY and obj.pricing_source != "dynamic"),
        key=lambda obj: obj.id,
    )


def offering_seed(
    config: GatewayConfig, catalog: dict[str, Any] | None, obj: ModelObject
) -> tuple[OfferingSeed, ModelCatalogEntry | None, str, str, str]:
    """One offering's identity seed, with the facts the seed was read from."""
    instance, model_id = split_offering_selector(obj)
    provider_type = config.provider_instance_type(instance)
    metadata = metadata_entry(catalog, provider_type, model_id)
    seed = OfferingSeed(
        selector=obj.id,
        provider_type=provider_type,
        model_id=model_id,
        name=metadata.name if metadata else None,
    )
    return seed, metadata, instance, model_id, provider_type


async def rebuild_selector_index(db: AsyncSession, config: GatewayConfig, *, fetch: bool = False) -> None:
    """Index the short spellings from the deployment's own catalog view.

    The master key's view, which is every configured instance priced from the
    deployment's list and the defaults: what a bare slug resolves to must not
    depend on who asks, since the same request from two keys should reach the
    same offering.

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
        db, config, auth=(None, True), session_identity=None, cached_only=not fetch, model_provider=None
    )
    catalog = (
        await load_models_dev_catalog(config, serve_stale=background_catalog_enabled(config))
        if fetch
        else cached_models_dev_catalog(config)
    )
    rows: list[OfferingRow] = []
    seeds: list[OfferingSeed] = []
    for obj in real_offerings(merged):
        seed, _metadata, instance, model_id, provider_type = offering_seed(config, catalog, obj)
        seeds.append(seed)
        rows.append(
            OfferingRow(
                selector=obj.id,
                instance=instance,
                provider_type=provider_type,
                # Spelled the way the catalog spells a name, so the short
                # selector and the model id agree: `fireworks:glm-5.3-flash`.
                cleaned_id=slugify(clean_model_id(model_id).model),
                input_rate=obj.pricing.input_price_per_million if obj.pricing is not None else None,
            )
        )
    identities = {identity.id: (identity.key, identity.selectors) for identity in group_offerings(seeds).values()}
    set_selector_index(build_selector_index(rows, identities))


SELECTOR_INDEX_INTERVAL_SECONDS = 60.0


async def run_selector_index_refresher(config: GatewayConfig, interval: float | None = None) -> None:
    """Keep the selector index current with discovery, pricing and providers.

    Rebuilt on a short fixed interval rather than hooked into every writer that
    could change it (a price set, a provider added, a discovery tick): the
    build is one catalog read, and a spelling that lags a minute behind a
    change is a far smaller hazard than one hook missed.
    """
    while True:
        try:
            async with create_session() as session:
                await rebuild_selector_index(session, config)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("catalog selector index rebuild failed; retrying on the next tick", exc_info=True)
        await asyncio.sleep(interval if interval is not None else SELECTOR_INDEX_INTERVAL_SECONDS)
