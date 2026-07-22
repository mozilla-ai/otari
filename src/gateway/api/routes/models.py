"""OpenAI-compatible models listing endpoint with auto-discovery."""

import calendar
from collections.abc import Sequence
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, ModelPricing
from gateway.services.alias_service import effective_aliases
from gateway.services.model_access import is_model_allowed, resolve_request_allowlist
from gateway.services.model_catalog_service import (
    ModelCatalogEntry,
    build_metadata_map,
    load_models_dev_catalog,
)
from gateway.services.model_discovery_service import (
    discover_all_models,
    discover_models_with_status,
    get_model_cache,
)
from gateway.services.pricing_service import (
    default_model_pricing,
    default_pricing_enabled,
    model_context_window,
    normalize_effective_at,
)
from gateway.services.provider_kwargs import normalize_pricing_key

router = APIRouter(prefix="/v1", tags=["models"])

# ``owned_by`` reported for alias entries. Aliases intentionally hide the
# underlying provider, so the gateway itself is named as the owner rather than
# the real upstream.
ALIAS_OWNED_BY = "otari"


class ModelPricingInfo(BaseModel):
    """Pricing information for a model."""

    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float | None = None
    cache_write_price_per_million: float | None = None


class ModelObject(BaseModel):
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    pricing: ModelPricingInfo | None = None
    # Where ``pricing`` came from: "configured" (DB), "default" (genai-prices
    # fallback, only when default_pricing is enabled), or "none".
    pricing_source: str = "none"
    # Context-window token limit from the bundled genai-prices dataset, when it
    # knows the model. Metadata only (independent of the default_pricing toggle);
    # ``None`` when the dataset has no value for the model.
    context_window: int | None = None


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[ModelObject]


def _owner_from_key(model_key: str) -> str:
    """The provider a ``provider:model`` key names, or "unknown" for a bare name."""
    provider, separator, _ = model_key.partition(":")
    return provider if separator else "unknown"


def _context_window_for_key(model_key: str) -> int | None:
    """genai-prices context-window for a ``provider:model`` (or bare) key.

    Metadata, so it is filled whether or not the default-pricing fallback is on;
    ``None`` when the dataset does not know the model or lists no window for it.
    """
    provider_part, separator, model_part = model_key.partition(":")
    provider = provider_part if separator else None
    model_name = model_part if separator else model_key
    return model_context_window(provider, model_name)


class DiscoverableModel(BaseModel):
    """A model one provider instance reports as available."""

    id: str = Field(description="Bare model id as the provider reports it.")
    key: str = Field(description="Selector to send as `model`, in `instance:model` form.")


class DiscoverableProvider(BaseModel):
    """One provider instance's discovery result."""

    provider: str
    ok: bool = Field(description="False when this instance could not be queried.")
    error: str | None = Field(
        default=None,
        description="Why discovery failed. Null when `ok` is true.",
    )
    models: list[DiscoverableModel]


class DiscoverableModelsResponse(BaseModel):
    """Per-provider discovery results for operator model selection."""

    providers: list[DiscoverableProvider]


class ModelMetadata(BaseModel):
    """models.dev metadata for one model, for the dashboard's detail view."""

    name: str | None = None
    description: str | None = None
    family: str | None = None
    input_modalities: list[str] = Field(default_factory=list)
    output_modalities: list[str] = Field(default_factory=list)
    reasoning: bool = False
    tool_call: bool = False
    structured_output: bool = False
    attachment: bool = False
    temperature: bool = False
    context_window: int | None = None
    max_output_tokens: int | None = None
    knowledge_cutoff: str | None = None
    release_date: str | None = None
    last_updated: str | None = None
    open_weights: bool = False
    deprecated: bool = False
    cost_input: float | None = None
    cost_output: float | None = None


class ModelMetadataResponse(BaseModel):
    """models.dev metadata keyed by ``provider:model``."""

    source: str = "models.dev"
    available: bool = Field(
        description="False when metadata could not be loaded (enrichment disabled or models.dev unreachable).",
    )
    models: dict[str, ModelMetadata] = Field(default_factory=dict)


def _to_metadata_schema(entry: ModelCatalogEntry) -> ModelMetadata:
    return ModelMetadata(
        name=entry.name,
        description=entry.description,
        family=entry.family,
        input_modalities=entry.input_modalities,
        output_modalities=entry.output_modalities,
        reasoning=entry.reasoning,
        tool_call=entry.tool_call,
        structured_output=entry.structured_output,
        attachment=entry.attachment,
        temperature=entry.temperature,
        context_window=entry.context_window,
        max_output_tokens=entry.max_output_tokens,
        knowledge_cutoff=entry.knowledge_cutoff,
        release_date=entry.release_date,
        last_updated=entry.last_updated,
        open_weights=entry.open_weights,
        deprecated=entry.deprecated,
        cost_input=entry.cost_input,
        cost_output=entry.cost_output,
    )


def _model_from_pricing(pricing: ModelPricing) -> ModelObject:
    """Convert a ModelPricing row to an OpenAI-compatible ModelObject."""
    created = int(calendar.timegm(pricing.created_at.utctimetuple()))
    return ModelObject(
        id=pricing.model_key,
        created=created,
        owned_by=_owner_from_key(pricing.model_key),
        pricing=ModelPricingInfo(
            input_price_per_million=pricing.input_price_per_million,
            output_price_per_million=pricing.output_price_per_million,
            cache_read_price_per_million=pricing.cache_read_price_per_million,
            cache_write_price_per_million=pricing.cache_write_price_per_million,
        ),
        pricing_source="configured",
        context_window=_context_window_for_key(pricing.model_key),
    )


def _alias_model(
    config: GatewayConfig, alias: str, target: str, pricing_lookup: dict[str, ModelPricing]
) -> ModelObject:
    """Build a ModelObject for an alias, from config.yml or from storage.

    The alias id is what the caller sees; pricing is looked up from the resolved
    target's canonical key so an alias shows the real model's price without
    revealing the provider/model behind it. ``pricing_source`` describes where
    that price came from, just as for a real model; the alias itself is
    identified by ``owned_by``.
    """
    canonical_target = normalize_pricing_key(config, target)
    pricing = pricing_lookup.get(canonical_target)
    obj = ModelObject(
        id=alias,
        created=0,
        owned_by=ALIAS_OWNED_BY,
        pricing=ModelPricingInfo(
            input_price_per_million=pricing.input_price_per_million,
            output_price_per_million=pricing.output_price_per_million,
            cache_read_price_per_million=pricing.cache_read_price_per_million,
            cache_write_price_per_million=pricing.cache_write_price_per_million,
        )
        if pricing
        else None,
        pricing_source="configured" if pricing else "none",
        # From the resolved target, like pricing: an alias's display name is not a
        # model the dataset knows. Exposing the window does not reveal the target.
        context_window=_context_window_for_key(canonical_target),
    )
    # Priced from the target, never from the alias's display name: the fallback
    # keys on a real provider/model, and the display name is neither. Without
    # this an alias to an unpriced model would report no price while the gateway
    # billed it at the default rate.
    _apply_default_pricing(obj, pricing_selector=canonical_target)
    return obj


def _alias_target_keys(config: GatewayConfig, aliases: dict[str, str]) -> set[str]:
    """Canonical pricing keys of every alias target."""
    return {normalize_pricing_key(config, target) for target in aliases.values()}


def _pricing_key_candidates(config: GatewayConfig, target: str) -> list[str]:
    """Stored key forms that could hold pricing for ``target``.

    Keys are canonicalized on write, but rows predating that may still use the
    legacy ``provider/model`` separator, so both forms are offered.
    """
    canonical = normalize_pricing_key(config, target)
    return sorted({target, canonical, canonical.replace(":", "/", 1)})


def _normalized_pricing_lookup(config: GatewayConfig, pricing_map: dict[str, ModelPricing]) -> dict[str, ModelPricing]:
    """Re-key a pricing map by canonical model key, matching how targets resolve."""
    return {normalize_pricing_key(config, key): row for key, row in pricing_map.items()}


def _apply_default_pricing(obj: ModelObject, pricing_selector: str | None = None) -> None:
    """Fill the genai-prices default rate for a model that has no DB price.

    No-op when the fallback is disabled or the model already carries a price, so
    database pricing always takes precedence. Marks the source as "default".

    ``pricing_selector`` names the model to price when that differs from
    ``obj.id`` (an alias is priced from its target); it defaults to ``obj.id``.
    """
    if obj.pricing is not None or not default_pricing_enabled():
        return
    selector = pricing_selector if pricing_selector is not None else obj.id
    provider_part, separator, model_part = selector.partition(":")
    provider = provider_part if separator else None
    model_name = model_part if separator else selector
    default = default_model_pricing(provider, model_name, normalize_effective_at(None))
    if default is not None:
        obj.pricing = ModelPricingInfo(
            input_price_per_million=default.input_price_per_million,
            output_price_per_million=default.output_price_per_million,
            cache_read_price_per_million=default.cache_read_price_per_million,
            cache_write_price_per_million=default.cache_write_price_per_million,
        )
        obj.pricing_source = "default"


async def _get_pricing_map(
    db: AsyncSession,
    provider_filter: str | None = None,
    model_keys: Sequence[str] | None = None,
) -> dict[str, ModelPricing]:
    """Load latest pricing per model_key, optionally filtered by provider prefix or key set."""
    latest_effective = (
        select(
            ModelPricing.model_key.label("model_key"),
            func.max(ModelPricing.effective_at).label("effective_at"),
        )
        .group_by(ModelPricing.model_key)
        .subquery()
    )

    stmt = select(ModelPricing).join(
        latest_effective,
        (ModelPricing.model_key == latest_effective.c.model_key)
        & (ModelPricing.effective_at == latest_effective.c.effective_at),
    )

    if provider_filter:
        stmt = stmt.where(ModelPricing.model_key.startswith(f"{provider_filter}:"))

    if model_keys is not None:
        stmt = stmt.where(ModelPricing.model_key.in_(model_keys))

    stmt = stmt.order_by(ModelPricing.model_key)
    result = await db.execute(stmt)
    pricings = result.scalars().all()
    return {p.model_key: p for p in pricings}


@router.get("/models")
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    provider: Annotated[str | None, Query(description="Filter models by provider name")] = None,
) -> ModelListResponse:
    """List all available models.

    Returns models auto-discovered from configured providers, enriched with
    pricing data from the model_pricing table when available. Models that only
    exist in the pricing table are also included for backward compatibility.
    """
    pricing_map = await _get_pricing_map(db, provider_filter=provider)
    # Snapshot before phase 1 mutates ``pricing_map`` (it pops matched keys), so
    # alias pricing can still be looked up by the target's canonical key. Keys are
    # canonicalized because an alias target is always canonical while a stored row
    # may use the legacy "provider/model" form; without this a legacy-form row
    # would be withheld from the listing by phase 2 yet never match its alias, so
    # its price would show nowhere.
    pricing_lookup = _normalized_pricing_lookup(config, pricing_map)
    # Read once: phase 2 withholds these targets and phase 3 lists the names, and
    # the two must agree even if a write lands between them.
    aliases = effective_aliases(config)
    # Alias targets are withheld from every phase that could surface the real
    # model, discovery (phase 1) as well as pricing-only (phase 2): publishing the
    # target under either would expose the provider:model name the alias exists to
    # hide. Computed before phase 1 so discovery honors it too.
    alias_targets = _alias_target_keys(config, aliases)

    merged: dict[str, ModelObject] = {}

    # Phase 1: auto-discovered models from upstream providers.
    if config.model_discovery:
        try:
            discovered = await discover_all_models(config, provider_filter=provider)
        except Exception:
            logger.exception("Model discovery failed unexpectedly")
            discovered = []

        for provider_name, model in discovered:
            model_key = f"{provider_name}:{model.id}"
            if normalize_pricing_key(config, model_key) in alias_targets:
                continue
            pricing = pricing_map.pop(model_key, None)
            merged[model_key] = ModelObject(
                id=model_key,
                created=model.created,
                owned_by=provider_name,
                pricing=ModelPricingInfo(
                    input_price_per_million=pricing.input_price_per_million,
                    output_price_per_million=pricing.output_price_per_million,
                )
                if pricing
                else None,
                pricing_source="configured" if pricing else "none",
                context_window=_context_window_for_key(model_key),
            )

    # Phase 2: pricing-only models (not discovered but have pricing entries).
    # An alias target is skipped: billing keys on the real model, so aliasing one
    # forces a pricing entry for it, and publishing that entry here would expose
    # the very name the alias exists to hide. Whether real models are listed is
    # governed by ``model_discovery`` (phase 1), never by pricing config.
    for model_key, pricing in pricing_map.items():
        if model_key in merged or normalize_pricing_key(config, model_key) in alias_targets:
            continue
        merged[model_key] = _model_from_pricing(pricing)

    # Phase 3: fill the genai-prices default for unpriced models, so the catalog
    # shows the effective rate when the fallback is active. Database pricing
    # (phases 1-2) always wins; this only touches models still without a price.
    # Runs before aliases are added: this fills from ``id``, and an alias's id is
    # a display name the fallback must never be asked to price. Aliases fill
    # their own default from the resolved target in phase 4.
    for obj in merged.values():
        _apply_default_pricing(obj)

    # Phase 4: aliases, from config.yml and from storage alike. An alias is a
    # display name, not a provider, so it is only listed for the unfiltered
    # listing; a ``?provider=`` filter asks for one provider's real models and
    # must not leak the alias mapping.
    if provider is None:
        for alias, target in aliases.items():
            merged[alias] = _alias_model(config, alias, target, pricing_lookup)

    # Model access control: hide models the calling key may not use, so the
    # catalog never advertises a model that would 403 at inference. Both surfaces
    # feed the SAME matcher the SAME canonical instance:model key; an alias id is a
    # display name, so it is matched on its resolved target. Master key sees all.
    api_key, is_master_key = auth
    key_allowlist = None if is_master_key else await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None:

        def _canonical(model_id: str) -> str:
            target = aliases[model_id] if model_id in aliases else model_id
            return normalize_pricing_key(config, target)

        merged = {mid: obj for mid, obj in merged.items() if is_model_allowed(key_allowlist, _canonical(mid))}

    sorted_models = sorted(merged.values(), key=lambda m: m.id)
    return ModelListResponse(data=sorted_models)


# Declared before GET /models/{model_id:path}: FastAPI matches routes in
# registration order, so a later declaration would be shadowed and "discoverable"
# would arrive as a model id. The corollary is that a provider model literally
# named "discoverable" is unreachable via GET /v1/models/discoverable; that is
# accepted, and such a model is still listed by GET /v1/models.
@router.get("/models/discoverable", dependencies=[Depends(verify_master_key)])
async def list_discoverable_models(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> DiscoverableModelsResponse:
    """List every model the configured provider credentials can reach.

    Operator-facing counterpart to GET /v1/models, which serves a curated catalog
    to API callers. This reports each provider separately and keeps its error, so
    a provider with a bad key is distinguishable from one with no models. It is
    master-key gated because a provider error message describes the gateway's own
    configuration.
    """
    discoveries = await discover_models_with_status(config)
    providers = [
        DiscoverableProvider(
            provider=discovery.provider,
            ok=discovery.error is None,
            error=discovery.error,
            models=sorted(
                (
                    DiscoverableModel(id=model.id, key=f"{discovery.provider}:{model.id}")
                    for model in discovery.models
                ),
                key=lambda m: m.id,
            ),
        )
        for discovery in discoveries
    ]
    return DiscoverableModelsResponse(providers=sorted(providers, key=lambda p: p.provider))


# Declared before GET /models/{model_id:path} for the same route-order reason as
# /models/discoverable above.
@router.get("/models/metadata", dependencies=[Depends(verify_master_key)])
async def list_model_metadata(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ModelMetadataResponse:
    """Per-model metadata for the dashboard's detail view, from models.dev.

    Covers every model models.dev lists under a configured provider, keyed by the
    ``instance:model`` selector the dashboard uses. ``available`` is false when
    enrichment is disabled (``models_dev_metadata``) or models.dev could not be
    reached; the response is then empty and the UI falls back to bundled data.
    Master-key gated: it describes the gateway's configured providers.
    """
    catalog = await load_models_dev_catalog(config)
    entries = build_metadata_map(config, catalog)
    return ModelMetadataResponse(
        available=catalog is not None,
        models={key: _to_metadata_schema(entry) for key, entry in entries.items()},
    )


@router.get("/models/{model_id:path}")
async def get_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
) -> ModelObject:
    """Get details for a specific model."""
    aliases = effective_aliases(config)

    # Model access control: a denied model returns 404, indistinguishable from a
    # missing one, so this endpoint cannot be used to probe which models exist
    # behind an allow-list. Uses the same matcher/canonical key as the listing and
    # inference. Master key bypasses.
    api_key, is_master_key = auth
    key_allowlist = None if is_master_key else await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None:
        target = aliases.get(model_id, model_id)
        if not is_model_allowed(key_allowlist, normalize_pricing_key(config, target)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_id}' not found")

    # An alias resolves to its own entry, with pricing read from the resolved
    # target so the underlying provider/model stays hidden. Only the target's own
    # rows are loaded rather than the whole pricing table.
    alias_target = aliases.get(model_id)
    if alias_target is not None:
        pricing_map = await _get_pricing_map(db, model_keys=_pricing_key_candidates(config, alias_target))
        return _alias_model(config, model_id, alias_target, _normalized_pricing_lookup(config, pricing_map))

    # Check the pricing table first.
    stmt = (
        select(ModelPricing)
        .where(ModelPricing.model_key == model_id)
        .order_by(ModelPricing.effective_at.desc())
        .limit(1)
    )
    pricing = (await db.execute(stmt)).scalar_one_or_none()

    # Check the discovery cache for this model (respecting TTL).
    # Parse provider from model_id ("provider:model_name") for a targeted lookup
    # instead of scanning all cached providers.
    discovered_model = None
    discovered_provider = None
    if config.model_discovery and ":" in model_id:
        provider_prefix, model_name = model_id.split(":", 1)
        cache = get_model_cache()
        ttl = config.model_cache_ttl_seconds
        cached_models = cache.get(provider_prefix, ttl)
        if cached_models is not None:
            for model in cached_models:
                if model.id == model_name:
                    discovered_model = model
                    discovered_provider = provider_prefix
                    break

    if not pricing and not discovered_model:
        # Neither priced nor discoverable, yet the gateway may still serve this
        # model and bill it at the genai-prices default: request-time lookup
        # consults the same fallback. Reporting 404 for a model that is being
        # charged for is the lie phase 3 of the listing exists to avoid, so
        # answer with the effective rate when there is one.
        fallback = ModelObject(
            id=model_id,
            created=0,
            owned_by=_owner_from_key(model_id),
            context_window=_context_window_for_key(model_id),
        )
        _apply_default_pricing(fallback)
        if fallback.pricing is not None:
            return fallback
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found",
        )

    # Build the response, merging both sources.
    if discovered_model:
        assert discovered_provider is not None
        model_key = f"{discovered_provider}:{discovered_model.id}"
        obj = ModelObject(
            id=model_key,
            created=discovered_model.created,
            owned_by=discovered_provider,
            pricing=ModelPricingInfo(
                input_price_per_million=pricing.input_price_per_million,
                output_price_per_million=pricing.output_price_per_million,
            )
            if pricing
            else None,
            pricing_source="configured" if pricing else "none",
            context_window=_context_window_for_key(model_key),
        )
        _apply_default_pricing(obj)
        return obj

    # Pricing-only model (no discovery data).
    assert pricing is not None
    return _model_from_pricing(pricing)
