"""OpenAI-compatible models listing endpoint with auto-discovery."""

import calendar
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key, verify_master_key
from gateway.api.routes.pricing import PricingTier
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, ModelPricing
from gateway.models.money import as_float
from gateway.models.routing import PolicySpec
from gateway.services.alias_service import effective_aliases
from gateway.services.model_access import is_model_allowed, resolve_request_allowlist
from gateway.services.model_catalog_service import (
    ModelCatalogEntry,
    background_catalog_enabled,
    build_metadata_map,
    load_models_dev_catalog,
)
from gateway.services.model_discovery_service import (
    background_discovery_enabled,
    discover_all_models,
    discover_models_with_status,
    get_model_cache,
)
from gateway.services.policy_store import effective_policies
from gateway.services.pricing_service import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    default_model_pricing,
    default_pricing_enabled,
    model_context_window,
    normalize_effective_at,
)
from gateway.services.provider_kwargs import normalize_pricing_key

if TYPE_CHECKING:
    from any_llm.types.model import Model

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
    cache_write_1h_price_per_million: float | None = None
    # The same tiers SetPricingRequest accepts, so the response says what the
    # request already promised. The permissive arm stays for the same reason as
    # the billing shapes on a usage row, with one addition specific to here:
    # dropping it would make Pydantic validate every stored tier against
    # PricingTier's rules on read, which is validation this field never used to
    # do, so a rule tightened later would turn old rows into a 500 on /v1/models.
    #
    # Deliberately left on pydantic's smart union, unlike the charge lines in
    # _billing_schemas, which pin `union_mode="left_to_right"`. Smart mode takes
    # the dict arm for a stored tier, which is what keeps the wire format clean.
    # Do not "make it consistent" by adding left_to_right here: PricingTier is
    # not one of the defaulted shapes, so it would write four null price keys per
    # tier, and it would start rejecting the catalog-derived tiers that
    # pricing_service._pricing_tiers emits (a `min_input_tokens` of 0 fails
    # `gt=0`, and a tier with no rate override fails validate_has_rate_override).
    pricing_tiers: Sequence[PricingTier | dict[str, float | int]] = Field(default_factory=list)


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


def _created_timestamp(model: "Model") -> int:
    """A discovered model's creation timestamp, or 0 when the provider omits one.

    ``Model.created`` is typed ``int`` but the OpenAI SDK builds response models
    without validation, so an OpenAI-compatible provider that reports ``null``
    (or a non-numeric value) reaches us as-is. Falling back to 0 matches what the
    other catalog phases publish for models with no known creation time; raising
    here would fail the whole listing over one provider's payload.
    """
    try:
        return int(model.created)
    except (TypeError, ValueError):
        return 0


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
    discovery_unsupported: bool = Field(
        default=False,
        description=(
            "True when discovery failed only because this backend serves no model-listing endpoint. "
            "The provider may still handle requests for models declared in config."
        ),
    )
    checked_at: str | None = Field(
        default=None,
        description=(
            "When this instance was last dialed, ISO 8601. Null when it has not been checked yet, "
            "which is what the first read after a restart sees while the background refresh runs."
        ),
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
            input_price_per_million=float(pricing.input_price_per_million),
            output_price_per_million=float(pricing.output_price_per_million),
            cache_read_price_per_million=as_float(pricing.cache_read_price_per_million),
            cache_write_price_per_million=as_float(pricing.cache_write_price_per_million),
            cache_write_1h_price_per_million=as_float(pricing.cache_write_1h_price_per_million),
            pricing_tiers=pricing.pricing_tiers or [],
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
            input_price_per_million=float(pricing.input_price_per_million),
            output_price_per_million=float(pricing.output_price_per_million),
            cache_read_price_per_million=as_float(pricing.cache_read_price_per_million),
            cache_write_price_per_million=as_float(pricing.cache_write_price_per_million),
            cache_write_1h_price_per_million=as_float(pricing.cache_write_1h_price_per_million),
            pricing_tiers=pricing.pricing_tiers or [],
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


def _dynamic_policy_model(policy_name: str) -> ModelObject:
    """Build a ModelObject for a policy whose candidate depends on the request.

    A router-driven or condition-driven policy has no single target, so it has no
    single price either. Reporting the default candidate's price would be a guess
    that happens to be wrong whenever the policy does its job. ``pricing_source``
    already exists for exactly this kind of statement, so it carries ``"dynamic"``
    rather than inventing a second field; ``pricing`` stays null, and a client that
    does not know the new value sees "unpriced" instead of a fabricated rate.
    """
    return ModelObject(
        id=policy_name,
        created=0,
        owned_by=ALIAS_OWNED_BY,
        pricing=None,
        pricing_source="dynamic",
    )


def _policy_catalog_entries(
    config: GatewayConfig, caller_user_id: str | None, caller_workspace_id: uuid.UUID | None
) -> tuple[dict[str, str], dict[str, PolicySpec]]:
    """Split the policies in force into ``{name: target}`` and dynamic ones.

    Reads through :func:`effective_policies`, so stored policies are listed
    alongside the ones from ``config.yml`` and a caller's user-scoped policy wins,
    exactly as it does at request time. Listing only the configured ones would mean
    a policy created in the dashboard worked but was invisible in the catalog.

    A static policy is an alias in every way the catalog cares about (one name, one
    target, priced from the target), so it is folded into the alias map and listed
    that way. Its target stays in the listing though, unlike an alias's: see the
    note in :func:`list_models`. A dynamic one is listed separately by
    :func:`_dynamic_policy_model`.
    """
    static: dict[str, str] = {}
    dynamic: dict[str, PolicySpec] = {}
    for name, spec in effective_policies(config, caller_user_id, workspace_id=caller_workspace_id).items():
        if spec.is_dynamic:
            dynamic[name] = spec
        else:
            static[name] = spec.default_target
    return static, dynamic


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
            input_price_per_million=float(default.input_price_per_million),
            output_price_per_million=float(default.output_price_per_million),
            cache_read_price_per_million=as_float(default.cache_read_price_per_million),
            cache_write_price_per_million=as_float(default.cache_write_price_per_million),
            cache_write_1h_price_per_million=as_float(default.cache_write_1h_price_per_million),
            pricing_tiers=default.pricing_tiers or [],
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
    # Aliases are scoped, so the catalog is too: a caller sees their workspace's
    # aliases and the configured ones, plus their own user-scoped layer, never
    # another user's and never another workspace's. A master-key caller has no key
    # to read either off, so it sees the configured layer plus the default
    # workspace's, which is where its own writes land.
    caller_user_id = auth[0].user_id if auth[0] is not None else None
    caller_workspace_id = auth[0].workspace_id if auth[0] is not None else None
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
    # A static policy is an alias for catalog purposes (one name, one target,
    # priced from the target), so it joins the alias map and is listed the same
    # way. Startup validation refuses a policy that collides with an alias name,
    # so this merge cannot silently shadow one.
    configured_aliases = effective_aliases(config, caller_user_id, workspace_id=caller_workspace_id)
    static_policies, dynamic_policies = _policy_catalog_entries(config, caller_user_id, caller_workspace_id)
    aliases = {**configured_aliases, **static_policies}
    # Alias targets are withheld from every phase that could surface the real
    # model, discovery (phase 1) as well as pricing-only (phase 2): publishing the
    # target under either would expose the provider:model name the alias exists to
    # hide. Computed before phase 1 so discovery honors it too.
    #
    # A *policy's* candidates are not withheld, which is why only the alias map
    # feeds this. The two indirections look alike but exist for different reasons:
    # an alias is a naming device whose whole purpose is to stand in for a target,
    # while a policy decides where traffic goes among models the caller may name
    # directly anyway. Hiding its candidates cost more than it bought: one policy
    # naming a fallback chain could empty most of the catalog, a model priced by
    # the genai-prices default then vanished from the dashboard along with its
    # rate, and GET /v1/models/{key} served the same model with its price all
    # along, so nothing was actually kept off the wire.
    alias_targets = _alias_target_keys(config, configured_aliases)

    merged: dict[str, ModelObject] = {}

    # Phase 1: auto-discovered models from upstream providers.
    if config.model_discovery:
        try:
            # Cache-only when a refresher owns the dialing. This endpoint is
            # reachable with any API key, so it deliberately has no ``refresh``
            # escape hatch: forcing a fanout across every configured provider is
            # an operator action, and lives on the master-key-gated
            # /v1/models/discoverable and /v1/providers/health instead.
            discovered = await discover_all_models(
                config,
                provider_filter=provider,
                serve_stale=background_discovery_enabled(config),
            )
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
                created=_created_timestamp(model),
                owned_by=provider_name,
                pricing=ModelPricingInfo(
                    input_price_per_million=float(pricing.input_price_per_million),
                    output_price_per_million=float(pricing.output_price_per_million),
                    cache_read_price_per_million=as_float(pricing.cache_read_price_per_million),
                    cache_write_price_per_million=as_float(pricing.cache_write_price_per_million),
                    cache_write_1h_price_per_million=as_float(pricing.cache_write_1h_price_per_million),
                    pricing_tiers=pricing.pricing_tiers or [],
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
        # A gateway-run tool is priced under the reserved ``otari:`` provider (see
        # ``gateway_tool_pricing_key``). It is not a model: publishing it would put a
        # selectable entry in the OpenAI-compatible catalog whose per-request rate
        # reads as a per-million-token price, and calling it would fail.
        if model_key.startswith(f"{GATEWAY_TOOL_PRICING_PROVIDER}:"):
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
        # Phase 5: policies whose candidate is decided per request. Listed last so
        # nothing else can price them, and excluded from a ``?provider=`` filter
        # for the same reason aliases are.
        for policy_name in dynamic_policies:
            merged[policy_name] = _dynamic_policy_model(policy_name)

    # Model access control: hide models the calling key may not use, so the
    # catalog never advertises a model that would 403 at inference. Both surfaces
    # feed the SAME matcher the SAME canonical instance:model key; an alias id is a
    # display name, so it is matched on its resolved target. Master key sees all.
    api_key, is_master_key = auth
    key_allowlist = None if is_master_key else await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None:
        # A dynamic policy is listed when the key may use *any* of its candidates,
        # which is what the compiler will do at request time: it drops the ones the
        # key cannot use and serves from the rest. Hiding it unless every candidate
        # were permitted would withhold a policy the caller can in fact call.
        dynamic_reachable = {
            name: [normalize_pricing_key(config, selector) for selector in spec.static_selectors()]
            for name, spec in dynamic_policies.items()
        }

        def _permitted(model_id: str) -> bool:
            candidates = dynamic_reachable.get(model_id)
            if candidates is not None:
                return any(is_model_allowed(key_allowlist, candidate) for candidate in candidates)
            target = aliases[model_id] if model_id in aliases else model_id
            return is_model_allowed(key_allowlist, normalize_pricing_key(config, target))

        merged = {mid: obj for mid, obj in merged.items() if _permitted(mid)}

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
    refresh: Annotated[
        bool,
        Query(description="Re-dial every provider instead of answering from the discovery cache."),
    ] = False,
) -> DiscoverableModelsResponse:
    """List every model the configured provider credentials can reach.

    Operator-facing counterpart to GET /v1/models, which serves a curated catalog
    to API callers. This reports each provider separately and keeps its error, so
    a provider with a bad key is distinguishable from one with no models. It is
    master-key gated because a provider error message describes the gateway's own
    configuration.

    Answers from the discovery cache, which a background refresher keeps warm, so
    the call does not wait on a slow or unreachable provider. Each provider
    carries the ``checked_at`` its result was produced at; a null one has not been
    dialed yet. Pass ``refresh=true`` to force a live re-dial of every provider.
    """
    serve_stale = background_discovery_enabled(config) and not refresh
    discoveries = await discover_models_with_status(config, serve_stale=serve_stale, force=refresh)
    cache = get_model_cache()
    providers = [
        DiscoverableProvider(
            provider=discovery.provider,
            ok=discovery.error is None,
            error=discovery.error,
            discovery_unsupported=discovery.discovery_unsupported,
            checked_at=checked.isoformat() if (checked := cache.checked_at(discovery.provider)) else None,
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

    Answers from the cached catalog, kept warm by a background refresher, so the
    dashboard never waits on the models.dev fetch timeout.
    """
    catalog = await load_models_dev_catalog(config, serve_stale=background_catalog_enabled(config))
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
    api_key, is_master_key = auth
    # Same scoping as the listing: the caller's own aliases, plus their
    # workspace's and the configured ones. A master-key caller has neither, so it
    # reads the configured layer and the default workspace's.
    aliases = effective_aliases(
        config,
        api_key.user_id if api_key is not None else None,
        workspace_id=api_key.workspace_id if api_key is not None else None,
    )

    # Model access control: a denied model returns 404, indistinguishable from a
    # missing one, so this endpoint cannot be used to probe which models exist
    # behind an allow-list. Uses the same matcher/canonical key as the listing and
    # inference. Master key bypasses.
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

    # Check the discovery cache for this model. Parse provider from model_id
    # ("provider:model_name") for a targeted lookup instead of scanning all
    # cached providers.
    #
    # Read stale-tolerantly, matching the listing above, because nothing on the
    # request path renews ``cached_at`` any more: the refresher sleeps its
    # interval *after* each round finishes, so an entry stored at T is already
    # expired when the next round starts and stays expired until that round's
    # dials complete. A TTL-bounded peek here would 404 a model that GET
    # /v1/models is listing in the same instant, for any provider model with no
    # pricing row and no genai-prices fallback. This endpoint never dials, so
    # serving the last known answer is the only way to agree with the listing.
    discovered_model = None
    discovered_provider = None
    if config.model_discovery and ":" in model_id:
        provider_prefix, model_name = model_id.split(":", 1)
        cache = get_model_cache()
        if background_discovery_enabled(config):
            stale = cache.stale(provider_prefix)
            cached_models = stale.models if stale is not None and stale.error is None else None
        else:
            cached_models = cache.get(provider_prefix, config.model_cache_ttl_seconds)
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
            created=_created_timestamp(discovered_model),
            owned_by=discovered_provider,
            pricing=ModelPricingInfo(
                input_price_per_million=float(pricing.input_price_per_million),
                output_price_per_million=float(pricing.output_price_per_million),
                cache_read_price_per_million=as_float(pricing.cache_read_price_per_million),
                cache_write_price_per_million=as_float(pricing.cache_write_price_per_million),
                cache_write_1h_price_per_million=as_float(pricing.cache_write_1h_price_per_million),
                pricing_tiers=pricing.pricing_tiers or [],
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
