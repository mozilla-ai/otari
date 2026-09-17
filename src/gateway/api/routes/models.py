"""OpenAI-compatible models listing endpoint with auto-discovery."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    ModelProviderPortDep,
    get_config,
    get_db,
    get_session_identity,
    require_deployment_operator,
    verify_catalog_reader,
)
from gateway.core.config import GatewayConfig
from gateway.core.surface import Surface
from gateway.models.api_keys import APIKey
from gateway.models.pricing import ModelPricing
from gateway.models.tenancy import User as TenancyUser
from gateway.services.merged_catalog_service import (
    ModelObject,
    alias_model,
    apply_default_pricing,
    build_merged_catalog,
    catalog_aliases,
    catalog_scope,
    context_window_for_key,
    created_timestamp,
    get_pricing_map,
    mark_deployment_managed,
    model_from_pricing,
    normalized_pricing_lookup,
    owner_from_key,
    pricing_info,
    pricing_key_candidates,
)
from gateway.services.model_access import is_model_allowed
from gateway.services.model_catalog_service import (
    ModelCatalogEntry,
    background_catalog_enabled,
    build_metadata_map,
    load_models_dev_catalog,
)
from gateway.services.model_discovery_service import (
    background_discovery_enabled,
    discover_models_with_status,
    get_model_cache,
)
from gateway.services.provider_kwargs import normalize_pricing_key

if TYPE_CHECKING:
    pass

# Two routers under one prefix, one per authorization rule, so that adding a
# route defaults to refusing a caller who is not a deployment operator and
# opening one up to any catalog reader has to be spelled by the router it is
# declared on. The catalog reads name ``verify_catalog_reader`` a second time as
# a parameter because they use what it resolves; ``Depends`` caching means it
# still runs once per request.
operator_router = APIRouter(
    tags=["models"],
    dependencies=[Depends(require_deployment_operator)],
)
catalog_router = APIRouter(
    tags=["models"],
    dependencies=[Depends(verify_catalog_reader)],
)

SURFACE = Surface("models")


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[ModelObject]


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


@catalog_router.get("/models")
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_catalog_reader)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    model_provider: ModelProviderPortDep,
    provider: Annotated[str | None, Query(description="Filter models by provider name")] = None,
) -> ModelListResponse:
    """List all available models.

    Returns models auto-discovered from configured providers, enriched with
    pricing data from the model_pricing table when available. Models that only
    exist in the pricing table are also included for backward compatibility.
    """
    catalog = await build_merged_catalog(
        db, config, auth=auth, session_identity=session_identity, provider=provider, model_provider=model_provider
    )
    return ModelListResponse(data=sorted(catalog.models.values(), key=lambda m: m.id))


# Served before GET /models/{model_id:path}, which FastAPI would otherwise match
# first and hand "discoverable" to as a model id. The two are now on different
# routers, so what keeps this true is the order ``api/main.py`` mounts them in,
# not the order they are declared in here. The corollary is that a provider model
# literally named "discoverable" is unreachable via GET /api/v1/models/discoverable;
# that is accepted, and such a model is still listed by GET /api/v1/models.
@operator_router.get("/models/discoverable")
async def list_discoverable_models(
    config: Annotated[GatewayConfig, Depends(get_config)],
    refresh: Annotated[
        bool,
        Query(description="Re-dial every provider instead of answering from the discovery cache."),
    ] = False,
) -> DiscoverableModelsResponse:
    """List every model the configured provider credentials can reach.

    Operator-facing counterpart to GET /api/v1/models, which serves a curated catalog
    to API callers. This reports each provider separately and keeps its error, so
    a provider with a bad key is distinguishable from one with no models. It is
    operator-gated because a provider error message describes the gateway's own
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


# Ahead of GET /models/{model_id:path} for the same reason as /models/discoverable.
@operator_router.get("/models/metadata")
async def list_model_metadata(
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ModelMetadataResponse:
    """Per-model metadata for the dashboard's detail view, from models.dev.

    Covers every model models.dev lists under a configured provider, keyed by the
    ``instance:model`` selector the dashboard uses. ``available`` is false when
    enrichment is disabled (``models_dev_metadata``) or models.dev could not be
    reached; the response is then empty and the UI falls back to bundled data.
    Operator-gated: it describes the gateway's configured providers.

    Answers from the cached catalog, kept warm by a background refresher, so the
    dashboard never waits on the models.dev fetch timeout.
    """
    catalog = await load_models_dev_catalog(config, serve_stale=background_catalog_enabled(config))
    entries = build_metadata_map(config, catalog)
    return ModelMetadataResponse(
        available=catalog is not None,
        models={key: _to_metadata_schema(entry) for key, entry in entries.items()},
    )


@catalog_router.get("/models/{model_id:path}")
async def get_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    auth: Annotated[tuple[APIKey | None, bool], Depends(verify_catalog_reader)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    model_provider: ModelProviderPortDep,
) -> ModelObject:
    """Get details for a specific model."""
    api_key, _is_master_key = auth
    # Same scoping as the listing, the workspace layer included: the caller's own
    # aliases, plus their workspace's and the configured ones. A master-key caller
    # has neither, so it reads the configured layer and the default workspace's.
    scope = await catalog_scope(db, config, auth=auth, session_identity=session_identity, model_provider=model_provider)
    aliases = catalog_aliases(
        config,
        caller_user_id=api_key.user_id if api_key is not None else None,
        caller_workspace_id=api_key.workspace_id if api_key is not None else None,
        workspace_layer=scope.reads_workspace_layer,
    )

    # Model access control: a denied model returns 404, indistinguishable from a
    # missing one, so this endpoint cannot be used to probe which models exist
    # behind an allow-list. Uses the same matcher/canonical key as the listing and
    # inference. Master key bypasses.
    key_allowlist = scope.allowlist
    if key_allowlist is not None:
        target = aliases.get(model_id, model_id)
        if not is_model_allowed(key_allowlist, normalize_pricing_key(config, target)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_id}' not found")

    # An alias resolves to its own entry, with pricing read from the resolved
    # target so the underlying provider/model stays hidden. Only the target's own
    # rows are loaded rather than the whole pricing table.
    alias_target = aliases.get(model_id)
    if alias_target is not None:
        pricing_map = await get_pricing_map(db, model_keys=pricing_key_candidates(config, alias_target))
        return alias_model(config, model_id, alias_target, normalized_pricing_lookup(config, pricing_map))

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
    # /api/v1/models is listing in the same instant, for any provider model with no
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
            owned_by=owner_from_key(model_id),
            context_window=context_window_for_key(model_id),
        )
        apply_default_pricing(fallback)
        if fallback.pricing is not None:
            return mark_deployment_managed(
                config, fallback, deployment_supplied_providers=scope.deployment_supplied_providers
            )
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
            created=created_timestamp(discovered_model),
            owned_by=discovered_provider,
            pricing=pricing_info(pricing)
            if pricing
            else None,
            pricing_source="configured" if pricing else "none",
            context_window=context_window_for_key(model_key),
        )
        apply_default_pricing(obj)
        return mark_deployment_managed(config, obj, deployment_supplied_providers=scope.deployment_supplied_providers)

    # Pricing-only model (no discovery data).
    assert pricing is not None
    return mark_deployment_managed(
        config, model_from_pricing(pricing), deployment_supplied_providers=scope.deployment_supplied_providers
    )
