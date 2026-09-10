"""The catalog grouped by model: one entry per model, one offering per selector.

``GET /v1/models`` answers an SDK, so it is flat and OpenAI-shaped: one object per
selector, and ``nebius:zai-org/GLM-5.3`` and ``fireworks:accounts/fireworks/models/glm-5p3``
are two unrelated rows. This router answers a person choosing a model. It reads
the same merged catalog (``models.build_merged_catalog``, so the two cannot
disagree about which selectors the caller may see), folds the selectors by
``services.model_identity``, and joins what a chooser needs onto each: the
models.dev description and capabilities, the provider's context and output
limits, and the price *this viewer* would be charged, with the rung it comes
from named.

That last part is the one thing the flat listing gets wrong for a tenant: it
prices from the deployment list, while a request from an organization holding a
negotiated override is billed at the override. Here the organization is
resolved the way settlement resolves it (the session's active organization, or
the API key's workspace's), never from a header.

Metadata is served to every catalog reader, where ``GET /v1/models/metadata``
is operator-only. That gate is inherited from the deployment-wide router the
route sits on and describes the operator's configured providers; a model's
public description and capabilities describe the model, and a member choosing
one needs them as much as an operator does.
"""

import asyncio
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import (
    get_config,
    get_db,
    get_session_identity,
    require_deployment_operator,
    verify_catalog_reader_or_public,
)
from gateway.api.routes.models import (
    ALIAS_OWNED_BY,
    MergedCatalog,
    ModelObject,
    ModelPricingInfo,
    _pricing_info,
    build_merged_catalog,
)
from gateway.core.config import GatewayConfig
from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.models.entities import APIKey, PricingSnapshot, UsageLog
from gateway.models.tenancy import User as TenancyUser
from gateway.models.tenancy import Workspace
from gateway.services.catalog_selectors import (
    build_selector_index,
    current_selector_index,
    model_selector_for_slug,
    set_selector_index,
    short_selector_for,
)
from gateway.services.model_catalog_service import (
    ModelCatalogEntry,
    background_catalog_enabled,
    cached_models_dev_catalog,
    load_models_dev_catalog,
    models_dev_provider_id,
    parse_entry,
)
from gateway.services.model_identity import (
    ModelIdentity,
    OfferingSeed,
    clean_model_id,
    group_offerings,
    identity_key,
)
from gateway.services.pricing_refresh_service import GENAI_PRICES_SOURCE
from gateway.services.pricing_service import (
    default_pricing_enabled,
    default_pricing_reference,
    load_organization_override_index,
    normalize_effective_at,
    resolve_organization_override,
)
from gateway.services.workspace_scope import organization_for_key_id

# ``verify_catalog_reader_or_public`` rather than ``verify_catalog_reader``: a
# visitor reads too while ``public_catalog`` is on, and is answered from the
# configured instances alone. Every other route in the process keeps its gate.
router = APIRouter(
    prefix="/v1/catalog",
    tags=["catalog"],
    dependencies=[Depends(verify_catalog_reader_or_public)],
)
# The one write on the catalog: an operator asking for the short spellings to
# be re-indexed now rather than on the next tick, after pricing or providers
# changed.
operator_router = APIRouter(prefix="/v1/catalog", tags=["catalog"], dependencies=[Depends(require_deployment_operator)])

# The anonymous caller, as the dependency hands it over; the routes below read
# it as "nobody" rather than as a key that failed to verify.
CatalogCaller = tuple[APIKey | None, bool] | None

PriceSource = Literal["organization", "deployment", "defaults"]
# ``hosted`` is the reserved instance a hosted edition serves deployment-owned
# offerings under (``RESERVED_PROVIDER_INSTANCE_NAMES``); the base gateway never
# configures one, so the label only ever appears where an overlay contributes it.
Credential = Literal["deployment", "organization", "hosted"]

# The window the viewer's own usage is rolled up over on a detail read.
_USAGE_WINDOW = timedelta(days=30)


class CatalogCapabilities(BaseModel):
    """What a model can do, as models.dev reports it. Any offering's yes is the model's."""

    reasoning: bool = False
    tool_call: bool = False
    structured_output: bool = False
    attachment: bool = False
    temperature: bool = False


class OfferingUsage(BaseModel):
    """What the viewer's organization actually paid for one offering, last 30 days.

    The listed rate is what a token costs; this is what the tokens cost, which is
    lower wherever prompt caching hit. Absent for a visitor and for an offering
    the organization never called.
    """

    requests: int
    total_tokens: int
    cache_read_tokens: int
    spend_usd: float
    cache_hit_rate: float | None = Field(
        description="Cache-read tokens over prompt tokens. Null when no prompt tokens."
    )
    effective_price_per_million: float | None = Field(
        description="Spend over every token served, per million. Null when no tokens were served."
    )


class CatalogOffering(BaseModel):
    """One way this deployment can call a model: a selector on a provider."""

    selector: str = Field(description="What to send as `model`, in `instance:model` form.")
    short_selector: str | None = Field(
        default=None,
        description=(
            "A shorter spelling the gateway also accepts: the instance with the model's cleaned id "
            "(`fireworks:gpt-oss-120b`). Null where two offerings on the instance would share it, or "
            "until the gateway has indexed the catalog."
        ),
    )
    provider: str = Field(description="The provider instance the selector names.")
    provider_type: str = Field(description="The any-llm implementation behind the instance.")
    credential: Credential = Field(
        description=(
            "Whose key serves it: `deployment` for a `providers:` instance the operator configured, "
            "`organization` for a key the viewer's organization holds."
        ),
    )
    discovered: bool = Field(description="Whether the provider itself reported this model.")
    context_window: int | None = None
    max_output_tokens: int | None = None
    quantization: str | None = Field(default=None, description="From the provider's id, when it names one.")
    pricing: ModelPricingInfo | None = None
    price_source: PriceSource | None = Field(
        default=None,
        description=(
            "Which price list `pricing` came from, for this viewer: the organization's own override, the "
            "deployment's stored row, or the genai-prices defaults. Null when nothing prices it."
        ),
    )
    price_reference: str | None = Field(
        default=None,
        description="For a default, the genai-prices `provider:model` entry that matched; the selector otherwise.",
    )
    metadata_input_price_per_million: float | None = Field(
        default=None,
        description=(
            "What models.dev lists this provider charging, for a cross-check. Not billed from: two "
            "independent datasets disagreeing is the cheapest stale-price detector there is."
        ),
    )
    metadata_output_price_per_million: float | None = None
    usage_30d: OfferingUsage | None = None


class CatalogModelSummary(BaseModel):
    """One model, as the list shows it."""

    id: str = Field(description="URL-safe id, derived from the display name.")
    selector: str | None = Field(
        default=None,
        description=(
            "The id as a selector: send it as `model` and the model's cheapest offering answers. "
            "Null until the gateway has indexed the catalog."
        ),
    )
    resolves_to: str | None = Field(default=None, description="The offering `selector` resolves to.")
    name: str
    vendor: str | None
    description: str | None = Field(default=None, description="models.dev's, from the offering that named the model.")
    family: str | None = None
    capabilities: CatalogCapabilities
    input_modalities: list[str]
    output_modalities: list[str]
    context_window: int | None = Field(default=None, description="The largest any offering serves.")
    max_output_tokens: int | None = Field(default=None, description="The largest any offering serves.")
    release_date: str | None = None
    knowledge_cutoff: str | None = None
    open_weights: bool = False
    deprecated: bool = Field(default=False, description="True only when every offering with metadata says so.")
    offering_count: int
    provider_count: int
    providers: list[str] = Field(description="The provider instances offering it, sorted.")
    selectors: list[str] = Field(description="Every offering's selector, so the list can be searched by one.")
    price_sources: list[PriceSource] = Field(
        description="Which price lists the priced offerings came from, distinct and sorted."
    )
    unpriced_count: int = Field(description="How many offerings carry no price for this caller.")
    discovered: bool = Field(description="Whether any offering was discovered from its provider.")
    min_input_price_per_million: float | None = Field(
        default=None,
        description="The cheapest offering's, at the comparison context where one was asked for.",
    )
    min_output_price_per_million: float | None = None


class CatalogElsewhere(BaseModel):
    """A provider models.dev lists for this model that this deployment has not configured."""

    provider_type: str
    name: str


class CatalogModelDetail(CatalogModelSummary):
    """One model with everything the detail page shows."""

    offerings: list[CatalogOffering]
    also_available_from: list[CatalogElsewhere]


class CatalogResponse(BaseModel):
    """The grouped catalog, and the facts a reader needs to interpret its prices."""

    default_pricing: bool = Field(description="Whether an unpriced model is metered at the genai-prices default.")
    defaults_as_of: datetime | None = Field(
        description="When the accepted genai-prices snapshot was taken. Null while the bundled dataset serves.",
    )
    metadata_available: bool = Field(
        description="False when models.dev could not be read; descriptions are then absent."
    )
    models: list[CatalogModelSummary]


@dataclass
class _Offering:
    """An offering with the metadata it carried, before the wire shape drops it."""

    wire: CatalogOffering
    metadata: ModelCatalogEntry | None
    model_id: str
    """The provider's own id, which a usage row carries beside the instance."""


def _split_selector(obj: ModelObject) -> tuple[str, str]:
    """The instance and the provider's own model id a merged selector names."""
    instance, separator, model_id = obj.id.partition(":")
    if not separator:
        return obj.owned_by, obj.id
    return instance, model_id


def _metadata_entry(catalog: dict[str, Any] | None, provider_type: str, model_id: str) -> ModelCatalogEntry | None:
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


async def _viewer_organization(
    db: AsyncSession, caller: CatalogCaller, session_identity: TenancyUser | None
) -> uuid.UUID | None:
    """The organization whose overrides price this viewer's requests.

    Resolved the way settlement resolves it: a session acts in its active
    organization, an API key in its workspace's, and a master key in the default
    workspace's. Never from the request. A visitor has none.
    """
    if session_identity is not None:
        return session_identity.active_organization_id
    if caller is None:
        return None
    api_key, _ = caller
    return await organization_for_key_id(db, api_key.id if api_key is not None else None)


async def _usage_by_selector(
    db: AsyncSession, organization_id: uuid.UUID, offerings: Iterable[tuple[str, str, str]]
) -> dict[str, OfferingUsage]:
    """The organization's last 30 days against each offering, keyed by selector.

    One grouped query over the organization's workspaces. A usage row carries
    the instance and the provider's model id separately, and older rows carry
    the whole selector in ``model``, so both spellings are matched.
    """
    since = datetime.now(UTC) - _USAGE_WINDOW
    by_pair = {(instance, model_id): selector for selector, instance, model_id in offerings}
    if not by_pair:
        return {}
    matches = [
        (UsageLog.provider == instance) & (UsageLog.model == model_id) | (UsageLog.model == selector)
        for (instance, model_id), selector in by_pair.items()
    ]
    stmt = (
        select(
            UsageLog.provider,
            UsageLog.model,
            func.count().label("requests"),
            func.coalesce(func.sum(UsageLog.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0).label("prompt_tokens"),
            func.coalesce(func.sum(UsageLog.cache_read_tokens), 0).label("cache_read_tokens"),
            func.coalesce(func.sum(UsageLog.cost), 0).label("spend"),
        )
        .join(Workspace, col(Workspace.id) == UsageLog.workspace_id)
        .where(
            col(Workspace.organization_id) == organization_id,
            UsageLog.timestamp >= since,
            UsageLog.status == "success",
            or_(*matches),
        )
        .group_by(UsageLog.provider, UsageLog.model)
    )
    # Both spellings of one offering come back as separate groups, so the rows
    # are summed per selector rather than assigned: a row set that carries the
    # legacy spelling as well would otherwise report whichever group the
    # database happened to return last.
    selectors = set(by_pair.values())
    totals: dict[str, list[float]] = {}
    for row in (await db.execute(stmt)).all():
        selector = by_pair.get((row.provider or "", row.model))
        if selector is None and row.model in selectors:
            selector = row.model
        if selector is None:
            continue
        running = totals.setdefault(selector, [0.0, 0.0, 0.0, 0.0, 0.0])
        running[0] += int(row.requests)
        running[1] += int(row.total_tokens)
        running[2] += int(row.prompt_tokens)
        running[3] += int(row.cache_read_tokens)
        running[4] += float(row.spend)
    usage: dict[str, OfferingUsage] = {}
    for selector, (requests, total, prompt, cached, spend) in totals.items():
        usage[selector] = OfferingUsage(
            requests=int(requests),
            total_tokens=int(total),
            cache_read_tokens=int(cached),
            spend_usd=spend,
            cache_hit_rate=round(cached / prompt, 4) if prompt > 0 else None,
            effective_price_per_million=round(spend / total * 1_000_000, 6) if total > 0 else None,
        )
    return usage


async def _defaults_as_of(db: AsyncSession) -> datetime | None:
    stmt = select(PricingSnapshot.updated_at).where(PricingSnapshot.source == GENAI_PRICES_SOURCE)
    taken = (await db.execute(stmt)).scalar_one_or_none()
    return normalize_effective_at(taken) if taken is not None else None


@dataclass
class _Grouped:
    identities: dict[str, ModelIdentity]
    offerings: dict[str, _Offering]
    """By selector."""
    catalog: dict[str, Any] | None
    configured_types: set[str]
    organization_id: uuid.UUID | None
    """Whose overrides priced the offerings; None for a visitor."""


def _real_offerings(merged: MergedCatalog) -> list[ModelObject]:
    """The selectors themselves, sorted so grouping is order-stable.

    Aliases and policies are names over selectors and live on Routing.
    """
    return sorted(
        (obj for obj in merged.models.values() if obj.owned_by != ALIAS_OWNED_BY and obj.pricing_source != "dynamic"),
        key=lambda obj: obj.id,
    )


def _seed(
    config: GatewayConfig, catalog: dict[str, Any] | None, obj: ModelObject
) -> tuple[OfferingSeed, ModelCatalogEntry | None, str, str, str]:
    """One offering's identity seed, with the facts the seed was read from."""
    instance, model_id = _split_selector(obj)
    provider_type = config.provider_instance_type(instance)
    metadata = _metadata_entry(catalog, provider_type, model_id)
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
    loop, and the cache is warm within a tick of any page load anyway.
    """
    merged = await build_merged_catalog(db, config, auth=(None, True), session_identity=None)
    catalog = (
        await load_models_dev_catalog(config, serve_stale=background_catalog_enabled(config))
        if fetch
        else cached_models_dev_catalog(config)
    )
    rows: list[tuple[str, str, str, float | None]] = []
    seeds: list[OfferingSeed] = []
    for obj in _real_offerings(merged):
        seed, _metadata, instance, model_id, _provider_type = _seed(config, catalog, obj)
        seeds.append(seed)
        rate = obj.pricing.input_price_per_million if obj.pricing is not None else None
        rows.append((obj.id, instance, clean_model_id(model_id).model, rate))
    identities = {
        identity.slug: (identity.key, identity.selectors) for identity in group_offerings(seeds).values()
    }
    set_selector_index(build_selector_index(rows, identities))


SELECTOR_INDEX_INTERVAL_SECONDS = 60.0


class SelectorIndexResponse(BaseModel):
    """What the rebuilt index knows."""

    offerings: int = Field(description="Selectors the deployment serves.")
    short_selectors: int = Field(description="Offerings with an unambiguous short spelling.")
    models: int = Field(description="Slugs that resolve to an offering.")


@operator_router.post("/selectors/refresh", response_model=SelectorIndexResponse)
async def refresh_selector_index(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> SelectorIndexResponse:
    """Re-index the short spellings now, rather than on the refresher's next tick."""
    await rebuild_selector_index(db, config, fetch=True)
    index = current_selector_index()
    return SelectorIndexResponse(
        offerings=len(index.full), short_selectors=len(index.short), models=len(index.models)
    )


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


async def _group(
    db: AsyncSession,
    config: GatewayConfig,
    merged: MergedCatalog,
    *,
    caller: CatalogCaller,
    session_identity: TenancyUser | None,
) -> _Grouped:
    catalog = await load_models_dev_catalog(config, serve_stale=background_catalog_enabled(config))
    now = normalize_effective_at(None)
    real = _real_offerings(merged)

    organization_id = await _viewer_organization(db, caller, session_identity)
    overrides = (
        await load_organization_override_index(db, organization_id, (obj.id for obj in real))
        if organization_id is not None
        else {}
    )
    configured_types = {models_dev_provider_id(config.provider_instance_type(i)) for i in config.providers}
    seeds: list[OfferingSeed] = []
    offerings: dict[str, _Offering] = {}
    for obj in real:
        seed, metadata, instance, model_id, provider_type = _seed(config, catalog, obj)
        seeds.append(seed)

        override = resolve_organization_override(overrides, [obj.id], now)
        pricing: ModelPricingInfo | None
        source: PriceSource | None
        reference: str | None
        if override is not None:
            pricing, source, reference = _pricing_info(override), "organization", obj.id
        elif obj.pricing is not None and obj.pricing_source == "configured":
            pricing, source, reference = obj.pricing, "deployment", obj.id
        elif obj.pricing is not None and obj.pricing_source == "default":
            pricing, source = obj.pricing, "defaults"
            reference = default_pricing_reference(instance, model_id, now)
        else:
            pricing, source, reference = None, None, None

        offerings[obj.id] = _Offering(
            wire=CatalogOffering(
                selector=obj.id,
                short_selector=short_selector_for(obj.id),
                provider=instance,
                provider_type=provider_type,
                credential=_credential(config, instance),
                discovered=obj.id in merged.discovered_keys,
                context_window=(metadata.context_window if metadata else None) or obj.context_window,
                max_output_tokens=metadata.max_output_tokens if metadata else None,
                quantization=clean_model_id(model_id).quantization,
                pricing=pricing,
                price_source=source,
                price_reference=reference,
                metadata_input_price_per_million=metadata.cost_input if metadata else None,
                metadata_output_price_per_million=metadata.cost_output if metadata else None,
            ),
            metadata=metadata,
            model_id=model_id,
        )

    return _Grouped(
        identities=group_offerings(seeds),
        offerings=offerings,
        catalog=catalog,
        configured_types=configured_types,
        organization_id=organization_id,
    )


async def _with_usage(db: AsyncSession, grouped: _Grouped, members: list[_Offering]) -> list[CatalogOffering]:
    """One model's offerings with the viewer's 30-day usage of each.

    Queried for these selectors alone rather than the whole catalog's: a usage
    row is matched on two unindexed columns, and a detail read asks about one
    model. A visitor has no organization and gets the offerings as they are.
    """
    if grouped.organization_id is None:
        return [member.wire for member in members]
    usage = await _usage_by_selector(
        db,
        grouped.organization_id,
        ((member.wire.selector, member.wire.provider, member.model_id) for member in members),
    )
    return [member.wire.model_copy(update={"usage_30d": usage.get(member.wire.selector)}) for member in members]


def _credential(config: GatewayConfig, instance: str) -> Credential:
    """Whose key an instance runs on, from its name alone."""
    if instance == "hosted":
        return "hosted"
    return "deployment" if instance in config.providers else "organization"


def _first(values: Iterable[str | None]) -> str | None:
    return next((value for value in values if value), None)


def _rates_at_context(pricing: ModelPricingInfo, at_context: int | None) -> tuple[float, float]:
    """The input and output rate a request of ``at_context`` tokens is metered at.

    The tier is the one settlement would pick: the highest cliff at or below
    the request's input tokens, each of its rates falling back to the base
    where the tier leaves one unset. No context asked for means the base rates.
    """
    rates = (pricing.input_price_per_million, pricing.output_price_per_million)
    if at_context is None:
        return rates
    # A tier arrives as the model or, off a stored row, as the dict it was kept as.
    tiers = [
        tier if isinstance(tier, dict) else tier.model_dump()
        for tier in pricing.pricing_tiers
    ]
    eligible = [
        tier
        for tier in tiers
        if isinstance(tier.get("min_input_tokens"), int | float) and tier["min_input_tokens"] <= at_context
    ]
    if not eligible:
        return rates
    tier = max(eligible, key=lambda t: float(t["min_input_tokens"]))
    input_rate = tier.get("input_price_per_million")
    output_rate = tier.get("output_price_per_million")
    return (
        float(input_rate) if isinstance(input_rate, int | float) else rates[0],
        float(output_rate) if isinstance(output_rate, int | float) else rates[1],
    )


def _summary(identity: ModelIdentity, members: list[_Offering], at_context: int | None = None) -> CatalogModelSummary:
    """Fold a group's offerings into the model they are offerings of.

    A limit is the largest any offering serves, because the model can do that
    much somewhere; the list page says "up to". A capability is true when any
    offering reports it. Deprecated is the one conjunction: a model is retired
    when everyone who describes it says so, not when one reseller has moved on.
    """
    described = [member.metadata for member in members if member.metadata is not None]
    priced = [member.wire.pricing for member in members if member.wire.pricing is not None]
    rates = [_rates_at_context(pricing, at_context) for pricing in priced]
    # The description from the offering whose spelling won the name, so the two
    # read as one source; any offering's when none of them named the model.
    winners = [entry for entry in described if entry.name and entry.name.rsplit("/", 1)[-1] == identity.name]
    contexts = [member.wire.context_window for member in members if member.wire.context_window is not None]
    outputs = [member.wire.max_output_tokens for member in members if member.wire.max_output_tokens is not None]
    resolves_to = model_selector_for_slug(identity.slug)
    return CatalogModelSummary(
        id=identity.slug,
        selector=identity.slug if resolves_to is not None else None,
        resolves_to=resolves_to,
        name=identity.name,
        vendor=identity.vendor,
        description=_first(entry.description for entry in [*winners, *described]),
        family=_first(entry.family for entry in described),
        capabilities=CatalogCapabilities(
            reasoning=any(entry.reasoning for entry in described),
            tool_call=any(entry.tool_call for entry in described),
            structured_output=any(entry.structured_output for entry in described),
            attachment=any(entry.attachment for entry in described),
            temperature=any(entry.temperature for entry in described),
        ),
        input_modalities=sorted({modality for entry in described for modality in entry.input_modalities}),
        output_modalities=sorted({modality for entry in described for modality in entry.output_modalities}),
        context_window=max(contexts) if contexts else None,
        max_output_tokens=max(outputs) if outputs else None,
        release_date=min((entry.release_date for entry in described if entry.release_date), default=None),
        knowledge_cutoff=_first(entry.knowledge_cutoff for entry in described),
        open_weights=any(entry.open_weights for entry in described),
        deprecated=bool(described) and all(entry.deprecated for entry in described),
        offering_count=len(members),
        provider_count=len({member.wire.provider for member in members}),
        providers=sorted({member.wire.provider for member in members}),
        selectors=sorted(member.wire.selector for member in members),
        price_sources=sorted({member.wire.price_source for member in members if member.wire.price_source}),
        unpriced_count=sum(1 for member in members if member.wire.pricing is None),
        discovered=any(member.wire.discovered for member in members),
        min_input_price_per_million=min((rate[0] for rate in rates), default=None),
        min_output_price_per_million=min((rate[1] for rate in rates), default=None),
    )


def _elsewhere(grouped: _Grouped, key: str, offered_types: set[str]) -> list[CatalogElsewhere]:
    """Providers models.dev lists for this model that nothing here reaches.

    A scan of the whole dataset, a few thousand entries, once per detail read.
    Cheap, and it answers "should I add a provider" with the same identity rule
    that grouped the offerings, so it cannot name a provider that would then
    land in a different group.
    """
    if not grouped.catalog:
        return []
    found: dict[str, str] = {}
    for provider_type, provider in grouped.catalog.items():
        if provider_type in offered_types or provider_type in grouped.configured_types:
            continue
        if not isinstance(provider, dict):
            continue
        models = provider.get("models")
        if not isinstance(models, dict):
            continue
        for model_id, model in models.items():
            if not isinstance(model, dict) or not isinstance(model_id, str):
                continue
            name = model.get("name")
            if identity_key(provider_type, model_id, name if isinstance(name, str) else None) == key:
                found[provider_type] = str(provider.get("name") or provider_type)
                break
    return [
        CatalogElsewhere(provider_type=pid, name=name)
        for pid, name in sorted(found.items(), key=lambda i: i[1].lower())
    ]


async def _merged_for(
    db: AsyncSession, config: GatewayConfig, caller: CatalogCaller, session_identity: TenancyUser | None
) -> MergedCatalog:
    if caller is None:
        return await build_merged_catalog(db, config, auth=(None, False), session_identity=None, anonymous=True)
    return await build_merged_catalog(db, config, auth=caller, session_identity=session_identity)


@router.get("/models")
async def list_catalog(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    caller: Annotated[CatalogCaller, Depends(verify_catalog_reader_or_public)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    at_context: Annotated[
        int | None,
        Query(
            ge=1,
            description=(
                "Compare prices for a request of this many input tokens: each model's minimum is taken "
                "from the pricing tier that request would settle at. Omitted, the base rates compare."
            ),
        ),
    ] = None,
) -> CatalogResponse:
    """The models this caller may use, one entry each however many providers serve it.

    Prices are the caller's: an organization's override where one applies, else
    the deployment's row, else the genai-prices default. Aliases and routing
    policies are not models and are not listed; see Routing. A visitor, where
    the catalog is public, sees the configured instances at the deployment's
    rates and nothing that belongs to a tenant.
    """
    merged = await _merged_for(db, config, caller, session_identity)
    grouped = await _group(db, config, merged, caller=caller, session_identity=session_identity)
    models = [
        _summary(identity, [grouped.offerings[selector] for selector in identity.selectors], at_context)
        for identity in grouped.identities.values()
    ]
    return CatalogResponse(
        default_pricing=default_pricing_enabled(),
        defaults_as_of=await _defaults_as_of(db),
        metadata_available=grouped.catalog is not None,
        models=sorted(models, key=lambda m: (m.name.lower(), m.id)),
    )


@router.get("/models/{model_id:path}")
async def get_catalog_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    caller: Annotated[CatalogCaller, Depends(verify_catalog_reader_or_public)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
) -> CatalogModelDetail:
    """One model and every offering of it this caller may use.

    A model the caller may not see answers 404, the same as one that does not
    exist, so the route cannot be used to probe the catalog behind an allow-list.
    A signed-in caller's offerings also carry their organization's own usage of
    each over the last 30 days.
    """
    merged = await _merged_for(db, config, caller, session_identity)
    grouped = await _group(db, config, merged, caller=caller, session_identity=session_identity)
    identity = next((identity for identity in grouped.identities.values() if identity.slug == model_id), None)
    if identity is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_id}' not found")

    members = [grouped.offerings[selector] for selector in identity.selectors]
    summary = _summary(identity, members)
    # The cheapest offering first, unpriced ones last, so the comparison the
    # page exists for is the order the rows arrive in.
    offerings = sorted(
        await _with_usage(db, grouped, members),
        key=lambda o: (o.pricing is None, o.pricing.input_price_per_million if o.pricing else 0.0, o.selector),
    )
    return CatalogModelDetail(
        **summary.model_dump(),
        offerings=offerings,
        also_available_from=_elsewhere(
            grouped, identity.key, {models_dev_provider_id(o.provider_type) for o in offerings}
        ),
    )
