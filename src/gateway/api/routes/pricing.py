from datetime import datetime
from typing import Annotated, Literal

from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import distinct, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, require_deployment_operator, verify_catalog_reader
from gateway.core.config import GatewayConfig
from gateway.models.entities import ModelPricing
from gateway.models.money import as_float, to_usd, to_usd_or_none
from gateway.services.alias_service import all_alias_names, resolve_effective_alias
from gateway.services.policy_store import all_policy_names, resolve_effective_policy
from gateway.services.pricing_refresh_service import (
    PricingRefreshError,
    PricingRefreshPreview,
    confirm_price_refresh,
    list_accepted_snapshots,
    prepare_price_refresh,
    preview_pending_refresh,
    reject_price_refresh,
)
from gateway.services.pricing_service import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    default_model_pricing,
    default_pricing_enabled,
    default_pricing_reference,
    normalize_effective_at,
)
from gateway.services.provider_kwargs import normalize_pricing_key, provider_key, split_selector

# Two routers under one prefix, one per authorization rule, so that adding a
# route defaults to refusing a caller who is not a deployment operator and
# opening one up to any catalog reader has to be spelled by the router it is
# declared on. See ``api/deps.verify_catalog_reader`` for why the reads are open
# at all.
operator_router = APIRouter(
    prefix="/pricing",
    tags=["pricing"],
    dependencies=[Depends(require_deployment_operator)],
)
catalog_router = APIRouter(
    prefix="/pricing",
    tags=["pricing"],
    dependencies=[Depends(verify_catalog_reader)],
)


class PricingTier(BaseModel):
    """Whole-request price cliff selected by total billable input tokens."""

    min_input_tokens: int = Field(gt=0)
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)
    cache_read_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_price_per_million: float | None = Field(default=None, ge=0)
    cache_write_1h_price_per_million: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_has_rate_override(self) -> "PricingTier":
        rates = (
            self.input_price_per_million,
            self.output_price_per_million,
            self.cache_read_price_per_million,
            self.cache_write_price_per_million,
            self.cache_write_1h_price_per_million,
        )
        if all(rate is None for rate in rates):
            raise ValueError("pricing tier must override at least one price field")
        return self


class SetPricingRequest(BaseModel):
    """Create a versioned per-model price, with optional cache and context tiers."""

    model_key: str = Field(description="Model identifier in format 'provider:model'")
    input_price_per_million: float = Field(ge=0, description="Price per 1M input tokens")
    output_price_per_million: float = Field(ge=0, description="Price per 1M output tokens")
    cache_read_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M cached-input tokens"
    )
    cache_write_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M cache-write (creation) tokens"
    )
    cache_write_1h_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M Anthropic 1-hour cache-write tokens"
    )
    pricing_tiers: list[PricingTier] | None = Field(
        default=None,
        description="Whole-request context thresholds. Fields omitted by a tier inherit the base rate.",
    )
    effective_at: datetime | None = Field(
        default=None,
        description="ISO 8601 datetime from which this price applies. Defaults to now if omitted.",
    )
    unit: Literal["tokens", "requests", "images"] = Field(
        default="tokens",
        description=(
            "What the rates are per: 'tokens' for a model, 'requests' for a gateway-run tool or a "
            "moderation call (USD per million requests), 'images' for image generation."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_tier_thresholds(self) -> "SetPricingRequest":
        if self.pricing_tiers is not None:
            thresholds = [tier.min_input_tokens for tier in self.pricing_tiers]
            if len(thresholds) != len(set(thresholds)):
                raise ValueError("pricing_tiers must not repeat min_input_tokens")
        return self


class PricingResponse(BaseModel):
    """Response model for model pricing."""

    model_key: str
    effective_at: str
    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float | None
    cache_write_price_per_million: float | None
    cache_write_1h_price_per_million: float | None
    pricing_tiers: list[PricingTier]
    unit: str = Field(description="What the rates are per: tokens, requests, or images.")
    origin: str | None = Field(
        description="Which writer set this row: config, api, or migration. Null when recorded before origins were.",
    )
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, pricing: "ModelPricing") -> "PricingResponse":
        """Create a PricingResponse from a ModelPricing ORM model."""
        return cls(
            model_key=pricing.model_key,
            effective_at=pricing.effective_at.isoformat(),
            input_price_per_million=float(pricing.input_price_per_million),
            output_price_per_million=float(pricing.output_price_per_million),
            cache_read_price_per_million=as_float(pricing.cache_read_price_per_million),
            cache_write_price_per_million=as_float(pricing.cache_write_price_per_million),
            cache_write_1h_price_per_million=as_float(pricing.cache_write_1h_price_per_million),
            pricing_tiers=[PricingTier.model_validate(tier) for tier in pricing.pricing_tiers or []],
            unit=pricing.unit or "tokens",
            origin=pricing.origin,
            created_at=pricing.created_at.isoformat(),
            updated_at=pricing.updated_at.isoformat(),
        )


class PricingRefreshChangeResponse(BaseModel):
    """One default model price changed by a pending refresh."""

    model_key: str
    change: str


class PricingRefreshPreviewResponse(BaseModel):
    """Reviewable summary of a pending genai-prices refresh."""

    fetched_at: datetime
    added_count: int
    changed_count: int
    removed_count: int
    protected_model_count: int
    changes: list[PricingRefreshChangeResponse]
    changes_truncated: bool


class PricingRefreshConfirmationResponse(BaseModel):
    """Result of activating a reviewed genai-prices refresh."""

    applied: bool = True


class AcceptedSnapshotResponse(BaseModel):
    """One accepted genai-prices snapshot in the history."""

    id: str
    accepted_at: datetime
    accepted_by: str = Field(description="`operator` for a dashboard confirm, `schedule` for the auto policy.")
    model_count: int


class PricingDriftRow(BaseModel):
    """A stored deployment rate beside the default it shadows."""

    model_key: str
    unit: str
    origin: str | None
    effective_at: str
    input_price_per_million: float
    output_price_per_million: float
    default_input_price_per_million: float | None = Field(
        description="What genai-prices would meter this key at today. Null when the dataset does not know it."
    )
    default_output_price_per_million: float | None
    default_reference: str | None = Field(description="The genai-prices entry the default came from.")
    input_delta_percent: float | None = Field(description="(stored - default) / default, as a percentage.")
    output_delta_percent: float | None


def _delta_percent(stored: float, default: float | None) -> float | None:
    if default is None:
        return None
    if default == 0:
        return None if stored == 0 else 100.0
    return round((stored - default) / default * 100, 1)


def _preview_response(preview: PricingRefreshPreview, protected_model_count: int) -> PricingRefreshPreviewResponse:
    return PricingRefreshPreviewResponse(
        fetched_at=preview.fetched_at,
        added_count=preview.added_count,
        changed_count=preview.changed_count,
        removed_count=preview.removed_count,
        protected_model_count=protected_model_count,
        changes=[
            PricingRefreshChangeResponse(model_key=change.model_key, change=change.change)
            for change in preview.changes
        ],
        changes_truncated=preview.changes_truncated,
    )


def _candidate_model_keys(raw_key: str) -> list[str]:
    """Return possible stored keys for a provided selector.

    Handles both real-provider selectors and instance-scoped ones. Instance
    names (e.g. ``home_lab``) are not any-llm providers, so the colon-normalized
    form is offered directly from the raw selector and the any-llm split is
    treated as best-effort: ``AnyLLMError`` (unknown provider) is caught like
    ``ValueError`` so an instance key never bubbles up as a 500.
    """

    candidates = [raw_key]
    # Normalize a "prefix/remainder" selector to its canonical colon form so the
    # legacy slash separator resolves for instance-scoped keys too.
    split = split_selector(raw_key)
    if split is not None:
        colon_form = f"{split[0]}:{split[1]}"
        if colon_form not in candidates:
            candidates.append(colon_form)

    try:
        provider, model_name = AnyLLM.split_model_provider(raw_key)
    except (ValueError, AnyLLMError):
        return candidates

    provider_value = provider_key(provider)
    if not provider_value:
        return candidates

    for key in (f"{provider_value}:{model_name}", f"{provider_value}/{model_name}"):
        if key not in candidates:
            candidates.append(key)
    return candidates


@operator_router.post("/refresh", response_model=PricingRefreshPreviewResponse)
async def preview_pricing_refresh(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PricingRefreshPreviewResponse:
    """Fetch the latest defaults and hold them for operator review."""

    try:
        preview = await prepare_price_refresh(db)
    except PricingRefreshError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to fetch the latest genai-prices data",
        ) from None

    protected_model_count = (
        await db.execute(select(func.count(distinct(ModelPricing.model_key))))
    ).scalar_one()
    return _preview_response(preview, protected_model_count)


@operator_router.get("/refresh/pending", response_model=PricingRefreshPreviewResponse)
async def get_pending_pricing_refresh(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PricingRefreshPreviewResponse:
    """The update the scheduled refresh has left waiting for review, if any.

    What the dashboard's notice reads. 404 when nothing is pending, so a page can
    ask on load without treating the common case as an error banner.
    """
    try:
        preview = await preview_pending_refresh(db)
    except PricingRefreshError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The pending genai-prices data is invalid",
        ) from None
    if preview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No pending genai-prices refresh")
    protected_model_count = (
        await db.execute(select(func.count(distinct(ModelPricing.model_key))))
    ).scalar_one()
    return _preview_response(preview, protected_model_count)


@operator_router.get("/snapshots", response_model=list[AcceptedSnapshotResponse])
async def list_pricing_snapshots(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[AcceptedSnapshotResponse]:
    """The accepted default-price snapshots, newest first."""
    return [
        AcceptedSnapshotResponse(
            id=str(row.id), accepted_at=row.accepted_at, accepted_by=row.accepted_by, model_count=row.model_count
        )
        for row in await list_accepted_snapshots(db, limit=limit)
    ]


@operator_router.get("/drift", response_model=list[PricingDriftRow])
async def list_pricing_drift(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PricingDriftRow]:
    """Every stored deployment rate in force today, beside today's default for it.

    A stored row shadows the genai-prices default silently and forever, whether
    it was a deliberate override or a copy of a then-current default. This is
    what makes the difference visible: a row that matches the default is a row
    that could be deleted, and a row far from it is one worth a second look.
    Tool rows (``otari:``) are per request and have no default to drift from,
    so they are left out.
    """
    now = normalize_effective_at(None)
    latest_effective = (
        select(ModelPricing.model_key.label("model_key"), func.max(ModelPricing.effective_at).label("effective_at"))
        .where(ModelPricing.effective_at <= now)
        .group_by(ModelPricing.model_key)
        .subquery()
    )
    stmt = (
        select(ModelPricing)
        .join(
            latest_effective,
            (ModelPricing.model_key == latest_effective.c.model_key)
            & (ModelPricing.effective_at == latest_effective.c.effective_at),
        )
        .where(ModelPricing.model_key.notlike(f"{GATEWAY_TOOL_PRICING_PROVIDER}:%"))
        .order_by(ModelPricing.model_key)
    )
    rows: list[PricingDriftRow] = []
    defaults_on = default_pricing_enabled()
    for pricing in (await db.execute(stmt)).scalars():
        provider_part, separator, model_part = pricing.model_key.partition(":")
        provider = provider_part if separator else None
        model_name = model_part if separator else pricing.model_key
        default = default_model_pricing(provider, model_name, now) if defaults_on else None
        default_input = float(default.input_price_per_million) if default is not None else None
        default_output = float(default.output_price_per_million) if default is not None else None
        rows.append(
            PricingDriftRow(
                model_key=pricing.model_key,
                unit=pricing.unit or "tokens",
                origin=pricing.origin,
                effective_at=pricing.effective_at.isoformat(),
                input_price_per_million=float(pricing.input_price_per_million),
                output_price_per_million=float(pricing.output_price_per_million),
                default_input_price_per_million=default_input,
                default_output_price_per_million=default_output,
                default_reference=default_pricing_reference(provider, model_name, now) if default is not None else None,
                input_delta_percent=_delta_percent(float(pricing.input_price_per_million), default_input),
                output_delta_percent=_delta_percent(float(pricing.output_price_per_million), default_output),
            )
        )
    return rows


@operator_router.post("/refresh/confirm", response_model=PricingRefreshConfirmationResponse)
async def confirm_pricing_refresh(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PricingRefreshConfirmationResponse:
    """Activate the latest reviewed default-price snapshot."""

    try:
        applied = await confirm_price_refresh(db)
    except PricingRefreshError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save the latest genai-prices data",
        ) from None
    if not applied:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No pending genai-prices refresh to apply")
    return PricingRefreshConfirmationResponse()


@operator_router.post("/refresh/reject", status_code=status.HTTP_204_NO_CONTENT)
async def reject_pricing_refresh(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Discard a reviewed default-price snapshot without applying it."""

    try:
        rejected = await reject_price_refresh(db)
    except PricingRefreshError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to discard the pending genai-prices data",
        ) from None
    if not rejected:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No pending genai-prices refresh to reject")


async def _get_effective_pricing(
    db: AsyncSession,
    model_keys: list[str],
    as_of: datetime,
) -> ModelPricing | None:
    for key in model_keys:
        stmt = (
            select(ModelPricing)
            .where(
                ModelPricing.model_key == key,
                ModelPricing.effective_at <= as_of,
            )
            .order_by(ModelPricing.effective_at.desc())
            .limit(1)
        )
        pricing = (await db.execute(stmt)).scalar_one_or_none()
        if pricing:
            return pricing
    return None


async def _get_pricing_history(db: AsyncSession, model_keys: list[str]) -> list[ModelPricing]:
    for key in model_keys:
        stmt = select(ModelPricing).where(ModelPricing.model_key == key).order_by(ModelPricing.effective_at.desc())
        pricings = list((await db.execute(stmt)).scalars().all())
        if pricings:
            return pricings
    return []


def _policy_pricing_detail(config: GatewayConfig, request: SetPricingRequest) -> str:
    """400 detail for a price aimed at a routing policy name.

    Names the candidates when they are knowable, because the operator's next
    action is to price them: a policy resolves to one of its own selectors, and
    that is the key the price has to be stored under.

    "Knowable" means resolvable in the default workspace, which is where this
    master-key surface reads (``services/policy_store``). A policy that lives in
    another workspace still triggers the refusal, because the name check above is
    scope-blind, and falls back to the generic wording: the price would be dead
    data either way, and guessing which workspace the operator meant would be
    worse than not naming its candidates.
    """
    spec = resolve_effective_policy(config, request.model_key)
    if spec is None:
        return (
            f"'{request.model_key}' is a routing policy, not a model. Pricing keys on the model a request "
            "resolves to, so set the price for the policy's candidates instead."
        )
    if not spec.is_dynamic:
        return (
            f"'{request.model_key}' is a routing policy for '{spec.default_target}', not a model. Pricing keys "
            f"on the model a request resolves to, so set the price for '{spec.default_target}' instead."
        )
    candidates = ", ".join(f"'{selector}'" for selector in spec.static_selectors())
    return (
        f"'{request.model_key}' is a routing policy, not a model, and it selects a candidate per request. "
        f"Pricing keys on the model a request resolves to, so set a price for each candidate instead: {candidates}."
    )


@operator_router.post("")
async def set_pricing(
    request: SetPricingRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> PricingResponse:
    """Set or update pricing for a model.

    Rejects an alias or a routing policy: pricing, budgets, and usage all key on
    the model a request resolves to, so a row stored under either name would
    never be read.
    """
    # Both checked against the raw key: neither an alias nor a policy name can
    # contain a selector delimiter (see ``validate_alias``), so normalization would
    # leave it unchanged anyway, and this reads as the same lookup request dispatch
    # does. Scope-blind: a name that is an indirection for even one user is still
    # not a model key, so a pricing row stored under it would never be read.
    if request.model_key in all_policy_names(config):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_policy_pricing_detail(config, request))
    if request.model_key in all_alias_names(config):
        alias_target = resolve_effective_alias(config, request.model_key)
        detail = (
            f"'{request.model_key}' is an alias, not a model. Pricing keys on the resolved target, "
            "so set the price for that instead."
        )
        if alias_target is not None:
            detail = (
                f"'{request.model_key}' is an alias for '{alias_target}', not a model. "
                f"Pricing keys on the resolved target, so set the price for '{alias_target}' instead."
            )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    normalized_key = normalize_pricing_key(config, request.model_key)
    effective_at = normalize_effective_at(request.effective_at)

    # Resolve the cache rates to persist. A field the client omits inherits the
    # model's most recent stored value, so a partial input/output update never
    # silently wipes cache pricing (each POST without an explicit effective_at
    # creates a new version, so the inherited value must carry forward). An
    # explicit null still clears the rate.
    cache_read_set = "cache_read_price_per_million" in request.model_fields_set
    cache_write_set = "cache_write_price_per_million" in request.model_fields_set
    cache_write_1h_set = "cache_write_1h_price_per_million" in request.model_fields_set
    tiers_set = "pricing_tiers" in request.model_fields_set
    latest: ModelPricing | None = None
    if not (cache_read_set and cache_write_set and cache_write_1h_set and tiers_set):
        latest = (
            await db.execute(
                select(ModelPricing)
                .where(ModelPricing.model_key == normalized_key)
                .order_by(ModelPricing.effective_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    # ``to_usd`` on the way in: the request carries JSON numbers (floats) and
    # the columns are exact, so the conversion is spelled here rather than left
    # to happen invisibly at flush.
    cache_read = to_usd_or_none(
        request.cache_read_price_per_million
        if cache_read_set
        else (latest.cache_read_price_per_million if latest else None)
    )
    cache_write = to_usd_or_none(
        request.cache_write_price_per_million
        if cache_write_set
        else (latest.cache_write_price_per_million if latest else None)
    )
    cache_write_1h = to_usd_or_none(
        request.cache_write_1h_price_per_million
        if cache_write_1h_set
        else (latest.cache_write_1h_price_per_million if latest else None)
    )
    pricing_tiers = (
        [tier.model_dump(exclude_none=True) for tier in request.pricing_tiers]
        if request.pricing_tiers is not None
        else ([] if tiers_set else (latest.pricing_tiers if latest else []))
    )

    result = await db.execute(
        select(ModelPricing).where(
            ModelPricing.model_key == normalized_key,
            ModelPricing.effective_at == effective_at,
        )
    )
    pricing = result.scalar_one_or_none()

    if pricing:
        pricing.input_price_per_million = to_usd(request.input_price_per_million)
        pricing.output_price_per_million = to_usd(request.output_price_per_million)
        pricing.cache_read_price_per_million = cache_read
        pricing.cache_write_price_per_million = cache_write
        pricing.cache_write_1h_price_per_million = cache_write_1h
        pricing.pricing_tiers = pricing_tiers
        pricing.unit = request.unit
        pricing.origin = "api"
    else:
        pricing = ModelPricing(
            model_key=normalized_key,
            effective_at=effective_at,
            input_price_per_million=to_usd(request.input_price_per_million),
            output_price_per_million=to_usd(request.output_price_per_million),
            cache_read_price_per_million=cache_read,
            cache_write_price_per_million=cache_write,
            cache_write_1h_price_per_million=cache_write_1h,
            pricing_tiers=pricing_tiers,
            unit=request.unit,
            origin="api",
        )
        db.add(pricing)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(pricing)

    return PricingResponse.from_model(pricing)


@catalog_router.get("")
async def list_pricing(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[PricingResponse]:
    """List all model pricing."""
    stmt = (
        select(ModelPricing)
        .order_by(ModelPricing.model_key, ModelPricing.effective_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    pricings = result.scalars().all()

    return [PricingResponse.from_model(pricing) for pricing in pricings]


@catalog_router.get("/{model_key:path}/history")
async def get_pricing_history(
    model_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PricingResponse]:
    """Return the full pricing history for a model."""

    candidates = _candidate_model_keys(model_key)
    pricings = await _get_pricing_history(db, candidates)
    if not pricings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

    return [PricingResponse.from_model(pricing) for pricing in pricings]


@catalog_router.get("/{model_key:path}")
async def get_pricing(
    model_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    as_of: Annotated[datetime | None, Query(description="ISO datetime for effective lookup", title="as_of")] = None,
) -> PricingResponse:
    """Get pricing for a specific model as of a timestamp."""

    candidates = _candidate_model_keys(model_key)
    pricing = await _get_effective_pricing(db, candidates, normalize_effective_at(as_of))

    if not pricing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

    return PricingResponse.from_model(pricing)


@operator_router.delete("/{model_key:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pricing(
    model_key: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    effective_at: Annotated[
        datetime | None,
        Query(
            description="ISO datetime identifying a specific pricing row to delete",
        ),
    ] = None,
) -> None:
    """Delete pricing entries for a model."""

    candidates = _candidate_model_keys(model_key)
    targets: list[ModelPricing] = []

    if effective_at is not None:
        normalized_effective_at = normalize_effective_at(effective_at)
        for key in candidates:
            stmt = (
                select(ModelPricing)
                .where(
                    ModelPricing.model_key == key,
                    ModelPricing.effective_at == normalized_effective_at,
                )
                .limit(1)
            )
            pricing = (await db.execute(stmt)).scalar_one_or_none()
            if pricing:
                targets = [pricing]
                break
        if not targets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=(
                    f"Pricing for model '{model_key}' with effective_at {normalized_effective_at.isoformat()} not found"
                ),
            )
    else:
        targets = await _get_pricing_history(db, candidates)
        if not targets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pricing for model '{model_key}' not found",
            )

    for pricing in targets:
        await db.delete(pricing)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
