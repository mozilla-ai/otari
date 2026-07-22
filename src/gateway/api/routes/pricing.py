from datetime import datetime
from typing import Annotated

from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import distinct, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.models.entities import ModelPricing
from gateway.services.alias_service import resolve_effective_alias
from gateway.services.pricing_refresh_service import (
    PricingRefreshError,
    confirm_price_refresh,
    prepare_price_refresh,
    reject_price_refresh,
)
from gateway.services.pricing_service import normalize_effective_at
from gateway.services.provider_kwargs import normalize_pricing_key, split_selector

router = APIRouter(prefix="/v1/pricing", tags=["pricing"])


class SetPricingRequest(BaseModel):
    """Request model for setting model pricing."""

    model_key: str = Field(description="Model identifier in format 'provider:model'")
    input_price_per_million: float = Field(ge=0, description="Price per 1M input tokens")
    output_price_per_million: float = Field(ge=0, description="Price per 1M output tokens")
    cache_read_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M cached-input tokens"
    )
    cache_write_price_per_million: float | None = Field(
        default=None, ge=0, description="Price per 1M cache-write (creation) tokens"
    )
    effective_at: datetime | None = Field(
        default=None,
        description="ISO 8601 datetime from which this price applies. Defaults to now if omitted.",
    )


class PricingResponse(BaseModel):
    """Response model for model pricing."""

    model_key: str
    effective_at: str
    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float | None
    cache_write_price_per_million: float | None
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, pricing: "ModelPricing") -> "PricingResponse":
        """Create a PricingResponse from a ModelPricing ORM model."""
        return cls(
            model_key=pricing.model_key,
            effective_at=pricing.effective_at.isoformat(),
            input_price_per_million=pricing.input_price_per_million,
            output_price_per_million=pricing.output_price_per_million,
            cache_read_price_per_million=pricing.cache_read_price_per_million,
            cache_write_price_per_million=pricing.cache_write_price_per_million,
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

    provider_value = provider.value if provider else None
    if not provider_value:
        return candidates

    for key in (f"{provider_value}:{model_name}", f"{provider_value}/{model_name}"):
        if key not in candidates:
            candidates.append(key)
    return candidates


@router.post(
    "/refresh",
    response_model=PricingRefreshPreviewResponse,
    dependencies=[Depends(verify_master_key)],
)
async def preview_pricing_refresh(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PricingRefreshPreviewResponse:
    """Fetch the latest defaults and hold them for operator review."""

    try:
        preview = await prepare_price_refresh()
    except PricingRefreshError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to fetch the latest genai-prices data",
        ) from None

    protected_model_count = (
        await db.execute(select(func.count(distinct(ModelPricing.model_key))))
    ).scalar_one()
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


@router.post(
    "/refresh/confirm",
    response_model=PricingRefreshConfirmationResponse,
    dependencies=[Depends(verify_master_key)],
)
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


@router.post(
    "/refresh/reject",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
async def reject_pricing_refresh() -> None:
    """Discard a reviewed default-price snapshot without applying it."""

    if not reject_price_refresh():
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


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_pricing(
    request: SetPricingRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> PricingResponse:
    """Set or update pricing for a model.

    Rejects an alias: pricing, budgets, and usage all key on the resolved
    target, so a row stored under an alias name would never be read.
    """
    # Checked against the raw key: an alias name can never contain a selector
    # delimiter (see ``validate_alias``), so normalization would leave it
    # unchanged anyway, and this reads as the same lookup request dispatch does.
    alias_target = resolve_effective_alias(config, request.model_key)
    if alias_target is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"'{request.model_key}' is an alias for '{alias_target}', not a model. "
                f"Pricing keys on the resolved target, so set the price for '{alias_target}' instead."
            ),
        )

    normalized_key = normalize_pricing_key(config, request.model_key)
    effective_at = normalize_effective_at(request.effective_at)

    # Resolve the cache rates to persist. A field the client omits inherits the
    # model's most recent stored value, so a partial input/output update never
    # silently wipes cache pricing (each POST without an explicit effective_at
    # creates a new version, so the inherited value must carry forward). An
    # explicit null still clears the rate.
    cache_read_set = "cache_read_price_per_million" in request.model_fields_set
    cache_write_set = "cache_write_price_per_million" in request.model_fields_set
    latest: ModelPricing | None = None
    if not (cache_read_set and cache_write_set):
        latest = (
            await db.execute(
                select(ModelPricing)
                .where(ModelPricing.model_key == normalized_key)
                .order_by(ModelPricing.effective_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
    cache_read = (
        request.cache_read_price_per_million
        if cache_read_set
        else (latest.cache_read_price_per_million if latest else None)
    )
    cache_write = (
        request.cache_write_price_per_million
        if cache_write_set
        else (latest.cache_write_price_per_million if latest else None)
    )

    result = await db.execute(
        select(ModelPricing).where(
            ModelPricing.model_key == normalized_key,
            ModelPricing.effective_at == effective_at,
        )
    )
    pricing = result.scalar_one_or_none()

    if pricing:
        pricing.input_price_per_million = request.input_price_per_million
        pricing.output_price_per_million = request.output_price_per_million
        pricing.cache_read_price_per_million = cache_read
        pricing.cache_write_price_per_million = cache_write
    else:
        pricing = ModelPricing(
            model_key=normalized_key,
            effective_at=effective_at,
            input_price_per_million=request.input_price_per_million,
            output_price_per_million=request.output_price_per_million,
            cache_read_price_per_million=cache_read,
            cache_write_price_per_million=cache_write,
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


@router.get("", dependencies=[Depends(verify_api_key_or_master_key)])
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


@router.get("/{model_key:path}/history", dependencies=[Depends(verify_api_key_or_master_key)])
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


@router.get("/{model_key:path}", dependencies=[Depends(verify_api_key_or_master_key)])
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


@router.delete(
    "/{model_key:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_master_key)],
)
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
