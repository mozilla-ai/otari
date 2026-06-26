from datetime import datetime
from typing import Annotated

from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.models.entities import ModelPricing
from gateway.services.pricing_service import normalize_effective_at
from gateway.services.provider_kwargs import normalize_pricing_key, split_selector

router = APIRouter(prefix="/v1/pricing", tags=["pricing"])


class SetPricingRequest(BaseModel):
    """Request model for setting model pricing."""

    model_key: str = Field(description="Model identifier in format 'provider:model'")
    input_price_per_million: float = Field(ge=0, description="Price per 1M input tokens")
    output_price_per_million: float = Field(ge=0, description="Price per 1M output tokens")
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
            created_at=pricing.created_at.isoformat(),
            updated_at=pricing.updated_at.isoformat(),
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

    provider_value = provider.value if provider else None
    if not provider_value:
        return candidates

    for key in (f"{provider_value}:{model_name}", f"{provider_value}/{model_name}"):
        if key not in candidates:
            candidates.append(key)
    return candidates


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
    """Set or update pricing for a model."""
    normalized_key = normalize_pricing_key(config, request.model_key)
    effective_at = normalize_effective_at(request.effective_at)
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
    else:
        pricing = ModelPricing(
            model_key=normalized_key,
            effective_at=effective_at,
            input_price_per_million=request.input_price_per_million,
            output_price_per_million=request.output_price_per_million,
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
