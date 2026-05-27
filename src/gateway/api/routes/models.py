"""OpenAI-compatible models listing endpoint with auto-discovery."""

import calendar
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelPricing
from gateway.services.model_discovery_service import discover_all_models, get_model_cache

router = APIRouter(prefix="/v1", tags=["models"])


class ModelPricingInfo(BaseModel):
    """Pricing information for a model."""

    input_price_per_million: float
    output_price_per_million: float


class ModelObject(BaseModel):
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int
    owned_by: str
    pricing: ModelPricingInfo | None = None


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[ModelObject]


def _model_from_pricing(pricing: ModelPricing) -> ModelObject:
    """Convert a ModelPricing row to an OpenAI-compatible ModelObject."""
    parts = pricing.model_key.split(":", 1)
    owned_by = parts[0] if len(parts) > 1 else "unknown"
    created = int(calendar.timegm(pricing.created_at.utctimetuple()))
    return ModelObject(
        id=pricing.model_key,
        created=created,
        owned_by=owned_by,
        pricing=ModelPricingInfo(
            input_price_per_million=pricing.input_price_per_million,
            output_price_per_million=pricing.output_price_per_million,
        ),
    )


async def _get_pricing_map(db: AsyncSession, provider_filter: str | None = None) -> dict[str, ModelPricing]:
    """Load latest pricing per model_key, optionally filtered by provider prefix."""
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

    stmt = stmt.order_by(ModelPricing.model_key)
    result = await db.execute(stmt)
    pricings = result.scalars().all()
    return {p.model_key: p for p in pricings}


@router.get("/models", dependencies=[Depends(verify_api_key_or_master_key)])
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    provider: Annotated[str | None, Query(description="Filter models by provider name")] = None,
) -> ModelListResponse:
    """List all available models.

    Returns models auto-discovered from configured providers, enriched with
    pricing data from the model_pricing table when available. Models that only
    exist in the pricing table are also included for backward compatibility.
    """
    pricing_map = await _get_pricing_map(db, provider_filter=provider)

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
            )

    # Phase 2: pricing-only models (not discovered but have pricing entries).
    for model_key, pricing in pricing_map.items():
        if model_key not in merged:
            merged[model_key] = _model_from_pricing(pricing)

    sorted_models = sorted(merged.values(), key=lambda m: m.id)
    return ModelListResponse(data=sorted_models)


@router.get("/models/{model_id:path}", dependencies=[Depends(verify_api_key_or_master_key)])
async def get_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ModelObject:
    """Get details for a specific model."""
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found",
        )

    # Build the response, merging both sources.
    if discovered_model:
        assert discovered_provider is not None
        model_key = f"{discovered_provider}:{discovered_model.id}"
        return ModelObject(
            id=model_key,
            created=discovered_model.created,
            owned_by=discovered_provider,
            pricing=ModelPricingInfo(
                input_price_per_million=pricing.input_price_per_million,
                output_price_per_million=pricing.output_price_per_million,
            )
            if pricing
            else None,
        )

    # Pricing-only model (no discovery data).
    assert pricing is not None
    return _model_from_pricing(pricing)
