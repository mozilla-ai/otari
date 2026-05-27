"""Model listing endpoints with OpenAI and gateway-catalog shapes."""

import calendar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import ModelPricing
from gateway.services.model_discovery_service import discover_all_models, get_model_cache

router = APIRouter(prefix="/v1", tags=["models"])

T = TypeVar("T")


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


class GatewayCatalogPricing(BaseModel):
    """Merge-style pricing metadata for a vendor/model pair."""

    currency: str = "USD"
    input_per_million: float
    output_per_million: float


class GatewayCatalogCapabilities(BaseModel):
    """Best-effort capability metadata for a vendor/model pair."""

    input: list[str] = Field(default_factory=lambda: ["text"])
    output: list[str] = Field(default_factory=lambda: ["text", "tool_use"])
    supports_tool_calling: bool = True
    supports_tool_choice: bool = True
    supports_structured_outputs: bool = True
    streaming: bool = True


class GatewayVendorModelMetadata(BaseModel):
    """Execution metadata for a model served by one vendor."""

    launch_date: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    availability_status: str = "available"
    zero_data_retention: bool = False
    capabilities: GatewayCatalogCapabilities = Field(default_factory=GatewayCatalogCapabilities)
    pricing: GatewayCatalogPricing | None = None


class GatewayCatalogModel(BaseModel):
    """Merge Gateway-style canonical model catalog object."""

    model: str
    provider: str
    display_name: str
    vendors: dict[str, GatewayVendorModelMetadata]
    availability_status: str = "available"
    created_at: str | None = None
    updated_at: str | None = None


class GatewayCatalogListResponse(BaseModel):
    """Paginated Merge-style model catalog response."""

    object: str = "list"
    data: list[GatewayCatalogModel]
    has_more: bool = False
    next_cursor: str | None = None


class GatewayVendorResponse(BaseModel):
    """Merge Gateway-style execution vendor object."""

    vendor: str
    name: str
    models: list[str]
    supports_zdr: bool = False
    supports_byok: bool = True
    availability_status: str = "active"


class GatewayVendorListResponse(BaseModel):
    """Paginated Merge-style vendor catalog response."""

    object: str = "list"
    data: list[GatewayVendorResponse]
    has_more: bool = False
    next_cursor: str | None = None


@dataclass(frozen=True)
class _CatalogRecord:
    model_key: str
    provider: str
    provider_model: str
    created: int
    created_at: str | None
    updated_at: str | None
    pricing: ModelPricingInfo | None


_VENDOR_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "azureopenai": "Azure OpenAI",
    "bedrock": "AWS Bedrock",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "vertexai": "Google Vertex AI",
    "vertexaianthropic": "Vertex AI Anthropic",
}


def _split_model_key(model_key: str) -> tuple[str, str]:
    """Split an Otari or Merge-style model selector into provider/model parts."""
    if ":" in model_key:
        provider, model_name = model_key.split(":", 1)
        return provider or "unknown", model_name
    if "/" in model_key:
        provider, model_name = model_key.split("/", 1)
        return provider or "unknown", model_name
    return "unknown", model_key


def _canonical_model_id(model_key: str) -> str:
    """Return Merge-style provider/model ID for a stored model key."""
    provider, model_name = _split_model_key(model_key)
    return f"{provider}/{model_name}"


def _stored_model_key(selector: str) -> str:
    """Return the canonical storage key for either provider:model or provider/model."""
    provider, model_name = _split_model_key(selector)
    return f"{provider}:{model_name}"


def _model_pricing_info(pricing: ModelPricing | None) -> ModelPricingInfo | None:
    if pricing is None:
        return None
    return ModelPricingInfo(
        input_price_per_million=pricing.input_price_per_million,
        output_price_per_million=pricing.output_price_per_million,
    )


def _created_epoch(value: datetime | None) -> int:
    if value is None:
        return 0
    return int(calendar.timegm(value.utctimetuple()))


def _iso_from_epoch(epoch_seconds: int | None) -> str | None:
    if epoch_seconds is None or epoch_seconds <= 0:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat().replace("+00:00", "Z")


def _display_name(model_key: str) -> str:
    _provider, model_name = _split_model_key(model_key)
    words = model_name.replace("/", " / ").replace("-", " ").replace("_", " ").split()
    fixed: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in {"gpt", "llm", "glm", "mcp"}:
            fixed.append(lower.upper())
        elif lower.endswith("o") and any(char.isdigit() for char in lower):
            fixed.append(lower)
        else:
            fixed.append(word[:1].upper() + word[1:])
    return " ".join(fixed)


def _vendor_name(vendor: str) -> str:
    return _VENDOR_DISPLAY_NAMES.get(vendor, vendor.replace("_", " ").replace("-", " ").title())


def _model_from_catalog_record(record: _CatalogRecord) -> ModelObject:
    """Convert an internal catalog record to an OpenAI-compatible ModelObject."""
    return ModelObject(
        id=record.model_key,
        created=record.created,
        owned_by=record.provider,
        pricing=record.pricing,
    )


def _gateway_model_from_catalog_record(record: _CatalogRecord) -> GatewayCatalogModel:
    """Convert an internal catalog record to a Merge-style model object."""
    pricing = None
    if record.pricing is not None:
        pricing = GatewayCatalogPricing(
            input_per_million=record.pricing.input_price_per_million,
            output_per_million=record.pricing.output_price_per_million,
        )
    return GatewayCatalogModel(
        model=_canonical_model_id(record.model_key),
        provider=record.provider,
        display_name=_display_name(record.model_key),
        vendors={
            record.provider: GatewayVendorModelMetadata(
                launch_date=record.created_at[:10] if record.created_at else None,
                pricing=pricing,
            )
        },
        created_at=record.created_at,
        updated_at=record.updated_at,
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


async def _load_catalog_records(
    db: AsyncSession,
    config: GatewayConfig,
    provider_filter: str | None = None,
) -> list[_CatalogRecord]:
    """Load latest models from discovery and pricing into one internal catalog."""
    pricing_map = await _get_pricing_map(db, provider_filter=provider_filter)
    records: dict[str, _CatalogRecord] = {}

    if config.model_discovery:
        try:
            discovered = await discover_all_models(config, provider_filter=provider_filter)
        except Exception:
            logger.exception("Model discovery failed unexpectedly")
            discovered = []

        for provider_name, model in discovered:
            model_key = f"{provider_name}:{model.id}"
            pricing = pricing_map.pop(model_key, None)
            created_at = _iso_from_epoch(model.created)
            records[model_key] = _CatalogRecord(
                model_key=model_key,
                provider=provider_name,
                provider_model=model.id,
                created=model.created,
                created_at=created_at,
                updated_at=pricing.updated_at.isoformat() if pricing else created_at,
                pricing=_model_pricing_info(pricing),
            )

    for model_key, pricing in pricing_map.items():
        if model_key in records:
            continue
        provider, provider_model = _split_model_key(model_key)
        records[model_key] = _CatalogRecord(
            model_key=model_key,
            provider=provider,
            provider_model=provider_model,
            created=_created_epoch(pricing.created_at),
            created_at=pricing.created_at.isoformat() if pricing.created_at else None,
            updated_at=pricing.updated_at.isoformat() if pricing.updated_at else None,
            pricing=_model_pricing_info(pricing),
        )

    return sorted(records.values(), key=lambda record: record.model_key)


def _paginate(items: list[T], *, cursor: str | None, limit: int) -> tuple[list[T], bool, str | None]:
    """Apply offset-cursor pagination to catalog responses."""
    try:
        offset = int(cursor or "0")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="cursor must be an integer offset",
        ) from None
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="cursor must be non-negative",
        )
    page = items[offset : offset + limit]
    next_offset = offset + limit
    has_more = next_offset < len(items)
    return page, has_more, str(next_offset) if has_more else None


def _filter_catalog_records(
    records: list[_CatalogRecord],
    *,
    provider: str | None,
    vendor: str | None,
    model: str | None,
) -> list[_CatalogRecord]:
    """Filter catalog records using Merge-style provider/vendor/model parameters."""
    selected = records
    if provider:
        selected = [record for record in selected if record.provider == provider]
    if vendor:
        selected = [record for record in selected if record.provider == vendor]
    if model:
        stored_model_key = _stored_model_key(model)
        selected = [record for record in selected if record.model_key == stored_model_key]
    return selected


def _vendor_catalog_from_records(records: list[_CatalogRecord], config: GatewayConfig) -> list[GatewayVendorResponse]:
    """Build execution-vendor objects from catalog records."""
    by_vendor: dict[str, set[str]] = {}
    for record in records:
        by_vendor.setdefault(record.provider, set()).add(_canonical_model_id(record.model_key))

    for provider_name in config.providers:
        by_vendor.setdefault(provider_name, set())

    vendors = [
        GatewayVendorResponse(
            vendor=vendor,
            name=_vendor_name(vendor),
            models=sorted(models),
            supports_byok=vendor in config.providers,
            availability_status="active" if models or vendor in config.providers else "unavailable",
        )
        for vendor, models in by_vendor.items()
    ]
    return sorted(vendors, key=lambda item: item.vendor)


@router.get("/models", dependencies=[Depends(verify_api_key_or_master_key)])
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    provider: Annotated[str | None, Query(description="Filter models by provider name")] = None,
    vendor: Annotated[str | None, Query(description="Filter gateway catalog models by execution vendor")] = None,
    model: Annotated[str | None, Query(description="Fetch one gateway catalog model by provider/model id")] = None,
    catalog_format: Annotated[
        str,
        Query(alias="format", description="Response format: 'openai' (default) or 'gateway'/'merge' catalog shape"),
    ] = "openai",
    cursor: Annotated[str | None, Query(description="Gateway catalog pagination cursor")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Gateway catalog page size")] = 50,
) -> ModelListResponse | GatewayCatalogListResponse | GatewayCatalogModel:
    """List all available models.

    Returns models auto-discovered from configured providers, enriched with
    pricing data from the model_pricing table when available. Models that only
    exist in the pricing table are also included for backward compatibility.
    """
    normalized_format = catalog_format.lower()
    if normalized_format not in {"openai", "gateway", "merge"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="format must be 'openai', 'gateway', or 'merge'",
        )

    records = await _load_catalog_records(db, config, provider_filter=provider)

    use_gateway_catalog = normalized_format in {"gateway", "merge"} or vendor is not None or model is not None
    if use_gateway_catalog:
        filtered = _filter_catalog_records(records, provider=provider, vendor=vendor, model=model)
        if model is not None:
            if not filtered:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model}' not found",
                )
            return _gateway_model_from_catalog_record(filtered[0])
        page, has_more, next_cursor = _paginate(filtered, cursor=cursor, limit=limit)
        return GatewayCatalogListResponse(
            data=[_gateway_model_from_catalog_record(record) for record in page],
            has_more=has_more,
            next_cursor=next_cursor,
        )

    return ModelListResponse(data=[_model_from_catalog_record(record) for record in records])


@router.get("/models/{model_id:path}", dependencies=[Depends(verify_api_key_or_master_key)])
async def get_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ModelObject:
    """Get details for a specific model."""
    normalized_model_id = _stored_model_key(model_id)
    # Check the pricing table first.
    stmt = (
        select(ModelPricing)
        .where(ModelPricing.model_key == normalized_model_id)
        .order_by(ModelPricing.effective_at.desc())
        .limit(1)
    )
    pricing = (await db.execute(stmt)).scalar_one_or_none()

    # Check the discovery cache for this model (respecting TTL).
    # Parse provider from model_id ("provider:model_name") for a targeted lookup
    # instead of scanning all cached providers.
    discovered_model = None
    discovered_provider = None
    if config.model_discovery and ":" in normalized_model_id:
        provider_prefix, model_name = normalized_model_id.split(":", 1)
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
    provider, provider_model = _split_model_key(pricing.model_key)
    return _model_from_catalog_record(
        _CatalogRecord(
            model_key=pricing.model_key,
            provider=provider,
            provider_model=provider_model,
            created=_created_epoch(pricing.created_at),
            created_at=pricing.created_at.isoformat() if pricing.created_at else None,
            updated_at=pricing.updated_at.isoformat() if pricing.updated_at else None,
            pricing=_model_pricing_info(pricing),
        )
    )


@router.get("/vendors", dependencies=[Depends(verify_api_key_or_master_key)], tags=["vendors"])
async def list_vendors(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    cursor: Annotated[str | None, Query(description="Pagination cursor")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Page size")] = 50,
) -> GatewayVendorListResponse:
    """List Merge-style execution vendors and the models they can serve."""
    records = await _load_catalog_records(db, config)
    vendors = _vendor_catalog_from_records(records, config)
    page, has_more, next_cursor = _paginate(vendors, cursor=cursor, limit=limit)
    return GatewayVendorListResponse(data=page, has_more=has_more, next_cursor=next_cursor)


@router.get("/vendors/{vendor_id}", dependencies=[Depends(verify_api_key_or_master_key)], tags=["vendors"])
async def get_vendor(
    vendor_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> GatewayVendorResponse:
    """Fetch one Merge-style execution vendor by id."""
    records = await _load_catalog_records(db, config)
    vendors = _vendor_catalog_from_records(records, config)
    for vendor in vendors:
        if vendor.vendor == vendor_id:
            return vendor
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Vendor '{vendor_id}' not found",
    )
