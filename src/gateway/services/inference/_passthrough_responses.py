"""Gateway-owned response bodies for the pass-through endpoints, built from the provider's result."""

from decimal import Decimal
from typing import Any

from any_llm.types.completion import CreateEmbeddingResponse
from any_llm.types.image import ImagesResponse
from any_llm.types.rerank import RerankResponse as ProviderRerankResponse

from gateway.core.config import COST_USD_HEADER, PRICING_SOURCE_HEADER, USAGE_UNIT_HEADER, USAGE_UNITS_HEADER
from gateway.core.metered_pricing import quantize_cost
from gateway.schemas.inference import (
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    InlineCost,
    RerankMeta,
    RerankResponse,
    RerankResult,
    RerankUsage,
)
from gateway.services.pricing import UsageCharge
from gateway.services.pricing_service import ResolvedPricing, metered_search_units


def rerank_response(
    result: ProviderRerankResponse,
    *,
    pricing: ResolvedPricing | None,
    cost: Decimal | None,
) -> RerankResponse:
    """The rerank response for ``result``, with what the request was metered at and cost.

    ``pricing`` is the rate found for the request and ``cost`` the amount
    written to its usage row. ``meta`` is the provider's billing report as it sent it; ``usage`` reports
    the provider's tokens, the search units billed when the model is priced per
    request, and the inline cost.
    """
    billed_units = result.meta.billed_units if result.meta else None
    cost_fields = _inline_cost(cost, pricing)
    return RerankResponse(
        id=result.id,
        model=result.model,
        results=[RerankResult(index=r.index, relevance_score=r.relevance_score) for r in result.results],
        meta=RerankMeta(billed_units=result.meta.billed_units, tokens=result.meta.tokens) if result.meta else None,
        usage=RerankUsage(
            total_tokens=result.usage.total_tokens if result.usage else None,
            search_units=metered_search_units(pricing.pricing if pricing else None, billed_units),
            cost_usd=cost_fields.cost_usd,
            pricing_source=cost_fields.pricing_source,
        ),
    )


def embeddings_response(
    result: CreateEmbeddingResponse,
    *,
    pricing: ResolvedPricing | None,
    cost: Decimal | None,
) -> EmbeddingResponse:
    """The embeddings response for ``result``, in OpenAI's shape with the inline cost on ``usage``.

    ``pricing`` and ``cost`` are as for :func:`rerank_response`. A provider that
    reported no usage gets none back, and so no inline cost.
    """
    cost_fields = _inline_cost(cost, pricing)
    return EmbeddingResponse(
        data=[EmbeddingData(index=item.index, embedding=item.embedding) for item in result.data],
        model=result.model,
        usage=EmbeddingUsage(
            prompt_tokens=result.usage.prompt_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=cost_fields.cost_usd,
            pricing_source=cost_fields.pricing_source,
        )
        if result.usage is not None
        else None,
    )


def images_response(
    result: ImagesResponse,
    *,
    pricing: ResolvedPricing | None,
    cost: Decimal | None,
) -> dict[str, Any]:
    """The images response for ``result``: the provider's body with the inline cost on ``usage``.

    ``pricing`` and ``cost`` are as for :func:`rerank_response`. The rest of the
    provider's body is returned as it came, so a priced request with no ``usage``
    from the provider (OpenAI's DALL-E models send none) gets one holding only the
    two cost fields, and one with token counts keeps them beside the fields. A
    ``cost_usd`` or ``pricing_source`` the provider sent itself is always dropped,
    so a caller never reads an upstream figure as the gateway's settled cost.
    """
    body = result.model_dump()
    usage = body.get("usage")
    if isinstance(usage, dict):
        usage = {name: value for name, value in usage.items() if name not in InlineCost.model_fields}
    if cost is not None and pricing is not None:
        usage = {**(usage or {}), **_inline_cost(cost, pricing).model_dump()}
    body["usage"] = usage
    return body


def speech_cost_headers(*, pricing: ResolvedPricing | None, charge: UsageCharge | None) -> dict[str, str]:
    """The headers a speech response carries its settled cost on, since its body is audio.

    ``pricing`` is the rate found for the request and ``charge`` what it settled
    at. The cost and its source follow the inline-cost rule: both when the
    request was priced, neither when it was not. The meter and the quantity
    billed are the charge's own, so they name what the usage row's
    ``billing_meters`` names, and a charge that billed nothing (a zero rate)
    names no meter.
    """
    if charge is None or pricing is None:
        return {}
    headers = {COST_USD_HEADER: f"{quantize_cost(charge.cost):.6f}", PRICING_SOURCE_HEADER: pricing.source}
    if len(charge.meters) == 1:
        ((meter, quantity),) = charge.meters.items()
        headers[USAGE_UNIT_HEADER] = meter
        headers[USAGE_UNITS_HEADER] = str(quantity)
    return headers


def _inline_cost(cost: Decimal | None, pricing: ResolvedPricing | None) -> InlineCost:
    """The ``cost_usd`` and ``pricing_source`` a pass-through response carries.

    Both are set only when both are known, the rule a completion's inline usage
    follows, so an unpriced request carries neither.
    """
    if cost is None or pricing is None:
        return InlineCost()
    return InlineCost(cost_usd=f"{quantize_cost(cost):.6f}", pricing_source=pricing.source)
