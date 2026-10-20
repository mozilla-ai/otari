"""Gateway-owned response bodies for the pass-through endpoints, built from the provider's result."""

from decimal import Decimal

from any_llm.types.completion import CreateEmbeddingResponse
from any_llm.types.rerank import RerankResponse as ProviderRerankResponse

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
from gateway.services.pricing_service import ResolvedPricing, metered_search_units


def rerank_response(
    result: ProviderRerankResponse,
    *,
    pricing: ResolvedPricing | None,
    cost: Decimal | None,
    inline_cost: InlineCost | None = None,
) -> RerankResponse:
    """The rerank response for ``result``, with what the request was metered at and cost.

    ``pricing`` is the rate found for the request and ``cost`` the amount
    written to its usage row. ``inline_cost``, when given, is a cost settled
    elsewhere (a hybrid gateway's control plane) and is carried as it is.
    ``meta`` is the provider's billing report as it sent it; ``usage`` reports
    the provider's tokens, the search units billed when the model is priced per
    request, and the inline cost.
    """
    billed_units = result.meta.billed_units if result.meta else None
    cost_fields = inline_cost or _inline_cost(cost, pricing)
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
    inline_cost: InlineCost | None = None,
) -> EmbeddingResponse:
    """The embeddings response for ``result``, in OpenAI's shape with the inline cost on ``usage``.

    ``pricing``, ``cost`` and ``inline_cost`` are as for :func:`rerank_response`. A provider that
    reported no usage gets none back, and so no inline cost.
    """
    cost_fields = inline_cost or _inline_cost(cost, pricing)
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


def _inline_cost(cost: Decimal | None, pricing: ResolvedPricing | None) -> InlineCost:
    """The ``cost_usd`` and ``pricing_source`` a pass-through response carries.

    Both are set only when both are known, the rule a completion's inline usage
    follows, so an unpriced request carries neither.
    """
    if cost is None or pricing is None:
        return InlineCost()
    return InlineCost(cost_usd=f"{quantize_cost(cost):.6f}", pricing_source=pricing.source)
