"""Rerank endpoint — reorder documents by relevance to a query."""

from decimal import Decimal
from typing import Annotated, Any

from any_llm import arerank
from any_llm.types.rerank import RerankResponse as ProviderRerankResponse
from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    get_config,
    get_db_if_needed,
    get_log_writer,
    verify_api_key_or_master_key_if_needed,
)
from gateway.api.routes._passthrough import run_passthrough
from gateway.api.routes._pipeline import record_inline_cost_settlement
from gateway.core.config import GatewayConfig
from gateway.models.api_keys import APIKey
from gateway.models.pricing import ModelPricing
from gateway.schemas.inference import RerankResponse
from gateway.services.budgets import rerank_estimate
from gateway.services.inference import RERANK_BILLING, InferenceService
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(tags=["rerank"])

# See chat.USAGE_ENDPOINT.
USAGE_ENDPOINT = "/v1/rerank"


class RerankRequest(BaseModel):
    """Rerank request."""

    model: str = Field(description="Provider-prefixed model ID, e.g. 'cohere:rerank-v3.5'")
    query: str = Field(description="The search query to rerank documents against")
    documents: list[str] = Field(description="List of document strings to rerank", min_length=1)
    top_n: int | None = Field(default=None, description="Maximum number of results to return", gt=0)
    max_tokens_per_doc: int | None = Field(default=None, description="Per-document truncation limit", gt=0)
    user: str | None = Field(default=None, description="User ID for usage attribution")


@router.post("/rerank")
async def create_rerank(
    raw_request: Request,
    response: Response,
    request: RerankRequest,
    auth_result: Annotated[tuple[APIKey | None, bool] | None, Depends(verify_api_key_or_master_key_if_needed)],
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> RerankResponse:
    """Rerank documents by relevance to a query.

    Billing: a model priced per request (``unit: requests``) is charged per
    search unit the provider reports, or for one unit when it reports none. A
    model priced per token is charged on the input tokens the provider reports.

    ``usage`` carries what the gateway metered: the provider's tokens, the
    search units billed when the model is priced per request, and, when the
    request was priced, ``cost_usd`` and ``pricing_source`` as a completion's
    usage does. ``meta`` is the provider's own billing report, unchanged.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """

    def estimate(pricing: ModelPricing | None) -> Decimal:
        return rerank_estimate(
            pricing, query_chars=len(request.query), document_chars=[len(doc) for doc in request.documents]
        )

    async def call_provider(resolved: ResolvedProvider) -> ProviderRerankResponse:
        rerank_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "query": request.query,
            "documents": request.documents,
            "provider": resolved.provider,
            **resolved.kwargs,
        }
        if request.top_n is not None:
            rerank_kwargs["top_n"] = request.top_n
        if request.max_tokens_per_doc is not None:
            rerank_kwargs["max_tokens_per_doc"] = request.max_tokens_per_doc
        return await arerank(**rerank_kwargs)

    # Rerank results are returned verbatim and carry ``model``, which would echo
    # the target an alias exists to hide, so it is relabeled to the alias.
    outcome = await run_passthrough(
        endpoint=USAGE_ENDPOINT,
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request.model,
        user=request.user,
        call_provider=call_provider,
        request_params=request.model_fields_set,
        estimate=estimate,
        enforce_require_pricing=True,
        billing=RERANK_BILLING,
        inline_cost=True,
    )
    if outcome.inline_cost is not None:
        record_inline_cost_settlement("attached")
    return InferenceService.rerank_response(
        outcome.result, pricing=outcome.pricing, cost=outcome.cost, inline_cost=outcome.inline_cost
    )
