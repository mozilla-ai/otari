"""Rerank endpoint — reorder documents by relevance to a query."""

from typing import Annotated, Any

from any_llm import arerank
from any_llm.types.rerank import RerankResponse
from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._passthrough import run_passthrough
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, ModelPricing
from gateway.services.budget_service import estimate_cost
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import input_token_cost
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(prefix="/v1", tags=["rerank"])


class RerankRequest(BaseModel):
    """Rerank request."""

    model: str = Field(description="Provider-prefixed model ID, e.g. 'cohere:rerank-v3.5'")
    query: str = Field(description="The search query to rerank documents against")
    documents: list[str] = Field(description="List of document strings to rerank", min_length=1)
    top_n: int | None = Field(default=None, description="Maximum number of results to return", gt=0)
    max_tokens_per_doc: int | None = Field(default=None, description="Per-document truncation limit", gt=0)
    user: str | None = Field(default=None, description="User ID for usage attribution")


@router.post("/rerank", response_model=None)
async def create_rerank(
    raw_request: Request,
    response: Response,
    request: RerankRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> Any:
    """Rerank documents by relevance to a query.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """
    # Rerank bills on total tokens (input only). Estimate from query + documents.
    prompt_chars = len(request.query) + sum(len(doc) for doc in request.documents)

    def estimate(pricing: ModelPricing | None) -> float:
        return estimate_cost(pricing, prompt_chars=prompt_chars, max_output_tokens=None, default_output_tokens=0)

    def usage_tokens(result: RerankResponse) -> tuple[int | None, int | None, int | None]:
        total_tokens = result.usage.total_tokens if result.usage else None
        return (total_tokens, 0, total_tokens)

    def compute_cost(result: RerankResponse, pricing: ModelPricing | None) -> float | None:
        total_tokens = result.usage.total_tokens if result.usage else None
        if result.usage and pricing and total_tokens:
            return input_token_cost(total_tokens, pricing)
        return None

    async def call_provider(resolved: ResolvedProvider) -> RerankResponse:
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

    # Rerank results are returned verbatim and some providers (Voyage, Jina)
    # carry ``model``, which would echo the target an alias exists to hide. The
    # relabeling is a no-op on results without the field.
    outcome = await run_passthrough(
        endpoint="/v1/rerank",
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request.model,
        user=request.user,
        call_provider=call_provider,
        estimate=estimate,
        enforce_require_pricing=True,
        usage_tokens=usage_tokens,
        compute_cost=compute_cost,
    )
    return outcome.result
