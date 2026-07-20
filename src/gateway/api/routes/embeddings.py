"""OpenAI-compatible embeddings endpoint."""

from typing import Annotated, Any

from any_llm import aembedding
from any_llm.types.completion import CreateEmbeddingResponse
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

router = APIRouter(prefix="/v1", tags=["embeddings"])


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str
    input: str | list[str] = Field(description="Input text to embed")
    user: str | None = None
    encoding_format: str | None = None
    dimensions: int | None = None


@router.post("/embeddings", response_model=None)
async def create_embedding(
    raw_request: Request,
    response: Response,
    request: EmbeddingRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> CreateEmbeddingResponse:
    """OpenAI-compatible embeddings endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    # Embeddings have no generated output, so only the prompt contributes cost.
    prompt_chars = sum(len(s) for s in request.input) if isinstance(request.input, list) else len(request.input)

    def estimate(pricing: ModelPricing | None) -> float:
        return estimate_cost(pricing, prompt_chars=prompt_chars, max_output_tokens=None, default_output_tokens=0)

    def usage_tokens(result: CreateEmbeddingResponse) -> tuple[int | None, int | None, int | None]:
        return (
            result.usage.prompt_tokens if result.usage else None,
            0,
            result.usage.total_tokens if result.usage else None,
        )

    def compute_cost(result: CreateEmbeddingResponse, pricing: ModelPricing | None) -> float | None:
        if result.usage and pricing:
            return input_token_cost(result.usage.prompt_tokens, pricing)
        return None

    async def call_provider(resolved: ResolvedProvider) -> CreateEmbeddingResponse:
        embedding_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "inputs": request.input,
            "provider": resolved.provider,
            **resolved.kwargs,
        }
        if request.encoding_format is not None:
            embedding_kwargs["encoding_format"] = request.encoding_format
        if request.dimensions is not None:
            embedding_kwargs["dimensions"] = request.dimensions
        return await aembedding(**embedding_kwargs)

    outcome = await run_passthrough(
        endpoint="/v1/embeddings",
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
