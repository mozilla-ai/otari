"""OpenAI-compatible embeddings endpoint."""

from decimal import Decimal
from typing import Annotated, Any

from any_llm import aembedding
from any_llm.types.completion import CreateEmbeddingResponse
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
from gateway.schemas.inference import EmbeddingResponse
from gateway.services.budgets import estimate_cost
from gateway.services.inference import EMBEDDINGS_BILLING, InferenceService
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(tags=["embeddings"])

# See chat.USAGE_ENDPOINT.
USAGE_ENDPOINT = "/v1/embeddings"


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str
    input: str | list[str] = Field(description="Input text to embed")
    user: str | None = None
    encoding_format: str | None = None
    dimensions: int | None = None
    input_type: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "What the input will be used for, for providers whose embedding depends on it: "
            "Cohere (`search_document`, `search_query`, `classification`, `clustering`, `image`) "
            "and Voyage (`query`, `document`). The provider validates the value. Any other provider "
            "answers 400, rather than embedding the input as if the field were absent."
        ),
    )


@router.post("/embeddings")
async def create_embedding(
    raw_request: Request,
    response: Response,
    request: EmbeddingRequest,
    auth_result: Annotated[tuple[APIKey | None, bool] | None, Depends(verify_api_key_or_master_key_if_needed)],
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> EmbeddingResponse:
    """OpenAI-compatible embeddings endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user

    A priced response carries its cost on `usage` as `cost_usd` (a six-decimal
    USD string) and `pricing_source`, as a chat completion does; an unpriced one
    carries neither.
    """
    # Embeddings have no generated output, so only the prompt contributes cost.
    prompt_chars = sum(len(s) for s in request.input) if isinstance(request.input, list) else len(request.input)

    def estimate(pricing: ModelPricing | None) -> Decimal:
        return estimate_cost(pricing, prompt_chars=prompt_chars, max_output_tokens=None, default_output_tokens=0)

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
        embedding_kwargs.update(
            InferenceService.embedding_provider_options(
                resolved.provider, selector=request.model, input_type=request.input_type
            )
        )
        return await aembedding(**embedding_kwargs)

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
        billing=EMBEDDINGS_BILLING,
        inline_cost=True,
    )
    body = InferenceService.embeddings_response(
        outcome.result, pricing=outcome.pricing, cost=outcome.cost, inline_cost=outcome.inline_cost
    )
    if outcome.inline_cost is not None or (outcome.cost is not None and outcome.pricing is not None):
        record_inline_cost_settlement("attached" if body.usage is not None else "unattached")
    return body
