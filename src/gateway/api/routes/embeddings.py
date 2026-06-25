"""OpenAI-compatible embeddings endpoint."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import aembedding
from any_llm.types.completion import CreateEmbeddingResponse
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes.chat import rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import (
    estimate_cost,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing, pricing_required_but_missing
from gateway.services.provider_kwargs import resolve_provider_selector

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
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_user_id(
        user_id_from_request=request.user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When using master key, 'user' field is required in request body",
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation failed",
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key has no associated user",
        ),
        forbidden_user_error=HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="'user' field does not match the authenticated API key's user",
        ),
        reject_mismatch=config.reject_user_mismatch,
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    resolved = resolve_provider_selector(config, request.model)
    provider, model = resolved.provider, resolved.model

    # Resolve pricing for the reservation estimate and the final cost.
    pricing = await find_model_pricing(db, resolved.instance, model)

    # Embeddings have no generated output, so only the prompt contributes cost.
    prompt_chars = sum(len(s) for s in request.input) if isinstance(request.input, list) else len(request.input)
    estimate = estimate_cost(pricing, prompt_chars=prompt_chars, max_output_tokens=None, default_output_tokens=0)
    # Reserve first so user/blocked/budget rejections (404/403) precede the
    # missing-pricing rejection (402); refund if we then reject for no pricing.
    reservation = await reserve_budget(
        db, user_id, estimate, model=request.model, strategy=config.budget_strategy
    )
    if pricing_required_but_missing(pricing, require_pricing=config.require_pricing):
        await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"No pricing configured for model '{resolved.instance}:{model}'",
        )

    provider_kwargs = resolved.kwargs

    embedding_kwargs: dict[str, Any] = {
        "model": model,
        "inputs": request.input,
        "provider": provider,
        **provider_kwargs,
    }
    if request.encoding_format is not None:
        embedding_kwargs["encoding_format"] = request.encoding_format
    if request.dimensions is not None:
        embedding_kwargs["dimensions"] = request.dimensions

    try:
        result = await aembedding(**embedding_kwargs)

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=resolved.instance,
            endpoint="/v1/embeddings",
            status="success",
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=0,
            total_tokens=result.usage.total_tokens if result.usage else None,
        )

        cost = 0.0
        if result.usage and pricing:
            cost = (result.usage.prompt_tokens / 1_000_000) * pricing.input_price_per_million
            usage_log.cost = cost

        await log_writer.put(usage_log)
        await reconcile_reservation(db, reservation, cost)

    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=resolved.instance,
            endpoint="/v1/embeddings",
            status="error",
            error_message=str(e),
        )
        await log_writer.put(error_log)
        await refund_reservation(db, reservation)

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result
