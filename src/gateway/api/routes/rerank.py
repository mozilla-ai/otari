"""Rerank endpoint — reorder documents by relevance to a query."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import arerank
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

    pricing = await find_model_pricing(db, resolved.instance, model)

    # Rerank bills on total tokens (input only). Estimate from query + documents.
    prompt_chars = len(request.query) + sum(len(doc) for doc in request.documents)
    estimate = estimate_cost(pricing, prompt_chars=prompt_chars, max_output_tokens=None, default_output_tokens=0)
    # Reserve first so 404/403 precede the missing-pricing 402; refund on reject.
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

    rerank_kwargs: dict[str, Any] = {
        "model": model,
        "query": request.query,
        "documents": request.documents,
        "provider": provider,
        **provider_kwargs,
    }
    if request.top_n is not None:
        rerank_kwargs["top_n"] = request.top_n
    if request.max_tokens_per_doc is not None:
        rerank_kwargs["max_tokens_per_doc"] = request.max_tokens_per_doc

    try:
        result = await arerank(**rerank_kwargs)

        total_tokens = result.usage.total_tokens if result.usage else None

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=resolved.instance,
            endpoint="/v1/rerank",
            status="success",
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        )

        cost = 0.0
        if result.usage and pricing and total_tokens:
            cost = (total_tokens / 1_000_000) * pricing.input_price_per_million
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
            endpoint="/v1/rerank",
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
