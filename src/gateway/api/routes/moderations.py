"""OpenAI-compatible moderations endpoint."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import amoderation
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
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
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import find_model_pricing
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.types.moderation import ModerationResponse

# Locked phrasing — cross-SDK error contract. Do not reword.
UNSUPPORTED_MODERATION_SUBSTRING = "does not support moderation"

router = APIRouter(prefix="/v1", tags=["moderations"])


class ModerationRequest(BaseModel):
    """OpenAI-compatible moderation request."""

    model: str
    input: str | list[str] | list[dict[str, Any]] = Field(
        description="Text, list of texts, or list of content-part dicts to moderate",
    )
    user: str | None = None


@router.post("/moderations", response_model=ModerationResponse)
async def create_moderation(
    raw_request: Request,
    response: Response,
    request: ModerationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    include_raw: Annotated[bool, Query()] = False,
) -> ModerationResponse:
    """OpenAI-compatible moderations endpoint.

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

    # Moderations is exempt from require_pricing: it is free at most providers
    # and is intentionally treated as $0 when unpriced (see below). Pricing, when
    # present, is a flat per-request rate stored in input_price_per_million.
    pricing = await find_model_pricing(db, resolved.instance, model)
    flat_cost = (pricing.input_price_per_million / 1_000_000) if pricing and pricing.input_price_per_million else 0.0
    reservation = await reserve_budget(
        db, user_id, flat_cost, model=request.model, strategy=config.budget_strategy
    )

    provider_kwargs = resolved.kwargs

    moderation_kwargs: dict[str, Any] = {
        "model": model,
        "input": request.input,
        "provider": provider,
        "include_raw": include_raw,
        **provider_kwargs,
    }

    try:
        result = await amoderation(**moderation_kwargs)

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=resolved.instance,
            endpoint="/v1/moderations",
            status="success",
            prompt_tokens=None,
            completion_tokens=0,
            total_tokens=None,
        )

        # Flat per-request rate (moderation has no token usage); intentionally
        # no "No pricing configured" warning here — free at most providers.
        usage_log.cost = flat_cost

        await log_writer.put(usage_log)
        await reconcile_reservation(db, reservation, flat_cost)

    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except NotImplementedError as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=resolved.instance,
            endpoint="/v1/moderations",
            status="error",
            error_message=str(e),
        )
        await log_writer.put(error_log)
        await refund_reservation(db, reservation)
        if UNSUPPORTED_MODERATION_SUBSTRING in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e
        logger.error("Provider implementation gap for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=resolved.instance,
            endpoint="/v1/moderations",
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
