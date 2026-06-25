"""OpenAI-compatible image generation endpoint."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import aimage_generation
from any_llm.types.image import ImageGenerationParams, ImagesResponse
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
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
from gateway.services.pricing_service import find_model_pricing, pricing_required_but_missing
from gateway.services.provider_kwargs import resolve_provider_selector

router = APIRouter(prefix="/v1", tags=["images"])


class ImageGenerationRequest(derive_request_base(ImageGenerationParams)):  # type: ignore[misc]
    """OpenAI-compatible image generation request.

    Fields are derived from any-llm's ``ImageGenerationParams`` (see
    ``_schema_derive``) so the schema cannot silently drop a param any-llm
    forwards.
    """

    # any-llm types this as a ``Literal['url', 'b64_json']``; keep the permissive
    # ``str`` the gateway has always accepted across providers.
    response_format: str | None = None


@router.post("/images/generations", response_model=None)
async def create_image(
    raw_request: Request,
    response: Response,
    request: ImageGenerationRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """OpenAI-compatible image generation endpoint.

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

    # Image pricing repurposes input_price_per_million as price-per-image.
    pricing = await find_model_pricing(db, resolved.instance, model)

    # Reserve for the requested image count; reconciled to the actual count below.
    # `is None` (not `or`) so an explicit n=0 isn't silently treated as 1.
    requested_images = request.n if request.n is not None else 1
    estimate = requested_images * pricing.input_price_per_million if pricing else 0.0
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

    # Forward every field the schema accepts (it is derived from
    # ImageGenerationParams), so a new any-llm param is passed through without a
    # code change. `model` is replaced by the split short name passed explicitly;
    # gateway-internal (`user`) and sensitive fields are stripped.
    forward = _strip_gateway_fields(request.model_dump(exclude_unset=True))
    forward.pop("model", None)
    image_kwargs: dict[str, Any] = {
        "model": model,
        "provider": provider,
        **provider_kwargs,
        **forward,
    }

    try:
        result: ImagesResponse = await aimage_generation(**image_kwargs)

        n_images = len(result.data) if result.data else requested_images

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=resolved.instance,
            endpoint="/v1/images/generations",
            status="success",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        # Image pricing: repurpose input_price_per_million as price-per-image
        cost = 0.0
        if pricing:
            cost = n_images * pricing.input_price_per_million
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
            endpoint="/v1/images/generations",
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

    return result.model_dump()
