"""OpenAI-compatible image generation endpoint."""

from decimal import Decimal
from typing import Annotated, Any

from any_llm import aimage_generation
from any_llm.types.image import ImageGenerationParams, ImagesResponse
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer, verify_api_key_or_master_key_if_needed
from gateway.api.routes._passthrough import run_passthrough
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.models.api_keys import APIKey
from gateway.models.pricing import ModelPricing
from gateway.services.budgets import count_estimate
from gateway.services.inference import IMAGES_BILLING
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(tags=["images"])

# See chat.USAGE_ENDPOINT.
USAGE_ENDPOINT = "/v1/images/generations"


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
    auth_result: Annotated[tuple[APIKey | None, bool] | None, Depends(verify_api_key_or_master_key_if_needed)],
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """OpenAI-compatible image generation endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """
    # Reserve for the requested image count; reconciled to the actual count below.
    # `is None` (not `or`) so an explicit n=0 isn't silently treated as 1.
    requested_images = request.n if request.n is not None else 1

    def estimate(pricing: ModelPricing | None) -> Decimal:
        return count_estimate(pricing, meter="images", count=requested_images)

    async def call_provider(resolved: ResolvedProvider) -> ImagesResponse:
        # Forward every field the schema accepts (it is derived from
        # ImageGenerationParams), so a new any-llm param is passed through without
        # a code change. `model` is replaced by the split short name passed
        # explicitly; gateway-internal (`user`) and sensitive fields are stripped.
        forward = _strip_gateway_fields(request.model_dump(exclude_unset=True))
        forward.pop("model", None)
        image_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "provider": resolved.provider,
            **resolved.kwargs,
            **forward,
        }
        return await aimage_generation(**image_kwargs)

    # Only an explicitly configured rate counts (pricing_use_defaults=False), for
    # the same unit mismatch audio, moderations, and search avoid: a token rate is
    # still read per image here for now (see ``price_usage``'s legacy reading)
    # while models.dev quotes USD per million tokens. gpt-image-1 is in that
    # dataset at 5.0, which would bill $5.00 for one image and, because this route
    # reserves its estimate, hold that $5.00 against the budget before the call.
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
        pricing_use_defaults=False,
        estimate=estimate,
        enforce_require_pricing=True,
        billing=IMAGES_BILLING,
        relabel=False,
    )
    return outcome.result.model_dump()
