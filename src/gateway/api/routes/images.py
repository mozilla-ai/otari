"""OpenAI-compatible image generation endpoint."""

from typing import Annotated, Any

from any_llm import aimage_generation
from any_llm.types.image import ImageGenerationParams, ImagesResponse
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._passthrough import run_passthrough
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, ModelPricing
from gateway.services.log_writer import LogWriter
from gateway.services.pricing_service import per_image_cost
from gateway.services.provider_kwargs import ResolvedProvider

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
    # Reserve for the requested image count; reconciled to the actual count below.
    # `is None` (not `or`) so an explicit n=0 isn't silently treated as 1.
    requested_images = request.n if request.n is not None else 1

    def estimate(pricing: ModelPricing | None) -> float:
        return per_image_cost(requested_images, pricing) if pricing else 0.0

    def compute_cost(result: ImagesResponse, pricing: ModelPricing | None) -> float | None:
        if pricing is None:
            return None
        n_images = len(result.data) if result.data else requested_images
        return per_image_cost(n_images, pricing)

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

    outcome = await run_passthrough(
        endpoint="/v1/images/generations",
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
        compute_cost=compute_cost,
        relabel=False,
    )
    return outcome.result.model_dump()
