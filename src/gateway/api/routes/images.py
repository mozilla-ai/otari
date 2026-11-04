"""OpenAI-compatible image generation and edit endpoints."""

from decimal import Decimal
from typing import Annotated, Any

from any_llm import aimage_edit, aimage_generation
from any_llm.types.image import ImageGenerationParams, ImagesResponse, sniff_image_format
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
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
USAGE_ENDPOINT_EDITS = "/v1/images/edits"

# OpenAI's limits for gpt-image edits: at most 16 images, each under 50 MB.
_MAX_EDIT_IMAGES = 16
_MAX_EDIT_IMAGE_BYTES = 50 * 1024 * 1024


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


async def _upload_part(upload: UploadFile, field: str) -> tuple[str, bytes, str | None]:
    """Read one uploaded file into the ``(filename, content, content_type)`` tuple any-llm takes.

    A tuple, never a ``str``: any-llm reads a string image as a path on this
    server. A missing or generic content type is replaced by the type the bytes
    show, because OpenAI rejects ``application/octet-stream`` image parts and
    Gemini needs the type.
    """
    data = await upload.read()
    if len(data) > _MAX_EDIT_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"{field} exceeds the maximum size of {_MAX_EDIT_IMAGE_BYTES // (1024 * 1024)} MB",
        )
    content_type = upload.content_type
    if not content_type or content_type == "application/octet-stream":
        sniffed = sniff_image_format(data)
        content_type = sniffed[1] if sniffed else None
    return upload.filename or field, data, content_type


# The form fields are declared by hand: ``derive_request_base`` cannot build a
# model from ``ImageEditParams``, whose ``image`` is a file type pydantic has no
# schema for. ``test_edit_form_accepts_every_image_edit_param`` fails when
# any-llm adds a field this route does not accept.
@router.post("/images/edits", response_model=None)
async def edit_image(
    raw_request: Request,
    response: Response,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    model: Annotated[str, Form()],
    prompt: Annotated[str, Form()],
    image: Annotated[list[UploadFile] | None, File(description="The image or images to edit.")] = None,
    # OpenAI's SDKs send several images as ``image[]``.
    image_list: Annotated[list[UploadFile] | None, File(alias="image[]", description="Same as image.")] = None,
    mask: Annotated[UploadFile | None, File()] = None,
    n: Annotated[int | None, Form()] = None,
    size: Annotated[str | None, Form()] = None,
    quality: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
    user: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """OpenAI-compatible image edit endpoint, taking ``multipart/form-data``.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """
    uploads = [*(image or []), *(image_list or [])]
    if not uploads:
        raise RequestValidationError([{"type": "missing", "loc": ("body", "image"), "msg": "Field required"}])
    if len(uploads) > _MAX_EDIT_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At most {_MAX_EDIT_IMAGES} images can be edited in one request",
        )
    images = [await _upload_part(upload, "image") for upload in uploads]
    mask_part = await _upload_part(mask, "mask") if mask is not None else None

    form = {"n": n, "size": size, "quality": quality, "response_format": response_format, "user": user}
    set_fields: dict[str, Any] = {name: value for name, value in form.items() if value is not None}
    if mask_part is not None:
        set_fields["mask"] = mask_part
    forward = _strip_gateway_fields(dict(set_fields))
    requested_images = n if n is not None else 1

    def estimate(pricing: ModelPricing | None) -> Decimal:
        return count_estimate(pricing, meter="images", count=requested_images)

    async def call_provider(resolved: ResolvedProvider) -> ImagesResponse:
        edit_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "provider": resolved.provider,
            **resolved.kwargs,
            **forward,
            # One image goes as one, not a one-item list, as a direct any-llm call would send it.
            "image": images[0] if len(images) == 1 else images,
            "prompt": prompt,
        }
        return await aimage_edit(**edit_kwargs)

    # Billed exactly as generation is, per image returned; see create_image for
    # why only an explicitly configured rate counts.
    outcome = await run_passthrough(
        endpoint=USAGE_ENDPOINT_EDITS,
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=model,
        user=user,
        call_provider=call_provider,
        request_params=set_fields.keys(),
        pricing_use_defaults=False,
        estimate=estimate,
        enforce_require_pricing=True,
        billing=IMAGES_BILLING,
        relabel=False,
    )
    return outcome.result.model_dump()
