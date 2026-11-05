"""OpenAI-compatible image generation and edit endpoints."""

from decimal import Decimal
from typing import Annotated, Any

from any_llm import aimage_edit, aimage_generation
from any_llm.types.image import ImageGenerationParams, ImagesResponse, sniff_image_format
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import FormData, UploadFile
from starlette.requests import ClientDisconnect

from gateway.api.deps import (
    get_config,
    get_db,
    get_log_writer,
    verify_api_key_or_master_key,
    verify_api_key_or_master_key_for_upload,
)
from gateway.api.routes._helpers import bounded_body
from gateway.api.routes._passthrough import PassthroughOutcome, run_passthrough
from gateway.api.routes._pipeline import record_inline_cost_settlement
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.pricing import ModelPricing
from gateway.schemas.inference import ImageEditForm
from gateway.services.budgets import report_estimate
from gateway.services.inference import InferenceService, image_billing, image_quantities
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(tags=["images"])

# See chat.USAGE_ENDPOINT.
USAGE_ENDPOINT = "/v1/images/generations"
USAGE_ENDPOINT_EDITS = "/v1/images/edits"

# OpenAI's limits for gpt-image edits: at most 16 images, each under 50 MB. The
# whole upload is held to one image's limit, so an edit can never hold more than
# that in memory, however many files it sends.
_MAX_EDIT_IMAGES = 16
_MAX_EDIT_UPLOAD_BYTES = 50 * 1024 * 1024
# Room for the multipart framing and the text fields around the files.
_MAX_EDIT_BODY_BYTES = _MAX_EDIT_UPLOAD_BYTES + 1024 * 1024
_EDIT_TOO_LARGE = f"The upload is larger than {_MAX_EDIT_UPLOAD_BYTES // (1024 * 1024)} MB"


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

    A priced response carries its cost on `usage` as `cost_usd` (a six-decimal
    USD string) and `pricing_source`, as a chat completion does; a provider that
    sends no `usage` gets one holding just those two fields. An unpriced
    response carries neither.
    """
    # Reserve for the requested image count at the size asked for; reconciled to
    # the images returned below. `is None` (not `or`) so an explicit n=0 isn't
    # silently treated as 1.
    requested_images = request.n if request.n is not None else 1
    billing = image_billing(request.size)

    def estimate(pricing: ModelPricing | None) -> Decimal:
        return report_estimate(
            pricing,
            slot=billing.slot,
            quantities=image_quantities(request.size, requested_images),
            units={"images": requested_images},
        )

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
        billing=billing,
        relabel=False,
    )
    return _priced_body(outcome)


def _priced_body(outcome: PassthroughOutcome[ImagesResponse]) -> dict[str, Any]:
    """The provider's body with the settled cost on ``usage``, never failing a served image over the cost."""
    if outcome.cost is None or outcome.pricing is None:
        return InferenceService.images_response(outcome.result, pricing=None, cost=None)
    try:
        body = InferenceService.images_response(outcome.result, pricing=outcome.pricing, cost=outcome.cost)
    except Exception as exc:  # noqa: BLE001 - the image is served and paid for; never fail it over the cost
        logger.warning("Failed to attach inline cost: %s", exc, exc_info=True)
        record_inline_cost_settlement("unattached")
        return InferenceService.images_response(outcome.result, pricing=None, cost=None)
    record_inline_cost_settlement("attached")
    return body


# ``image[]`` is how OpenAI's SDKs send several images.
EDIT_IMAGE_FIELDS = ("image", "image[]")
EDIT_FILE_FIELDS = (*EDIT_IMAGE_FIELDS, "mask")

_FILE_SCHEMA = {"type": "string", "contentMediaType": "application/octet-stream"}
_EDIT_FORM_SCHEMA = ImageEditForm.model_json_schema()
_EDIT_UPLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        **_EDIT_FORM_SCHEMA["properties"],
        "image": {"type": "array", "items": _FILE_SCHEMA, "description": "The image or images to edit."},
        "image[]": {"type": "array", "items": _FILE_SCHEMA, "description": "Same as image."},
        "mask": {**_FILE_SCHEMA, "description": "Where to edit: the transparent areas of this PNG."},
    },
    # ``image`` or ``image[]``, either one: the route checks that at least one file came.
    "required": _EDIT_FORM_SCHEMA["required"],
}


def _invalid(field: str, msg: str, error_type: str = "value_error") -> RequestValidationError:
    return RequestValidationError([{"type": error_type, "loc": ("body", field), "msg": msg}])


def _edit_fields(form: FormData) -> ImageEditForm:
    """The text fields, refused the way FastAPI refuses a body it cannot validate."""
    values = {name: form.get(name) for name in ImageEditForm.model_fields if name in form}
    for name, value in values.items():
        if not isinstance(value, str):
            raise _invalid(name, "Expected a text field, not a file")
    try:
        return ImageEditForm.model_validate(values)
    except ValidationError as exc:
        raise RequestValidationError(
            [{"type": e["type"], "loc": ("body", *e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        ) from exc


def _edit_files(form: FormData) -> tuple[list[UploadFile], UploadFile | None]:
    """The images and the mask, each refused when sent as text.

    A text ``image`` must never reach any-llm, which reads a string image as a
    path on this server.
    """
    uploads: list[UploadFile] = []
    for name in EDIT_IMAGE_FIELDS:
        for value in form.getlist(name):
            if not isinstance(value, UploadFile):
                raise _invalid("image", "Expected a file, not a text field")
            uploads.append(value)
    if not uploads:
        raise _invalid("image", "Field required", "missing")
    if len(uploads) > _MAX_EDIT_IMAGES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"At most {_MAX_EDIT_IMAGES} images can be edited in one request"
        )
    mask = form.get("mask")
    if mask is not None and not isinstance(mask, UploadFile):
        raise _invalid("mask", "Expected a file, not a text field")
    # The body cap leaves room for the form around the files, so the files get their own.
    files = [*uploads, *([mask] if mask is not None else [])]
    if sum(upload.size or 0 for upload in files) > _MAX_EDIT_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, _EDIT_TOO_LARGE)
    return uploads, mask


async def _upload_part(upload: UploadFile, field: str) -> tuple[str, bytes, str | None]:
    """Read one uploaded file into the ``(filename, content, content_type)`` tuple any-llm takes.

    A tuple, never a ``str``: any-llm reads a string image as a path on this
    server. A missing or generic content type is replaced by the type the bytes
    show, because OpenAI rejects ``application/octet-stream`` image parts and
    Gemini needs the type.
    """
    await upload.seek(0)
    data = await upload.read()
    content_type = upload.content_type
    if not content_type or content_type == "application/octet-stream":
        sniffed = sniff_image_format(data)
        content_type = sniffed[1] if sniffed else None
    return upload.filename or field, data, content_type


@router.post(
    "/images/edits",
    response_model=None,
    responses={
        413: {"description": f"The upload is larger than {_MAX_EDIT_UPLOAD_BYTES // (1024 * 1024)} MB."},
        422: {"description": "A required field is missing, or a field is the wrong type."},
    },
    openapi_extra={
        "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": _EDIT_UPLOAD_SCHEMA}}}
    },
)
async def edit_image(
    raw_request: Request,
    response: Response,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key_for_upload)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """OpenAI-compatible image edit endpoint, taking ``multipart/form-data``.

    Authentication modes:
    - Master key: the ``user`` field is required and names an existing user.
    - API key: usage and spend go to the key's own user. A ``user`` naming a
      different user is refused with 403, or ignored when mismatch rejection is off.

    A priced response carries its cost on `usage` as `cost_usd` (a six-decimal
    USD string) and `pricing_source`, as a chat completion does; a provider that
    sends no `usage` gets one holding just those two fields. An unpriced
    response carries neither.
    """
    # Read here rather than as File parameters, which FastAPI parses before the
    # auth check runs, and through a capped body, so nobody can make the gateway
    # store more than one upload's limit.
    try:
        async with bounded_body(raw_request, _MAX_EDIT_BODY_BYTES, _EDIT_TOO_LARGE).form(
            max_files=_MAX_EDIT_IMAGES + 1, max_fields=len(ImageEditForm.model_fields) + 16
        ) as form:
            return await _run_edit(raw_request, response, auth_result, db, config, log_writer, form)
    except ClientDisconnect:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The upload was interrupted.") from None


async def _run_edit(
    raw_request: Request,
    response: Response,
    auth_result: tuple[APIKey | None, bool],
    db: AsyncSession,
    config: GatewayConfig,
    log_writer: LogWriter,
    form: FormData,
) -> dict[str, Any]:
    fields = _edit_fields(form)
    uploads, mask = _edit_files(form)

    set_fields = fields.model_dump(exclude_unset=True, exclude={"model", "prompt"})
    forward = _strip_gateway_fields(dict(set_fields))
    request_params = {*set_fields, *(("mask",) if mask is not None else ())}
    requested_images = fields.n if fields.n is not None else 1
    billing = image_billing(fields.size)

    def estimate(pricing: ModelPricing | None) -> Decimal:
        return report_estimate(
            pricing,
            slot=billing.slot,
            quantities=image_quantities(fields.size, requested_images),
            units={"images": requested_images},
        )

    async def call_provider(resolved: ResolvedProvider) -> ImagesResponse:
        # Read here, after the rate limit, budget and pricing gates, so a refused
        # request never loads its files into memory.
        images = [await _upload_part(upload, "image") for upload in uploads]
        mask_part = {"mask": await _upload_part(mask, "mask")} if mask is not None else {}
        edit_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "provider": resolved.provider,
            **resolved.kwargs,
            **forward,
            **mask_part,
            # One image goes as one, not a one-item list, as a direct any-llm call would send it.
            "image": images[0] if len(images) == 1 else images,
            "prompt": fields.prompt,
        }
        return await aimage_edit(**edit_kwargs)

    # Billed exactly as generation is.
    outcome = await run_passthrough(
        endpoint=USAGE_ENDPOINT_EDITS,
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=fields.model,
        user=fields.user,
        call_provider=call_provider,
        request_params=request_params,
        estimate=estimate,
        enforce_require_pricing=True,
        billing=billing,
        relabel=False,
    )
    return _priced_body(outcome)
