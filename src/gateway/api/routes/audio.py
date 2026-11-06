"""OpenAI-compatible audio transcription and speech endpoints."""

import io
import mimetypes
from decimal import Decimal
from typing import Annotated, Any

from any_llm import aspeech, atranscription
from any_llm.types.audio import AudioSpeechParams, Transcription
from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    get_config,
    get_db,
    get_log_writer,
    verify_api_key_or_master_key,
    verify_api_key_or_master_key_for_upload,
)
from gateway.api.routes._helpers import bounded_body
from gateway.api.routes._passthrough import run_passthrough
from gateway.api.routes._request_body import MULTIPART_OVERHEAD_BYTES, form_field
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import (
    COST_USD_HEADER,
    PRICING_SOURCE_HEADER,
    USAGE_UNIT_HEADER,
    USAGE_UNITS_HEADER,
    GatewayConfig,
)
from gateway.models.api_keys import APIKey
from gateway.models.pricing import ModelPricing
from gateway.services.budgets import report_estimate
from gateway.services.inference import TRANSCRIPTION_BILLING, speech_billing, speech_cost_headers
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(tags=["audio"])

# See chat.USAGE_ENDPOINT.
USAGE_ENDPOINT_TRANSCRIPTIONS = "/v1/audio/transcriptions"
USAGE_ENDPOINT_SPEECH = "/v1/audio/speech"

# Maximum upload size for audio files (25 MB, matching OpenAI's limit)
_MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
_UPLOAD_FILE_ADAPTER: TypeAdapter[UploadFile] = TypeAdapter(UploadFile)
_STRING_ADAPTER: TypeAdapter[str] = TypeAdapter(str)
_OPTIONAL_STRING_ADAPTER: TypeAdapter[str | None] = TypeAdapter(str | None)
_OPTIONAL_FLOAT_ADAPTER: TypeAdapter[float | None] = TypeAdapter(float | None)
_TRANSCRIPTION_BODY_SCHEMA = {
    "title": "Body_audio-create_transcription",
    "type": "object",
    "properties": {
        "file": {"contentMediaType": "application/octet-stream", "title": "File", "type": "string"},
        "language": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Language"},
        "model": {"title": "Model", "type": "string"},
        "prompt": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Prompt"},
        "response_format": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Response Format"},
        "temperature": {"anyOf": [{"type": "number"}, {"type": "null"}], "title": "Temperature"},
        "user": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "User"},
    },
    "required": ["file", "model"],
}

# Mapping from response_format to MIME type for speech endpoint
_SPEECH_CONTENT_TYPES: dict[str | None, str] = {
    None: "audio/mpeg",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/L16",
}


def _named_audio(data: bytes, upload: UploadFile) -> io.BytesIO:
    """Wrap an upload so providers see its file name and type.

    OpenAI-style upstreams infer the audio format from the multipart file name;
    bare bytes arrive nameless and are rejected as an unsupported format.
    """
    name = upload.filename
    if not name:
        extension = mimetypes.guess_extension(upload.content_type or "") or ".mp3"
        name = f"audio{extension}"
    named = io.BytesIO(data)
    named.name = name
    return named


@router.post(
    "/audio/transcriptions",
    response_model=None,
    responses={
        413: {"description": "The audio upload exceeds the maximum request size."},
        422: {
            "description": "Validation Error",
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}},
        },
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": _TRANSCRIPTION_BODY_SCHEMA}},
        }
    },
)
async def create_transcription(
    raw_request: Request,
    response: Response,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key_for_upload)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """OpenAI-compatible audio transcription endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """
    async with bounded_body(
        raw_request,
        _MAX_AUDIO_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES,
        f"Audio upload request exceeds the maximum size of {_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB.",
    ).form() as form:
        file = form_field(form, "file", _UPLOAD_FILE_ADAPTER, required=True)
        model = form_field(form, "model", _STRING_ADAPTER, required=True)
        language = form_field(form, "language", _OPTIONAL_STRING_ADAPTER)
        prompt = form_field(form, "prompt", _OPTIONAL_STRING_ADAPTER)
        response_format = form_field(form, "response_format", _OPTIONAL_STRING_ADAPTER)
        temperature = form_field(form, "temperature", _OPTIONAL_FLOAT_ADAPTER)
        user = form_field(form, "user", _OPTIONAL_STRING_ADAPTER)

        async def call_provider(resolved: ResolvedProvider) -> Transcription:
            if file.size is not None and file.size > _MAX_AUDIO_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio file exceeds maximum upload size of {_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            await file.seek(0)
            file_bytes = await file.read()

            transcription_kwargs: dict[str, Any] = {
                "model": resolved.model,
                "file": _named_audio(file_bytes, file),
                "provider": resolved.provider,
                **resolved.kwargs,
            }
            if language is not None:
                transcription_kwargs["language"] = language
            if prompt is not None:
                transcription_kwargs["prompt"] = prompt
            if response_format is not None:
                transcription_kwargs["response_format"] = response_format
            if temperature is not None:
                transcription_kwargs["temperature"] = temperature

            return await atranscription(**transcription_kwargs)

        # Transcription is exempt from require_pricing. The reservation estimate
        # is 0, because the audio's length is not read here: the reservation still
        # enforces per-user state (user exists, not blocked, not already over
        # budget), and settlement bills the length or the tokens the provider
        # reports, or one request, as the rate prices them.
        outcome = await run_passthrough(
            endpoint=USAGE_ENDPOINT_TRANSCRIPTIONS,
            raw_request=raw_request,
            response=response,
            auth_result=auth_result,
            db=db,
            config=config,
            log_writer=log_writer,
            model=model,
            user=user,
            call_provider=call_provider,
            request_params=[
                name
                for name, value in (
                    ("language", language),
                    ("prompt", prompt),
                    ("response_format", response_format),
                    ("temperature", temperature),
                )
                if value is not None
            ],
            lookup_pricing=True,
            reserve_before_resolve=True,
            relabel=False,
            billing=TRANSCRIPTION_BILLING,
        )
        return outcome.result.model_dump()


class AudioSpeechRequest(derive_request_base(AudioSpeechParams)):  # type: ignore[misc]
    """OpenAI-compatible audio speech (TTS) request.

    The speech fields are derived from any-llm's ``AudioSpeechParams`` (see
    ``_schema_derive``) so the schema cannot silently drop a param any-llm
    forwards. ``user`` is gateway-only (billing / auth scoping); it is not an
    any-llm param and is stripped before the request is forwarded.
    """

    # any-llm types this as a provider-specific ``Literal``; keep the permissive
    # ``str`` the gateway has always accepted across providers.
    response_format: str | None = None
    user: str | None = None


_SPEECH_COST_HEADERS: dict[str, Any] = {
    COST_USD_HEADER: {
        "description": "What the request cost, as a six-decimal USD string; absent when it was not priced",
        "schema": {"type": "string"},
    },
    PRICING_SOURCE_HEADER: {
        "description": "Which rate priced it: organization, deployment or defaults; absent when it was not priced",
        "schema": {"type": "string"},
    },
    USAGE_UNIT_HEADER: {
        "description": "The meter the request was billed on, characters or requests; absent when nothing was billed",
        "schema": {"type": "string"},
    },
    USAGE_UNITS_HEADER: {
        "description": "How many of that meter were billed; absent when nothing was billed",
        "schema": {"type": "integer"},
    },
}


@router.post(
    "/audio/speech",
    response_model=None,
    responses={
        200: {
            "description": "Audio bytes in the requested format",
            "headers": _SPEECH_COST_HEADERS,
            "content": {
                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                "audio/opus": {"schema": {"type": "string", "format": "binary"}},
                "audio/aac": {"schema": {"type": "string", "format": "binary"}},
                "audio/flac": {"schema": {"type": "string", "format": "binary"}},
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "audio/L16": {"schema": {"type": "string", "format": "binary"}},
            },
        },
    },
)
async def create_speech(
    raw_request: Request,
    response: Response,
    request: AudioSpeechRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> StreamingResponse:
    """OpenAI-compatible audio speech (TTS) endpoint.

    Billed on the characters of ``input`` at a rate priced per character, or
    one request at a rate priced per request. A model with no rate is refused
    under ``require_pricing``. The body is audio, so the settled cost rides
    the response headers: ``Otari-Cost-USD`` and ``Otari-Pricing-Source``, the
    ``cost_usd`` and ``pricing_source`` other responses carry on ``usage``, and
    ``Otari-Usage-Unit`` and ``Otari-Usage-Units`` for the meter billed and
    its count.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """

    async def call_provider(resolved: ResolvedProvider) -> bytes:
        # Forward every field the schema accepts (it is derived from
        # AudioSpeechParams), so a new any-llm param is passed through without a
        # code change. `model` is replaced by the split short name passed
        # explicitly; gateway-internal (`user`) and sensitive fields are stripped.
        forward = _strip_gateway_fields(request.model_dump(exclude_unset=True))
        forward.pop("model", None)
        speech_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "provider": resolved.provider,
            **resolved.kwargs,
            **forward,
        }
        return await aspeech(**speech_kwargs)

    # The characters spoken are known before the call, so they are held at a
    # per-character rate, or one request at a per-request one; a speech result
    # reports nothing more to settle on, which is why a token rate prices it
    # nothing and ``require_pricing`` refuses it as it refuses a model with no rate.
    characters = len(request.input)
    billing = speech_billing(characters)

    def estimate(pricing: ModelPricing | None) -> Decimal:
        return report_estimate(pricing, slot=billing.slot, quantities={"character": characters}, units={"requests": 1})

    outcome = await run_passthrough(
        endpoint=USAGE_ENDPOINT_SPEECH,
        raw_request=raw_request,
        response=None,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request.model,
        user=request.user,
        call_provider=call_provider,
        request_params=request.model_fields_set,
        lookup_pricing=True,
        estimate=estimate,
        enforce_require_pricing=True,
        relabel=False,
        billing=billing,
    )

    content_type = _SPEECH_CONTENT_TYPES.get(request.response_format, "audio/mpeg")

    return StreamingResponse(
        content=iter([outcome.result]),
        media_type=content_type,
        headers={**outcome.headers, **speech_cost_headers(pricing=outcome.pricing, charge=outcome.charge)},
    )
