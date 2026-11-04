"""OpenAI-compatible audio transcription and speech endpoints."""

import io
import mimetypes
from typing import Annotated, Any

from any_llm import aspeech, atranscription
from any_llm.types.audio import AudioSpeechParams, Transcription
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import FormData, UploadFile
from starlette.requests import ClientDisconnect

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import (
    FORM_FIELD_SLACK,
    FORM_FRAMING_BYTES,
    bounded_body,
    form_fields,
    form_file,
    multipart_request_body,
)
from gateway.api.routes._passthrough import run_passthrough
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.models.api_keys import APIKey
from gateway.schemas.inference import TranscriptionForm
from gateway.services.inference import ONE_REQUEST_BILLING, TRANSCRIPTION_BILLING
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(tags=["audio"])

# See chat.USAGE_ENDPOINT.
USAGE_ENDPOINT_TRANSCRIPTIONS = "/v1/audio/transcriptions"
USAGE_ENDPOINT_SPEECH = "/v1/audio/speech"

# Maximum upload size for audio files (25 MB, matching OpenAI's limit)
_MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024
_MAX_AUDIO_BODY_BYTES = _MAX_AUDIO_UPLOAD_BYTES + FORM_FRAMING_BYTES
_AUDIO_TOO_LARGE = f"Audio file exceeds maximum upload size of {_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB"

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
        413: {"description": f"{_AUDIO_TOO_LARGE}."},
        422: {"description": "A required field is missing, or a field is the wrong type."},
    },
    openapi_extra=multipart_request_body(
        TranscriptionForm, {"file": "The audio to transcribe."}, required_files=("file",)
    ),
)
async def create_transcription(
    raw_request: Request,
    response: Response,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any]:
    """OpenAI-compatible audio transcription endpoint, taking ``multipart/form-data``.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use the shared "default" user
    """
    # Read here rather than as File parameters, which FastAPI parses before the
    # auth check runs, and through a capped body, so nobody can make the gateway
    # store more than one upload's limit.
    try:
        async with bounded_body(raw_request, _MAX_AUDIO_BODY_BYTES, _AUDIO_TOO_LARGE).form(
            max_files=1, max_fields=len(TranscriptionForm.model_fields) + FORM_FIELD_SLACK
        ) as form:
            return await _run_transcription(raw_request, response, auth_result, db, config, log_writer, form)
    except ClientDisconnect:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The upload was interrupted.") from None


async def _run_transcription(
    raw_request: Request,
    response: Response,
    auth_result: tuple[APIKey | None, bool],
    db: AsyncSession,
    config: GatewayConfig,
    log_writer: LogWriter,
    form: FormData,
) -> dict[str, Any]:
    fields = form_fields(form, TranscriptionForm)
    file = form_file(form, "file")
    # The body cap leaves room for the form around the file, so the file gets its own.
    if (file.size or 0) > _MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, _AUDIO_TOO_LARGE)
    forward = {name: value for name, value in fields.model_dump(exclude={"model", "user"}).items() if value is not None}

    async def call_provider(resolved: ResolvedProvider) -> Transcription:
        # Read here, after the rate limit and budget gates, so a refused request
        # never loads its file into memory.
        await file.seek(0)
        file_bytes = await file.read()
        transcription_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "file": _named_audio(file_bytes, file),
            "provider": resolved.provider,
            **resolved.kwargs,
            **forward,
        }
        return await atranscription(**transcription_kwargs)

    # Audio is exempt from require_pricing and has no measurable cost unit yet
    # (tokens, seconds, etc.), so the reservation estimate is 0; the reservation
    # still enforces existing per-user state (user exists, not blocked, not
    # already over budget). When pricing is configured, the cost is recorded as
    # an auditable per-request charge line. Only an explicitly configured rate
    # counts (pricing_use_defaults=False): models.dev quotes audio models such
    # as gpt-4o-transcribe per million tokens, which this per-request convention
    # would misread as a per-million-request rate.
    outcome = await run_passthrough(
        endpoint=USAGE_ENDPOINT_TRANSCRIPTIONS,
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=fields.model,
        user=fields.user,
        call_provider=call_provider,
        request_params=list(forward),
        lookup_pricing=True,
        pricing_use_defaults=False,
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


@router.post(
    "/audio/speech",
    response_model=None,
    responses={
        200: {
            "description": "Audio bytes in the requested format",
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

    # Audio is exempt from require_pricing and has no measurable cost unit yet
    # (tokens, seconds, characters, etc.), so the reservation estimate is 0; the
    # reservation still enforces existing per-user state. When pricing is
    # configured, the cost is recorded as an auditable per-request charge line.
    # Only an explicitly configured rate counts (pricing_use_defaults=False):
    # models.dev quotes speech models such as gpt-4o-mini-tts per million
    # tokens, which this per-request convention would misread as a
    # per-million-request rate.
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
        pricing_use_defaults=False,
        reserve_before_resolve=True,
        relabel=False,
        billing=ONE_REQUEST_BILLING,
    )

    content_type = _SPEECH_CONTENT_TYPES.get(request.response_format, "audio/mpeg")

    return StreamingResponse(
        content=iter([outcome.result]),
        media_type=content_type,
        headers=outcome.headers,
    )
