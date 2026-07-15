"""OpenAI-compatible audio transcription and speech endpoints."""

from typing import Annotated, Any

from any_llm import aspeech, atranscription
from any_llm.types.audio import AudioSpeechParams, Transcription
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._passthrough import run_passthrough
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.log_writer import LogWriter
from gateway.services.provider_kwargs import ResolvedProvider

router = APIRouter(prefix="/v1", tags=["audio"])

# Maximum upload size for audio files (25 MB, matching OpenAI's limit)
_MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024

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


@router.post("/audio/transcriptions", response_model=None)
async def create_transcription(
    raw_request: Request,
    response: Response,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str | None = Form(None),
    temperature: float | None = Form(None),
    user: str | None = Form(None),
) -> dict[str, Any]:
    """OpenAI-compatible audio transcription endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """

    async def call_provider(resolved: ResolvedProvider) -> Transcription:
        file_bytes = await file.read()
        if len(file_bytes) > _MAX_AUDIO_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio file exceeds maximum upload size of {_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB",
            )

        transcription_kwargs: dict[str, Any] = {
            "model": resolved.model,
            "file": file_bytes,
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

    # Audio is exempt from require_pricing and has no measurable cost unit yet
    # (tokens, seconds, etc.), so the reservation estimate is 0 and cost stays
    # unset; the reservation still enforces existing per-user state (user
    # exists, not blocked, not already over budget).
    outcome = await run_passthrough(
        endpoint="/v1/audio/transcriptions",
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=model,
        user=user,
        call_provider=call_provider,
        lookup_pricing=False,
        reserve_before_resolve=True,
        relabel=False,
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
    - API key without user field: Use virtual user created with API key
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
    # (tokens, seconds, characters, etc.), so the reservation estimate is 0 and
    # cost stays unset; the reservation still enforces existing per-user state.
    outcome = await run_passthrough(
        endpoint="/v1/audio/speech",
        raw_request=raw_request,
        response=None,
        auth_result=auth_result,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request.model,
        user=request.user,
        call_provider=call_provider,
        lookup_pricing=False,
        reserve_before_resolve=True,
        relabel=False,
    )

    content_type = _SPEECH_CONTENT_TYPES.get(request.response_format, "audio/mpeg")

    return StreamingResponse(
        content=iter([outcome.result]),
        media_type=content_type,
        headers=outcome.headers,
    )
