"""OpenAI-compatible audio transcription and speech endpoints."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from any_llm import aspeech, atranscription
from any_llm.types.audio import AudioSpeechParams, Transcription
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
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
from gateway.services.provider_kwargs import resolve_provider_selector

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
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_user_id(
        user_id_from_request=user,
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

    # Audio is exempt from require_pricing and has no measurable cost unit yet,
    # so the reservation estimate is 0 — it still enforces existing per-user
    # state (user exists, not blocked, not already over budget).
    reservation = await reserve_budget(db, user_id, 0.0, model=model, strategy=config.budget_strategy)

    resolved = resolve_provider_selector(config, model)
    provider, model_name = resolved.provider, resolved.model

    provider_kwargs = resolved.kwargs

    file_bytes = await file.read()
    if len(file_bytes) > _MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum upload size of {_MAX_AUDIO_UPLOAD_BYTES // (1024 * 1024)} MB",
        )

    transcription_kwargs: dict[str, Any] = {
        "model": model_name,
        "file": file_bytes,
        "provider": provider,
        **provider_kwargs,
    }
    if language is not None:
        transcription_kwargs["language"] = language
    if prompt is not None:
        transcription_kwargs["prompt"] = prompt
    if response_format is not None:
        transcription_kwargs["response_format"] = response_format
    if temperature is not None:
        transcription_kwargs["temperature"] = temperature

    try:
        result: Transcription = await atranscription(**transcription_kwargs)

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model_name,
            provider=resolved.instance,
            endpoint="/v1/audio/transcriptions",
            status="success",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        # Audio transcription lacks a measurable usage unit (tokens, seconds, etc.)
        # so cost is left unset until a dedicated pricing metric is available.

        await log_writer.put(usage_log)
        await reconcile_reservation(db, reservation, 0.0)

    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model_name,
            provider=resolved.instance,
            endpoint="/v1/audio/transcriptions",
            status="error",
            error_message=str(e),
        )
        await log_writer.put(error_log)
        await refund_reservation(db, reservation)

        logger.error("Provider call failed for %s:%s: %s", provider, model_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result.model_dump()


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

    # Audio is exempt from require_pricing and has no measurable cost unit yet,
    # so the reservation estimate is 0 — it still enforces existing per-user
    # state (user exists, not blocked, not already over budget).
    reservation = await reserve_budget(db, user_id, 0.0, model=request.model, strategy=config.budget_strategy)

    resolved = resolve_provider_selector(config, request.model)
    provider, model_name = resolved.provider, resolved.model

    provider_kwargs = resolved.kwargs

    # Forward every field the schema accepts (it is derived from
    # AudioSpeechParams), so a new any-llm param is passed through without a code
    # change. `model` is replaced by the split short name passed explicitly;
    # gateway-internal (`user`) and sensitive fields are stripped.
    forward = _strip_gateway_fields(request.model_dump(exclude_unset=True))
    forward.pop("model", None)
    speech_kwargs: dict[str, Any] = {
        "model": model_name,
        "provider": provider,
        **provider_kwargs,
        **forward,
    }

    try:
        audio_bytes: bytes = await aspeech(**speech_kwargs)

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model_name,
            provider=resolved.instance,
            endpoint="/v1/audio/speech",
            status="success",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        # Audio speech lacks a measurable usage unit (tokens, seconds, characters, etc.)
        # so cost is left unset until a dedicated pricing metric is available.

        await log_writer.put(usage_log)
        await reconcile_reservation(db, reservation, 0.0)

    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model_name,
            provider=resolved.instance,
            endpoint="/v1/audio/speech",
            status="error",
            error_message=str(e),
        )
        await log_writer.put(error_log)
        await refund_reservation(db, reservation)

        logger.error("Provider call failed for %s:%s: %s", provider, model_name, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    content_type = _SPEECH_CONTENT_TYPES.get(request.response_format, "audio/mpeg")

    headers: dict[str, str] = {}
    if rate_limit_info:
        headers.update(rate_limit_headers(rate_limit_info))

    return StreamingResponse(
        content=iter([audio_bytes]),
        media_type=content_type,
        headers=headers,
    )
