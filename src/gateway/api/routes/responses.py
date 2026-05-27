import json
import time
import uuid
from typing import Annotated, Any

from any_llm import AnyLLM, aresponses
from any_llm.types.completion import ChatCompletion, CompletionUsage
from any_llm.types.responses import ResponseStreamEvent
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseUsage
from openresponses_types.types import Usage as OpenResponsesUsage
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes.chat import (
    ChatCompletionRequest,
    chat_completions,
    get_provider_kwargs,
    log_usage,
    rate_limit_headers,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import validate_project_budget, validate_tag_budgets, validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.services.routing_policy_service import DEFAULT_ROUTING_MODEL, normalize_routing_model_selector
from gateway.streaming import RESPONSES_STREAM_FORMAT, streaming_generator

router = APIRouter(prefix="/v1", tags=["responses"])

_MASTER_KEY_USER_REQUIRED = "When using master key, 'user' field is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"


class ResponsesRequest(BaseModel):
    """OpenAI Responses API-compatible request."""

    model_config = ConfigDict(extra="allow")

    model: str = DEFAULT_ROUTING_MODEL
    input: Any
    instructions: str | None = None
    stream: bool = False
    user: str | None = None
    project_id: str | None = Field(default=None, description="Optional gateway project id for default routing")
    tags: dict[str, str] | None = Field(default=None, description="Optional trace tags for default routing")

    @field_validator("model", mode="before")
    @classmethod
    def normalize_model(cls, v: Any) -> str:
        """Accept Merge-style omitted/null/case-insensitive default_routing sentinels."""
        normalized = normalize_routing_model_selector(v)
        if not normalized:
            raise ValueError("model must not be blank")
        return normalized


def _usage_to_completion_usage(
    usage: ResponseUsage | OpenResponsesUsage | None,
) -> CompletionUsage | None:
    if usage is None:
        return None
    return CompletionUsage(
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _content_to_text(value: Any) -> str:
    """Convert Responses API input/content items into chat-compatible text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_content_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
        content = value.get("content")
        if content is not None:
            return _content_to_text(content)
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)
    return str(value)


def _response_input_to_chat_messages(input_payload: Any, instructions: str | None) -> list[dict[str, Any]]:
    """Translate a common Responses API input shape into chat messages."""
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    def append_message(role: Any, content: Any) -> None:
        role_value = str(role or "user")
        if role_value == "developer":
            role_value = "system"
        text = _content_to_text(content)
        if text:
            messages.append({"role": role_value, "content": text})

    if isinstance(input_payload, str):
        append_message("user", input_payload)
    elif isinstance(input_payload, dict):
        if "role" in input_payload:
            append_message(input_payload.get("role"), input_payload.get("content", input_payload.get("text")))
        else:
            append_message("user", input_payload)
    elif isinstance(input_payload, list):
        for item in input_payload:
            if isinstance(item, dict) and "role" in item:
                append_message(item.get("role"), item.get("content", item.get("text")))
            else:
                append_message("user", item)
    else:
        append_message("user", input_payload)

    if not messages:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Responses input must contain at least one message or text item",
        )
    return messages


def _chat_completion_text(completion: ChatCompletion) -> str:
    """Extract the first assistant text from a chat completion."""
    if not completion.choices:
        return ""
    message = completion.choices[0].message
    return _content_to_text(getattr(message, "content", None))


def _split_model_selector(model_selector: str) -> tuple[str | None, str]:
    """Split either provider:model or provider/model into provider/model parts."""
    if ":" in model_selector:
        provider, model = model_selector.split(":", 1)
        return provider or None, model
    if "/" in model_selector:
        provider, model = model_selector.split("/", 1)
        return provider or None, model
    return None, model_selector


def _served_metadata(provider: str, model: str) -> dict[str, str]:
    """Build Merge-style served model/vendor metadata."""
    return {
        "model": f"{provider}/{model}",
        "vendor": provider,
    }


def _metadata_from_model_selector(model_selector: str) -> dict[str, str] | None:
    """Build served metadata from a provider-qualified model selector."""
    provider, model = _split_model_selector(model_selector)
    if provider is None:
        return None
    return _served_metadata(provider, model)


def _set_served_headers(response: FastAPIResponse, metadata: dict[str, str]) -> None:
    """Expose the model/vendor that served a Responses request."""
    response.headers["X-Response-Model"] = metadata["model"]
    response.headers["X-Response-Vendor"] = metadata["vendor"]


def _chat_completion_to_response_payload(completion: ChatCompletion) -> dict[str, Any]:
    """Wrap a routed chat completion in an OpenAI Responses-like payload."""
    text = _chat_completion_text(completion)
    usage = completion.usage
    metadata = _metadata_from_model_selector(completion.model)
    payload: dict[str, Any] = {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": completion.created or int(time.time()),
        "status": "completed",
        "model": metadata["model"] if metadata else completion.model,
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
    }
    if metadata is not None:
        payload["vendor"] = metadata["vendor"]
    if usage is not None:
        payload["usage"] = {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    return payload


def _response_payload_with_served_metadata(
    payload: dict[str, Any],
    *,
    provider: str,
    requested_model: str,
) -> dict[str, Any]:
    """Attach Merge-style served metadata to a provider-native Responses payload."""
    model_value = payload.get("model")
    provider_from_payload: str | None = None
    model_from_payload = requested_model
    if isinstance(model_value, str):
        provider_from_payload, model_from_payload = _split_model_selector(model_value)

    served_provider = provider_from_payload or provider
    payload["model"] = _served_metadata(served_provider, model_from_payload)["model"]
    payload["vendor"] = served_provider
    return payload


def _chat_request_from_response_request(request_body: ResponsesRequest) -> ChatCompletionRequest:
    """Build a chat completion request for the standalone routing backend."""
    request_fields = request_body.model_dump(exclude_none=True)
    max_output_tokens = request_fields.get("max_output_tokens")
    max_tokens = max_output_tokens if isinstance(max_output_tokens, int) and max_output_tokens > 0 else None
    chat_request = ChatCompletionRequest(
        model=DEFAULT_ROUTING_MODEL,
        messages=_response_input_to_chat_messages(request_body.input, request_body.instructions),
        user=request_body.user,
        project_id=request_body.project_id,
        tags=request_body.tags,
        temperature=request_fields.get("temperature"),
        max_tokens=max_tokens,
        top_p=request_fields.get("top_p"),
        tools=request_fields.get("tools"),
        tool_choice=request_fields.get("tool_choice"),
        response_format=request_fields.get("response_format"),
    )
    chat_request.set_route_trace_endpoint("/v1/responses")
    return chat_request


@router.post("/responses", response_model=None)
async def create_response(
    raw_request: Request,
    response: FastAPIResponse,
    background_tasks: BackgroundTasks,
    request_body: ResponsesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible Responses endpoint."""
    if not config.is_platform_mode and request_body.model == DEFAULT_ROUTING_MODEL:
        if request_body.stream:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Routing policies do not support streaming responses yet",
            )
        chat_request = _chat_request_from_response_request(request_body)
        chat_result = await chat_completions(
            raw_request=raw_request,
            response=response,
            background_tasks=background_tasks,
            request=chat_request,
            db=db,
            config=config,
            log_writer=log_writer,
        )
        if isinstance(chat_result, StreamingResponse):
            return chat_result
        metadata = _metadata_from_model_selector(chat_result.model)
        if metadata is not None:
            _set_served_headers(response, metadata)
        return _chat_completion_to_response_payload(chat_result)

    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None

    user_id = resolve_user_id(
        user_id_from_request=request_body.user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_MASTER_KEY_USER_REQUIRED,
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_API_KEY_VALIDATION_FAILED,
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=_API_KEY_NO_USER,
        ),
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request_body.model, strategy=config.budget_strategy)
    if request_body.project_id is not None:
        _ = await validate_project_budget(
            db,
            request_body.project_id,
            request_body.model,
            strategy=config.budget_strategy,
        )
    _ = await validate_tag_budgets(db, request_body.tags, request_body.model, strategy=config.budget_strategy)
    if config.budget_strategy == "for_update":
        await db.rollback()

    provider, model = AnyLLM.split_model_provider(request_body.model)
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_RESPONSES", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider.value}' does not support the Responses API",
        )

    provider_kwargs = get_provider_kwargs(config, provider)

    request_fields = request_body.model_dump(exclude_none=True)
    input_payload = request_fields.pop("input")
    stream = bool(request_fields.pop("stream", False))
    request_fields.pop("model", None)
    request_fields.pop("user", None)
    request_fields.pop("project_id", None)
    request_fields.pop("tags", None)
    request_fields["user"] = user_id

    call_kwargs: dict[str, Any] = {**provider_kwargs}
    call_kwargs.update(request_fields)
    call_kwargs["model"] = model
    call_kwargs["provider"] = provider
    call_kwargs["input_data"] = input_payload

    try:
        if stream:
            call_kwargs["stream"] = True

            def _format_chunk(event: ResponseStreamEvent) -> str:
                return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

            def _extract_usage(event: ResponseStreamEvent) -> CompletionUsage | None:
                response_obj = getattr(event, "response", None)
                if response_obj and getattr(response_obj, "usage", None):
                    return _usage_to_completion_usage(response_obj.usage)
                return None

            async def _on_complete(usage_data: CompletionUsage) -> None:
                await log_usage(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/responses",
                    user_id=user_id,
                    project_id=request_body.project_id,
                    tags=request_body.tags,
                    usage_override=usage_data,
                )

            async def _on_error(error: str) -> None:
                await log_usage(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/responses",
                    user_id=user_id,
                    project_id=request_body.project_id,
                    tags=request_body.tags,
                    error=error,
                )

            stream_result = await aresponses(**call_kwargs)
            rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
            metadata = _served_metadata(provider.value, model)
            rl_headers["X-Response-Model"] = metadata["model"]
            rl_headers["X-Response-Vendor"] = metadata["vendor"]
            return StreamingResponse(
                streaming_generator(
                    stream=stream_result,  # type: ignore[arg-type]
                    format_chunk=_format_chunk,
                    extract_usage=_extract_usage,
                    fmt=RESPONSES_STREAM_FORMAT,
                    on_complete=_on_complete,
                    on_error=_on_error,
                    label=f"{provider}:{model}",
                ),
                media_type="text/event-stream",
                headers=rl_headers,
            )

        result = await aresponses(**call_kwargs)
        usage_data = _usage_to_completion_usage(getattr(result, "usage", None))
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            project_id=request_body.project_id,
            tags=request_body.tags,
            usage_override=usage_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            project_id=request_body.project_id,
            tags=request_body.tags,
            error=str(e),
        )
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    metadata = _served_metadata(provider.value, model)
    _set_served_headers(response, metadata)
    payload = result.model_dump(exclude_none=True)  # type: ignore[union-attr]
    return _response_payload_with_served_metadata(payload, provider=provider.value, requested_model=model)
