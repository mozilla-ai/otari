import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

import httpx
from any_llm import LLMProvider, acompletion
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CompletionUsage,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer
from gateway.api.routes._helpers import latest_user_text
from gateway.api.routes._normalize import normalize_request_messages
from gateway.api.routes._pipeline import (
    ALL_PROVIDERS_FAILED_DETAIL,
    ALL_PROVIDERS_TIMED_OUT_DETAIL,
    NO_RESOLVABLE_PROVIDER_DETAIL,
    PROVIDER_ERROR_DETAIL,
    SANDBOX_UNREACHABLE_DETAIL,
    WEB_SEARCH_UNREACHABLE_DETAIL,
    ErrorKind,
    classify_provider_error,
    default_attempt_kwargs,
    log_usage,
    prepare_gateway_tools,
    rate_limit_headers,
    resolve_request_context,
    run_platform_non_stream,
    run_single_attempt_stream,
    run_standalone_non_stream,
    run_streaming_with_fallback,
)
from gateway.api.routes._platform import ResolvedAttempt
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.core.usage import GatewayUsage
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import McpServerConfig
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import (
    MAX_TOOL_ITERATIONS_CAP,
    ToolBackend,
    inject_purpose_hints,
    mcp_tool_loop,
    mcp_tool_loop_stream,
)
from gateway.services.provider_kwargs import get_provider_kwargs as get_provider_kwargs  # noqa: F401
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import OPENAI_STREAM_FORMAT, StreamFormat

router = APIRouter(prefix="/v1/chat", tags=["chat"])

__all__ = [
    "ChatCompletionRequest",
    "chat_completions",
    "get_provider_kwargs",
    "log_usage",
    "rate_limit_headers",
    "router",
]


class ChatCompletionRequest(derive_request_base(CompletionParams)):  # type: ignore[misc]
    """OpenAI-compatible chat completion request.

    The completion-param fields are derived from any-llm's ``CompletionParams``
    (see ``_schema_derive``) so the schema cannot silently drop a param any-llm
    forwards. Fields below either tighten a derived field (``messages``,
    ``response_format``) or add gateway-internal behavior (``mcp_servers``,
    ``mcp_server_ids``, ``guardrails``, ``tools_header``, ``max_tool_iterations``)
    that is stripped before the request is forwarded upstream.
    """

    messages: list[dict[str, Any]] = Field(min_length=1)
    # any-llm types this as ``dict | type | None``; the wire body only ever
    # carries the dict form.
    response_format: dict[str, Any] | None = None
    # any-llm types ``stream`` as ``bool | None``; keep the OpenAI wire contract
    # (a non-nullable boolean defaulting to false) for stable SDK generation.
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def validate_message_structure(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, message in enumerate(v):
            if "role" not in message:
                msg = f"messages[{i}]: 'role' is required"
                raise ValueError(msg)
        return v

    mcp_servers: list[McpServerConfig] | None = None
    mcp_server_ids: list[uuid.UUID] | None = None
    guardrails: list[GuardrailConfig] | None = Field(default=None, max_length=8)
    tools_header: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "Optional override for the lead-in that the gateway prepends before the "
            "per-tool hint block in the system message. Useful for expressing "
            "global tool-selection policy (e.g. 'prefer MCP tools over code_execution'). "
            "Falls back to OTARI_TOOLS_HEADER env, then to the built-in default."
        ),
    )
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)


class _ChatAdapter:
    """OpenAI Chat Completions edges of the shared pipeline.

    Provider-call and tool-loop functions are resolved as module globals at
    call time so tests can monkeypatch ``gateway.api.routes.chat.acompletion``
    and friends.
    """

    name = "chat"
    endpoint = "/v1/chat/completions"
    stream_format: StreamFormat = OPENAI_STREAM_FORMAT
    log_success_without_usage = True

    def error(self, status_code: int, message: str, kind: ErrorKind = ErrorKind.API) -> HTTPException:
        return HTTPException(status_code=status_code, detail=message)

    def provider_error(self, exc: BaseException) -> HTTPException:
        mapping = classify_provider_error(exc)
        if mapping is not None:
            return HTTPException(status_code=mapping.status_code, detail=mapping.detail)
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=PROVIDER_ERROR_DETAIL,
        )

    def format_chunk(self, chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    def extract_stream_usage(self, chunk: ChatCompletionChunk) -> CompletionUsage | None:
        if not chunk.usage:
            return None
        details = chunk.usage.prompt_tokens_details
        return GatewayUsage(
            prompt_tokens=chunk.usage.prompt_tokens or 0,
            completion_tokens=chunk.usage.completion_tokens or 0,
            total_tokens=chunk.usage.total_tokens or 0,
            prompt_tokens_details=details,
            cache_read_tokens=(details.cached_tokens or 0) if details is not None else 0,
        )

    def extract_usage(self, result: ChatCompletion) -> CompletionUsage | None:
        if result.usage is None:
            return None
        return GatewayUsage.from_completion_usage(result.usage)

    async def call_provider(self, kwargs: dict[str, Any]) -> ChatCompletion:
        return await acompletion(**kwargs)  # type: ignore[return-value]

    async def open_provider_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        return await acompletion(**kwargs)  # type: ignore[return-value]

    def prepare_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("stream_options") is None:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def run_tool_loop(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        on_first_response: Callable[[], None] | None = None,
    ) -> ChatCompletion:
        # Standalone dispatch has no lock-in callback; only pass the kwarg on
        # the platform-attempt path so test fakes can mirror each call shape.
        extra: dict[str, Any] = {}
        if on_first_response is not None:
            extra["on_first_response"] = on_first_response
        return await mcp_tool_loop(
            completion_kwargs=kwargs,
            pool=pool,
            max_iterations=max_iterations,
            **extra,
        )

    def open_tool_loop_stream(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
    ) -> AsyncIterator[ChatCompletionChunk]:
        return mcp_tool_loop_stream(
            completion_kwargs=kwargs,
            pool=pool,
            max_iterations=max_iterations,
        )

    def inject_hints(
        self,
        kwargs: dict[str, Any],
        hints: list[tuple[str, str]],
        *,
        header: str | None,
    ) -> dict[str, Any]:
        return {
            **kwargs,
            "messages": inject_purpose_hints(kwargs["messages"], hints, header=header),
        }

    def attempt_kwargs(
        self,
        attempt: ResolvedAttempt,
        base_request_fields: dict[str, Any],
    ) -> dict[str, Any]:
        return default_attempt_kwargs(attempt, base_request_fields)

    def prepare_platform_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs


_ADAPTER = _ChatAdapter()

_MASTER_KEY_USER_REQUIRED = "When using master key, 'user' field is required in request body"
_USER_FORBIDDEN = "'user' field does not match the authenticated API key's user"


@router.post("/completions", response_model=None)
async def chat_completions(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: ChatCompletionRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    Handles reasoning content from otari providers.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    if not request.model.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request: model is required",
        )

    async def _normalize(
        user_id: str, provider: LLMProvider | None, model: str, instance: str | None
    ) -> tuple[int, CompletionUsage | None]:
        # Resolve uploaded file/image blocks into the wire payload (extract to
        # text for text-only models, inline for natively-capable ones) before
        # the cost estimate. Standalone only; no-op when the files feature is
        # off or the request has no attachments.
        request.messages, stats = await normalize_request_messages(
            request.messages,
            fmt="openai",
            config=config,
            provider=provider,
            model=model,
            db=db,
            raw_request=raw_request,
            user_id=user_id,
            instance=instance,
        )
        return len(str(request.messages)), stats.vision_usage()

    ctx = await resolve_request_context(
        adapter=_ADAPTER,
        raw_request=raw_request,
        response=response,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request.model,
        user_id_from_request=request.user,
        estimate_prompt_chars=len(str(request.messages)),
        estimate_max_output_tokens=(
            request.max_tokens if request.max_tokens is not None else request.max_completion_tokens
        ),
        master_key_user_required_detail=_MASTER_KEY_USER_REQUIRED,
        user_forbidden_detail=_USER_FORBIDDEN,
        normalize_messages=_normalize,
    )

    tool_ctx = await prepare_gateway_tools(
        adapter=_ADAPTER,
        ctx=ctx,
        response=response,
        guardrails=request.guardrails,
        guardrail_text=latest_user_text(request.messages),
        tools=request.tools,
        mcp_servers=request.mcp_servers,
        mcp_server_ids=request.mcp_server_ids,
        max_tool_iterations=request.max_tool_iterations,
        tools_header=request.tools_header,
    )

    request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tool_ctx.tools_extracted,
        remaining_user_tools=tool_ctx.remaining_user_tools,
    )

    # ------------------------------------------------------------------
    # Streaming path: in hybrid mode, iterate `route.attempts` before any
    # bytes are flushed, then commit to the first attempt that yields a chunk
    # (tool modes included, so they get per-attempt fallback up to the
    # lock-in point). Mid-stream failover (after first chunk) is out of
    # scope: errors after first chunk propagate to the client.
    # ------------------------------------------------------------------
    if request.stream:
        if ctx.hybrid_mode:
            route = ctx.route
            if route is None or not route.attempts:
                if route is not None:
                    logger.error(
                        "Platform returned empty attempts list request_id=%s",
                        route.request_id,
                    )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=NO_RESOLVABLE_PROVIDER_DETAIL,
                )
            try:
                return await run_streaming_with_fallback(
                    adapter=_ADAPTER,
                    route=route,
                    base_request_fields=request_fields,
                    config=config,
                    background_tasks=background_tasks,
                    rate_limit_info=ctx.rate_limit_info,
                    tool_ctx=tool_ctx,
                )
            except HTTPException:
                raise
            except SandboxNotReachableError as exc:
                logger.error("Sandbox unreachable request_id=%s: %s", route.request_id, exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=SANDBOX_UNREACHABLE_DETAIL,
                ) from exc
            except WebSearchNotReachableError as exc:
                logger.error("Web search backend unreachable request_id=%s: %s", route.request_id, exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=WEB_SEARCH_UNREACHABLE_DETAIL,
                ) from exc
            except Exception as exc:
                # Every attempt failed before any bytes were flushed.
                logger.error(
                    "All streaming attempts failed request_id=%s: %s",
                    route.request_id,
                    exc,
                )
                # Classify when there is a single attempt, or when the failure is
                # a non-retryable invalid request (400/422): that short-circuits
                # the fallback and is definitive regardless of attempt count, so
                # it matches the non-streaming path. Otherwise surface the
                # multi-attempt aggregate.
                mapping = classify_provider_error(exc)
                invalid_request = mapping is not None and mapping.status_code == status.HTTP_400_BAD_REQUEST
                if invalid_request or len(route.attempts) <= 1:
                    raise _ADAPTER.provider_error(exc) from exc
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=ALL_PROVIDERS_TIMED_OUT_DETAIL,
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=ALL_PROVIDERS_FAILED_DETAIL,
                ) from exc

        # Standalone path: single attempt, no fallback (no `route.attempts`).
        # Resolve the instance to its implementation; dispatch any-llm against
        # ``implementation:model`` while billing/logging key on the instance.
        resolved = resolve_provider_selector(config, request.model)
        call_kwargs = {**resolved.kwargs, **request_fields, "model": resolved.dispatch_model}
        return await run_single_attempt_stream(
            adapter=_ADAPTER,
            ctx=ctx,
            tool_ctx=tool_ctx,
            call_kwargs=call_kwargs,
            provider=resolved.instance,
            model=resolved.model,
        )

    # ------------------------------------------------------------------
    # Non-streaming path. Hybrid mode iterates `route.attempts` with
    # pre-lock-in fallback semantics: once an attempt's tool loop has
    # received its first assistant message, subsequent failures terminate
    # the request; we never swap providers between tool-use rounds.
    # ------------------------------------------------------------------
    if ctx.hybrid_mode:
        route = ctx.route
        if route is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: missing route context",
            )
        return await run_platform_non_stream(
            adapter=_ADAPTER,
            route=route,
            base_request_fields=request_fields,
            tool_ctx=tool_ctx,
            response=response,
            background_tasks=background_tasks,
            config=config,
            rate_limit_info=ctx.rate_limit_info,
        )

    resolved = resolve_provider_selector(config, request.model)
    call_kwargs = {**resolved.kwargs, **request_fields, "model": resolved.dispatch_model}
    completion = await run_standalone_non_stream(
        adapter=_ADAPTER,
        ctx=ctx,
        tool_ctx=tool_ctx,
        call_kwargs=call_kwargs,
        provider=resolved.instance,
        model=resolved.model,
    )

    if ctx.rate_limit_info:
        for key, value in rate_limit_headers(ctx.rate_limit_info).items():
            response.headers[key] = value

    return completion
