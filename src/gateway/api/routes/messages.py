import asyncio
import math
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

import httpx
from any_llm import LLMProvider, amessages
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageResponse,
    MessagesParams,
    MessageStartEvent,
    MessageStreamEvent,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import latest_user_text
from gateway.api.routes._normalize import normalize_request_messages
from gateway.api.routes._pipeline import (
    ALL_PROVIDERS_FAILED_DETAIL,
    ALL_PROVIDERS_TIMED_OUT_DETAIL,
    DB_UNAVAILABLE_DETAIL,
    NO_RESOLVABLE_PROVIDER_DETAIL,
    SANDBOX_UNREACHABLE_DETAIL,
    WEB_SEARCH_UNREACHABLE_DETAIL,
    ErrorKind,
    classify_provider_error,
    default_attempt_kwargs,
    prepare_gateway_tools,
    rate_limit_headers,
    resolve_request_context,
    run_platform_non_stream,
    run_single_attempt_stream,
    run_standalone_non_stream,
    run_streaming_with_fallback,
)
from gateway.api.routes._platform import (
    ResolvedAttempt,
    _extract_platform_user_token,
    _resolve_platform_credentials,
)
from gateway.api.routes._schema_derive import derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.core.usage import GatewayUsage
from gateway.log_config import logger
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import McpServerConfig
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import ToolBackend
from gateway.services.mcp_loop_messages import (
    MAX_TOOL_ITERATIONS_CAP,
    anthropic_tool_loop,
    anthropic_tool_loop_stream,
)
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.tool_format import inject_purpose_hints_anthropic, openai_to_anthropic_tools
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import ANTHROPIC_STREAM_FORMAT, StreamFormat

router = APIRouter(prefix="/v1", tags=["messages"])


class MessagesRequest(derive_request_base(MessagesParams)):  # type: ignore[misc]
    """Anthropic Messages API-compatible request.

    The wire fields are derived from any-llm's ``MessagesParams`` (see
    ``_schema_derive``) so the schema cannot silently drop a param any-llm
    forwards. Gateway-internal fields (``mcp_servers``, ``mcp_server_ids``,
    ``guardrails``, ``tools_header``, ``max_tool_iterations``) opt the request
    into gateway-managed MCP / sandbox / web_search / guardrails without
    changing the upstream wire shape. They're stripped before the request is
    forwarded.
    """

    messages: list[dict[str, Any]] = Field(min_length=1)
    # any-llm types ``stream`` as ``bool | None``; keep the Anthropic wire
    # contract (a non-nullable boolean defaulting to false) for stable SDK
    # generation.
    stream: bool = False

    # Gateway-internal: identical semantics to ChatCompletionRequest.
    mcp_servers: list[McpServerConfig] | None = None
    mcp_server_ids: list[uuid.UUID] | None = None
    guardrails: list[GuardrailConfig] | None = Field(default=None, max_length=8)
    tools_header: str | None = None
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)


class CountTokensRequest(BaseModel):
    """Anthropic ``/v1/messages/count_tokens`` request.

    A subset of :class:`MessagesRequest`: the input fields that affect the token
    count, minus ``max_tokens`` and the streaming/sampling controls, since the
    endpoint only counts input tokens. Clients such as Claude Code call this on
    every turn to keep their prompt within the model's context window.
    """

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None


class CountTokensResponse(BaseModel):
    """Anthropic ``/v1/messages/count_tokens`` response."""

    input_tokens: int


def _anthropic_error(error_type: str, message: str, status_code: int) -> HTTPException:
    """Create an HTTPException with Anthropic-style error body."""
    return HTTPException(
        status_code=status_code,
        detail={"type": "error", "error": {"type": error_type, "message": message}},
    )


_ERR_INVALID_REQUEST = "invalid_request_error"
_ERR_API = "api_error"
_ERR_PERMISSION = "permission_error"
_ERR_AUTHENTICATION = "authentication_error"
_ERR_NOT_FOUND = "not_found_error"
_ERR_RATE_LIMIT = "rate_limit_error"

# Anthropic error.type keyed by HTTP status, used when re-wrapping a plain-string
# HTTPException into the Anthropic envelope: classified provider failures plus
# preamble auth/permission/resolve rejections. Unlisted statuses (e.g. the 502
# used for a credentials fault, or a 500) fall back to api_error.
_STATUS_TO_ANTHROPIC_TYPE = {
    400: _ERR_INVALID_REQUEST,
    401: _ERR_AUTHENTICATION,
    403: _ERR_PERMISSION,
    404: _ERR_NOT_FOUND,
    429: _ERR_RATE_LIMIT,
}


def _ensure_anthropic_error(exc: HTTPException) -> HTTPException:
    """Re-wrap a plain-string ``HTTPException`` in the Anthropic error envelope,
    preserving the status code and headers (e.g. a 429's ``Retry-After``).

    HTTPExceptions already carrying the Anthropic ``detail`` dict (raised via
    ``_anthropic_error``) pass through unchanged, so this is safe to apply to any
    HTTPException on the ``/v1/messages`` path, including format-agnostic ones
    raised by the hybrid preamble (platform resolve/auth) and the shared
    execution runners.
    """
    if not isinstance(exc.detail, str):
        return exc
    error_type = _STATUS_TO_ANTHROPIC_TYPE.get(exc.status_code, _ERR_API)
    return HTTPException(
        status_code=exc.status_code,
        detail={"type": "error", "error": {"type": error_type, "message": exc.detail}},
        headers=exc.headers,
    )


_MASTER_KEY_USER_REQUIRED = "When using master key, 'metadata.user_id' is required in request body"
_USER_FORBIDDEN = "'metadata.user_id' does not match the authenticated API key's user"
_PROVIDER_ERROR = "The request could not be completed by the provider"

_ERROR_KIND_TO_ANTHROPIC_TYPE = {
    ErrorKind.INVALID_REQUEST: _ERR_INVALID_REQUEST,
    ErrorKind.API: _ERR_API,
    ErrorKind.PERMISSION: _ERR_PERMISSION,
}


def _messages_stream_usage(event: MessageStreamEvent) -> CompletionUsage | None:
    if isinstance(event, MessageDeltaEvent):
        input_tokens = event.usage.input_tokens or 0
        output_tokens = event.usage.output_tokens or 0
        return GatewayUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cache_read_tokens=event.usage.cache_read_input_tokens or 0,
            cache_write_tokens=event.usage.cache_creation_input_tokens or 0,
        )
    if isinstance(event, MessageStartEvent):
        usage = event.message.usage
        input_tokens = usage.input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        if input_tokens or cache_read or cache_write:
            return GatewayUsage(
                prompt_tokens=input_tokens,
                completion_tokens=0,
                total_tokens=input_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            )
    return None


class _MessagesAdapter:
    """Anthropic Messages edges of the shared pipeline.

    Provider-call and tool-loop functions are resolved as module globals at
    call time so tests can monkeypatch ``gateway.api.routes.messages.amessages``
    and friends.
    """

    name = "messages"
    endpoint = "/v1/messages"
    stream_format: StreamFormat = ANTHROPIC_STREAM_FORMAT
    # A successful non-streaming call without provider usage data skips the
    # usage-log row (only the reservation is settled), matching the wire
    # behavior this endpoint has always had.
    log_success_without_usage = False

    def error(self, status_code: int, message: str, kind: ErrorKind = ErrorKind.API) -> HTTPException:
        return _anthropic_error(_ERROR_KIND_TO_ANTHROPIC_TYPE[kind], message, status_code)

    def provider_error(self, exc: BaseException) -> HTTPException:
        mapping = classify_provider_error(exc)
        if mapping is not None:
            error_type = _STATUS_TO_ANTHROPIC_TYPE.get(mapping.status_code, _ERR_API)
            return _anthropic_error(error_type, mapping.detail, mapping.status_code)
        return _anthropic_error(_ERR_API, _PROVIDER_ERROR, status.HTTP_500_INTERNAL_SERVER_ERROR)

    def format_chunk(self, chunk: MessageStreamEvent) -> str:
        return f"event: {chunk.type}\ndata: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def extract_stream_usage(self, chunk: MessageStreamEvent) -> CompletionUsage | None:
        return _messages_stream_usage(chunk)

    def extract_usage(self, result: MessageResponse) -> CompletionUsage | None:
        if not result.usage:
            return None
        return GatewayUsage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.input_tokens + result.usage.output_tokens,
            cache_read_tokens=result.usage.cache_read_input_tokens or 0,
            cache_write_tokens=result.usage.cache_creation_input_tokens or 0,
        )

    async def call_provider(self, kwargs: dict[str, Any]) -> MessageResponse:
        return await amessages(**kwargs)  # type: ignore[return-value]

    async def open_provider_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[MessageStreamEvent]:
        return await amessages(**kwargs)  # type: ignore[return-value]

    def prepare_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs["stream"] = True
        return kwargs

    async def run_tool_loop(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        on_first_response: Callable[[], None] | None = None,
    ) -> MessageResponse:
        # Standalone dispatch has no lock-in callback; only pass the kwarg on
        # the platform-attempt path so test fakes can mirror each call shape.
        extra: dict[str, Any] = {}
        if on_first_response is not None:
            extra["on_first_response"] = on_first_response
        return await anthropic_tool_loop(
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
    ) -> AsyncIterator[MessageStreamEvent]:
        return anthropic_tool_loop_stream(
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
        return inject_purpose_hints_anthropic({**kwargs}, hints, header=header)

    def attempt_kwargs(
        self,
        attempt: ResolvedAttempt,
        base_request_fields: dict[str, Any],
    ) -> dict[str, Any]:
        return default_attempt_kwargs(attempt, base_request_fields)

    def prepare_platform_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs


_ADAPTER = _MessagesAdapter()


@router.post("/messages", response_model=None)
async def create_message(
    raw_request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    request: MessagesRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API-compatible endpoint.

    Supports MCP tool-use loops, sandboxed code execution, and SearXNG
    web_search in both standalone mode and hybrid mode. Hybrid-mode
    requests resolve credentials via the platform service and (for
    non-tool-loop requests) get multi-attempt fallback across the resolved
    route. Tool-loop requests collapse to a single attempt — once
    ``on_first_response`` lock-in plumbing lands across the codebase, a
    follow-up will enable pre-lock-in fallback for tool-loop requests too.
    """
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None

    async def _normalize(
        user_id: str, provider: LLMProvider | None, model: str, instance: str | None
    ) -> tuple[int, CompletionUsage | None]:
        # Resolve uploaded file/image blocks into the Anthropic wire payload
        # before the cost estimate. Standalone only; no-op when the files
        # feature is off or the request has no attachments.
        request.messages, stats = await normalize_request_messages(
            request.messages,
            fmt="anthropic",
            config=config,
            provider=provider,
            model=model,
            db=db,
            raw_request=raw_request,
            user_id=user_id,
            instance=instance,
        )
        return len(str(request.messages)) + len(str(request.system or "")), stats.vision_usage()

    try:
        ctx = await resolve_request_context(
            adapter=_ADAPTER,
            raw_request=raw_request,
            response=response,
            db=db,
            config=config,
            log_writer=log_writer,
            model=request.model,
            user_id_from_request=str(user_from_metadata) if user_from_metadata else None,
            estimate_prompt_chars=len(str(request.messages)) + len(str(request.system or "")),
            estimate_max_output_tokens=request.max_tokens,
            master_key_user_required_detail=_MASTER_KEY_USER_REQUIRED,
            user_forbidden_detail=_USER_FORBIDDEN,
            normalize_messages=_normalize,
        )
    except HTTPException as exc:
        # The hybrid preamble (platform resolve / auth) raises format-agnostic
        # plain-string HTTPExceptions (some with a Retry-After header); re-wrap
        # them in the Anthropic envelope so /v1/messages errors stay structured.
        raise _ensure_anthropic_error(exc) from exc

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

    # Strip gateway-internal fields, convert any caller-supplied OpenAI-shaped
    # tools to Anthropic shape so a mixed list works.
    request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tool_ctx.tools_extracted,
        remaining_user_tools=tool_ctx.remaining_user_tools,
    )
    if request_fields.get("tools"):
        request_fields["tools"] = openai_to_anthropic_tools(request_fields["tools"])

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if request.stream:
        # Tool-loop streaming collapses to a single attempt for this format
        # (chat already runs tool loops through the multi-attempt fallback;
        # wiring the same here is a planned follow-up).
        if ctx.hybrid_mode and not tool_ctx.use_tool_loop:
            route = ctx.route
            assert route is not None  # guaranteed by the hybrid-mode preamble
            if not route.attempts:
                logger.error("Platform returned empty attempts list request_id=%s", route.request_id)
                raise _anthropic_error(
                    _ERR_API,
                    NO_RESOLVABLE_PROVIDER_DETAIL,
                    status.HTTP_502_BAD_GATEWAY,
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
            except HTTPException as exc:
                # Hybrid terminal failures arrive as format-agnostic plain-string
                # HTTPExceptions; ensure the Anthropic envelope (dict details pass
                # through unchanged).
                converted = _ensure_anthropic_error(exc)
                if converted is exc:
                    raise
                raise converted from exc
            except Exception as exc:
                logger.error("All streaming attempts failed request_id=%s: %s", route.request_id, exc)
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
                    raise _anthropic_error(
                        _ERR_API,
                        ALL_PROVIDERS_TIMED_OUT_DETAIL,
                        status.HTTP_504_GATEWAY_TIMEOUT,
                    ) from exc
                raise _anthropic_error(
                    _ERR_API,
                    ALL_PROVIDERS_FAILED_DETAIL,
                    status.HTTP_502_BAD_GATEWAY,
                ) from exc

        # Standalone (or hybrid + tool-loop): single attempt streaming.
        platform_correlation_id: str | None = None
        platform_request_id: str | None = None
        if ctx.hybrid_mode:
            # Tool-loop hybrid path: build call_kwargs from the primary
            # attempt and keep the platform contract (X-Correlation-ID,
            # X-Otari-Request-ID, usage reported via _report_platform_usage).
            route = ctx.route
            assert route is not None  # guaranteed by the hybrid-mode preamble
            if not route.attempts:
                raise _anthropic_error(
                    _ERR_API,
                    NO_RESOLVABLE_PROVIDER_DETAIL,
                    status.HTTP_502_BAD_GATEWAY,
                )
            attempt = route.attempts[0]
            model = attempt.model
            call_kwargs = default_attempt_kwargs(attempt, request_fields)
            platform_correlation_id = attempt.attempt_id
            platform_request_id = route.request_id
            billing_provider: Any = LLMProvider(attempt.provider)
        else:
            resolved = resolve_provider_selector(config, request.model)
            model = resolved.model
            call_kwargs = {**resolved.kwargs, **request_fields, "model": resolved.dispatch_model}
            billing_provider = resolved.instance

        return await run_single_attempt_stream(
            adapter=_ADAPTER,
            ctx=ctx,
            tool_ctx=tool_ctx,
            call_kwargs=call_kwargs,
            provider=billing_provider,
            model=model,
            platform_correlation_id=platform_correlation_id,
            platform_request_id=platform_request_id,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    if ctx.hybrid_mode:
        route = ctx.route
        assert route is not None  # guaranteed by the hybrid-mode preamble
        try:
            result = await run_platform_non_stream(
                adapter=_ADAPTER,
                route=route,
                base_request_fields=request_fields,
                tool_ctx=tool_ctx,
                response=response,
                background_tasks=background_tasks,
                config=config,
                rate_limit_info=ctx.rate_limit_info,
            )
        except HTTPException as exc:
            # Hybrid terminal failures arrive as format-agnostic plain-string
            # HTTPExceptions; ensure the Anthropic envelope (dict details pass
            # through unchanged).
            converted = _ensure_anthropic_error(exc)
            if converted is exc:
                raise
            raise converted from exc
        except SandboxNotReachableError as e:
            logger.error("Sandbox unreachable: %s", e)
            raise _anthropic_error(
                _ERR_API,
                SANDBOX_UNREACHABLE_DETAIL,
                status.HTTP_502_BAD_GATEWAY,
            ) from e
        except WebSearchNotReachableError as e:
            logger.error("Web search backend unreachable: %s", e)
            raise _anthropic_error(
                _ERR_API,
                WEB_SEARCH_UNREACHABLE_DETAIL,
                status.HTTP_502_BAD_GATEWAY,
            ) from e
        return result.model_dump(exclude_none=True)

    # Standalone non-stream path
    resolved = resolve_provider_selector(config, request.model)
    call_kwargs = {**resolved.kwargs, **request_fields, "model": resolved.dispatch_model}
    result = await run_standalone_non_stream(
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

    return result.model_dump(exclude_none=True)


# The gateway has no tokenizer (see budget_service.estimate_cost), so input
# tokens are approximated as ``chars / 4`` — the same heuristic used for budget
# pre-debit. count_tokens callers (e.g. Claude Code) use the result only to
# gauge headroom against the context window, so an approximate count is fine.
# Round up: an over-count keeps callers safely inside the context window, while
# an under-count could let a prompt slip over the limit.
_CHARS_PER_TOKEN = 4


def _estimate_input_tokens(request: CountTokensRequest) -> int:
    """Approximate the prompt's input-token count from its serialized length."""
    chars = len(str(request.messages))
    if request.system:
        chars += len(str(request.system))
    if request.tools:
        chars += len(str(request.tools))
    return max(1, math.ceil(chars / _CHARS_PER_TOKEN))


@router.post("/messages/count_tokens")
async def count_message_tokens(
    raw_request: Request,
    request: CountTokensRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CountTokensResponse:
    """Anthropic ``/v1/messages/count_tokens``-compatible endpoint.

    Returns ``{"input_tokens": N}`` without contacting an upstream provider:
    counting is local, so there is no budget reservation, pricing, or usage
    logging. Authentication mirrors :func:`create_message` — hybrid mode
    resolves the caller's token against the platform, standalone mode validates
    the API key — so the endpoint is not an open token-counting oracle.
    """
    try:
        if config.is_hybrid_mode:
            # Resolve against the platform purely to authenticate the caller (same
            # as create_message); the routing plan is discarded since counting is
            # local. Without this, any non-empty bearer string would be accepted.
            user_token = _extract_platform_user_token(raw_request)
            await _resolve_platform_credentials(
                config=config,
                user_token=user_token,
                model_selector=request.model,
            )
        else:
            if db is None:
                raise _anthropic_error(_ERR_API, DB_UNAVAILABLE_DETAIL, status.HTTP_500_INTERNAL_SERVER_ERROR)
            await verify_api_key_or_master_key(raw_request, db, config)
    except HTTPException as exc:
        # Keep /v1/messages/count_tokens auth errors in the Anthropic envelope too.
        raise _ensure_anthropic_error(exc) from exc

    return CountTokensResponse(input_tokens=_estimate_input_tokens(request))
