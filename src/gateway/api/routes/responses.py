import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

from any_llm import AnyLLM, LLMProvider, aresponses
from any_llm.types.completion import CompletionUsage
from any_llm.types.responses import Response as ResponsesResponse
from any_llm.types.responses import ResponsesParams, ResponseStreamEvent
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseUsage
from openresponses_types.types import Usage as OpenResponsesUsage
from pydantic import ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer
from gateway.api.routes._helpers import latest_user_text, text_from_content
from gateway.api.routes._normalize import normalize_request_messages
from gateway.api.routes._pipeline import (
    NO_RESOLVABLE_PROVIDER_DETAIL,
    PROVIDER_ERROR_DETAIL,
    ErrorKind,
    classify_provider_error,
    prepare_gateway_tools,
    raise_all_streaming_attempts_failed,
    release_reservation,
    resolve_dispatch_provider,
    resolve_request_context,
    run_platform_non_stream,
    run_single_attempt_stream,
    run_standalone_non_stream,
    run_streaming_with_fallback,
)
from gateway.api.routes._platform import ResolvedAttempt
from gateway.api.routes._schema_derive import SESSION_LABEL_DESC, SESSION_LABEL_MAX_LENGTH, derive_request_base
from gateway.api.routes._tools import _strip_gateway_fields
from gateway.core.config import GatewayConfig
from gateway.core.usage import GatewayUsage
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import McpServerConfig
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_loop import ToolBackend
from gateway.services.mcp_loop_responses import (
    MAX_TOOL_ITERATIONS_CAP,
    responses_tool_loop,
    responses_tool_loop_stream,
)
from gateway.services.tool_format import inject_purpose_hints_responses, openai_to_responses_tools
from gateway.streaming import RESPONSES_STREAM_FORMAT, StreamFormat

router = APIRouter(prefix="/v1", tags=["responses"])

_MASTER_KEY_USER_REQUIRED = "When using master key, 'user' field is required in request body"
_USER_FORBIDDEN = "'user' field does not match the authenticated API key's user"
_CODEX_CLIENT_METADATA_FIELD = "client_metadata"
_CODEX_INPUT_METADATA_FIELD = "internal_chat_message_metadata_passthrough"
_CODEX_EXTRA_BODY_FIELD = "_codex_extra_body"


class ResponsesRequest(derive_request_base(ResponsesParams)):  # type: ignore[misc]
    """OpenAI Responses API-compatible request.

    The wire fields are derived from any-llm's ``ResponsesParams`` (see
    ``_schema_derive``) so the schema cannot silently drop a param any-llm
    forwards. Gateway-internal fields (``mcp_servers``, ``mcp_server_ids``,
    ``guardrails``, ``tools_header``, ``max_tool_iterations``) opt the request
    into gateway-managed MCP / sandbox / web_search / guardrails without
    changing the upstream wire shape. They're stripped before the request is
    forwarded.
    """

    model_config = ConfigDict(extra="allow")

    # any-llm types ``input`` as a large union of OpenAI item shapes; keep the
    # loose ``Any`` the gateway has always accepted (still required). Likewise
    # ``response_format`` only ever carries the dict form on the wire.
    input: Any
    response_format: dict[str, Any] | None = None
    # any-llm types ``stream`` as ``bool | None``; keep the OpenAI Responses wire
    # contract (a non-nullable boolean defaulting to false) for stable SDK
    # generation.
    stream: bool = False

    # Gateway-internal: identical semantics to ChatCompletionRequest.
    mcp_servers: list[McpServerConfig] | None = None
    mcp_server_ids: list[uuid.UUID] | None = None
    guardrails: list[GuardrailConfig] | None = Field(default=None, max_length=8)
    tools_header: str | None = None
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)
    session_label: str | None = Field(default=None, max_length=SESSION_LABEL_MAX_LENGTH, description=SESSION_LABEL_DESC)


def _responses_input_text(value: Any) -> str:
    """Flatten the Responses ``input`` field to plain text for guardrail checks.

    ``input`` may be a bare string or a list of input items sharing the
    ``role``/``content`` shape used by chat/messages (text parts look like
    ``{"type": "input_text", "text": "..."}``).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return latest_user_text(value)
    return text_from_content(value)


def _split_codex_input_metadata(value: Any) -> tuple[Any, bool]:
    """Return input without Codex metadata and whether any was removed.

    ``any-llm`` validates Responses input against the public OpenAI item
    shapes, while Codex's private extension must be preserved for OpenAI's
    own Responses endpoint. The caller sends the cleaned input through the
    typed path and, where supported, restores the original input via the
    provider SDK's raw-body extension.
    """
    if isinstance(value, list):
        cleaned_items: list[Any] = []
        changed = False
        for item in value:
            cleaned_item, item_changed = _split_codex_input_metadata(item)
            cleaned_items.append(cleaned_item)
            changed = changed or item_changed
        return (cleaned_items if changed else value), changed
    if isinstance(value, dict):
        cleaned_fields: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            if key == _CODEX_INPUT_METADATA_FIELD:
                changed = True
                continue
            cleaned_item, item_changed = _split_codex_input_metadata(item)
            cleaned_fields[key] = cleaned_item
            changed = changed or item_changed
        return (cleaned_fields if changed else value), changed
    return value, False


def _codex_extra_body(input_data: Any, client_metadata: Any | None) -> tuple[Any, dict[str, Any] | None]:
    """Prepare Codex metadata for OpenAI's raw Responses request body."""
    cleaned_input, input_has_codex_metadata = _split_codex_input_metadata(input_data)
    extra_body: dict[str, Any] = {}
    if input_has_codex_metadata:
        extra_body["input"] = input_data
    if client_metadata is not None:
        extra_body[_CODEX_CLIENT_METADATA_FIELD] = client_metadata
    return cleaned_input, extra_body or None


def _with_codex_extra_body(fields: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    """Restore Codex extensions only for the OpenAI Responses implementation."""
    result = {key: value for key, value in fields.items() if key != _CODEX_EXTRA_BODY_FIELD}
    extra_body = fields.get(_CODEX_EXTRA_BODY_FIELD)
    if provider == LLMProvider.OPENAI and extra_body is not None:
        result["extra_body"] = extra_body
    return result


def _usage_to_completion_usage(
    usage: ResponseUsage | OpenResponsesUsage | None,
) -> CompletionUsage | None:
    if usage is None:
        return None
    details = getattr(usage, "input_tokens_details", None)
    cache_read_tokens = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return GatewayUsage(
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        cache_read_tokens=cache_read_tokens,
    )


def _ensure_provider_supports_responses(provider: LLMProvider) -> None:
    provider_class = AnyLLM.get_provider_class(provider)
    if not getattr(provider_class, "SUPPORTS_RESPONSES", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider.value}' does not support the Responses API",
        )


class _ResponsesAdapter:
    """OpenAI Responses edges of the shared pipeline.

    Provider-call and tool-loop functions are resolved as module globals at
    call time so tests can monkeypatch ``gateway.api.routes.responses.aresponses``
    and friends.
    """

    name = "responses"
    endpoint = "/v1/responses"
    stream_format: StreamFormat = RESPONSES_STREAM_FORMAT
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

    def format_chunk(self, chunk: ResponseStreamEvent) -> str:
        return f"event: {chunk.type}\ndata: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def extract_stream_usage(self, chunk: ResponseStreamEvent) -> CompletionUsage | None:
        response_obj = getattr(chunk, "response", None)
        if response_obj and getattr(response_obj, "usage", None):
            return _usage_to_completion_usage(response_obj.usage)
        return None

    def extract_usage(self, result: ResponsesResponse) -> CompletionUsage | None:
        return _usage_to_completion_usage(getattr(result, "usage", None))

    async def call_provider(self, kwargs: dict[str, Any]) -> ResponsesResponse:
        return await aresponses(**kwargs)  # type: ignore[return-value]

    async def open_provider_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ResponseStreamEvent]:
        return await aresponses(**kwargs)  # type: ignore[return-value]

    def prepare_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs["stream"] = True
        return kwargs

    async def run_tool_loop(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        on_first_response: Callable[[], None] | None = None,
    ) -> ResponsesResponse:
        # Standalone dispatch has no lock-in callback; only pass the kwarg on
        # the platform-attempt path so test fakes can mirror each call shape.
        extra: dict[str, Any] = {}
        if on_first_response is not None:
            extra["on_first_response"] = on_first_response
        return await responses_tool_loop(
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
    ) -> AsyncIterator[ResponseStreamEvent]:
        return responses_tool_loop_stream(
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
        return inject_purpose_hints_responses({**kwargs}, hints, header=header)

    def attempt_kwargs(
        self,
        attempt: ResolvedAttempt,
        base_request_fields: dict[str, Any],
    ) -> dict[str, Any]:
        # base_request_fields carries input_data (plus provider in standalone
        # mode, stripped here); per attempt we set provider and model from the
        # resolved attempt in the keyword shape ``aresponses`` expects.
        kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            kwargs["api_base"] = attempt.api_base
        request_fields = _with_codex_extra_body(base_request_fields, LLMProvider(attempt.provider))
        return {
            **kwargs,
            **{k: v for k, v in request_fields.items() if k != "provider"},
            "model": attempt.model,
            "provider": LLMProvider(attempt.provider),
        }

    def prepare_platform_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # run_platform_attempts hands us ``"model"`` as ``"provider:model"``;
        # split it back out for the aresponses signature.
        merged_model = kwargs.pop("model")
        provider_str, model_str = merged_model.split(":", 1)
        kwargs["model"] = model_str
        kwargs["provider"] = LLMProvider(provider_str)
        return kwargs


_ADAPTER = _ResponsesAdapter()


@router.post("/responses", response_model=None)
async def create_response(
    raw_request: Request,
    response: FastAPIResponse,
    background_tasks: BackgroundTasks,
    request_body: ResponsesRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible Responses endpoint.

    Supports MCP tool-use loops, sandboxed code execution, and SearXNG
    web_search in both standalone mode and hybrid mode. Hybrid-mode requests
    resolve credentials via the platform service and get multi-attempt
    fallback across the resolved route, tool-loop requests included (fallback
    applies up to the pre-lock-in point, same as chat).
    """
    # max_output_tokens comes from an extra="allow" body, so it may be absent,
    # non-int, or negative — only trust a non-negative int for the estimate.
    raw_max_output = getattr(request_body, "max_output_tokens", None)
    max_output_tokens = raw_max_output if isinstance(raw_max_output, int) and raw_max_output >= 0 else None

    async def _normalize(
        user_id: str, provider: LLMProvider | None, model: str, instance: str | None
    ) -> tuple[int, CompletionUsage | None]:
        # Resolve uploaded file/image blocks into the Responses input payload
        # before the cost estimate. Standalone only; no-op when the files
        # feature is off or the request has no attachments.
        request_body.input, stats = await normalize_request_messages(
            request_body.input,
            fmt="responses",
            config=config,
            provider=provider,
            model=model,
            db=db,
            raw_request=raw_request,
            user_id=user_id,
            instance=instance,
        )
        chars = len(str(request_body.input)) + len(str(getattr(request_body, "instructions", "") or ""))
        return chars, stats.vision_usage()

    ctx = await resolve_request_context(
        adapter=_ADAPTER,
        raw_request=raw_request,
        response=response,
        db=db,
        config=config,
        log_writer=log_writer,
        model=request_body.model,
        user_id_from_request=request_body.user,
        estimate_prompt_chars=len(str(request_body.input)) + len(str(getattr(request_body, "instructions", "") or "")),
        estimate_max_output_tokens=max_output_tokens,
        master_key_user_required_detail=_MASTER_KEY_USER_REQUIRED,
        user_forbidden_detail=_USER_FORBIDDEN,
        normalize_messages=_normalize,
    )

    # Provider-support guard: an unsupported provider would just fail
    # downstream, so surface a clearer 400 upfront. In hybrid mode validate
    # *every* resolved attempt so a fallback that lands on an unsupported
    # provider (e.g. primary OpenAI, fallback Anthropic) fails fast here
    # instead of crashing mid-fallback when the runner calls ``aresponses``
    # on a provider that doesn't speak the Responses API.
    if ctx.hybrid_mode:
        route = ctx.route
        assert route is not None  # guaranteed by the hybrid-mode preamble
        for attempt in route.attempts:
            _ensure_provider_supports_responses(LLMProvider(attempt.provider))
    else:
        resolved = resolve_dispatch_provider(ctx, config, request_body.model)
        # ``provider`` is the underlying implementation handed to any-llm;
        # ``billing_instance`` is the otari routing key pricing/usage key on.
        provider, model = resolved.provider, resolved.model
        provider_kwargs = resolved.kwargs
        billing_instance = resolved.instance
        try:
            _ensure_provider_supports_responses(provider)
        except HTTPException:
            # The preamble already pre-debited the estimate; release it before
            # rejecting so the held amount does not shrink the budget.
            await release_reservation(ctx)
            raise

    tool_ctx = await prepare_gateway_tools(
        adapter=_ADAPTER,
        ctx=ctx,
        response=response,
        guardrails=request_body.guardrails,
        guardrail_text=_responses_input_text(request_body.input),
        tools=request_body.tools,
        mcp_servers=request_body.mcp_servers,
        mcp_server_ids=request_body.mcp_server_ids,
        max_tool_iterations=request_body.max_tool_iterations,
        tools_header=request_body.tools_header,
    )

    # Strip gateway-internal fields, flatten any caller-supplied function tools
    # to the Responses shape.
    request_fields = _strip_gateway_fields(
        request_body.model_dump(exclude_none=True),
        tools_extracted=tool_ctx.tools_extracted,
        remaining_user_tools=tool_ctx.remaining_user_tools,
    )
    # This is an internal handoff populated below, never a client request field.
    # ``ResponsesRequest`` permits extra fields for OpenAI compatibility, so
    # remove a caller-supplied value before constructing the provider kwargs.
    request_fields.pop(_CODEX_EXTRA_BODY_FIELD, None)
    client_metadata = request_fields.pop(_CODEX_CLIENT_METADATA_FIELD, None)
    if request_fields.get("tools"):
        request_fields["tools"] = openai_to_responses_tools(request_fields["tools"])

    input_payload, codex_extra_body = _codex_extra_body(request_fields.pop("input"), client_metadata)
    stream = bool(request_fields.pop("stream", False))
    request_fields.pop("model", None)
    request_fields.pop("user", None)
    # Standalone mode forwards the resolved user_id to the upstream provider
    # for analytics; hybrid mode handles attribution server-side via the
    # correlation id.
    if not ctx.hybrid_mode and ctx.user_id:
        request_fields["user"] = ctx.user_id

    # base_request_fields is what gets merged with per-attempt creds; it
    # includes ``provider`` and ``input_data`` so each attempt has the full
    # call shape.
    base_request_fields = {**request_fields}
    if not ctx.hybrid_mode:
        base_request_fields["provider"] = provider
    base_request_fields["input_data"] = input_payload
    if codex_extra_body is not None:
        base_request_fields[_CODEX_EXTRA_BODY_FIELD] = codex_extra_body

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if stream:
        if ctx.hybrid_mode:
            route = ctx.route
            assert route is not None  # guaranteed by the hybrid-mode preamble
            if not route.attempts:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=NO_RESOLVABLE_PROVIDER_DETAIL,
                )
            try:
                return await run_streaming_with_fallback(
                    adapter=_ADAPTER,
                    route=route,
                    base_request_fields=base_request_fields,
                    config=config,
                    background_tasks=background_tasks,
                    rate_limit_info=ctx.rate_limit_info,
                    tool_ctx=tool_ctx,
                    session_label=request_body.session_label,
                )
            except HTTPException:
                raise
            except Exception as exc:
                raise_all_streaming_attempts_failed(_ADAPTER, exc, route)

        # Standalone: single attempt streaming.
        call_kwargs = {**provider_kwargs, **_with_codex_extra_body(base_request_fields, provider), "model": model}
        return await run_single_attempt_stream(
            adapter=_ADAPTER,
            ctx=ctx,
            tool_ctx=tool_ctx,
            call_kwargs=call_kwargs,
            provider=billing_instance,
            model=model,
            session_label=request_body.session_label,
            display_model=resolved.alias,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    if ctx.hybrid_mode:
        route = ctx.route
        assert route is not None  # guaranteed by the hybrid-mode preamble
        result = await run_platform_non_stream(
            adapter=_ADAPTER,
            route=route,
            base_request_fields=base_request_fields,
            tool_ctx=tool_ctx,
            response=response,
            background_tasks=background_tasks,
            config=config,
            rate_limit_info=ctx.rate_limit_info,
            session_label=request_body.session_label,
        )
        return result.model_dump(exclude_none=True)

    # Standalone non-stream path
    call_kwargs = {**provider_kwargs, **_with_codex_extra_body(base_request_fields, provider), "model": model}
    result = await run_standalone_non_stream(
        adapter=_ADAPTER,
        ctx=ctx,
        tool_ctx=tool_ctx,
        call_kwargs=call_kwargs,
        response=response,
        provider=billing_instance,
        model=model,
        display_model=resolved.alias,
    )

    return result.model_dump(exclude_none=True)
