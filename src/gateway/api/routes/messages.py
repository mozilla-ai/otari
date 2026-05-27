import os
from collections.abc import AsyncIterator
from typing import Annotated, Any

from any_llm import AnyLLM, amessages
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStreamEvent,
)
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._tools import (
    _build_web_search_backend,
    _extract_code_execution_tool,
    _extract_web_search_tool,
    _resolve_sandbox_purpose_hint,
    _strip_gateway_fields,
)
from gateway.api.routes.chat import get_provider_kwargs, log_usage, rate_limit_headers
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import validate_user_budget
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_loop_messages import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    anthropic_tool_loop,
    anthropic_tool_loop_stream,
)
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.tool_format import inject_purpose_hints_anthropic, openai_to_anthropic_tools
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import ANTHROPIC_STREAM_FORMAT, streaming_generator

router = APIRouter(prefix="/v1", tags=["messages"])


class MessagesRequest(BaseModel):
    """Anthropic Messages API-compatible request.

    Gateway-internal fields (``mcp_servers``, ``tools_header``,
    ``max_tool_iterations``) are accepted on top of the Anthropic wire shape
    so the same client can opt into gateway-managed MCP / sandbox / web_search
    without abandoning the Messages API. These fields are stripped before the
    request is forwarded upstream.
    """

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None

    # Gateway-internal: identical semantics to ChatCompletionRequest.
    mcp_servers: list[McpServerConfig] | None = None
    tools_header: str | None = None
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)


def _anthropic_error(error_type: str, message: str, status_code: int) -> HTTPException:
    """Create an HTTPException with Anthropic-style error body."""
    return HTTPException(
        status_code=status_code,
        detail={"type": "error", "error": {"type": error_type, "message": message}},
    )


_ERR_INVALID_REQUEST = "invalid_request_error"
_ERR_API = "api_error"
_MASTER_KEY_USER_REQUIRED = "When using master key, 'metadata.user_id' is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"
_PROVIDER_ERROR = "The request could not be completed by the provider"


@router.post("/messages", response_model=None)
async def create_message(
    raw_request: Request,
    response: Response,
    request: MessagesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API-compatible endpoint.

    Supports MCP tool-use loops, sandboxed code execution, and SearXNG
    web_search on the standalone-mode path. Platform-mode multi-attempt
    fallback is not wired here yet (handled by a follow-up PR); requests in
    platform mode currently pass through to the upstream provider unchanged.
    """
    api_key, is_master_key = auth_result
    api_key_id = api_key.id if api_key else None
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None
    user_id = resolve_user_id(
        user_id_from_request=str(user_from_metadata) if user_from_metadata else None,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=_anthropic_error(
            _ERR_INVALID_REQUEST,
            _MASTER_KEY_USER_REQUIRED,
            status.HTTP_400_BAD_REQUEST,
        ),
        no_api_key_error=_anthropic_error(
            _ERR_API,
            _API_KEY_VALIDATION_FAILED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
        no_user_error=_anthropic_error(
            _ERR_API,
            _API_KEY_NO_USER,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request.model, strategy=config.budget_strategy)
    if config.budget_strategy == "for_update":
        await db.rollback()

    provider, model = AnyLLM.split_model_provider(request.model)
    provider_kwargs = get_provider_kwargs(config, provider)

    # Tool extraction mirrors chat.py: sandbox / web_search / mcp_servers are
    # mutually exclusive for now (multi-backend dispatch is a follow-up).
    sandbox_tool_entry, tools_after_sandbox = _extract_code_execution_tool(request.tools)
    sandbox_url: str | None = os.environ.get("GATEWAY_SANDBOX_URL") or None
    use_sandbox = False
    if sandbox_tool_entry is not None:
        if sandbox_url is None:
            raise _anthropic_error(
                _ERR_INVALID_REQUEST,
                (
                    "code_execution tool requested but no sandbox is configured on this gateway. "
                    "Set GATEWAY_SANDBOX_URL on the gateway, or remove code_execution from `tools`."
                ),
                status.HTTP_400_BAD_REQUEST,
            )
        if request.mcp_servers:
            raise _anthropic_error(
                _ERR_INVALID_REQUEST,
                (
                    "code_execution and mcp_servers cannot be combined in the same request yet; "
                    "pick one. Multi-backend dispatch is a planned refinement."
                ),
                status.HTTP_400_BAD_REQUEST,
            )
        use_sandbox = True

    web_search_tool_entry, remaining_user_tools = _extract_web_search_tool(tools_after_sandbox)
    web_search_url: str | None = os.environ.get("GATEWAY_WEB_SEARCH_URL") or None
    use_web_search = False
    if web_search_tool_entry is not None:
        if web_search_url is None:
            raise _anthropic_error(
                _ERR_INVALID_REQUEST,
                (
                    "web_search tool requested but no search backend is configured on this gateway. "
                    "Set GATEWAY_WEB_SEARCH_URL on the gateway, or remove web_search from `tools`."
                ),
                status.HTTP_400_BAD_REQUEST,
            )
        if use_sandbox or request.mcp_servers:
            raise _anthropic_error(
                _ERR_INVALID_REQUEST,
                (
                    "web_search cannot be combined with code_execution or mcp_servers in the same "
                    "request yet; pick one."
                ),
                status.HTTP_400_BAD_REQUEST,
            )
        use_web_search = True

    mcp_server_configs = request.mcp_servers
    max_tool_iterations = min(
        request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
        MAX_TOOL_ITERATIONS_CAP,
    )

    # Strip gateway-internal fields, convert any caller-supplied OpenAI-shaped
    # tools to Anthropic shape so a mixed list works, and rebuild kwargs.
    request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
        remaining_user_tools=remaining_user_tools,
    )
    if request_fields.get("tools"):
        request_fields["tools"] = openai_to_anthropic_tools(request_fields["tools"])
    call_kwargs: dict[str, Any] = {**provider_kwargs, **request_fields}

    use_tool_loop = bool(mcp_server_configs) or use_sandbox or use_web_search

    if request.stream:
        return await _stream_messages(
            call_kwargs=call_kwargs,
            request=request,
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            provider=provider,
            model=model,
            rate_limit_info=rate_limit_info,
            use_tool_loop=use_tool_loop,
            mcp_server_configs=mcp_server_configs,
            use_sandbox=use_sandbox,
            sandbox_tool_entry=sandbox_tool_entry,
            sandbox_url=sandbox_url,
            use_web_search=use_web_search,
            web_search_tool_entry=web_search_tool_entry,
            web_search_url=web_search_url,
            max_tool_iterations=max_tool_iterations,
        )

    try:
        result = await _run_messages_non_stream(
            call_kwargs=call_kwargs,
            tools_header=request.tools_header,
            use_tool_loop=use_tool_loop,
            mcp_server_configs=mcp_server_configs,
            use_sandbox=use_sandbox,
            sandbox_tool_entry=sandbox_tool_entry,
            sandbox_url=sandbox_url,
            use_web_search=use_web_search,
            web_search_tool_entry=web_search_tool_entry,
            web_search_url=web_search_url,
            max_tool_iterations=max_tool_iterations,
        )

        if result.usage:
            usage_data = CompletionUsage(
                prompt_tokens=result.usage.input_tokens,
                completion_tokens=result.usage.output_tokens,
                total_tokens=result.usage.input_tokens + result.usage.output_tokens,
            )
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/messages",
                user_id=user_id,
                usage_override=usage_data,
            )

    except HTTPException:
        raise
    except MaxToolIterationsExceeded as e:
        # Gateway-side cap hit, not an upstream failure. 422 keeps the same
        # semantics as the chat endpoint so callers can tell a runaway tool
        # loop from a provider outage.
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            user_id=user_id,
            error=str(e),
        )
        raise _anthropic_error(_ERR_INVALID_REQUEST, str(e), status.HTTP_422_UNPROCESSABLE_ENTITY) from e
    except SandboxNotReachableError as e:
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, e)
        raise _anthropic_error(
            _ERR_API,
            "code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            status.HTTP_502_BAD_GATEWAY,
        ) from e
    except WebSearchNotReachableError as e:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, e)
        raise _anthropic_error(
            _ERR_API,
            "web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
            status.HTTP_502_BAD_GATEWAY,
        ) from e
    except Exception as e:
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            user_id=user_id,
            error=str(e),
        )
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise _anthropic_error(
            _ERR_API,
            _PROVIDER_ERROR,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result.model_dump(exclude_none=True)


async def _run_messages_non_stream(
    *,
    call_kwargs: dict[str, Any],
    tools_header: str | None,
    use_tool_loop: bool,
    mcp_server_configs: list[McpServerConfig] | None,
    use_sandbox: bool,
    sandbox_tool_entry: dict[str, Any] | None,
    sandbox_url: str | None,
    use_web_search: bool,
    web_search_tool_entry: dict[str, Any] | None,
    web_search_url: str | None,
    max_tool_iterations: int,
) -> MessageResponse:
    """Dispatch the non-streaming Anthropic call.

    Plain ``amessages`` when no gateway tools are in play. Otherwise the
    appropriate backend is opened (MCP pool, sandbox, or web_search) and
    ``anthropic_tool_loop`` drives the tool round-trips.
    """
    if not use_tool_loop:
        return await amessages(**call_kwargs)  # type: ignore[return-value]

    if mcp_server_configs:
        async with MCPClientPool(mcp_server_configs) as pool:
            kwargs = inject_purpose_hints_anthropic(
                {**call_kwargs},
                pool.purpose_hints(),
                header=tools_header,
            )
            return await anthropic_tool_loop(
                completion_kwargs=kwargs,
                pool=pool,
                max_iterations=max_tool_iterations,
            )

    if use_sandbox:
        assert sandbox_url is not None
        sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
        async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
            kwargs = inject_purpose_hints_anthropic(
                {**call_kwargs},
                backend.purpose_hints(),
                header=tools_header,
            )
            return await anthropic_tool_loop(
                completion_kwargs=kwargs,
                pool=backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
            )

    assert use_web_search
    assert web_search_url is not None
    assert web_search_tool_entry is not None
    async with _build_web_search_backend(
        base_url=web_search_url,
        tool_entry=web_search_tool_entry,
    ) as web_backend:
        kwargs = inject_purpose_hints_anthropic(
            {**call_kwargs},
            web_backend.purpose_hints(),
            header=tools_header,
        )
        return await anthropic_tool_loop(
            completion_kwargs=kwargs,
            pool=web_backend,  # type: ignore[arg-type]
            max_iterations=max_tool_iterations,
        )


async def _stream_messages(
    *,
    call_kwargs: dict[str, Any],
    request: MessagesRequest,
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    user_id: str,
    provider: Any,
    model: str,
    rate_limit_info: Any,
    use_tool_loop: bool,
    mcp_server_configs: list[McpServerConfig] | None,
    use_sandbox: bool,
    sandbox_tool_entry: dict[str, Any] | None,
    sandbox_url: str | None,
    use_web_search: bool,
    web_search_tool_entry: dict[str, Any] | None,
    web_search_url: str | None,
    max_tool_iterations: int,
) -> StreamingResponse:
    """Streaming Anthropic endpoint dispatch.

    Plain ``amessages(stream=True)`` when no gateway tools are in play.
    Otherwise the backend is opened **eagerly** (before the StreamingResponse
    is constructed) so a backend-unreachable error surfaces as an HTTP error
    rather than a 200 OK followed by an in-band SSE error. Same rationale as
    the chat.py sandbox/web_search streaming paths.
    """
    call_kwargs["stream"] = True

    def _format_chunk(event: MessageStreamEvent) -> str:
        return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

    def _extract_usage(event: MessageStreamEvent) -> CompletionUsage | None:
        if isinstance(event, MessageDeltaEvent):
            input_tokens = event.usage.input_tokens or 0
            output_tokens = event.usage.output_tokens or 0
            return CompletionUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        if isinstance(event, MessageStartEvent):
            input_tokens = event.message.usage.input_tokens or 0
            if input_tokens:
                return CompletionUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=0,
                    total_tokens=input_tokens,
                )
        return None

    async def _on_complete(usage_data: CompletionUsage) -> None:
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            user_id=user_id,
            usage_override=usage_data,
        )

    async def _on_error(error: str) -> None:
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            user_id=user_id,
            error=error,
        )

    try:
        if not use_tool_loop:
            msg_stream = await amessages(**call_kwargs)
            msg_stream_typed: AsyncIterator[MessageStreamEvent] = msg_stream  # type: ignore[assignment]
        else:
            msg_stream_typed = await _open_tool_loop_stream(
                call_kwargs=call_kwargs,
                tools_header=request.tools_header,
                mcp_server_configs=mcp_server_configs,
                use_sandbox=use_sandbox,
                sandbox_tool_entry=sandbox_tool_entry,
                sandbox_url=sandbox_url,
                use_web_search=use_web_search,
                web_search_tool_entry=web_search_tool_entry,
                web_search_url=web_search_url,
                max_tool_iterations=max_tool_iterations,
            )
    except SandboxNotReachableError as exc:
        # Surfaced here (rather than from the in-band SSE channel) because the
        # backend is opened eagerly — see ``_open_tool_loop_stream``'s
        # docstring. Mirrors the non-streaming error mapping.
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
        raise _anthropic_error(
            _ERR_API,
            "code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            status.HTTP_502_BAD_GATEWAY,
        ) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        raise _anthropic_error(
            _ERR_API,
            "web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
            status.HTTP_502_BAD_GATEWAY,
        ) from exc

    rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
    return StreamingResponse(
        streaming_generator(
            stream=msg_stream_typed,
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=ANTHROPIC_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
        ),
        media_type="text/event-stream",
        headers=rl_headers,
    )


async def _open_tool_loop_stream(
    *,
    call_kwargs: dict[str, Any],
    tools_header: str | None,
    mcp_server_configs: list[McpServerConfig] | None,
    use_sandbox: bool,
    sandbox_tool_entry: dict[str, Any] | None,
    sandbox_url: str | None,
    use_web_search: bool,
    web_search_tool_entry: dict[str, Any] | None,
    web_search_url: str | None,
    max_tool_iterations: int,
) -> AsyncIterator[MessageStreamEvent]:
    """Return an async iterator that yields events for the entire tool loop.

    For the sandbox and web_search paths the backend is opened **eagerly**
    (the ``await ... __aenter__()`` runs before the iterator is returned), so
    a backend-unreachable error surfaces as an HTTP 502 rather than landing
    in the SSE channel after the response has already committed to 200 OK.

    The MCP path is different: ``MCPClientPool`` is entered lazily inside the
    iterator's ``async with`` block. An ``MCPClientPool`` dial failure surfaces
    once the client starts pulling events. A future improvement would
    AsyncExitStack-share the pool across attempts and eager-open it for the
    same UX as the other two backends; for now MCP keeps the simpler
    enter-inside-generator pattern.
    """
    if mcp_server_configs:
        pool_cfgs = mcp_server_configs

        async def _mcp_iter() -> AsyncIterator[MessageStreamEvent]:
            async with MCPClientPool(pool_cfgs) as pool:
                kwargs = inject_purpose_hints_anthropic(
                    {**call_kwargs},
                    pool.purpose_hints(),
                    header=tools_header,
                )
                async for event in anthropic_tool_loop_stream(
                    completion_kwargs=kwargs,
                    pool=pool,
                    max_iterations=max_tool_iterations,
                ):
                    yield event

        return _mcp_iter()

    if use_sandbox:
        assert sandbox_url is not None
        sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
        sandbox_backend = SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint)
        await sandbox_backend.__aenter__()  # eager-open: may raise SandboxNotReachableError

        async def _sandbox_iter() -> AsyncIterator[MessageStreamEvent]:
            try:
                kwargs = inject_purpose_hints_anthropic(
                    {**call_kwargs},
                    sandbox_backend.purpose_hints(),
                    header=tools_header,
                )
                async for event in anthropic_tool_loop_stream(
                    completion_kwargs=kwargs,
                    pool=sandbox_backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                ):
                    yield event
            finally:
                await sandbox_backend.__aexit__(None, None, None)

        return _sandbox_iter()

    assert use_web_search
    assert web_search_url is not None
    assert web_search_tool_entry is not None
    web_search_backend = _build_web_search_backend(
        base_url=web_search_url,
        tool_entry=web_search_tool_entry,
    )
    await web_search_backend.__aenter__()  # eager-open

    async def _web_search_iter() -> AsyncIterator[MessageStreamEvent]:
        try:
            kwargs = inject_purpose_hints_anthropic(
                {**call_kwargs},
                web_search_backend.purpose_hints(),
                header=tools_header,
            )
            async for event in anthropic_tool_loop_stream(
                completion_kwargs=kwargs,
                pool=web_search_backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
            ):
                yield event
        finally:
            await web_search_backend.__aexit__(None, None, None)

    return _web_search_iter()
