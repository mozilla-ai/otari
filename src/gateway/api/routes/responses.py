import os
from collections.abc import AsyncIterator
from typing import Annotated, Any

from any_llm import AnyLLM, aresponses
from any_llm.types.completion import CompletionUsage
from any_llm.types.responses import Response as ResponsesResponse
from any_llm.types.responses import ResponseStreamEvent
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseUsage
from openresponses_types.types import Usage as OpenResponsesUsage
from pydantic import BaseModel, ConfigDict
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
from gateway.services.mcp_loop_responses import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    responses_tool_loop,
    responses_tool_loop_stream,
)
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.tool_format import inject_purpose_hints_responses, openai_to_responses_tools
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import RESPONSES_STREAM_FORMAT, streaming_generator

router = APIRouter(prefix="/v1", tags=["responses"])

_MASTER_KEY_USER_REQUIRED = "When using master key, 'user' field is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"


class ResponsesRequest(BaseModel):
    """OpenAI Responses API-compatible request.

    Gateway-internal fields (``mcp_servers``, ``tools_header``,
    ``max_tool_iterations``) opt the request into gateway-managed MCP /
    sandbox / web_search without changing the upstream wire shape. They're
    stripped before the request is forwarded.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any
    stream: bool = False
    user: str | None = None
    tools: list[dict[str, Any]] | None = None

    # Gateway-internal — same semantics as ChatCompletionRequest /
    # MessagesRequest. Stripped from upstream call_kwargs at the boundary.
    mcp_servers: list[McpServerConfig] | None = None
    tools_header: str | None = None
    max_tool_iterations: int | None = None


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


@router.post("/responses", response_model=None)
async def create_response(
    raw_request: Request,
    response: FastAPIResponse,
    request_body: ResponsesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible Responses endpoint.

    Supports gateway-managed MCP tool-use loops, sandboxed code execution,
    and SearXNG web_search on the standalone path. Platform-mode
    multi-attempt fallback is handled in a follow-up PR.
    """

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

    # Tool extraction mirrors chat.py / messages.py.
    sandbox_tool_entry, tools_after_sandbox = _extract_code_execution_tool(request_body.tools)
    sandbox_url: str | None = os.environ.get("GATEWAY_SANDBOX_URL") or None
    use_sandbox = False
    if sandbox_tool_entry is not None:
        if sandbox_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "code_execution tool requested but no sandbox is configured on this gateway. "
                    "Set GATEWAY_SANDBOX_URL on the gateway, or remove code_execution from `tools`."
                ),
            )
        if request_body.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "code_execution and mcp_servers cannot be combined in the same request yet; "
                    "pick one. Multi-backend dispatch is a planned refinement."
                ),
            )
        use_sandbox = True

    web_search_tool_entry, remaining_user_tools = _extract_web_search_tool(tools_after_sandbox)
    web_search_url: str | None = os.environ.get("GATEWAY_WEB_SEARCH_URL") or None
    use_web_search = False
    if web_search_tool_entry is not None:
        if web_search_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "web_search tool requested but no search backend is configured on this gateway. "
                    "Set GATEWAY_WEB_SEARCH_URL on the gateway, or remove web_search from `tools`."
                ),
            )
        if use_sandbox or request_body.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "web_search cannot be combined with code_execution or mcp_servers in the same "
                    "request yet; pick one."
                ),
            )
        use_web_search = True

    mcp_server_configs = request_body.mcp_servers
    max_tool_iterations = min(
        request_body.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
        MAX_TOOL_ITERATIONS_CAP,
    )

    request_fields = _strip_gateway_fields(
        request_body.model_dump(exclude_none=True),
        tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
        remaining_user_tools=remaining_user_tools,
    )
    # Caller-supplied function tools are still in OpenAI Chat-Completions
    # nested shape; convert to Responses flat shape so the upstream call
    # accepts them.
    if request_fields.get("tools"):
        request_fields["tools"] = openai_to_responses_tools(request_fields["tools"])

    input_payload = request_fields.pop("input")
    stream = bool(request_fields.pop("stream", False))
    request_fields.pop("model", None)
    request_fields.pop("user", None)
    request_fields["user"] = user_id

    call_kwargs: dict[str, Any] = {**provider_kwargs}
    call_kwargs.update(request_fields)
    call_kwargs["model"] = model
    call_kwargs["provider"] = provider
    call_kwargs["input_data"] = input_payload

    use_tool_loop = bool(mcp_server_configs) or use_sandbox or use_web_search

    if stream:
        return await _stream_responses(
            call_kwargs=call_kwargs,
            tools_header=request_body.tools_header,
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
        result = await _run_responses_non_stream(
            call_kwargs=call_kwargs,
            tools_header=request_body.tools_header,
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
        usage_data = _usage_to_completion_usage(getattr(result, "usage", None))
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            usage_override=usage_data,
        )

    except HTTPException:
        raise
    except MaxToolIterationsExceeded as e:
        # Gateway-side cap hit, not an upstream failure. 422 lets callers
        # distinguish a runaway tool loop from a provider outage.
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except SandboxNotReachableError as e:
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
        ) from e
    except WebSearchNotReachableError as e:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        ) from e
    except Exception as e:
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
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

    return result.model_dump(exclude_none=True)


async def _run_responses_non_stream(
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
) -> ResponsesResponse:
    """Dispatch the non-streaming Responses call.

    Plain ``aresponses`` when no gateway tools are in play. Otherwise the
    appropriate backend is opened and ``responses_tool_loop`` drives the tool
    round-trips.
    """
    if not use_tool_loop:
        return await aresponses(**call_kwargs)  # type: ignore[return-value]

    if mcp_server_configs:
        async with MCPClientPool(mcp_server_configs) as pool:
            kwargs = inject_purpose_hints_responses(
                {**call_kwargs},
                pool.purpose_hints(),
                header=tools_header,
            )
            return await responses_tool_loop(
                completion_kwargs=kwargs,
                pool=pool,
                max_iterations=max_tool_iterations,
            )

    if use_sandbox:
        assert sandbox_url is not None
        sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
        async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
            kwargs = inject_purpose_hints_responses(
                {**call_kwargs},
                backend.purpose_hints(),
                header=tools_header,
            )
            return await responses_tool_loop(
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
        kwargs = inject_purpose_hints_responses(
            {**call_kwargs},
            web_backend.purpose_hints(),
            header=tools_header,
        )
        return await responses_tool_loop(
            completion_kwargs=kwargs,
            pool=web_backend,  # type: ignore[arg-type]
            max_iterations=max_tool_iterations,
        )


async def _stream_responses(
    *,
    call_kwargs: dict[str, Any],
    tools_header: str | None,
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
    """Streaming Responses dispatch.

    Plain ``aresponses(stream=True)`` when no gateway tools are in play.
    Otherwise the backend is opened eagerly so a backend-unreachable failure
    surfaces as an HTTP error rather than a 200 + in-band SSE error. Same
    rationale as the chat.py / messages.py streaming paths.
    """
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
            error=error,
        )

    try:
        if not use_tool_loop:
            stream_result = await aresponses(**call_kwargs)
            stream_iter: AsyncIterator[ResponseStreamEvent] = stream_result  # type: ignore[assignment]
        else:
            stream_iter = await _open_tool_loop_stream(
                call_kwargs=call_kwargs,
                tools_header=tools_header,
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
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
        ) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        ) from exc

    rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
    return StreamingResponse(
        streaming_generator(
            stream=stream_iter,
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
) -> AsyncIterator[ResponseStreamEvent]:
    """Open the right backend, eagerly enter it, and return an iterator that
    yields all events for the entire tool loop.
    """
    if mcp_server_configs:
        pool_cfgs = mcp_server_configs

        async def _mcp_iter() -> AsyncIterator[ResponseStreamEvent]:
            async with MCPClientPool(pool_cfgs) as pool:
                kwargs = inject_purpose_hints_responses(
                    {**call_kwargs},
                    pool.purpose_hints(),
                    header=tools_header,
                )
                async for event in responses_tool_loop_stream(
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
        await sandbox_backend.__aenter__()  # eager-open

        async def _sandbox_iter() -> AsyncIterator[ResponseStreamEvent]:
            try:
                kwargs = inject_purpose_hints_responses(
                    {**call_kwargs},
                    sandbox_backend.purpose_hints(),
                    header=tools_header,
                )
                async for event in responses_tool_loop_stream(
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

    async def _web_search_iter() -> AsyncIterator[ResponseStreamEvent]:
        try:
            kwargs = inject_purpose_hints_responses(
                {**call_kwargs},
                web_search_backend.purpose_hints(),
                header=tools_header,
            )
            async for event in responses_tool_loop_stream(
                completion_kwargs=kwargs,
                pool=web_search_backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
            ):
                yield event
        finally:
            await web_search_backend.__aexit__(None, None, None)

    return _web_search_iter()
