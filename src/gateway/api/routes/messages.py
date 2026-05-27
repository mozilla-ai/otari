import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

import httpx
from any_llm import AnyLLM, LLMProvider, amessages
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStreamEvent,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.api.routes._platform import (
    ResolvedAttempt,
    ResolvedRoute,
    _classify_upstream_error,
    _extract_platform_user_token,
    _report_platform_usage,
    _resolve_platform_credentials,
    _resolve_platform_mcp_servers,
    run_platform_attempts,
)
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
from gateway.rate_limit import RateLimitInfo, check_rate_limit
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
from gateway.streaming import (
    ANTHROPIC_STREAM_FORMAT,
    StreamingAttemptFailure,
    iterate_streaming_attempts,
    streaming_generator,
)

router = APIRouter(prefix="/v1", tags=["messages"])


class MessagesRequest(BaseModel):
    """Anthropic Messages API-compatible request.

    Gateway-internal fields (``mcp_servers``, ``mcp_server_ids``,
    ``tools_header``, ``max_tool_iterations``) opt the request into
    gateway-managed MCP / sandbox / web_search without changing the upstream
    wire shape. They're stripped before the request is forwarded.
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
    mcp_server_ids: list[uuid.UUID] | None = None
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
    background_tasks: BackgroundTasks,
    request: MessagesRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API-compatible endpoint.

    Supports MCP tool-use loops, sandboxed code execution, and SearXNG
    web_search in both standalone mode and platform mode. Platform-mode
    requests resolve credentials via the platform service and (for
    non-tool-loop requests) get multi-attempt fallback across the resolved
    route. Tool-loop requests collapse to a single attempt — once
    ``on_first_response`` lock-in plumbing lands across the codebase, a
    follow-up will enable pre-lock-in fallback for tool-loop requests too.
    """
    api_key: APIKey | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    platform_mode = config.is_platform_mode
    route: ResolvedRoute | None = None
    user_token: str | None = None

    if platform_mode:
        user_token = _extract_platform_user_token(raw_request)
        start_time = time.perf_counter()
        route = await _resolve_platform_credentials(
            config=config,
            user_token=user_token,
            model_selector=request.model,
        )
        resolve_latency_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Otari-Request-ID"] = route.request_id
        logger.info(
            "Platform resolve succeeded request_id=%s attempts=%d fallback_enabled=%s resolve_latency_ms=%.2f",
            route.request_id,
            len(route.attempts),
            route.fallback_enabled,
            resolve_latency_ms,
        )
    else:
        if db is None:
            raise _anthropic_error(_ERR_API, "Database session unavailable", status.HTTP_500_INTERNAL_SERVER_ERROR)
        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
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

    # mcp_server_ids is platform-only — standalone has no platform to
    # resolve them against, so we reject with a 400 rather than silently
    # ignoring the field.
    if request.mcp_server_ids and not platform_mode:
        raise _anthropic_error(
            _ERR_INVALID_REQUEST,
            "mcp_server_ids is only available in platform mode",
            status.HTTP_400_BAD_REQUEST,
        )
    if platform_mode and request.mcp_server_ids:
        assert user_token is not None
        resolved_mcp_servers = await _resolve_platform_mcp_servers(
            config=config,
            user_token=user_token,
            mcp_server_ids=request.mcp_server_ids,
        )
        request.mcp_servers = (request.mcp_servers or []) + resolved_mcp_servers

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
    use_tool_loop = bool(mcp_server_configs) or use_sandbox or use_web_search

    # Strip gateway-internal fields, convert any caller-supplied OpenAI-shaped
    # tools to Anthropic shape so a mixed list works.
    request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
        remaining_user_tools=remaining_user_tools,
    )
    if request_fields.get("tools"):
        request_fields["tools"] = openai_to_anthropic_tools(request_fields["tools"])

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if request.stream:
        # Tool-loop streaming collapses to a single attempt (same as chat.py
        # until on_first_response lock-in is wired across all three loops).
        if platform_mode and not use_tool_loop:
            assert route is not None
            if not route.attempts:
                logger.error("Platform returned empty attempts list request_id=%s", route.request_id)
                raise _anthropic_error(
                    _ERR_API,
                    "Authorization service returned no resolvable provider",
                    status.HTTP_502_BAD_GATEWAY,
                )
            try:
                return await _run_streaming_with_fallback_messages(
                    route=route,
                    base_request_fields=request_fields,
                    response=response,
                    config=config,
                    background_tasks=background_tasks,
                    rate_limit_info=rate_limit_info,
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.error("All streaming attempts failed request_id=%s: %s", route.request_id, exc)
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                    raise _anthropic_error(
                        _ERR_API,
                        "LLM provider timeout" if len(route.attempts) <= 1 else "All upstream providers timed out",
                        status.HTTP_504_GATEWAY_TIMEOUT,
                    ) from exc
                raise _anthropic_error(
                    _ERR_API,
                    "LLM provider error" if len(route.attempts) <= 1 else "All upstream providers failed",
                    status.HTTP_502_BAD_GATEWAY,
                ) from exc

        # Standalone (or platform + tool-loop): single attempt streaming.
        if platform_mode:
            # Tool-loop platform path: build call_kwargs from the primary attempt.
            assert route is not None
            if not route.attempts:
                raise _anthropic_error(
                    _ERR_API,
                    "Authorization service returned no resolvable provider",
                    status.HTTP_502_BAD_GATEWAY,
                )
            attempt = route.attempts[0]
            provider = LLMProvider(attempt.provider)
            model = attempt.model
            attempt_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
            if attempt.api_base:
                attempt_kwargs["api_base"] = attempt.api_base
            call_kwargs: dict[str, Any] = {
                **attempt_kwargs,
                **request_fields,
                "model": f"{provider.value}:{model}",
            }
        else:
            provider, model = AnyLLM.split_model_provider(request.model)
            provider_kwargs = get_provider_kwargs(config, provider)
            call_kwargs = {**provider_kwargs, **request_fields}

        # Platform tool-loop streaming is currently single-attempt; pass the
        # primary attempt's correlation id + the route's request id so the
        # response still carries the platform contract (X-Correlation-ID,
        # X-Otari-Request-ID, usage reported via _report_platform_usage).
        platform_correlation_id: str | None = None
        platform_request_id: str | None = None
        platform_config: GatewayConfig | None = None
        if platform_mode and route is not None and route.attempts:
            platform_correlation_id = route.attempts[0].attempt_id
            platform_request_id = route.request_id
            platform_config = config

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
            platform_correlation_id=platform_correlation_id,
            platform_request_id=platform_request_id,
            platform_config=platform_config,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    if platform_mode:
        assert route is not None
        platform_route = route
        attempts_to_try = platform_route.attempts
        if not attempts_to_try:
            logger.error("Platform returned empty attempts list request_id=%s", platform_route.request_id)
            raise _anthropic_error(
                _ERR_API,
                "Authorization service returned no resolvable provider",
                status.HTTP_502_BAD_GATEWAY,
            )
        try:
            return await _run_platform_non_stream_messages(
                route=platform_route,
                attempts=attempts_to_try,
                base_request_fields=request_fields,
                tools_header=request.tools_header,
                response=response,
                background_tasks=background_tasks,
                config=config,
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
        except HTTPException:
            raise
        except SandboxNotReachableError as e:
            logger.error("Sandbox unreachable: %s", e)
            raise _anthropic_error(
                _ERR_API,
                "code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
                status.HTTP_502_BAD_GATEWAY,
            ) from e
        except WebSearchNotReachableError as e:
            logger.error("Web search backend unreachable: %s", e)
            raise _anthropic_error(
                _ERR_API,
                "web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
                status.HTTP_502_BAD_GATEWAY,
            ) from e

    # Standalone non-stream path
    provider, model = AnyLLM.split_model_provider(request.model)
    provider_kwargs = get_provider_kwargs(config, provider)
    call_kwargs = {**provider_kwargs, **request_fields}

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

        if db is not None and result.usage:
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
        if db is not None:
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
        if db is not None:
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


async def _run_platform_non_stream_messages(
    *,
    route: ResolvedRoute,
    attempts: list[ResolvedAttempt],
    base_request_fields: dict[str, Any],
    tools_header: str | None,
    response: Response,
    background_tasks: BackgroundTasks,
    config: GatewayConfig,
    rate_limit_info: RateLimitInfo | None,
    use_tool_loop: bool,
    mcp_server_configs: list[McpServerConfig] | None,
    use_sandbox: bool,
    sandbox_tool_entry: dict[str, Any] | None,
    sandbox_url: str | None,
    use_web_search: bool,
    web_search_tool_entry: dict[str, Any] | None,
    web_search_url: str | None,
    max_tool_iterations: int,
) -> dict[str, Any]:
    """Drive the multi-attempt platform-mode non-streaming path via the
    shared ``run_platform_attempts`` runner.
    """

    async def _run_attempt(
        completion_kwargs: dict[str, Any],
        on_first_response: Callable[[], None],
    ) -> MessageResponse:
        if not use_tool_loop:
            return await amessages(**completion_kwargs)  # type: ignore[return-value]
        if mcp_server_configs:
            async with MCPClientPool(mcp_server_configs) as pool:
                kwargs = inject_purpose_hints_anthropic(
                    {**completion_kwargs},
                    pool.purpose_hints(),
                    header=tools_header,
                )
                return await anthropic_tool_loop(
                    completion_kwargs=kwargs,
                    pool=pool,
                    max_iterations=max_tool_iterations,
                    on_first_response=on_first_response,
                )
        if use_sandbox:
            assert sandbox_url is not None
            sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
            async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
                kwargs = inject_purpose_hints_anthropic(
                    {**completion_kwargs},
                    backend.purpose_hints(),
                    header=tools_header,
                )
                return await anthropic_tool_loop(
                    completion_kwargs=kwargs,
                    pool=backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                    on_first_response=on_first_response,
                )
        assert use_web_search
        assert web_search_url is not None
        assert web_search_tool_entry is not None
        async with _build_web_search_backend(
            base_url=web_search_url,
            tool_entry=web_search_tool_entry,
        ) as web_backend:
            kwargs = inject_purpose_hints_anthropic(
                {**completion_kwargs},
                web_backend.purpose_hints(),
                header=tools_header,
            )
            return await anthropic_tool_loop(
                completion_kwargs=kwargs,
                pool=web_backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
                on_first_response=on_first_response,
            )

    def _extract_usage(result: MessageResponse) -> CompletionUsage | None:
        if not result.usage:
            return None
        return CompletionUsage(
            prompt_tokens=result.usage.input_tokens,
            completion_tokens=result.usage.output_tokens,
            total_tokens=result.usage.input_tokens + result.usage.output_tokens,
        )

    def _report_attempt_outcome(
        attempt: ResolvedAttempt,
        outcome: str,
        usage: Any,
        error_class: str | None,
    ) -> None:
        background_tasks.add_task(
            _report_platform_usage,
            config,
            attempt.attempt_id,
            outcome,
            usage,
            error_class,
        )

    def _on_attempt_success(attempt: ResolvedAttempt) -> None:
        response.headers["X-Correlation-ID"] = attempt.attempt_id
        if rate_limit_info:
            for key, value in rate_limit_headers(rate_limit_info).items():
                response.headers[key] = value

    result = await run_platform_attempts(
        route=route,
        attempts=attempts,
        base_request_fields=base_request_fields,
        run_attempt=_run_attempt,
        extract_usage=_extract_usage,
        classify_error=_classify_upstream_error,
        report_attempt_outcome=_report_attempt_outcome,
        on_success=_on_attempt_success,
        max_tool_iterations=max_tool_iterations,
    )
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
    """Standalone-mode non-streaming dispatch.

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
    db: AsyncSession | None,
    log_writer: LogWriter,
    api_key_id: str | None,
    user_id: str | None,
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
    platform_correlation_id: str | None = None,
    platform_request_id: str | None = None,
    platform_config: GatewayConfig | None = None,
) -> StreamingResponse:
    """Streaming dispatch for single-attempt requests.

    Used for standalone mode and for the tool-loop path in platform mode
    (where streaming fallback is still single-attempt — multi-attempt
    streaming for non-tool-loop uses ``_run_streaming_with_fallback_messages``).

    When ``platform_correlation_id`` / ``platform_request_id`` /
    ``platform_config`` are set (platform mode, single-attempt path), the
    response includes ``X-Correlation-ID`` / ``X-Otari-Request-ID`` headers
    and usage is reported to the platform on complete/error. When unset, the
    response logs usage to the local DB via ``log_usage`` (standalone mode).
    """
    call_kwargs["stream"] = True
    platform_mode_active = platform_correlation_id is not None and platform_config is not None

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
        if platform_mode_active:
            assert platform_config is not None
            assert platform_correlation_id is not None
            asyncio.create_task(
                _report_platform_usage(
                    config=platform_config,
                    correlation_id=platform_correlation_id,
                    outcome="success",
                    usage=usage_data,
                )
            )
            return
        if db is None:
            return
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
        if platform_mode_active:
            assert platform_config is not None
            assert platform_correlation_id is not None
            asyncio.create_task(
                _report_platform_usage(
                    config=platform_config,
                    correlation_id=platform_correlation_id,
                    outcome="error",
                    usage=None,
                )
            )
            return
        if db is None:
            return
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

    headers: dict[str, str] = {}
    if rate_limit_info:
        headers.update(rate_limit_headers(rate_limit_info))
    if platform_correlation_id:
        headers["X-Correlation-ID"] = platform_correlation_id
    if platform_request_id:
        headers["X-Otari-Request-ID"] = platform_request_id

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
        headers=headers,
    )


async def _run_streaming_with_fallback_messages(
    *,
    route: ResolvedRoute,
    base_request_fields: dict[str, Any],
    response: Response,
    config: GatewayConfig,
    background_tasks: BackgroundTasks,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Multi-attempt streaming for platform-mode non-tool-loop requests.

    Mirrors chat.py's ``_run_streaming_with_fallback``: iterates
    ``route.attempts`` and falls through on any attempt that fails before its
    first chunk arrives. Once an attempt yields its first chunk, the request
    locks in and any further errors land in the SSE channel as today.
    """
    base_request_fields = dict(base_request_fields)  # we mutate stream_options
    first_chunk_timeout_seconds = int(config.platform.get("streaming_first_chunk_timeout_ms", 2000)) / 1000

    async def _build_for_attempt(
        attempt: ResolvedAttempt,
    ) -> AsyncIterator[MessageStreamEvent]:
        attempt_provider = LLMProvider(attempt.provider)
        provider_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            provider_kwargs["api_base"] = attempt.api_base
        completion_kwargs = {
            **provider_kwargs,
            **base_request_fields,
            "model": f"{attempt_provider.value}:{attempt.model}",
            "stream": True,
        }
        return await amessages(**completion_kwargs)  # type: ignore[return-value]

    async def _on_attempt_failed(attempt: ResolvedAttempt, failure: StreamingAttemptFailure) -> None:
        background_tasks.add_task(
            _report_platform_usage,
            config,
            attempt.attempt_id,
            "error",
            None,
            failure.error_class,
        )
        logger.warning(
            "Streaming attempt failed request_id=%s position=%d provider=%s model=%s error=%s",
            route.request_id,
            attempt.position,
            attempt.provider,
            attempt.model,
            failure.error_class,
        )

    chosen, stream = await iterate_streaming_attempts(
        attempts=route.attempts,
        build_stream=_build_for_attempt,
        classify_error=_classify_upstream_error,
        on_attempt_failed=_on_attempt_failed,
        first_chunk_timeout_seconds=first_chunk_timeout_seconds,
    )

    return _build_streaming_response_messages(
        stream=stream,
        provider=LLMProvider(chosen.provider),
        model=chosen.model,
        correlation_id=chosen.attempt_id,
        request_id=route.request_id,
        config=config,
        rate_limit_info=rate_limit_info,
    )


def _build_streaming_response_messages(
    *,
    stream: AsyncIterator[MessageStreamEvent],
    provider: LLMProvider,
    model: str,
    correlation_id: str,
    request_id: str,
    config: GatewayConfig,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Wrap an already-opened Anthropic stream in an SSE response for
    platform-mode requests. Sets correlation / request-id headers and reports
    usage to the platform on complete.
    """

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
        asyncio.create_task(
            _report_platform_usage(
                config=config,
                correlation_id=correlation_id,
                outcome="success",
                usage=usage_data,
            )
        )

    async def _on_error(error: str) -> None:
        asyncio.create_task(
            _report_platform_usage(
                config=config,
                correlation_id=correlation_id,
                outcome="error",
                usage=None,
            )
        )

    headers: dict[str, str] = {}
    if rate_limit_info:
        headers.update(rate_limit_headers(rate_limit_info))
    headers["X-Correlation-ID"] = correlation_id
    headers["X-Otari-Request-ID"] = request_id

    return StreamingResponse(
        streaming_generator(
            stream=stream,
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=ANTHROPIC_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
        ),
        media_type="text/event-stream",
        headers=headers,
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
        await sandbox_backend.__aenter__()  # eager-open

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
