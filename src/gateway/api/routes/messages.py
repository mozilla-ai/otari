import asyncio
import math
import os
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
from gateway.api.routes._helpers import apply_input_guardrails, latest_user_text
from gateway.api.routes._mode_strategy import (
    RequestModeStrategy,
    RequestSettlement,
    ResolveErrors,
    ResolveSpec,
    select_request_mode_strategy,
)
from gateway.api.routes._platform import (
    ResolvedAttempt,
    ResolvedRoute,
    _classify_upstream_error,
    _extract_platform_user_token,
    _report_platform_usage,
    _resolve_platform_credentials,
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
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo
from gateway.services.budget_service import (
    reconcile_reservation,
    refund_reservation,
)
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
    ``guardrails``, ``tools_header``, ``max_tool_iterations``) opt the request
    into gateway-managed MCP / sandbox / web_search / guardrails without
    changing the upstream wire shape. They're stripped before the request is
    forwarded.
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
_MASTER_KEY_USER_REQUIRED = "When using master key, 'metadata.user_id' is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"
_USER_FORBIDDEN = "'metadata.user_id' does not match the authenticated API key's user"
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
    # Mode seam: select the strategy once, then resolve credentials (platform)
    # or authenticate + reserve budget (standalone). See chat.py for the shared
    # rationale; the only differences here are the Anthropic error envelope and
    # the metadata-based user id.
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None
    strategy = select_request_mode_strategy(config, db, log_writer)
    await strategy.resolve(
        raw_request=raw_request,
        response=response,
        spec=ResolveSpec(
            model_selector=request.model,
            user_id_from_request=str(user_from_metadata) if user_from_metadata else None,
            prompt_chars=len(str(request.messages)) + len(str(request.system or "")),
            max_output_tokens=request.max_tokens,
            errors=ResolveErrors(
                db_unavailable=_anthropic_error(
                    _ERR_API, "Database session unavailable", status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                master_key_user_required=_anthropic_error(
                    _ERR_INVALID_REQUEST, _MASTER_KEY_USER_REQUIRED, status.HTTP_400_BAD_REQUEST
                ),
                api_key_validation_failed=_anthropic_error(
                    _ERR_API, _API_KEY_VALIDATION_FAILED, status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                no_user=_anthropic_error(_ERR_API, _API_KEY_NO_USER, status.HTTP_500_INTERNAL_SERVER_ERROR),
                forbidden_user=_anthropic_error(_ERR_PERMISSION, _USER_FORBIDDEN, status.HTTP_403_FORBIDDEN),
                no_pricing=_anthropic_error(
                    _ERR_INVALID_REQUEST,
                    f"No pricing configured for model '{request.model}'",
                    status.HTTP_402_PAYMENT_REQUIRED,
                ),
            ),
        ),
    )
    platform_mode = strategy.is_platform
    route = strategy.route
    rate_limit_info = strategy.rate_limit_info
    # Used only by the standalone non-streaming tail below (platform returns
    # earlier via the multi-attempt runner); the streaming path settles through
    # the strategy's settlement object instead.
    api_key_id = strategy.api_key_id
    user_id = strategy.user_id
    reservation = strategy.reservation

    # Caller-requested input guardrails run before any provider/tool dispatch
    # (see chat.py for the rationale). Stripped before forwarding upstream.
    await apply_input_guardrails(
        request.guardrails,
        latest_user_text(request.messages),
        response=response,
    )

    # mcp_server_ids is platform-only — the platform strategy resolves them,
    # the standalone strategy rejects the field with a 400.
    if request.mcp_server_ids:
        resolved_mcp_servers = await strategy.resolve_mcp_servers(
            request.mcp_server_ids,
            reject_error=_anthropic_error(
                _ERR_INVALID_REQUEST,
                "mcp_server_ids is only available in platform mode",
                status.HTTP_400_BAD_REQUEST,
            ),
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
                    "otari_code_execution tool requested but no sandbox is configured on this gateway. "
                    "Set GATEWAY_SANDBOX_URL on the gateway, or remove otari_code_execution from `tools`."
                ),
                status.HTTP_400_BAD_REQUEST,
            )
        if request.mcp_servers:
            raise _anthropic_error(
                _ERR_INVALID_REQUEST,
                (
                    "otari_code_execution and mcp_servers cannot be combined in the same request yet; "
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
                    "otari_web_search tool requested but no search backend is configured on this gateway. "
                    "Set GATEWAY_WEB_SEARCH_URL on the gateway, or remove otari_web_search from `tools`."
                ),
                status.HTTP_400_BAD_REQUEST,
            )
        if use_sandbox or request.mcp_servers:
            raise _anthropic_error(
                _ERR_INVALID_REQUEST,
                (
                    "otari_web_search cannot be combined with otari_code_execution or mcp_servers in the same "
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
            assert route is not None  # guaranteed by the platform-mode branch above
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
                    strategy=strategy,
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
            assert route is not None  # guaranteed by the platform-mode branch above
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

        # Platform tool-loop streaming is currently single-attempt; carry the
        # primary attempt's correlation id + the route's request id so the
        # response still presents the platform contract (X-Correlation-ID,
        # X-Otari-Request-ID, usage reported via the platform settlement).
        correlation_id: str | None = None
        request_id: str | None = None
        if platform_mode and route is not None and route.attempts:
            correlation_id = route.attempts[0].attempt_id
            request_id = route.request_id

        settlement = strategy.make_settlement(
            provider=provider,
            model=model,
            endpoint="/v1/messages",
            correlation_id=correlation_id,
        )

        return await _stream_messages(
            call_kwargs=call_kwargs,
            request=request,
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
            settlement=settlement,
            correlation_id=correlation_id,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    if platform_mode:
        assert route is not None  # guaranteed by the platform-mode branch above
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

        if db is not None:
            actual_cost: float | None = None
            if result.usage:
                usage_data = CompletionUsage(
                    prompt_tokens=result.usage.input_tokens,
                    completion_tokens=result.usage.output_tokens,
                    total_tokens=result.usage.input_tokens + result.usage.output_tokens,
                )
                actual_cost = await log_usage(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/messages",
                    user_id=user_id,
                    usage_override=usage_data,
                )
            if reservation is not None:
                await reconcile_reservation(db, reservation, actual_cost or 0.0)

    except HTTPException:
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
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
            if reservation is not None:
                await refund_reservation(db, reservation)
        raise _anthropic_error(_ERR_INVALID_REQUEST, str(e), status.HTTP_422_UNPROCESSABLE_ENTITY) from e
    except SandboxNotReachableError as e:
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, e)
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
        raise _anthropic_error(
            _ERR_API,
            "code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            status.HTTP_502_BAD_GATEWAY,
        ) from e
    except WebSearchNotReachableError as e:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, e)
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
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
            if reservation is not None:
                await refund_reservation(db, reservation)
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
            assert sandbox_url is not None  # guaranteed past the missing-URL 400 above
            sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
            async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
                kwargs = inject_purpose_hints_anthropic(
                    {**completion_kwargs},
                    backend.purpose_hints(),
                    header=tools_header,
                )
                return await anthropic_tool_loop(
                    completion_kwargs=kwargs,
                    pool=backend,
                    max_iterations=max_tool_iterations,
                    on_first_response=on_first_response,
                )
        assert use_web_search
        assert web_search_url is not None  # guaranteed past the missing-URL 400 above
        assert web_search_tool_entry is not None  # guaranteed by the web_search opt-in above
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
                pool=web_backend,
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
        assert sandbox_url is not None  # guaranteed past the missing-URL 400 above
        sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
        async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
            kwargs = inject_purpose_hints_anthropic(
                {**call_kwargs},
                backend.purpose_hints(),
                header=tools_header,
            )
            return await anthropic_tool_loop(
                completion_kwargs=kwargs,
                pool=backend,
                max_iterations=max_tool_iterations,
            )

    assert use_web_search
    assert web_search_url is not None  # guaranteed past the missing-URL 400 above
    assert web_search_tool_entry is not None  # guaranteed by the web_search opt-in above
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
            pool=web_backend,
            max_iterations=max_tool_iterations,
        )


async def _stream_messages(
    *,
    call_kwargs: dict[str, Any],
    request: MessagesRequest,
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
    settlement: RequestSettlement,
    correlation_id: str | None = None,
    request_id: str | None = None,
) -> StreamingResponse:
    """Streaming dispatch for single-attempt requests.

    Used for standalone mode and for the tool-loop path in platform mode
    (where streaming fallback is still single-attempt — multi-attempt
    streaming for non-tool-loop uses ``_run_streaming_with_fallback_messages``).

    Usage settlement (platform usage report vs local log + reconcile/refund) is
    delegated to ``settlement``; ``correlation_id`` / ``request_id`` are set as
    headers when present (platform mode).
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
        await settlement.on_success(usage_data)

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
    except HTTPException:
        raise
    except Exception as exc:
        # A provider error raised before the stream starts must surface as an
        # HTTP error (Anthropic envelope) rather than escaping uncaught into a
        # 500. Matches the non-streaming handler.
        await settlement.on_provider_error_precommit(str(exc))
        logger.error("Provider call failed for %s:%s: %s", provider, model, exc)
        raise _anthropic_error(
            _ERR_API,
            _PROVIDER_ERROR,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc

    headers: dict[str, str] = {}
    if rate_limit_info:
        headers.update(rate_limit_headers(rate_limit_info))
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    if request_id:
        headers["X-Otari-Request-ID"] = request_id

    return StreamingResponse(
        streaming_generator(
            stream=msg_stream_typed,
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=ANTHROPIC_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=settlement.on_error,
            label=f"{provider}:{model}",
            on_no_usage=settlement.on_no_usage,
            on_incomplete=settlement.on_incomplete,
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
    strategy: RequestModeStrategy,
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
        settlement=strategy.make_settlement(
            provider=LLMProvider(chosen.provider),
            model=chosen.model,
            endpoint="/v1/messages",
            correlation_id=chosen.attempt_id,
        ),
        correlation_id=chosen.attempt_id,
        request_id=route.request_id,
        rate_limit_info=rate_limit_info,
    )


def _build_streaming_response_messages(
    *,
    stream: AsyncIterator[MessageStreamEvent],
    provider: LLMProvider,
    model: str,
    settlement: RequestSettlement,
    correlation_id: str,
    request_id: str,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Wrap an already-opened Anthropic stream in an SSE response for
    platform-mode requests. Sets correlation / request-id headers and reports
    usage through ``settlement`` on complete/error.
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
        await settlement.on_success(usage_data)

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
            on_error=settlement.on_error,
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
        assert sandbox_url is not None  # guaranteed past the missing-URL 400 above
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
                    pool=sandbox_backend,
                    max_iterations=max_tool_iterations,
                ):
                    yield event
            finally:
                await sandbox_backend.__aexit__(None, None, None)

        return _sandbox_iter()

    assert use_web_search
    assert web_search_url is not None  # guaranteed past the missing-URL 400 above
    assert web_search_tool_entry is not None  # guaranteed by the web_search opt-in above
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
                pool=web_search_backend,
                max_iterations=max_tool_iterations,
            ):
                yield event
        finally:
            await web_search_backend.__aexit__(None, None, None)

    return _web_search_iter()


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
    logging. Authentication mirrors :func:`create_message` — platform mode
    resolves the caller's token against the platform, standalone mode validates
    the API key — so the endpoint is not an open token-counting oracle.
    """
    if config.is_platform_mode:
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
            raise _anthropic_error(_ERR_API, "Database session unavailable", status.HTTP_500_INTERNAL_SERVER_ERROR)
        await verify_api_key_or_master_key(raw_request, db, config)

    return CountTokensResponse(input_tokens=_estimate_input_tokens(request))
