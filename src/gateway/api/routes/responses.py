import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

import httpx
from any_llm import AnyLLM, LLMProvider, aresponses
from any_llm.exceptions import AnyLLMError
from any_llm.types.completion import CompletionUsage
from any_llm.types.responses import Response as ResponsesResponse
from any_llm.types.responses import ResponseStreamEvent
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseUsage
from openresponses_types.types import Usage as OpenResponsesUsage
from pydantic import BaseModel, ConfigDict, Field
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
from gateway.services.budget_service import (
    ReservationHandle,
    estimate_cost,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_loop_responses import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    responses_tool_loop,
    responses_tool_loop_stream,
)
from gateway.services.pricing_service import find_model_pricing, pricing_required_but_missing
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.tool_format import inject_purpose_hints_responses, openai_to_responses_tools
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import (
    RESPONSES_STREAM_FORMAT,
    StreamingAttemptFailure,
    iterate_streaming_attempts,
    streaming_generator,
)

router = APIRouter(prefix="/v1", tags=["responses"])

_MASTER_KEY_USER_REQUIRED = "When using master key, 'user' field is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"
_USER_FORBIDDEN = "'user' field does not match the authenticated API key's user"


class ResponsesRequest(BaseModel):
    """OpenAI Responses API-compatible request.

    Gateway-internal fields (``mcp_servers``, ``mcp_server_ids``,
    ``tools_header``, ``max_tool_iterations``) opt the request into
    gateway-managed MCP / sandbox / web_search without changing the upstream
    wire shape. They're stripped before the request is forwarded.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any
    stream: bool = False
    user: str | None = None
    tools: list[dict[str, Any]] | None = None

    # Gateway-internal: identical semantics to ChatCompletionRequest.
    mcp_servers: list[McpServerConfig] | None = None
    mcp_server_ids: list[uuid.UUID] | None = None
    tools_header: str | None = None
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)


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
    background_tasks: BackgroundTasks,
    request_body: ResponsesRequest,
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible Responses endpoint.

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
    # Budget pre-debit for the standalone (local-DB) path only; platform mode
    # reports usage upstream instead. Settled (reconciled/refunded) at every
    # completion and error hook below.
    reservation: ReservationHandle | None = None

    if platform_mode:
        user_token = _extract_platform_user_token(raw_request)
        start_time = time.perf_counter()
        route = await _resolve_platform_credentials(
            config=config,
            user_token=user_token,
            model_selector=request_body.model,
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
        # Provider-support guard: even in platform mode, an unsupported provider
        # would just fail downstream — surface a clearer 400 upfront. Validate
        # *every* resolved attempt so a fallback that lands on an unsupported
        # provider (e.g. primary OpenAI, fallback Anthropic) fails fast here
        # instead of crashing mid-fallback when the runner calls ``aresponses``
        # on a provider that doesn't speak the Responses API.
        for attempt in route.attempts:
            provider = LLMProvider(attempt.provider)
            provider_class = AnyLLM.get_provider_class(provider)
            if not getattr(provider_class, "SUPPORTS_RESPONSES", False):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Provider '{provider.value}' does not support the Responses API",
                )
    else:
        if db is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session unavailable",
            )
        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
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
            forbidden_user_error=HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=_USER_FORBIDDEN,
            ),
            reject_mismatch=config.reject_user_mismatch,
        )
        rate_limit_info = check_rate_limit(raw_request, user_id)
        # Tolerate an unparseable selector: the budget gate (404/403) and the
        # downstream call surface those; an unparseable model simply has no pricing.
        try:
            gate_provider, gate_model = AnyLLM.split_model_provider(request_body.model)
        except (ValueError, AnyLLMError):
            gate_provider, gate_model = None, request_body.model
        gate_pricing = await find_model_pricing(db, gate_provider, gate_model)
        # max_output_tokens comes from an extra="allow" body, so it may be absent,
        # non-int, or negative — only trust a non-negative int for the estimate.
        raw_max_output = getattr(request_body, "max_output_tokens", None)
        max_output_tokens = raw_max_output if isinstance(raw_max_output, int) and raw_max_output >= 0 else None
        estimate = estimate_cost(
            gate_pricing,
            prompt_chars=len(str(request_body.input))
            + len(str(getattr(request_body, "instructions", "") or "")),
            max_output_tokens=max_output_tokens,
            default_output_tokens=config.budget_estimate_default_output_tokens,
        )
        # Reserve first so user/blocked/budget rejections (404/403) precede the
        # missing-pricing rejection (402); refund if we then reject for no pricing.
        reservation = await reserve_budget(
            db, user_id, estimate, model=request_body.model, strategy=config.budget_strategy
        )
        if pricing_required_but_missing(gate_pricing, require_pricing=config.require_pricing):
            await refund_reservation(db, reservation)
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"No pricing configured for model '{request_body.model}'",
            )
        provider, model = AnyLLM.split_model_provider(request_body.model)
        provider_class = AnyLLM.get_provider_class(provider)
        if not getattr(provider_class, "SUPPORTS_RESPONSES", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider '{provider.value}' does not support the Responses API",
            )

    # mcp_server_ids is platform-only.
    if request_body.mcp_server_ids and not platform_mode:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mcp_server_ids is only available in platform mode",
        )
    if platform_mode and request_body.mcp_server_ids:
        assert user_token is not None
        resolved_mcp_servers = await _resolve_platform_mcp_servers(
            config=config,
            user_token=user_token,
            mcp_server_ids=request_body.mcp_server_ids,
        )
        request_body.mcp_servers = (request_body.mcp_servers or []) + resolved_mcp_servers

    # Tool extraction mirrors chat.py / messages.py.
    sandbox_tool_entry, tools_after_sandbox = _extract_code_execution_tool(request_body.tools)
    sandbox_url: str | None = os.environ.get("GATEWAY_SANDBOX_URL") or None
    use_sandbox = False
    if sandbox_tool_entry is not None:
        if sandbox_url is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "otari_code_execution tool requested but no sandbox is configured on this gateway. "
                    "Set GATEWAY_SANDBOX_URL on the gateway, or remove otari_code_execution from `tools`."
                ),
            )
        if request_body.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "otari_code_execution and mcp_servers cannot be combined in the same request yet; "
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
                    "otari_web_search tool requested but no search backend is configured on this gateway. "
                    "Set GATEWAY_WEB_SEARCH_URL on the gateway, or remove otari_web_search from `tools`."
                ),
            )
        if use_sandbox or request_body.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "otari_web_search cannot be combined with otari_code_execution or mcp_servers in the same "
                    "request yet; pick one."
                ),
            )
        use_web_search = True

    mcp_server_configs = request_body.mcp_servers
    max_tool_iterations = min(
        request_body.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
        MAX_TOOL_ITERATIONS_CAP,
    )
    use_tool_loop = bool(mcp_server_configs) or use_sandbox or use_web_search

    # Strip gateway-internal fields, flatten any caller-supplied function tools
    # to the Responses shape.
    request_fields = _strip_gateway_fields(
        request_body.model_dump(exclude_none=True),
        tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
        remaining_user_tools=remaining_user_tools,
    )
    if request_fields.get("tools"):
        request_fields["tools"] = openai_to_responses_tools(request_fields["tools"])

    input_payload = request_fields.pop("input")
    stream = bool(request_fields.pop("stream", False))
    request_fields.pop("model", None)
    request_fields.pop("user", None)
    # Standalone mode forwards the resolved user_id to the upstream provider
    # for analytics; platform mode handles attribution server-side via the
    # correlation id.
    if not platform_mode and user_id:
        request_fields["user"] = user_id

    # base_request_fields is what gets merged with per-attempt creds; it
    # includes ``provider`` and ``input_data`` so each attempt has the full
    # call shape.
    base_request_fields = {**request_fields}
    if not platform_mode:
        base_request_fields["provider"] = provider
    base_request_fields["input_data"] = input_payload

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------
    if stream:
        if platform_mode and not use_tool_loop:
            assert route is not None
            if not route.attempts:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Authorization service returned no resolvable provider",
                )
            try:
                return await _run_streaming_with_fallback_responses(
                    route=route,
                    base_request_fields=base_request_fields,
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
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=(
                            "LLM provider timeout" if len(route.attempts) <= 1 else "All upstream providers timed out"
                        ),
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        "LLM provider error" if len(route.attempts) <= 1 else "All upstream providers failed"
                    ),
                ) from exc

        # Single-attempt streaming (standalone, or platform + tool-loop).
        if platform_mode:
            assert route is not None
            if not route.attempts:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Authorization service returned no resolvable provider",
                )
            attempt = route.attempts[0]
            attempt_provider = LLMProvider(attempt.provider)
            model = attempt.model
            provider = attempt_provider
            call_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
            if attempt.api_base:
                call_kwargs["api_base"] = attempt.api_base
            call_kwargs.update(request_fields)
            call_kwargs["model"] = model
            call_kwargs["provider"] = provider
            call_kwargs["input_data"] = input_payload
        else:
            provider_kwargs = get_provider_kwargs(config, provider)
            call_kwargs = {**provider_kwargs}
            call_kwargs.update(request_fields)
            call_kwargs["model"] = model
            call_kwargs["provider"] = provider
            call_kwargs["input_data"] = input_payload

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

        return await _stream_responses(
            call_kwargs=call_kwargs,
            tools_header=request_body.tools_header,
            config=config,
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
            reservation=reservation,
        )

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------
    if platform_mode:
        assert route is not None
        platform_route = route
        attempts_to_try = platform_route.attempts
        if not attempts_to_try:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Authorization service returned no resolvable provider",
            )
        try:
            return await _run_platform_non_stream_responses(
                route=platform_route,
                attempts=attempts_to_try,
                base_request_fields=base_request_fields,
                tools_header=request_body.tools_header,
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
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            ) from e
        except WebSearchNotReachableError as e:
            logger.error("Web search backend unreachable: %s", e)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
            ) from e

    # Standalone non-stream path
    provider_kwargs = get_provider_kwargs(config, provider)
    call_kwargs = {**provider_kwargs}
    call_kwargs.update(request_fields)
    call_kwargs["model"] = model
    call_kwargs["provider"] = provider
    call_kwargs["input_data"] = input_payload

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
        if db is not None:
            actual_cost = await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/responses",
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
                endpoint="/v1/responses",
                user_id=user_id,
                error=str(e),
            )
            if reservation is not None:
                await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except SandboxNotReachableError as e:
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, e)
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
        ) from e
    except WebSearchNotReachableError as e:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, e)
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        ) from e
    except Exception as e:
        if db is not None:
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
            if reservation is not None:
                await refund_reservation(db, reservation)
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result.model_dump(exclude_none=True)


async def _run_platform_non_stream_responses(
    *,
    route: ResolvedRoute,
    attempts: list[ResolvedAttempt],
    base_request_fields: dict[str, Any],
    tools_header: str | None,
    response: FastAPIResponse,
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
    """Drive the multi-attempt platform-mode non-streaming Responses path via
    the shared ``run_platform_attempts`` runner.
    """

    async def _run_attempt(
        completion_kwargs: dict[str, Any],
        on_first_response: Callable[[], None],
    ) -> ResponsesResponse:
        # run_platform_attempts hands us ``"model"`` as ``"provider:model"``;
        # split it back out for the aresponses signature.
        merged_model = completion_kwargs.pop("model")
        provider_str, model_str = merged_model.split(":", 1)
        completion_kwargs["model"] = model_str
        completion_kwargs["provider"] = LLMProvider(provider_str)

        if not use_tool_loop:
            return await aresponses(**completion_kwargs)  # type: ignore[return-value]
        if mcp_server_configs:
            async with MCPClientPool(mcp_server_configs) as pool:
                kwargs = inject_purpose_hints_responses(
                    {**completion_kwargs},
                    pool.purpose_hints(),
                    header=tools_header,
                )
                return await responses_tool_loop(
                    completion_kwargs=kwargs,
                    pool=pool,
                    max_iterations=max_tool_iterations,
                    on_first_response=on_first_response,
                )
        if use_sandbox:
            assert sandbox_url is not None
            sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
            async with SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint) as backend:
                kwargs = inject_purpose_hints_responses(
                    {**completion_kwargs},
                    backend.purpose_hints(),
                    header=tools_header,
                )
                return await responses_tool_loop(
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
            kwargs = inject_purpose_hints_responses(
                {**completion_kwargs},
                web_backend.purpose_hints(),
                header=tools_header,
            )
            return await responses_tool_loop(
                completion_kwargs=kwargs,
                pool=web_backend,  # type: ignore[arg-type]
                max_iterations=max_tool_iterations,
                on_first_response=on_first_response,
            )

    def _extract_usage(result: ResponsesResponse) -> CompletionUsage | None:
        return _usage_to_completion_usage(getattr(result, "usage", None))

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
    """Standalone-mode non-streaming dispatch.

    Plain ``aresponses`` when no gateway tools are in play. Otherwise the
    appropriate backend is opened and ``responses_tool_loop`` drives the
    tool round-trips.
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
    config: GatewayConfig,
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
    reservation: ReservationHandle | None = None,
) -> StreamingResponse:
    """Streaming dispatch for single-attempt requests.

    Used for standalone mode and the tool-loop path in platform mode (where
    streaming fallback is still single-attempt — multi-attempt streaming for
    non-tool-loop uses ``_run_streaming_with_fallback_responses``).

    When ``platform_correlation_id`` / ``platform_request_id`` /
    ``platform_config`` are set (platform mode, single-attempt path), the
    response includes ``X-Correlation-ID`` / ``X-Otari-Request-ID`` headers
    and usage is reported to the platform on complete/error. When unset, the
    response logs usage to the local DB via ``log_usage`` (standalone mode).
    """
    call_kwargs["stream"] = True
    platform_mode_active = platform_correlation_id is not None and platform_config is not None

    def _format_chunk(event: ResponseStreamEvent) -> str:
        return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

    def _extract_usage(event: ResponseStreamEvent) -> CompletionUsage | None:
        response_obj = getattr(event, "response", None)
        if response_obj and getattr(response_obj, "usage", None):
            return _usage_to_completion_usage(response_obj.usage)
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
        actual_cost = await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            usage_override=usage_data,
        )
        if reservation is not None:
            await reconcile_reservation(db, reservation, actual_cost or 0.0)

    async def _on_no_usage() -> None:
        # Stream completed but the provider sent no usage data. Settle the
        # reservation per stream_missing_usage_policy instead of billing $0.
        if db is None or log_writer is None or reservation is None:
            return
        policy = config.stream_missing_usage_policy
        if policy == "allow_free":
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/responses",
                user_id=user_id,
            )
            await refund_reservation(db, reservation)
            return
        # 'estimate' and 'fail' both charge the up-front estimate; 'fail' also
        # records the request as errored.
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
            user_id=user_id,
            error="stream completed without usage data" if policy == "fail" else None,
            cost_override=reservation.estimate,
        )
        await reconcile_reservation(db, reservation, reservation.estimate)

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
            endpoint="/v1/responses",
            user_id=user_id,
            error=error,
        )
        if reservation is not None:
            await refund_reservation(db, reservation)

    async def _on_incomplete() -> None:
        # Client disconnected mid-stream — release the reservation.
        if db is None or reservation is None:
            return
        await refund_reservation(db, reservation)

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
    except HTTPException:
        raise
    except Exception as exc:
        # A provider error raised before the stream starts (e.g. auth failure,
        # 5xx, connection error) must surface as a 502 HTTP error rather than
        # escaping uncaught into a 500. Matches the non-streaming handler and
        # the pre-existing behavior before streaming was factored into this
        # helper.
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/responses",
                user_id=user_id,
                error=str(exc),
            )
        logger.error("Provider call failed for %s:%s: %s", provider, model, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
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
            stream=stream_iter,
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=RESPONSES_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
            on_no_usage=_on_no_usage,
            on_incomplete=_on_incomplete,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


async def _run_streaming_with_fallback_responses(
    *,
    route: ResolvedRoute,
    base_request_fields: dict[str, Any],
    response: FastAPIResponse,
    config: GatewayConfig,
    background_tasks: BackgroundTasks,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Multi-attempt streaming for platform-mode non-tool-loop requests.

    Mirrors the chat / messages equivalents: iterates ``route.attempts`` and
    falls through on any attempt that fails before its first chunk arrives.
    """
    first_chunk_timeout_seconds = int(config.platform.get("streaming_first_chunk_timeout_ms", 2000)) / 1000

    async def _build_for_attempt(
        attempt: ResolvedAttempt,
    ) -> AsyncIterator[ResponseStreamEvent]:
        attempt_provider = LLMProvider(attempt.provider)
        provider_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            provider_kwargs["api_base"] = attempt.api_base
        # base_request_fields carries input_data + provider stripped from the
        # request; per-attempt we replace provider and model.
        completion_kwargs = {
            **provider_kwargs,
            **{k: v for k, v in base_request_fields.items() if k != "provider"},
            "model": attempt.model,
            "provider": attempt_provider,
            "stream": True,
        }
        return await aresponses(**completion_kwargs)  # type: ignore[return-value]

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

    return _build_streaming_response_responses(
        stream=stream,
        provider=LLMProvider(chosen.provider),
        model=chosen.model,
        correlation_id=chosen.attempt_id,
        request_id=route.request_id,
        config=config,
        rate_limit_info=rate_limit_info,
    )


def _build_streaming_response_responses(
    *,
    stream: AsyncIterator[ResponseStreamEvent],
    provider: LLMProvider,
    model: str,
    correlation_id: str,
    request_id: str,
    config: GatewayConfig,
    rate_limit_info: RateLimitInfo | None,
) -> StreamingResponse:
    """Wrap an already-opened Responses stream in an SSE response for
    platform-mode requests. Sets correlation / request-id headers and reports
    usage to the platform on complete.
    """

    def _format_chunk(event: ResponseStreamEvent) -> str:
        return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

    def _extract_usage(event: ResponseStreamEvent) -> CompletionUsage | None:
        response_obj = getattr(event, "response", None)
        if response_obj and getattr(response_obj, "usage", None):
            return _usage_to_completion_usage(response_obj.usage)
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
            fmt=RESPONSES_STREAM_FORMAT,
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
) -> AsyncIterator[ResponseStreamEvent]:
    """Return an async iterator that yields events for the entire tool loop.

    For the sandbox and web_search paths the backend is opened **eagerly**
    (the ``await ... __aenter__()`` runs before the iterator is returned), so
    a backend-unreachable error surfaces as an HTTP 502 rather than landing
    in the SSE channel after the response has already committed to 200 OK.

    The MCP path is different: ``MCPClientPool`` is entered lazily inside the
    iterator's ``async with`` block — same trade-off as messages.py.
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
        await sandbox_backend.__aenter__()

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
    await web_search_backend.__aenter__()

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
