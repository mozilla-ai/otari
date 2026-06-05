import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.exceptions import AnyLLMError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db_if_needed, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._helpers import apply_input_guardrails, latest_user_text, resolve_user_id
from gateway.api.routes._platform import (
    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS,
    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP,
    _STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY,
    _STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY,
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
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.metrics import record_cost, record_tokens
from gateway.models.entities import APIKey, UsageLog
from gateway.models.guardrails import GuardrailConfig
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
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    inject_purpose_hints,
    mcp_tool_loop,
    mcp_tool_loop_stream,
)
from gateway.services.pricing_service import find_model_pricing, pricing_required_but_missing
from gateway.services.provider_kwargs import get_provider_kwargs as get_provider_kwargs  # noqa: F401
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import (
    OPENAI_STREAM_FORMAT,
    StreamingAttemptFailure,
    iterate_streaming_attempts,
    streaming_generator,
)

router = APIRouter(prefix="/v1/chat", tags=["chat"])

def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset)),
    }

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_message_structure(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, message in enumerate(v):
            if "role" not in message:
                msg = f"messages[{i}]: 'role' is required"
                raise ValueError(msg)
        return v

    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
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
            "Falls back to GATEWAY_TOOLS_HEADER env, then to the built-in default."
        ),
    )
    max_tool_iterations: int | None = Field(default=None, ge=1, le=MAX_TOOL_ITERATIONS_CAP)


async def log_usage(
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str | None,
    endpoint: str,
    user_id: str | None = None,
    response: ChatCompletion | AsyncIterator[ChatCompletionChunk] | None = None,
    usage_override: CompletionUsage | None = None,
    error: str | None = None,
    cost_override: float | None = None,
) -> float | None:
    """Log API usage to the database and return the computed cost.

    Spend is no longer written here — the budget reservation reconcile path owns
    ``users.spend``. This returns the cost it computed so the caller can reconcile
    the reservation with the actual amount.

    Args:
        db: Database session
        api_key_id: API key identifier (None if using master key)
        model: Model name
        provider: Provider name
        endpoint: Endpoint path
        user_id: User identifier for tracking
        response: Response object (if successful)
        usage_override: Usage data for streaming requests
        error: Error message (if failed)

    Returns:
        The computed cost for this request, or None when usage/pricing is absent.

    """
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )

    usage_data = usage_override
    if not usage_data and response and isinstance(response, ChatCompletion) and response.usage:
        usage_data = response.usage

    if usage_data:
        usage_log.prompt_tokens = usage_data.prompt_tokens
        usage_log.completion_tokens = usage_data.completion_tokens
        usage_log.total_tokens = usage_data.total_tokens

        record_tokens(
            str(provider or ""),
            model,
            usage_data.prompt_tokens,
            usage_data.completion_tokens,
        )

        pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
        if pricing:
            cost = (usage_data.prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
                usage_data.completion_tokens / 1_000_000
            ) * pricing.output_price_per_million
            usage_log.cost = cost
            record_cost(str(provider or ""), model, cost)
        else:
            model_ref = f"{provider}:{model}" if provider else model
            logger.warning(f"No pricing configured for '{model_ref}'. Usage will be tracked without cost.")

    # When the caller bills a fixed amount without provider usage (e.g. the
    # stream-missing-usage estimate policy), record that amount on the log row so
    # usage_logs.cost stays consistent with the spend that was reconciled.
    if cost_override is not None:
        usage_log.cost = cost_override

    await log_writer.put(usage_log)
    return usage_log.cost




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

    api_key: APIKey | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    platform_mode = config.is_platform_mode
    route: ResolvedRoute | None = None
    user_token: str | None = None  # set inside the platform_mode branch; referenced again later
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
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database session unavailable",
            )

        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
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

        # Tolerate an unparseable / unknown-provider selector here — the budget
        # check below and the downstream provider call surface those with their
        # own status codes. A model we can't parse simply has no pricing.
        try:
            gate_provider, gate_model = AnyLLM.split_model_provider(request.model)
        except (ValueError, AnyLLMError):
            gate_provider, gate_model = None, request.model
        gate_pricing = await find_model_pricing(db, gate_provider, gate_model)
        estimate = estimate_cost(
            gate_pricing,
            prompt_chars=len(str(request.messages)),
            max_output_tokens=request.max_tokens if request.max_tokens is not None else request.max_completion_tokens,
            default_output_tokens=config.budget_estimate_default_output_tokens,
        )
        # Reserve first so user/blocked/budget rejections (404/403) take
        # precedence over the missing-pricing rejection (402); refund if we then
        # reject for missing pricing.
        reservation = await reserve_budget(
            db, user_id, estimate, model=request.model, strategy=config.budget_strategy
        )
        if pricing_required_but_missing(gate_pricing, require_pricing=config.require_pricing):
            await refund_reservation(db, reservation)
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"No pricing configured for model '{request.model}'",
            )

    # Caller-requested input guardrails run before any provider/tool dispatch.
    # `block`-mode flags raise 403 here (provider never called); `monitor`-mode
    # flags annotate the response header and fall through. No-op when the caller
    # didn't send a `guardrails` field. The field is stripped before forwarding
    # upstream by `_strip_gateway_fields`.
    await apply_input_guardrails(
        request.guardrails,
        latest_user_text(request.messages),
        response=response,
    )

    # Workspace-scoped MCP server references (platform mode only). Callers
    # pass `mcp_server_ids: [uuid, ...]` instead of inlining each config; we
    # resolve them against the platform's `/gateway/mcp-servers/resolve`
    # endpoint and merge with any inline `mcp_servers` so the downstream
    # MCP loop sees a single list. In standalone mode there's no platform
    # to consult, so we reject the field with a 400 rather than silently
    # ignoring it.
    if request.mcp_server_ids and not platform_mode:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mcp_server_ids is only available in platform mode",
        )
    if platform_mode and request.mcp_server_ids:
        assert user_token is not None  # guaranteed by the platform_mode branch above
        resolved_mcp_servers = await _resolve_platform_mcp_servers(
            config=config,
            user_token=user_token,
            mcp_server_ids=request.mcp_server_ids,
        )
        request.mcp_servers = (request.mcp_servers or []) + resolved_mcp_servers

    # Per-request opt-in for sandboxed code execution. Matches Anthropic /
    # OpenAI's wire shape: caller adds {"type": "code_execution"} to their
    # `tools` array. The sandbox endpoint is operator-controlled — set
    # GATEWAY_SANDBOX_URL in the gateway's environment. We deliberately do
    # NOT honour a per-request URL override (e.g. `sandbox_url` on the tool
    # entry) because that would let an untrusted caller use the gateway as
    # an open HTTP client (SSRF / arbitrary-POST surface). Operators that
    # want per-tenant sandbox isolation should stand up multiple gateway
    # instances or proxy at a layer they control.
    # Mutually exclusive with `mcp_servers` for now (multi-backend dispatch
    # is the next iteration); the multi-attempt routing-policy fallback is
    # also bypassed when sandbox is in use.
    sandbox_tool_entry, tools_after_sandbox = _extract_code_execution_tool(request.tools)
    sandbox_url: str | None = os.environ.get("GATEWAY_SANDBOX_URL") or None
    # Auth to forward to the sandbox backend when configured (e.g. an authenticated
    # remote sandbox that derives the tenant from the token). Passed explicitly to
    # each SandboxBackend — never global state, so it can't leak across requests.
    sandbox_forward_auth = raw_request.headers.get("authorization") if config.sandbox_forward_auth else None
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
        if request.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "otari_code_execution and mcp_servers cannot be combined in the same request yet; "
                    "pick one. Multi-backend dispatch is a planned refinement."
                ),
            )
        use_sandbox = True

    # web_search opt-in mirrors the sandbox path; see comment above for the
    # threat-model rationale. GATEWAY_WEB_SEARCH_URL is operator-controlled —
    # see web_search_backend.py for the wire protocol the service must expose.
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
        if use_sandbox or request.mcp_servers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "otari_web_search cannot be combined with otari_code_execution or mcp_servers in the same "
                    "request yet; pick one."
                ),
            )
        use_web_search = True

    # ------------------------------------------------------------------
    # Streaming path: iterate `route.attempts` before any bytes are flushed,
    # then commit to the first attempt that yields a chunk. Implemented in
    # `_run_streaming_with_fallback` via `iterate_streaming_attempts`.
    #
    # Mid-stream failover (after first chunk) is out of scope: recovering
    # would require either silently buffering the prefix (delays first byte)
    # or a client-aware "restart" event (breaks OpenAI-SDK compatibility).
    # Errors after first chunk propagate to the client.
    # ------------------------------------------------------------------
    if request.stream:
        # Platform-mode streaming — tool modes (sandbox / web_search / MCP)
        # also flow through here so they get per-attempt fallback up to the
        # lock-in point (first chunk = first assistant message).
        if platform_mode:
            if route is None or not route.attempts:
                if route is not None:
                    logger.error(
                        "Platform returned empty attempts list request_id=%s",
                        route.request_id,
                    )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Authorization service returned no resolvable provider",
                )
            stream_mcp_configs = request.mcp_servers
            stream_max_tool_iterations = min(
                request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
                MAX_TOOL_ITERATIONS_CAP,
            )
            try:
                return await _run_streaming_with_fallback(
                    route=route,
                    request=request,
                    response=response,
                    config=config,
                    background_tasks=background_tasks,
                    rate_limit_info=rate_limit_info,
                    mcp_server_configs=stream_mcp_configs,
                    use_sandbox=use_sandbox,
                    sandbox_url=sandbox_url,
                    sandbox_forward_auth=sandbox_forward_auth,
                    sandbox_tool_entry=sandbox_tool_entry,
                    use_web_search=use_web_search,
                    web_search_url=web_search_url,
                    web_search_tool_entry=web_search_tool_entry,
                    remaining_user_tools=remaining_user_tools,
                    tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
                    max_tool_iterations=stream_max_tool_iterations,
                )
            except HTTPException:
                raise
            except SandboxNotReachableError as exc:
                logger.error("Sandbox unreachable request_id=%s: %s", route.request_id, exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
                ) from exc
            except WebSearchNotReachableError as exc:
                logger.error("Web search backend unreachable request_id=%s: %s", route.request_id, exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
                ) from exc
            except Exception as exc:
                # Every attempt failed before any bytes were flushed.
                logger.error(
                    "All streaming attempts failed request_id=%s: %s",
                    route.request_id,
                    exc,
                )
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=(
                            "LLM provider timeout" if len(route.attempts) <= 1 else "All upstream providers timed out"
                        ),
                    ) from exc
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=("LLM provider error" if len(route.attempts) <= 1 else "All upstream providers failed"),
                ) from exc

        # Standalone path: single attempt, no fallback (no `route.attempts`).
        # Platform-mode requests (including tool modes) take the multi-attempt
        # branch above; this block runs only when `model` is a literal
        # ``provider:model`` selector and no routing policy applies.
        provider, model = AnyLLM.split_model_provider(request.model)
        provider_kwargs = get_provider_kwargs(config, provider)

        mcp_server_configs = request.mcp_servers
        max_tool_iterations = min(
            request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
            MAX_TOOL_ITERATIONS_CAP,
        )

        request_fields = _strip_gateway_fields(
            request.model_dump(exclude_unset=True),
            tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
            remaining_user_tools=remaining_user_tools,
        )
        completion_kwargs = {**provider_kwargs, **request_fields}
        if completion_kwargs.get("stream_options") is None:
            completion_kwargs["stream_options"] = {"include_usage": True}

        try:
            if mcp_server_configs:
                # Bind the truthy value to a non-Optional local so mypy can
                # narrow inside the nested `_mcp_stream` closure.
                pool_configs = mcp_server_configs

                async def _mcp_stream() -> AsyncIterator[ChatCompletionChunk]:
                    async with MCPClientPool(pool_configs) as pool:
                        kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                pool.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        async for chunk in mcp_tool_loop_stream(
                            completion_kwargs=kwargs,
                            pool=pool,
                            max_iterations=max_tool_iterations,
                        ):
                            yield chunk

                stream: AsyncIterator[ChatCompletionChunk] = _mcp_stream()
            elif use_sandbox:
                # SandboxBackend duck-types as MCPClientPool — same tool-loop helper.
                # Eagerly open the backend *before* constructing the
                # StreamingResponse so that a failure to reach the sandbox
                # surfaces synchronously as an HTTP error. A lazy `async with`
                # inside the generator would only run after the response
                # was committed, and SandboxNotReachableError would land in
                # the SSE channel after a 200 OK header — confusing for
                # clients that expected a normal HTTP failure.
                assert sandbox_url is not None
                sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
                sandbox_backend = SandboxBackend(
                    sandbox_url=sandbox_url, purpose_hint=sandbox_hint, forward_auth=sandbox_forward_auth
                )
                await sandbox_backend.__aenter__()  # may raise SandboxNotReachableError

                async def _sandbox_stream() -> AsyncIterator[ChatCompletionChunk]:
                    try:
                        kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                sandbox_backend.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        async for chunk in mcp_tool_loop_stream(
                            completion_kwargs=kwargs,
                            pool=sandbox_backend,  # type: ignore[arg-type]
                            max_iterations=max_tool_iterations,
                        ):
                            yield chunk
                    finally:
                        await sandbox_backend.__aexit__(None, None, None)

                stream = _sandbox_stream()
            elif use_web_search:
                # Same eager-open rationale as the sandbox path above.
                assert web_search_url is not None
                assert web_search_tool_entry is not None
                web_search_backend = _build_web_search_backend(
                    base_url=web_search_url,
                    tool_entry=web_search_tool_entry,
                )
                await web_search_backend.__aenter__()

                async def _web_search_stream() -> AsyncIterator[ChatCompletionChunk]:
                    try:
                        kwargs = {
                            **completion_kwargs,
                            "messages": inject_purpose_hints(
                                completion_kwargs["messages"],
                                web_search_backend.purpose_hints(),
                                header=request.tools_header,
                            ),
                        }
                        async for chunk in mcp_tool_loop_stream(
                            completion_kwargs=kwargs,
                            pool=web_search_backend,  # type: ignore[arg-type]
                            max_iterations=max_tool_iterations,
                        ):
                            yield chunk
                    finally:
                        await web_search_backend.__aexit__(None, None, None)

                stream = _web_search_stream()
            else:
                stream = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        except HTTPException:
            raise
        except SandboxNotReachableError as exc:
            # The sandbox is part of the gateway's own infra, not the LLM
            # provider — surface a clearer status so operators don't chase
            # a "provider outage" that's actually the sandbox container
            # being down. 502 keeps "upstream dependency failed" semantics.
            logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
            if db is not None and reservation is not None:
                await refund_reservation(db, reservation)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
            ) from exc
        except WebSearchNotReachableError as exc:
            logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
            if db is not None and reservation is not None:
                await refund_reservation(db, reservation)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
            ) from exc
        except Exception as exc:
            if db is not None:
                await log_usage(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    model=model,
                    provider=provider,
                    endpoint="/v1/chat/completions",
                    user_id=user_id,
                    error=str(exc),
                )
                if reservation is not None:
                    await refund_reservation(db, reservation)
            logger.error("Stream creation failed for %s:%s: %s", provider, model, exc)
            if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="LLM provider timeout",
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="LLM provider error",
            ) from exc

        return _build_streaming_response(
            stream=stream,
            provider=provider,
            model=model,
            platform_mode=False,
            correlation_id=None,
            request_id=None,
            config=config,
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            rate_limit_info=rate_limit_info,
            reservation=reservation,
        )

    # ------------------------------------------------------------------
    # Non-streaming path. Iterates `route.attempts` with pre-lock-in
    # fallback semantics: if a tool-loop attempt fails *before* the model
    # has returned its first assistant message, we fall through to the
    # next attempt. Once locked in (first assistant message received),
    # subsequent failures terminate the request — we never swap providers
    # between tool-use rounds.
    # ------------------------------------------------------------------
    mcp_server_configs = request.mcp_servers
    max_tool_iterations = min(
        request.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
        MAX_TOOL_ITERATIONS_CAP,
    )

    if platform_mode:
        if route is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error: missing route context",
            )
        platform_route = route
        attempts_to_try = platform_route.attempts
        if not attempts_to_try:
            logger.error(
                "Platform returned empty attempts list request_id=%s",
                platform_route.request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Authorization service returned no resolvable provider",
            )

        base_request_fields = _strip_gateway_fields(
            request.model_dump(exclude_unset=True),
            tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
            remaining_user_tools=remaining_user_tools,
        )

        async def _run_chat_attempt(
            completion_kwargs: dict[str, Any],
            on_first_response: Callable[[], None],
        ) -> ChatCompletion:
            if mcp_server_configs:
                async with MCPClientPool(mcp_server_configs) as pool:
                    mcp_kwargs = {
                        **completion_kwargs,
                        "messages": inject_purpose_hints(
                            completion_kwargs["messages"],
                            pool.purpose_hints(),
                            header=request.tools_header,
                        ),
                    }
                    return await mcp_tool_loop(
                        completion_kwargs=mcp_kwargs,
                        pool=pool,
                        max_iterations=max_tool_iterations,
                        on_first_response=on_first_response,
                    )
            if use_sandbox:
                assert sandbox_url is not None
                sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
                async with SandboxBackend(
                    sandbox_url=sandbox_url, purpose_hint=sandbox_hint, forward_auth=sandbox_forward_auth
                ) as backend:
                    sandbox_kwargs = {
                        **completion_kwargs,
                        "messages": inject_purpose_hints(
                            completion_kwargs["messages"],
                            backend.purpose_hints(),
                            header=request.tools_header,
                        ),
                    }
                    return await mcp_tool_loop(
                        completion_kwargs=sandbox_kwargs,
                        pool=backend,  # type: ignore[arg-type]
                        max_iterations=max_tool_iterations,
                        on_first_response=on_first_response,
                    )
            if use_web_search:
                assert web_search_url is not None
                assert web_search_tool_entry is not None
                async with _build_web_search_backend(
                    base_url=web_search_url,
                    tool_entry=web_search_tool_entry,
                ) as web_backend:
                    web_kwargs = {
                        **completion_kwargs,
                        "messages": inject_purpose_hints(
                            completion_kwargs["messages"],
                            web_backend.purpose_hints(),
                            header=request.tools_header,
                        ),
                    }
                    return await mcp_tool_loop(
                        completion_kwargs=web_kwargs,
                        pool=web_backend,  # type: ignore[arg-type]
                        max_iterations=max_tool_iterations,
                        on_first_response=on_first_response,
                    )
            return await acompletion(**completion_kwargs)  # type: ignore[return-value]

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

        return await run_platform_attempts(
            route=platform_route,
            attempts=attempts_to_try,
            base_request_fields=base_request_fields,
            run_attempt=_run_chat_attempt,
            extract_usage=lambda completion: completion.usage,
            classify_error=_classify_upstream_error,
            report_attempt_outcome=_report_attempt_outcome,
            on_success=_on_attempt_success,
            max_tool_iterations=max_tool_iterations,
        )

    provider, model = AnyLLM.split_model_provider(request.model)
    provider_kwargs = get_provider_kwargs(config, provider)

    # Standalone path (no platform / no fallback). Same MCP / sandbox /
    # web_search semantics as platform mode above: if `mcp_servers` is set,
    # the request goes through the MCP tool-use loop; if `tools` includes a
    # code_execution entry, the request goes through the SandboxBackend
    # tool-use loop; if `tools` includes a web_search entry, the request
    # goes through the WebSearchBackend tool-use loop; otherwise a single
    # ``acompletion`` call.
    request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=sandbox_tool_entry is not None or web_search_tool_entry is not None,
        remaining_user_tools=remaining_user_tools,
    )
    completion_kwargs = {**provider_kwargs, **request_fields}

    try:
        if mcp_server_configs:
            async with MCPClientPool(mcp_server_configs) as pool:
                mcp_kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        pool.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                completion = await mcp_tool_loop(
                    completion_kwargs=mcp_kwargs,
                    pool=pool,
                    max_iterations=max_tool_iterations,
                )
        elif use_sandbox:
            assert sandbox_url is not None
            sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
            async with SandboxBackend(
                sandbox_url=sandbox_url, purpose_hint=sandbox_hint, forward_auth=sandbox_forward_auth
            ) as backend:
                sandbox_kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        backend.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                completion = await mcp_tool_loop(
                    completion_kwargs=sandbox_kwargs,
                    pool=backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                )
        elif use_web_search:
            assert web_search_url is not None
            assert web_search_tool_entry is not None
            async with _build_web_search_backend(
                base_url=web_search_url,
                tool_entry=web_search_tool_entry,
            ) as web_backend:
                web_kwargs = {
                    **completion_kwargs,
                    "messages": inject_purpose_hints(
                        completion_kwargs["messages"],
                        web_backend.purpose_hints(),
                        header=request.tools_header,
                    ),
                }
                completion = await mcp_tool_loop(
                    completion_kwargs=web_kwargs,
                    pool=web_backend,  # type: ignore[arg-type]
                    max_iterations=max_tool_iterations,
                )
        else:
            completion = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        if db is not None:
            actual_cost = await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                response=completion,
            )
            if reservation is not None:
                await reconcile_reservation(db, reservation, actual_cost or 0.0)
    except HTTPException:
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
        raise
    except SandboxNotReachableError as exc:
        # Sandbox is gateway-side infra, not an LLM provider. Clearer detail
        # so operators don't chase a provider outage that's really the
        # sandbox container being down.
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="code_execution sandbox unreachable — check GATEWAY_SANDBOX_URL",
        ) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        if db is not None and reservation is not None:
            await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="web_search backend unreachable — check GATEWAY_WEB_SEARCH_URL",
        ) from exc
    except MaxToolIterationsExceeded as e:
        # Gateway-owned cap, not an upstream provider failure. 422 lets
        # callers distinguish a runaway tool loop from a real outage.
        logger.warning("Tool loop iteration cap hit (standalone): cap=%d", max_tool_iterations)
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                error=str(e),
            )
            if reservation is not None:
                await refund_reservation(db, reservation)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        if db is not None:
            await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint="/v1/chat/completions",
                user_id=user_id,
                error=str(e),
            )
            if reservation is not None:
                await refund_reservation(db, reservation)

        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        if isinstance(e, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="LLM provider timeout",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider error",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return completion


def _build_streaming_response(
    *,
    stream: AsyncIterator[ChatCompletionChunk],
    provider: LLMProvider,
    model: str,
    platform_mode: bool,
    correlation_id: str | None,
    request_id: str | None,
    config: GatewayConfig,
    db: AsyncSession | None,
    log_writer: LogWriter | None,
    api_key_id: str | None,
    user_id: str | None,
    rate_limit_info: RateLimitInfo | None,
    reservation: ReservationHandle | None = None,
) -> StreamingResponse:
    """Wrap an already-opened upstream stream in an SSE response."""

    def _format_chunk(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    def _extract_usage(chunk: ChatCompletionChunk) -> CompletionUsage | None:
        if not chunk.usage:
            return None
        return CompletionUsage(
            prompt_tokens=chunk.usage.prompt_tokens or 0,
            completion_tokens=chunk.usage.completion_tokens or 0,
            total_tokens=chunk.usage.total_tokens or 0,
        )

    async def _on_complete(usage_data: CompletionUsage) -> None:
        if platform_mode and correlation_id:
            asyncio.create_task(
                _report_platform_usage(
                    config=config,
                    correlation_id=correlation_id,
                    outcome="success",
                    usage=usage_data,
                )
            )
            return
        if db is None or log_writer is None:
            return
        actual_cost = await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
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
                endpoint="/v1/chat/completions",
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
            endpoint="/v1/chat/completions",
            user_id=user_id,
            error="stream completed without usage data" if policy == "fail" else None,
            cost_override=reservation.estimate,
        )
        await reconcile_reservation(db, reservation, reservation.estimate)

    async def _on_error(error: str) -> None:
        if platform_mode and correlation_id:
            asyncio.create_task(
                _report_platform_usage(
                    config=config,
                    correlation_id=correlation_id,
                    outcome="error",
                    usage=None,
                )
            )
            return
        if db is None or log_writer is None:
            return
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
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

    rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
    # StreamingResponse builds its own response object, so headers we want on
    # the wire have to be passed in here — assigning to the dependency-injected
    # `Response` object doesn't propagate to streaming responses.
    headers = dict(rl_headers)
    if platform_mode and correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    if platform_mode and request_id:
        headers["X-Otari-Request-ID"] = request_id
    return StreamingResponse(
        streaming_generator(
            stream=stream,
            format_chunk=_format_chunk,
            extract_usage=_extract_usage,
            fmt=OPENAI_STREAM_FORMAT,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
            on_no_usage=_on_no_usage,
            on_incomplete=_on_incomplete,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


async def _run_streaming_with_fallback(
    *,
    route: ResolvedRoute,
    request: ChatCompletionRequest,
    response: Response,
    config: GatewayConfig,
    background_tasks: BackgroundTasks,
    rate_limit_info: RateLimitInfo | None,
    mcp_server_configs: list[McpServerConfig] | None = None,
    use_sandbox: bool = False,
    sandbox_url: str | None = None,
    sandbox_forward_auth: str | None = None,
    sandbox_tool_entry: dict[str, Any] | None = None,
    use_web_search: bool = False,
    web_search_url: str | None = None,
    web_search_tool_entry: dict[str, Any] | None = None,
    remaining_user_tools: list[dict[str, Any]] | None = None,
    tools_extracted: bool = False,
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
) -> StreamingResponse:
    """Iterate route.attempts for a streaming request, falling through on any
    attempt that fails before its first chunk arrives.

    Once an attempt yields its first chunk, we commit and start flushing to
    the client — errors past that point propagate to the SSE channel as
    today. This is the streaming analogue of the non-streaming
    pre-lock-in-fallback flow in the request handler above.

    Tool-loop modes (sandbox / web_search / MCP) are layered on top using
    the same pre-first-chunk fallback semantics: the upstream
    ``acompletion(stream=True)`` call inside ``mcp_tool_loop_stream`` runs
    lazily when ``iterate_streaming_attempts`` pulls the first chunk; if
    that call fails (or any retryable error fires before a chunk arrives)
    we move to the next attempt with a clean conversation slate.
    """
    tool_mode = bool(mcp_server_configs) or use_sandbox or use_web_search

    base_request_fields = _strip_gateway_fields(
        request.model_dump(exclude_unset=True),
        tools_extracted=tools_extracted,
        remaining_user_tools=remaining_user_tools,
    )

    # Tool-mode streams need more headroom on first-chunk wait: the model
    # may reason briefly before emitting tokens or a tool_call, especially
    # with extended thinking. Keep the existing tight default for plain
    # streams so failed-attempt latency stays low when no tools are in play.
    if tool_mode:
        first_chunk_timeout_seconds = (
            int(
                config.platform.get(
                    _STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY,
                    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP,
                )
            )
            / 1000
        )
    else:
        first_chunk_timeout_seconds = (
            int(
                config.platform.get(
                    _STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY,
                    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS,
                )
            )
            / 1000
        )

    # Open the tool backend(s) once before iterating attempts. Eager open
    # lets gateway-side dependency failures (sandbox unreachable, web
    # search backend unreachable) surface as a normal HTTP error instead
    # of as a mid-SSE error event after the 200 OK header. The exit stack
    # is closed when streaming completes (success or error path inside
    # the wrapped iterator) or, if no attempt commits, in the outer
    # except handler below.
    backend_stack = AsyncExitStack()
    pool_for_loop: Any = None
    try:
        if mcp_server_configs:
            pool_for_loop = await backend_stack.enter_async_context(MCPClientPool(mcp_server_configs))
        elif use_sandbox:
            assert sandbox_url is not None
            sandbox_hint = _resolve_sandbox_purpose_hint(sandbox_tool_entry)
            pool_for_loop = await backend_stack.enter_async_context(
                SandboxBackend(sandbox_url=sandbox_url, purpose_hint=sandbox_hint, forward_auth=sandbox_forward_auth),
            )
        elif use_web_search:
            assert web_search_url is not None
            assert web_search_tool_entry is not None
            pool_for_loop = await backend_stack.enter_async_context(
                _build_web_search_backend(base_url=web_search_url, tool_entry=web_search_tool_entry),
            )
    except BaseException:
        # Eager-open failure (e.g. SandboxNotReachableError) — propagate so
        # the route handler maps it to the existing HTTP status. Nothing to
        # clean up on the stack yet because the entry failed.
        await backend_stack.aclose()
        raise

    async def _build_for_attempt(
        attempt: ResolvedAttempt,
    ) -> AsyncIterator[ChatCompletionChunk]:
        attempt_provider = LLMProvider(attempt.provider)
        provider_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            provider_kwargs["api_base"] = attempt.api_base
        completion_kwargs = {
            **provider_kwargs,
            **base_request_fields,
            "model": f"{attempt_provider.value}:{attempt.model}",
        }
        if completion_kwargs.get("stream_options") is None:
            completion_kwargs["stream_options"] = {"include_usage": True}
        if pool_for_loop is None:
            return await acompletion(**completion_kwargs)  # type: ignore[return-value]
        kwargs = {
            **completion_kwargs,
            "messages": inject_purpose_hints(
                completion_kwargs["messages"],
                pool_for_loop.purpose_hints(),
                header=request.tools_header,
            ),
        }
        return mcp_tool_loop_stream(
            completion_kwargs=kwargs,
            pool=pool_for_loop,
            max_iterations=max_tool_iterations,
        )

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

    try:
        chosen, stream = await iterate_streaming_attempts(
            attempts=route.attempts,
            build_stream=_build_for_attempt,
            classify_error=_classify_upstream_error,
            on_attempt_failed=_on_attempt_failed,
            first_chunk_timeout_seconds=first_chunk_timeout_seconds,
        )
    except BaseException:
        # No attempt yielded a first chunk — close the tool backend before
        # propagating the failure. The route handler maps it to HTTP.
        await backend_stack.aclose()
        raise

    if tool_mode:
        logger.info(
            "Tool-loop streaming lock-in request_id=%s position=%d provider=%s model=%s",
            route.request_id,
            chosen.position,
            chosen.provider,
            chosen.model,
        )

    if pool_for_loop is not None:
        async def _stream_with_backend_cleanup() -> AsyncIterator[ChatCompletionChunk]:
            try:
                async for chunk in stream:
                    yield chunk
            finally:
                await backend_stack.aclose()

        stream_to_return: AsyncIterator[ChatCompletionChunk] = _stream_with_backend_cleanup()
    else:
        stream_to_return = stream

    return _build_streaming_response(
        stream=stream_to_return,
        provider=LLMProvider(chosen.provider),
        model=chosen.model,
        platform_mode=True,
        correlation_id=chosen.attempt_id,
        request_id=route.request_id,
        config=config,
        db=None,  # platform mode doesn't use the local DB
        log_writer=None,  # unused when db is None
        api_key_id=None,
        user_id=None,
        rate_limit_info=rate_limit_info,
    )
