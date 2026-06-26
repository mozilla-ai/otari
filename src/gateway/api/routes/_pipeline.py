"""Shared request-pipeline core for the chat / messages / responses routes.

The three completion-style endpoints speak different wire formats but run the
same pipeline: authenticate (platform resolve or local key + budget pre-debit),
apply input guardrails, extract gateway-managed tools, dispatch to the provider
(directly or through a tool-loop backend), and settle the budget reservation
when the request finishes. This module owns that pipeline once; each route
supplies a small :class:`FormatAdapter` for the format-specific edges (request
parsing, SSE chunk shape, error envelope, provider call, tool loop).

Settlement invariants owned here:

* ``reserve_budget`` happens in :func:`resolve_request_context` (standalone
  mode only); every downstream success path reconciles via
  :func:`reconcile_reservation` and every failure path refunds via
  :func:`refund_reservation`, including streaming completions, streams that
  end without usage data (``stream_missing_usage_policy``), client
  disconnects, and pre-stream dispatch failures.
* The streaming settlement callbacks (``on_complete`` / ``on_no_usage`` /
  ``on_error`` / ``on_incomplete``) are built in exactly one place,
  :func:`build_streaming_response`, and are wired identically for the
  single-attempt and platform-fallback paths of every format.
* Backend open semantics: sandbox and web_search backends open eagerly so an
  unreachable backend surfaces as an HTTP 502 before the 200 OK header; the
  MCP pool opens lazily inside the stream generator (single-attempt paths) or
  eagerly on an ``AsyncExitStack`` shared across attempts (platform fallback).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Generic, NamedTuple, Protocol, TypeVar
from urllib.parse import ParseResult, urlparse

import httpx
from any_llm import LLMProvider
from any_llm.exceptions import AnyLLMError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
)
from fastapi import BackgroundTasks, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import verify_api_key_or_master_key
from gateway.api.routes._helpers import apply_input_guardrails, resolve_user_id
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
    _resolve_platform_web_search,
    run_platform_attempts,
)
from gateway.api.routes._tools import (
    _build_web_search_backend,
    _extract_code_execution_tool,
    _extract_web_search_tool,
    _resolve_sandbox_purpose_hint,
)
from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.core.usage import cache_read_tokens_of, cache_write_tokens_of
from gateway.log_config import logger
from gateway.metrics import record_cost, record_tokens
from gateway.models.entities import UsageLog
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import (
    ReservationHandle,
    estimate_cost,
    increase_reservation,
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
    ToolBackend,
)
from gateway.services.pricing_service import find_model_pricing, pricing_required_but_missing
from gateway.services.provider_kwargs import resolve_provider_selector
from gateway.services.sandbox_backend import SandboxBackend, SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.streaming import (
    StreamFormat,
    StreamingAttemptFailure,
    iterate_streaming_attempts,
    streaming_generator,
)

ResultT = TypeVar("ResultT")
ChunkT = TypeVar("ChunkT")

# ---------------------------------------------------------------------------
# Shared wire-level detail strings. These are client-visible API contract
# values; do not edit them without a deprecation plan.
# ---------------------------------------------------------------------------
DB_UNAVAILABLE_DETAIL = "Database session unavailable"
API_KEY_VALIDATION_FAILED_DETAIL = "API key validation failed"
API_KEY_NO_USER_DETAIL = "API key has no associated user"
MCP_SERVER_IDS_HYBRID_ONLY_DETAIL = "mcp_server_ids is only available in hybrid mode"
NO_RESOLVABLE_PROVIDER_DETAIL = "Authorization service returned no resolvable provider"
PROVIDER_ERROR_DETAIL = "LLM provider error"
PROVIDER_TIMEOUT_DETAIL = "LLM provider timeout"
# Specific-but-safe provider-failure details. Each is a fixed string that never
# embeds the raw upstream message, so classifying an error cannot leak provider
# internals (see classify_provider_error and test_error_detail_leakage).
PROVIDER_BAD_REQUEST_DETAIL = "The provider rejected the request as invalid (check the model name and parameters)"
PROVIDER_MODEL_NOT_FOUND_DETAIL = "The requested model was not found on the provider"
PROVIDER_CREDENTIALS_DETAIL = "The provider rejected the gateway's credentials"
PROVIDER_RATE_LIMITED_DETAIL = "The provider rate-limited this request"
ALL_PROVIDERS_FAILED_DETAIL = "All upstream providers failed"
ALL_PROVIDERS_TIMED_OUT_DETAIL = "All upstream providers timed out"
SANDBOX_NOT_CONFIGURED_DETAIL = (
    "otari_code_execution tool requested but no sandbox is configured on this gateway. "
    "Set OTARI_SANDBOX_URL on the gateway, or remove otari_code_execution from `tools`."
)
SANDBOX_MCP_CONFLICT_DETAIL = (
    "otari_code_execution and mcp_servers cannot be combined in the same request yet; "
    "pick one. Multi-backend dispatch is a planned refinement."
)
WEB_SEARCH_NOT_CONFIGURED_DETAIL = (
    "otari_web_search tool requested but no search backend is configured on this gateway. "
    "Set OTARI_WEB_SEARCH_URL on the gateway, or remove otari_web_search from `tools`."
)
WEB_SEARCH_CONFLICT_DETAIL = (
    "otari_web_search cannot be combined with otari_code_execution or mcp_servers in the same request yet; pick one."
)
WEB_SEARCH_NOT_ENABLED_DETAIL = "web search is not enabled for this workspace"
SANDBOX_UNREACHABLE_DETAIL = "code_execution sandbox unreachable — check OTARI_SANDBOX_URL"
WEB_SEARCH_UNREACHABLE_DETAIL = "web_search backend unreachable — check OTARI_WEB_SEARCH_URL"


class ErrorKind(Enum):
    """Coarse error category an adapter maps onto its wire envelope.

    The chat and responses formats raise plain ``HTTPException`` and ignore
    the kind; the Anthropic messages format maps it to the ``error.type``
    field of its error body.
    """

    INVALID_REQUEST = auto()
    API = auto()
    PERMISSION = auto()


class ProviderErrorMapping(NamedTuple):
    """A safe, client-facing (status, detail) for a classified provider failure."""

    status_code: int
    detail: str


def _upstream_status_code(exc: BaseException) -> int | None:
    """Pull an HTTP status off an upstream exception, if it carries one.

    any-llm surfaces provider HTTP errors either as ``exc.status_code`` or via an
    attached ``exc.response.status_code``; everything else has neither.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def classify_provider_error(exc: BaseException) -> ProviderErrorMapping | None:
    """Map an upstream provider exception to a safe, specific (status, detail).

    Returns ``None`` when the failure carries no signal we can safely act on, so
    the caller falls back to its existing generic provider-error response. Every
    detail returned here is a fixed string: the raw provider message is never
    included, preserving the no-leak guarantee. The mapping is intentionally
    conservative, classifying only the cases a caller can act on and leaving
    everything else (including provider 5xx) to the generic 502.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return ProviderErrorMapping(status.HTTP_504_GATEWAY_TIMEOUT, PROVIDER_TIMEOUT_DETAIL)

    status_code = _upstream_status_code(exc)
    if status_code is None:
        return None
    if status_code in (400, 422):
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, PROVIDER_BAD_REQUEST_DETAIL)
    if status_code == 404:
        return ProviderErrorMapping(status.HTTP_404_NOT_FOUND, PROVIDER_MODEL_NOT_FOUND_DETAIL)
    # A provider rejecting the gateway's credentials is a gateway-config fault,
    # not the caller's: surface it as a 502, never as a client-facing 401/403.
    if status_code in (401, 403):
        return ProviderErrorMapping(status.HTTP_502_BAD_GATEWAY, PROVIDER_CREDENTIALS_DETAIL)
    # A provider 429 is surfaced as a client 429. Note this drops the upstream
    # Retry-After: the (status, detail) pair cannot carry it, so the caller
    # can't honor the provider's exact backoff window. Acceptable because the
    # gateway has no single correct value to forward (BYO vs shared keys differ)
    # and a bare 429 still tells the caller to back off.
    if status_code == 429:
        return ProviderErrorMapping(status.HTTP_429_TOO_MANY_REQUESTS, PROVIDER_RATE_LIMITED_DETAIL)
    return None


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalized_origin(parsed: ParseResult) -> tuple[str, str | None, int | None]:
    """(scheme, host, port) with the scheme's default port filled in.

    So ``https://h`` and ``https://h:443`` compare equal (and ``http`` / ``:80``),
    rather than failing on ``None != 443`` and silently not forwarding the token.
    """
    port = parsed.port if parsed.port is not None else _DEFAULT_PORTS.get(parsed.scheme)
    return (parsed.scheme, parsed.hostname, port)


def web_search_url_targets_platform(web_search_url: str, platform_base_url: str | None) -> bool:
    """True when ``web_search_url`` is the platform itself (same origin, under its base path).

    Gates forwarding the platform token to the web-search backend: it is only
    safe to hand that high-privilege credential to the platform — the host the
    gateway already trusts it with for resolve. A raw string prefix check is not
    enough: with a path-less ``PLATFORM_BASE_URL`` (e.g. ``https://api.otari.ai``)
    a confusable URL like ``https://api.otari.ai.evil.com`` or
    ``https://api.otari.ai@evil.com`` would satisfy ``startswith`` and leak the
    token. So compare the parsed (scheme, host, port) origin exactly — with
    default ports normalized — and require the search path to sit under the base
    path at a ``/`` boundary.
    """
    if not platform_base_url:
        return False
    base = urlparse(platform_base_url)
    target = urlparse(web_search_url)
    if _normalized_origin(target) != _normalized_origin(base):
        return False
    base_path = base.path.rstrip("/")
    return target.path == base_path or target.path.startswith(base_path + "/")


def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset)),
    }


class FormatAdapter(Protocol, Generic[ResultT, ChunkT]):
    """Per-format edges of the shared pipeline.

    One instance per wire format (chat / messages / responses) lives in the
    corresponding route module. Methods must resolve provider-call and
    tool-loop functions as module globals of the route module at call time so
    tests can monkeypatch them there.
    """

    name: str
    endpoint: str
    stream_format: StreamFormat

    def error(self, status_code: int, message: str, kind: ErrorKind = ErrorKind.API) -> HTTPException:
        """Build the format's wire error for ``status_code`` / ``message``."""
        ...

    def provider_error(self, exc: BaseException) -> HTTPException:
        """Map a single-attempt upstream failure to the format's wire error."""
        ...

    def format_chunk(self, chunk: ChunkT) -> str: ...

    def extract_stream_usage(self, chunk: ChunkT) -> CompletionUsage | None: ...

    def extract_usage(self, result: ResultT) -> CompletionUsage | None: ...

    # When True (chat, responses) a successful non-streaming call without
    # provider usage data still writes a usage-log row; messages skips the row.
    log_success_without_usage: bool

    async def call_provider(self, kwargs: dict[str, Any]) -> ResultT: ...

    async def open_provider_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ChunkT]: ...

    def prepare_stream_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Normalize per-call kwargs for a streaming dispatch (e.g. force
        ``stream=True`` or inject ``stream_options``)."""
        ...

    async def run_tool_loop(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        on_first_response: Callable[[], None] | None = None,
    ) -> ResultT: ...

    def open_tool_loop_stream(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
    ) -> AsyncIterator[ChunkT]: ...

    def inject_hints(
        self,
        kwargs: dict[str, Any],
        hints: list[tuple[str, str]],
        *,
        header: str | None,
    ) -> dict[str, Any]:
        """Prepend tool purpose hints to the format's system/instructions slot."""
        ...

    def attempt_kwargs(
        self,
        attempt: ResolvedAttempt,
        base_request_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge platform-attempt credentials and model into call kwargs."""
        ...

    def prepare_platform_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adjust the ``run_platform_attempts``-shaped kwargs for the format's
        provider call (the responses format re-splits ``provider:model``)."""
        ...


def default_attempt_kwargs(
    attempt: ResolvedAttempt,
    base_request_fields: dict[str, Any],
) -> dict[str, Any]:
    """Standard platform-attempt kwargs: credentials + ``provider:model`` selector."""
    attempt_provider = LLMProvider(attempt.provider)
    kwargs: dict[str, Any] = {"api_key": attempt.api_key}
    if attempt.api_base:
        kwargs["api_base"] = attempt.api_base
    return {
        **kwargs,
        **base_request_fields,
        "model": f"{attempt_provider.value}:{attempt.model}",
    }


# ---------------------------------------------------------------------------
# Request context (auth, budget reservation, platform route)
# ---------------------------------------------------------------------------


class RequestContext:
    """Everything the preamble resolved for one request."""

    def __init__(
        self,
        *,
        config: GatewayConfig,
        db: AsyncSession | None,
        log_writer: LogWriter,
        hybrid_mode: bool,
        route: ResolvedRoute | None,
        user_token: str | None,
        api_key_id: str | None,
        user_id: str | None,
        rate_limit_info: RateLimitInfo | None,
        reservation: ReservationHandle | None,
    ) -> None:
        self.config = config
        self.db = db
        self.log_writer = log_writer
        self.hybrid_mode = hybrid_mode
        self.route = route
        self.user_token = user_token
        self.api_key_id = api_key_id
        self.user_id = user_id
        self.rate_limit_info = rate_limit_info
        self.reservation = reservation


async def _bill_vision_side_call(
    *,
    db: AsyncSession,
    log_writer: LogWriter,
    config: GatewayConfig,
    api_key_id: str | None,
    user_id: str,
    endpoint: str,
    usage: CompletionUsage,
) -> None:
    """Meter and bill a vision describe side-call made during normalization.

    The describe model already ran (to caption an image for a text-only target
    model), so its cost is recorded as its own usage-log row for the configured
    vision model and committed directly to ``users.spend``. It is intentionally
    not gated or refundable: the cost is already incurred, so a budget reject
    here would lose it, and refunding the main request must not erase it.
    No-op when no vision model is configured or its selector can't be parsed.
    """
    model_selector = config.vision_describe_model
    if not model_selector:
        return
    try:
        resolved = resolve_provider_selector(config, model_selector)
    except (ValueError, AnyLLMError):
        logger.warning("vision billing: cannot parse vision_describe_model %r", model_selector)
        return
    # Key the side-call's usage/pricing on the instance, matching how the main
    # request is billed (the vision call itself routes via the same resolver).
    cost = await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=resolved.model,
        provider=resolved.instance,
        endpoint=endpoint,
        user_id=user_id,
        usage_override=usage,
    )
    # Commit the spend directly via an unreserved handle (no held estimate to
    # release): this just adds the actual cost to users.spend.
    await reconcile_reservation(
        db,
        ReservationHandle(user_id=user_id, estimate=0.0, reserved=False, strategy=config.budget_strategy),
        cost or 0.0,
    )


async def resolve_request_context(
    *,
    adapter: FormatAdapter[Any, Any],
    raw_request: Request,
    response: Response,
    db: AsyncSession | None,
    config: GatewayConfig,
    log_writer: LogWriter,
    model: str,
    user_id_from_request: str | None,
    estimate_prompt_chars: int,
    estimate_max_output_tokens: int | None,
    master_key_user_required_detail: str,
    user_forbidden_detail: str,
    normalize_messages: Callable[
        [str, LLMProvider | None, str, str | None], Awaitable[tuple[int, CompletionUsage | None]]
    ]
    | None = None,
) -> RequestContext:
    """Run the shared handler preamble up to (and including) budget pre-debit.

    Hybrid mode: extract the caller's bearer token and resolve the routing
    plan against the platform; no local DB state is touched.

    Standalone mode: validate the API key, resolve the billed user, check the
    rate limit, then reserve the estimated cost. The reservation is taken
    before the missing-pricing gate so user/blocked/budget rejections
    (404/403) take precedence over the 402; it is refunded if the request is
    then rejected for missing pricing.

    ``normalize_messages`` (standalone only) is an optional hook the file
    feature uses to resolve uploaded attachments into the wire payload before
    the cost estimate. It runs after the billed user is known (file access is
    user-scoped) and after the provider/model split (capability detection needs
    it), and returns the post-normalization prompt-char count so the reservation
    reflects any text extracted from attachments. It is never called in hybrid
    mode, where the files feature is unavailable. It returns
    ``(prompt_chars, vision_usage)``; any vision describe side-call it made is
    metered and billed here as committed spend (the call already happened, so it
    is not gated or refundable).
    """
    hybrid_mode = config.is_hybrid_mode
    route: ResolvedRoute | None = None
    user_token: str | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    reservation: ReservationHandle | None = None

    if hybrid_mode:
        user_token = _extract_platform_user_token(raw_request)
        start_time = time.perf_counter()
        route = await _resolve_platform_credentials(
            config=config,
            user_token=user_token,
            model_selector=model,
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
            raise adapter.error(500, DB_UNAVAILABLE_DETAIL, ErrorKind.API)
        api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
        api_key_id = api_key.id if api_key else None
        user_id = resolve_user_id(
            user_id_from_request=user_id_from_request,
            api_key=api_key,
            is_master_key=is_master_key,
            master_key_error=adapter.error(400, master_key_user_required_detail, ErrorKind.INVALID_REQUEST),
            no_api_key_error=adapter.error(500, API_KEY_VALIDATION_FAILED_DETAIL, ErrorKind.API),
            no_user_error=adapter.error(500, API_KEY_NO_USER_DETAIL, ErrorKind.API),
            forbidden_user_error=adapter.error(403, user_forbidden_detail, ErrorKind.PERMISSION),
            reject_mismatch=config.reject_user_mismatch,
        )
        rate_limit_info = check_rate_limit(raw_request, user_id)

        # Tolerate an unparseable / unknown-provider selector here: the budget
        # check below and the downstream provider call surface those with
        # their own status codes. A model we can't parse simply has no pricing.
        # Pricing/budget keys on the *instance* name (``instance:model``) while
        # capability detection needs the underlying implementation, so keep both.
        gate_instance: str | None
        gate_impl: LLMProvider | None
        try:
            resolved = resolve_provider_selector(config, model)
            gate_instance, gate_impl, gate_model = resolved.instance, resolved.provider, resolved.model
        except (ValueError, AnyLLMError):
            gate_instance, gate_impl, gate_model = None, None, model

        gate_pricing = await find_model_pricing(db, gate_instance, gate_model)
        estimate = estimate_cost(
            gate_pricing,
            prompt_chars=estimate_prompt_chars,
            max_output_tokens=estimate_max_output_tokens,
            default_output_tokens=config.budget_estimate_default_output_tokens,
        )
        # Reserve first so user/blocked/budget rejections (404/403) take
        # precedence over the missing-pricing rejection (402); refund if we
        # then reject for missing pricing.
        reservation = await reserve_budget(db, user_id, estimate, model=model, strategy=config.budget_strategy)
        if pricing_required_but_missing(gate_pricing, require_pricing=config.require_pricing):
            await refund_reservation(db, reservation)
            raise adapter.error(
                402,
                f"No pricing configured for model '{model}'",
                ErrorKind.INVALID_REQUEST,
            )

        # Resolve uploaded attachments only once the request is authorized
        # (user exists, not blocked, within budget, pricing OK). Done after the
        # budget gate so a blocked/over-budget user can't trigger extraction or
        # vision side-calls. Attachments may expand the prompt (extracted
        # document text, image captions), so top up the reservation to the
        # post-normalization size; the top-up rejects if it no longer fits.
        # Refund on any failure in this setup phase, which the downstream
        # provider-call settlement does not cover.
        if normalize_messages is not None:
            try:
                post_chars, vision_usage = await normalize_messages(
                    user_id, gate_impl, gate_model, gate_instance
                )
                post_estimate = estimate_cost(
                    gate_pricing,
                    prompt_chars=post_chars,
                    max_output_tokens=estimate_max_output_tokens,
                    default_output_tokens=config.budget_estimate_default_output_tokens,
                )
                await increase_reservation(
                    db,
                    reservation,
                    post_estimate - estimate,
                    model=model,
                    strategy=config.budget_strategy,
                )
                if vision_usage is not None:
                    await _bill_vision_side_call(
                        db=db,
                        log_writer=log_writer,
                        config=config,
                        api_key_id=api_key_id,
                        user_id=user_id,
                        endpoint=adapter.endpoint,
                        usage=vision_usage,
                    )
            except HTTPException:
                await refund_reservation(db, reservation)
                raise
            except Exception as exc:
                await refund_reservation(db, reservation)
                logger.error("Request setup failed after reservation: %s", exc)
                raise adapter.error(
                    500,
                    "Failed to process request attachments",
                    ErrorKind.API,
                ) from exc

    return RequestContext(
        config=config,
        db=db,
        log_writer=log_writer,
        hybrid_mode=hybrid_mode,
        route=route,
        user_token=user_token,
        api_key_id=api_key_id,
        user_id=user_id,
        rate_limit_info=rate_limit_info,
        reservation=reservation,
    )


# ---------------------------------------------------------------------------
# Gateway-managed tools (guardrails, MCP, sandbox, web_search)
# ---------------------------------------------------------------------------


class ToolContext:
    """Resolved gateway-tool configuration for one request."""

    def __init__(
        self,
        *,
        mcp_server_configs: list[McpServerConfig] | None,
        use_sandbox: bool,
        sandbox_tool_entry: dict[str, Any] | None,
        sandbox_url: str | None,
        sandbox_auth_token: str | None,
        use_web_search: bool,
        web_search_tool_entry: dict[str, Any] | None,
        web_search_url: str | None,
        web_search_auth_token: str | None,
        remaining_user_tools: list[dict[str, Any]] | None,
        max_tool_iterations: int,
        tools_header: str | None,
    ) -> None:
        self.mcp_server_configs = mcp_server_configs
        self.use_sandbox = use_sandbox
        self.sandbox_tool_entry = sandbox_tool_entry
        self.sandbox_url = sandbox_url
        self.sandbox_auth_token = sandbox_auth_token
        self.use_web_search = use_web_search
        self.web_search_tool_entry = web_search_tool_entry
        self.web_search_url = web_search_url
        self.web_search_auth_token = web_search_auth_token
        self.remaining_user_tools = remaining_user_tools
        self.max_tool_iterations = max_tool_iterations
        self.tools_header = tools_header

    @property
    def tools_extracted(self) -> bool:
        return self.sandbox_tool_entry is not None or self.web_search_tool_entry is not None

    @property
    def use_tool_loop(self) -> bool:
        return bool(self.mcp_server_configs) or self.use_sandbox or self.use_web_search


async def prepare_gateway_tools(
    *,
    adapter: FormatAdapter[Any, Any],
    ctx: RequestContext,
    response: Response,
    guardrails: list[GuardrailConfig] | None,
    guardrail_text: str,
    tools: list[dict[str, Any]] | None,
    mcp_servers: list[McpServerConfig] | None,
    mcp_server_ids: list[uuid.UUID] | None,
    max_tool_iterations: int | None,
    tools_header: str | None,
) -> ToolContext:
    """Guardrails, MCP server-id resolution, and gateway-tool extraction.

    Caller-requested input guardrails run before any provider/tool dispatch.
    ``block``-mode flags raise 403 here (provider never called);
    ``monitor``-mode flags annotate the response header and fall through.

    ``mcp_server_ids`` is hybrid-only: standalone mode has no platform to
    resolve the ids against, so the field is rejected with a 400 rather than
    silently ignored. The sandbox and web_search opt-ins follow the wire shape
    of Anthropic / OpenAI tool entries; their backend URLs are operator
    controlled (no per-request URL override, which would be an SSRF surface).
    The three backends are mutually exclusive for now.

    Any rejection raised here (guardrail block, unresolvable MCP ids,
    misconfigured or conflicting tool opt-ins) releases the budget
    reservation taken by :func:`resolve_request_context` before propagating.
    """
    try:
        await apply_input_guardrails(guardrails, guardrail_text, response=response)

        if mcp_server_ids and not ctx.hybrid_mode:
            raise adapter.error(400, MCP_SERVER_IDS_HYBRID_ONLY_DETAIL, ErrorKind.INVALID_REQUEST)
        if ctx.hybrid_mode and mcp_server_ids:
            assert ctx.user_token is not None  # guaranteed by the hybrid-mode preamble
            resolved_mcp_servers = await _resolve_platform_mcp_servers(
                config=ctx.config,
                user_token=ctx.user_token,
                mcp_server_ids=mcp_server_ids,
            )
            mcp_servers = (mcp_servers or []) + resolved_mcp_servers

        sandbox_tool_entry, tools_after_sandbox = _extract_code_execution_tool(tools)
        sandbox_url: str | None = otari_env("SANDBOX_URL") or None
        use_sandbox = False
        if sandbox_tool_entry is not None:
            if sandbox_url is None:
                raise adapter.error(400, SANDBOX_NOT_CONFIGURED_DETAIL, ErrorKind.INVALID_REQUEST)
            if mcp_servers:
                raise adapter.error(400, SANDBOX_MCP_CONFLICT_DETAIL, ErrorKind.INVALID_REQUEST)
            use_sandbox = True

        # Forwarded to the sandbox backend as `Authorization: Bearer`. Only set in
        # hybrid mode when the backend IS the platform (its URL is under the
        # platform base URL the gateway already trusts this token with for resolve):
        # the platform-hosted /v1/sandbox proxy authenticates the caller's workspace
        # token and derives tenancy + per-workspace code-exec policy from it. Never
        # leak it to a standalone exec-service an operator pointed the URL at.
        sandbox_auth_token: str | None = None
        if use_sandbox and ctx.hybrid_mode and sandbox_url is not None:
            assert ctx.user_token is not None  # guaranteed by the hybrid-mode preamble
            if web_search_url_targets_platform(sandbox_url, ctx.config.platform.get("base_url")):
                sandbox_auth_token = ctx.user_token

        web_search_tool_entry, remaining_user_tools = _extract_web_search_tool(tools_after_sandbox)
        web_search_url: str | None = otari_env("WEB_SEARCH_URL") or None
        # Forwarded to the search backend as `X-Gateway-Token`. Only set in
        # hybrid mode, where the backend may be the platform-hosted web-search
        # endpoint that authenticates the gateway. Standalone backends (SearXNG /
        # self-hosted adapter) get no token and ignore the header.
        web_search_auth_token: str | None = None
        use_web_search = False
        if web_search_tool_entry is not None:
            if web_search_url is None:
                raise adapter.error(400, WEB_SEARCH_NOT_CONFIGURED_DETAIL, ErrorKind.INVALID_REQUEST)
            if use_sandbox or mcp_servers:
                raise adapter.error(400, WEB_SEARCH_CONFLICT_DETAIL, ErrorKind.INVALID_REQUEST)
            use_web_search = True

            # Hybrid mode owns the per-workspace web-search policy (whether it's
            # enabled at all, plus workspace-default max_results / domain filters /
            # purpose hint / provider_options). Mirrors the mcp_server_ids resolve
            # above. Precedence is "per-request overrides workspace default":
            #  * top-level keys are applied only when the request didn't supply a
            #    meaningful (truthy) value of its own. An empty list / empty string
            #    reads as "no preference" and falls back to the workspace value
            #    rather than silently clearing the workspace's policy (e.g. a
            #    request `allowed_domains: []` must NOT wipe a workspace allow-list);
            #  * provider_options is shallow-merged so workspace defaults fill the
            #    keys the request omitted while per-request keys still win (rather
            #    than the request's dict replacing the workspace dict wholesale).
            # Standalone mode has no platform to consult.
            if ctx.hybrid_mode:
                assert ctx.user_token is not None  # guaranteed by the hybrid-mode preamble
                # Forward the platform token only when the search backend IS the
                # platform (its URL is under the platform base URL the gateway
                # already trusts this token with for resolve). Never leak this
                # high-privilege credential to a bundled SearXNG or a third-party
                # adapter that an operator happened to point GATEWAY_WEB_SEARCH_URL at.
                if web_search_url_targets_platform(web_search_url, ctx.config.platform.get("base_url")):
                    web_search_auth_token = ctx.config.platform_token
                web_search_policy = await _resolve_platform_web_search(
                    config=ctx.config,
                    user_token=ctx.user_token,
                )
                if not web_search_policy.get("enabled"):
                    raise adapter.error(403, WEB_SEARCH_NOT_ENABLED_DETAIL, ErrorKind.PERMISSION)
                for key in ("max_results", "allowed_domains", "blocked_domains", "purpose_hint"):
                    resolved_value = web_search_policy.get(key)
                    if not web_search_tool_entry.get(key) and resolved_value is not None:
                        web_search_tool_entry[key] = resolved_value
                workspace_options = web_search_policy.get("provider_options")
                if isinstance(workspace_options, dict):
                    request_options = web_search_tool_entry.get("provider_options")
                    web_search_tool_entry["provider_options"] = (
                        {**workspace_options, **request_options}
                        if isinstance(request_options, dict)
                        else workspace_options
                    )
    except HTTPException:
        await release_reservation(ctx)
        raise

    return ToolContext(
        mcp_server_configs=mcp_servers,
        use_sandbox=use_sandbox,
        sandbox_tool_entry=sandbox_tool_entry,
        sandbox_url=sandbox_url,
        sandbox_auth_token=sandbox_auth_token,
        use_web_search=use_web_search,
        web_search_tool_entry=web_search_tool_entry,
        web_search_url=web_search_url,
        web_search_auth_token=web_search_auth_token,
        remaining_user_tools=remaining_user_tools,
        max_tool_iterations=min(
            max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
            MAX_TOOL_ITERATIONS_CAP,
        ),
        tools_header=tools_header,
    )


# ---------------------------------------------------------------------------
# Usage logging and reservation settlement
# ---------------------------------------------------------------------------


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

    Spend is not written here; the budget reservation reconcile path owns
    ``users.spend``. This returns the cost it computed so the caller can
    reconcile the reservation with the actual amount.

    Args:
        db: Database session
        log_writer: Queueing usage-log writer
        api_key_id: API key identifier (None if using master key)
        model: Model name
        provider: Provider name
        endpoint: Endpoint path
        user_id: User identifier for tracking
        response: Response object (if successful)
        usage_override: Usage data for streaming requests
        error: Error message (if failed)
        cost_override: Fixed amount to record when billing without provider usage

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
        usage_log.cache_read_tokens = cache_read_tokens_of(usage_data)
        usage_log.cache_write_tokens = cache_write_tokens_of(usage_data)

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
            logger.warning("No pricing configured for '%s'. Usage will be tracked without cost.", model_ref)

    # When the caller bills a fixed amount without provider usage (e.g. the
    # stream-missing-usage estimate policy), record that amount on the log row
    # so usage_logs.cost stays consistent with the spend that was reconciled.
    if cost_override is not None:
        usage_log.cost = cost_override

    await log_writer.put(usage_log)
    return usage_log.cost


async def release_reservation(ctx: RequestContext) -> None:
    """Refund the request's budget reservation, if one was taken.

    No-op in hybrid mode and for requests that reserved nothing. Use this
    before raising on any path that rejects the request after
    :func:`resolve_request_context` pre-debited the estimate; otherwise the
    held amount shrinks the user's budget until the next reset (or forever,
    for budgets without a reset period).
    """
    if ctx.db is not None and ctx.reservation is not None:
        await refund_reservation(ctx.db, ctx.reservation)


async def _log_failure_and_refund(
    ctx: RequestContext,
    adapter: FormatAdapter[Any, Any],
    provider: Any,
    model: str,
    error: str,
) -> None:
    if ctx.db is None:
        return
    await log_usage(
        db=ctx.db,
        log_writer=ctx.log_writer,
        api_key_id=ctx.api_key_id,
        model=model,
        provider=provider,
        endpoint=adapter.endpoint,
        user_id=ctx.user_id,
        error=error,
    )
    if ctx.reservation is not None:
        await refund_reservation(ctx.db, ctx.reservation)


# ---------------------------------------------------------------------------
# Backend dispatch (the single copy of the mcp / sandbox / web_search ladder)
# ---------------------------------------------------------------------------


async def dispatch_non_stream(
    *,
    adapter: FormatAdapter[ResultT, Any],
    tool_ctx: ToolContext,
    call_kwargs: dict[str, Any],
    on_first_response: Callable[[], None] | None = None,
) -> ResultT:
    """Non-streaming dispatch: plain provider call, or the matching tool-loop
    backend (MCP pool / sandbox / web_search) opened for the duration of the
    loop.
    """
    if not tool_ctx.use_tool_loop:
        return await adapter.call_provider(call_kwargs)

    if tool_ctx.mcp_server_configs:
        async with MCPClientPool(tool_ctx.mcp_server_configs) as pool:
            kwargs = adapter.inject_hints(call_kwargs, pool.purpose_hints(), header=tool_ctx.tools_header)
            return await adapter.run_tool_loop(kwargs, pool, tool_ctx.max_tool_iterations, on_first_response)

    if tool_ctx.use_sandbox:
        assert tool_ctx.sandbox_url is not None  # guaranteed past the missing-URL 400 in prepare_gateway_tools
        sandbox_hint = _resolve_sandbox_purpose_hint(tool_ctx.sandbox_tool_entry)
        async with SandboxBackend(
            sandbox_url=tool_ctx.sandbox_url, purpose_hint=sandbox_hint, auth_token=tool_ctx.sandbox_auth_token
        ) as backend:
            kwargs = adapter.inject_hints(call_kwargs, backend.purpose_hints(), header=tool_ctx.tools_header)
            return await adapter.run_tool_loop(kwargs, backend, tool_ctx.max_tool_iterations, on_first_response)

    assert tool_ctx.use_web_search
    assert tool_ctx.web_search_url is not None  # guaranteed past the missing-URL 400 in prepare_gateway_tools
    assert tool_ctx.web_search_tool_entry is not None  # guaranteed by the web_search opt-in
    async with _build_web_search_backend(
        base_url=tool_ctx.web_search_url,
        tool_entry=tool_ctx.web_search_tool_entry,
        auth_token=tool_ctx.web_search_auth_token,
    ) as web_backend:
        kwargs = adapter.inject_hints(call_kwargs, web_backend.purpose_hints(), header=tool_ctx.tools_header)
        return await adapter.run_tool_loop(kwargs, web_backend, tool_ctx.max_tool_iterations, on_first_response)


async def _lazy_mcp_stream(
    adapter: FormatAdapter[Any, ChunkT],
    kwargs: dict[str, Any],
    configs: list[McpServerConfig],
    tool_ctx: ToolContext,
) -> AsyncIterator[ChunkT]:
    # The MCP pool is entered lazily inside the generator: a dial failure
    # surfaces once the client starts pulling events. Sandbox / web_search use
    # the eager-open path below for a pre-200 HTTP error instead.
    async with MCPClientPool(configs) as pool:
        hinted = adapter.inject_hints(kwargs, pool.purpose_hints(), header=tool_ctx.tools_header)
        async for event in adapter.open_tool_loop_stream(hinted, pool, tool_ctx.max_tool_iterations):
            yield event


async def _eager_backend_stream(
    adapter: FormatAdapter[Any, ChunkT],
    kwargs: dict[str, Any],
    backend: Any,
    tool_ctx: ToolContext,
) -> AsyncIterator[ChunkT]:
    # ``backend.__aenter__`` already ran in ``open_stream``; this generator
    # owns the matching ``__aexit__`` once the stream finishes or errors.
    try:
        hinted = adapter.inject_hints(kwargs, backend.purpose_hints(), header=tool_ctx.tools_header)
        async for event in adapter.open_tool_loop_stream(hinted, backend, tool_ctx.max_tool_iterations):
            yield event
    finally:
        await backend.__aexit__(None, None, None)


async def open_stream(
    *,
    adapter: FormatAdapter[Any, ChunkT],
    tool_ctx: ToolContext,
    call_kwargs: dict[str, Any],
) -> AsyncIterator[ChunkT]:
    """Open the upstream stream for a single-attempt streaming request.

    The sandbox and web_search backends are opened eagerly (their
    ``__aenter__`` runs before this function returns) so a backend-unreachable
    error surfaces as an HTTP 502 rather than landing in the SSE channel after
    the response has already committed to 200 OK. The MCP pool is entered
    lazily inside the returned iterator.
    """
    kwargs = adapter.prepare_stream_kwargs(call_kwargs)

    if not tool_ctx.use_tool_loop:
        return await adapter.open_provider_stream(kwargs)

    if tool_ctx.mcp_server_configs:
        return _lazy_mcp_stream(adapter, kwargs, tool_ctx.mcp_server_configs, tool_ctx)

    if tool_ctx.use_sandbox:
        assert tool_ctx.sandbox_url is not None  # guaranteed past the missing-URL 400 in prepare_gateway_tools
        sandbox_hint = _resolve_sandbox_purpose_hint(tool_ctx.sandbox_tool_entry)
        sandbox_backend = SandboxBackend(
            sandbox_url=tool_ctx.sandbox_url, purpose_hint=sandbox_hint, auth_token=tool_ctx.sandbox_auth_token
        )
        await sandbox_backend.__aenter__()  # may raise SandboxNotReachableError
        return _eager_backend_stream(adapter, kwargs, sandbox_backend, tool_ctx)

    assert tool_ctx.use_web_search
    assert tool_ctx.web_search_url is not None  # guaranteed past the missing-URL 400 in prepare_gateway_tools
    assert tool_ctx.web_search_tool_entry is not None  # guaranteed by the web_search opt-in
    web_search_backend = _build_web_search_backend(
        base_url=tool_ctx.web_search_url,
        tool_entry=tool_ctx.web_search_tool_entry,
        auth_token=tool_ctx.web_search_auth_token,
    )
    await web_search_backend.__aenter__()  # may raise WebSearchNotReachableError
    return _eager_backend_stream(adapter, kwargs, web_search_backend, tool_ctx)


# ---------------------------------------------------------------------------
# Streaming settlement (the single copy of the callback bundle)
# ---------------------------------------------------------------------------


def build_streaming_response(
    *,
    adapter: FormatAdapter[Any, ChunkT],
    stream: AsyncIterator[ChunkT],
    provider: Any,
    model: str,
    config: GatewayConfig,
    db: AsyncSession | None,
    log_writer: LogWriter | None,
    api_key_id: str | None,
    user_id: str | None,
    rate_limit_info: RateLimitInfo | None,
    reservation: ReservationHandle | None,
    platform_correlation_id: str | None = None,
    platform_request_id: str | None = None,
) -> StreamingResponse:
    """Wrap an already-opened upstream stream in an SSE response.

    This is the only place the streaming settlement callbacks are built, so
    every format and both the single-attempt and platform-fallback paths get
    identical reservation handling:

    * ``on_complete``: report usage upstream (platform) or write the usage log
      and reconcile the reservation against actual cost (standalone).
    * ``on_no_usage``: stream finished without usage data; settle per
      ``stream_missing_usage_policy`` instead of silently billing $0.
    * ``on_error``: report/log the failure and refund the reservation.
    * ``on_incomplete``: client disconnected mid-stream; refund so the
      reservation does not leak.
    """
    platform_active = platform_correlation_id is not None

    async def _on_complete(usage_data: CompletionUsage) -> None:
        if platform_active:
            assert platform_correlation_id is not None
            asyncio.create_task(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
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
            endpoint=adapter.endpoint,
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
                endpoint=adapter.endpoint,
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
            endpoint=adapter.endpoint,
            user_id=user_id,
            error="stream completed without usage data" if policy == "fail" else None,
            cost_override=reservation.estimate,
        )
        await reconcile_reservation(db, reservation, reservation.estimate)

    async def _on_error(error: str) -> None:
        if platform_active:
            assert platform_correlation_id is not None
            asyncio.create_task(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
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
            endpoint=adapter.endpoint,
            user_id=user_id,
            error=error,
        )
        if reservation is not None:
            await refund_reservation(db, reservation)

    async def _on_incomplete() -> None:
        # Client disconnected mid-stream: release the reservation.
        if db is None or reservation is None:
            return
        await refund_reservation(db, reservation)

    # StreamingResponse builds its own response object, so headers we want on
    # the wire have to be passed in here; assigning to the dependency-injected
    # ``Response`` object does not propagate to streaming responses.
    headers: dict[str, str] = dict(rate_limit_headers(rate_limit_info)) if rate_limit_info else {}
    if platform_correlation_id:
        headers["X-Correlation-ID"] = platform_correlation_id
    if platform_request_id:
        headers["X-Otari-Request-ID"] = platform_request_id

    return StreamingResponse(
        streaming_generator(
            stream=stream,
            format_chunk=adapter.format_chunk,
            extract_usage=adapter.extract_stream_usage,
            fmt=adapter.stream_format,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
            on_no_usage=_on_no_usage,
            on_incomplete=_on_incomplete,
        ),
        media_type="text/event-stream",
        headers=headers,
    )


def stream_first_chunk_timeout_seconds(config: GatewayConfig, *, tool_mode: bool) -> float:
    """First-chunk timeout for platform-fallback streaming, shared by all formats.

    Tool-mode streams get more headroom: the model may reason briefly before
    emitting tokens or a tool_call, especially with extended thinking. Plain
    streams keep a tight default so failed-attempt latency stays low.
    """
    if tool_mode:
        return (
            int(
                config.platform.get(
                    _STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY,
                    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP,
                )
            )
            / 1000
        )
    return (
        int(
            config.platform.get(
                _STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY,
                _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS,
            )
        )
        / 1000
    )


# ---------------------------------------------------------------------------
# Shared request runners
# ---------------------------------------------------------------------------


async def run_single_attempt_stream(
    *,
    adapter: FormatAdapter[Any, ChunkT],
    ctx: RequestContext,
    tool_ctx: ToolContext,
    call_kwargs: dict[str, Any],
    provider: Any,
    model: str,
    platform_correlation_id: str | None = None,
    platform_request_id: str | None = None,
) -> StreamingResponse:
    """Open a single-attempt stream and wrap it with settlement callbacks.

    Pre-stream failures settle here: gateway-side backend failures map to a
    502 with a backend-specific detail (clearer than a fake provider outage),
    provider failures go through the adapter's error mapping, and in both
    cases any budget reservation is refunded before the error surfaces.
    """
    try:
        stream = await open_stream(adapter=adapter, tool_ctx=tool_ctx, call_kwargs=call_kwargs)
    except HTTPException:
        await release_reservation(ctx)
        raise
    except SandboxNotReachableError as exc:
        # The sandbox is part of the gateway's own infra, not the LLM
        # provider; a distinct status stops operators chasing a "provider
        # outage" that is actually the sandbox container being down. 502
        # keeps "upstream dependency failed" semantics.
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
        await release_reservation(ctx)
        raise adapter.error(502, SANDBOX_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        await release_reservation(ctx)
        raise adapter.error(502, WEB_SEARCH_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    except Exception as exc:
        await _log_failure_and_refund(ctx, adapter, provider, model, str(exc))
        logger.error("Stream creation failed for %s:%s: %s", provider, model, exc)
        raise adapter.provider_error(exc) from exc

    return build_streaming_response(
        adapter=adapter,
        stream=stream,
        provider=provider,
        model=model,
        config=ctx.config,
        db=ctx.db,
        log_writer=ctx.log_writer,
        api_key_id=ctx.api_key_id,
        user_id=ctx.user_id,
        rate_limit_info=ctx.rate_limit_info,
        reservation=ctx.reservation,
        platform_correlation_id=platform_correlation_id,
        platform_request_id=platform_request_id,
    )


async def _flush_pending_usage_reports(
    config: GatewayConfig,
    pending_error_reports: list[tuple[str, str, Any, str | None]],
    request_id: str,
) -> None:
    """Send the per-attempt error reports inline on the all-failed path.

    FastAPI BackgroundTasks are dropped when the request ends in an error
    response, so on a fully-exhausted fallback chain these reports must be
    flushed before the terminal 502/504 (the queued background copies never
    run, so there is no double-report).

    The flush is bounded: this is best-effort telemetry and must not materially
    delay the already-failing response. Reports run concurrently, and the whole
    batch is capped at ``usage_timeout_ms`` so a degraded usage endpoint is cut
    off rather than stacking each report's full retry/backoff budget onto the
    response. Callers on the streaming path skip this entirely on cancellation.
    """
    if not pending_error_reports:
        return

    timeout_s = int(config.platform.get("usage_timeout_ms", 5000)) / 1000
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    _report_platform_usage(config, attempt_id, outcome, usage, error_class)
                    for attempt_id, outcome, usage, error_class in pending_error_reports
                ),
                return_exceptions=True,
            ),
            timeout=timeout_s,
        )
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "Inline usage-report flush timed out after %.1fs on the all-failed path request_id=%s",
            timeout_s,
            request_id,
        )
        return

    for result in results:
        if isinstance(result, BaseException):
            logger.warning(
                "Inline usage report failed on the all-failed path request_id=%s: %s",
                request_id,
                result,
            )


async def run_streaming_with_fallback(
    *,
    adapter: FormatAdapter[Any, ChunkT],
    route: ResolvedRoute,
    base_request_fields: dict[str, Any],
    config: GatewayConfig,
    background_tasks: BackgroundTasks,
    rate_limit_info: RateLimitInfo | None,
    tool_ctx: ToolContext,
) -> StreamingResponse:
    """Multi-attempt streaming for hybrid-mode requests.

    Iterates ``route.attempts`` and falls through on any attempt that fails
    before its first chunk arrives. Once an attempt yields its first chunk,
    the request locks in and starts flushing to the client; errors past that
    point land in the SSE channel. Mid-stream failover is out of scope:
    recovering would require silently buffering the prefix (delays first byte)
    or a client-aware "restart" event (breaks SDK compatibility).

    Tool-loop modes are layered on top with the same pre-first-chunk fallback
    semantics; the tool backend (including the MCP pool) is opened eagerly
    once on an ``AsyncExitStack`` shared across attempts, so gateway-side
    dependency failures surface as a normal HTTP error and each retried
    attempt starts with a clean conversation slate.
    """
    tool_mode = tool_ctx.use_tool_loop
    first_chunk_timeout = stream_first_chunk_timeout_seconds(config, tool_mode=tool_mode)

    backend_stack = AsyncExitStack()
    pool_for_loop: Any = None
    try:
        if tool_ctx.mcp_server_configs:
            pool_for_loop = await backend_stack.enter_async_context(MCPClientPool(tool_ctx.mcp_server_configs))
        elif tool_ctx.use_sandbox:
            assert tool_ctx.sandbox_url is not None  # guaranteed past the missing-URL 400 in prepare_gateway_tools
            sandbox_hint = _resolve_sandbox_purpose_hint(tool_ctx.sandbox_tool_entry)
            pool_for_loop = await backend_stack.enter_async_context(
                SandboxBackend(
                    sandbox_url=tool_ctx.sandbox_url, purpose_hint=sandbox_hint, auth_token=tool_ctx.sandbox_auth_token
                ),
            )
        elif tool_ctx.use_web_search:
            assert tool_ctx.web_search_url is not None  # guaranteed past the missing-URL 400
            assert tool_ctx.web_search_tool_entry is not None  # guaranteed by the web_search opt-in
            pool_for_loop = await backend_stack.enter_async_context(
                _build_web_search_backend(
                    base_url=tool_ctx.web_search_url,
                    tool_entry=tool_ctx.web_search_tool_entry,
                    auth_token=tool_ctx.web_search_auth_token,
                ),
            )
    except BaseException:
        # Eager-open failure (e.g. SandboxNotReachableError): propagate so the
        # route handler maps it to the existing HTTP status. Nothing to clean
        # up on the stack yet because the entry failed.
        await backend_stack.aclose()
        raise

    async def _build_for_attempt(attempt: ResolvedAttempt) -> AsyncIterator[ChunkT]:
        completion_kwargs = adapter.prepare_stream_kwargs(
            adapter.attempt_kwargs(attempt, base_request_fields),
        )
        if pool_for_loop is None:
            return await adapter.open_provider_stream(completion_kwargs)
        kwargs = adapter.inject_hints(
            completion_kwargs,
            pool_for_loop.purpose_hints(),
            header=tool_ctx.tools_header,
        )
        return adapter.open_tool_loop_stream(kwargs, pool_for_loop, tool_ctx.max_tool_iterations)

    # See run_platform_non_stream: BackgroundTasks only run after a successful
    # response, so if every attempt fails before its first chunk the queued
    # reports are dropped with the terminal 502/504. Keep the background task
    # for the success path (it flushes once the SSE response completes), but
    # also stash the error reports so they can be flushed inline on the
    # all-failed path below.
    pending_error_reports: list[tuple[str, str, Any, str | None]] = []

    async def _on_attempt_failed(attempt: ResolvedAttempt, failure: StreamingAttemptFailure) -> None:
        background_tasks.add_task(
            _report_platform_usage,
            config,
            attempt.attempt_id,
            "error",
            None,
            failure.error_class,
        )
        pending_error_reports.append((attempt.attempt_id, "error", None, failure.error_class))
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
            first_chunk_timeout_seconds=first_chunk_timeout,
        )
    except BaseException as exc:
        # No attempt yielded a first chunk: the request ends in an error
        # response, which drops the queued BackgroundTasks, so flush the
        # per-attempt error reports inline to keep the platform's per-attempt
        # record. Skip the flush on cancellation (reporting I/O must not delay
        # teardown), and always close the tool backend before propagating, even
        # if the flush raises or is interrupted.
        try:
            if not isinstance(exc, asyncio.CancelledError):
                await _flush_pending_usage_reports(config, pending_error_reports, route.request_id)
        finally:
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

    stream_to_return: AsyncIterator[ChunkT] = stream
    if pool_for_loop is not None:
        stream_to_return = _stream_with_stack_cleanup(stream, backend_stack)

    return build_streaming_response(
        adapter=adapter,
        stream=stream_to_return,
        provider=LLMProvider(chosen.provider),
        model=chosen.model,
        config=config,
        db=None,  # hybrid mode does not use the local DB
        log_writer=None,  # unused when db is None
        api_key_id=None,
        user_id=None,
        rate_limit_info=rate_limit_info,
        reservation=None,
        platform_correlation_id=chosen.attempt_id,
        platform_request_id=route.request_id,
    )


async def _stream_with_stack_cleanup(
    stream: AsyncIterator[ChunkT],
    backend_stack: AsyncExitStack,
) -> AsyncIterator[ChunkT]:
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await backend_stack.aclose()


async def run_platform_non_stream(
    *,
    adapter: FormatAdapter[ResultT, Any],
    route: ResolvedRoute,
    base_request_fields: dict[str, Any],
    tool_ctx: ToolContext,
    response: Response,
    background_tasks: BackgroundTasks,
    config: GatewayConfig,
    rate_limit_info: RateLimitInfo | None,
) -> ResultT:
    """Drive the multi-attempt hybrid-mode non-streaming path via the shared
    ``run_platform_attempts`` runner, dispatching each attempt through the
    shared backend ladder.
    """
    attempts = route.attempts
    if not attempts:
        logger.error("Platform returned empty attempts list request_id=%s", route.request_id)
        raise adapter.error(502, NO_RESOLVABLE_PROVIDER_DETAIL, ErrorKind.API)

    async def _run_attempt(
        completion_kwargs: dict[str, Any],
        on_first_response: Callable[[], None],
    ) -> ResultT:
        call_kwargs = adapter.prepare_platform_call_kwargs(completion_kwargs)
        return await dispatch_non_stream(
            adapter=adapter,
            tool_ctx=tool_ctx,
            call_kwargs=call_kwargs,
            on_first_response=on_first_response,
        )

    # FastAPI BackgroundTasks only run after a *successful* response. When every
    # attempt fails the runner raises (502/504) and the queued usage reports are
    # silently dropped, so the platform never records the failed attempts and
    # can't fire its fallback-exhausted accounting. Keep the background task for
    # the success-response path (non-blocking), but also stash the error reports
    # so they can be flushed inline if the request ends in an exception.
    pending_error_reports: list[tuple[str, str, Any, str | None]] = []

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
        if outcome != "success":
            pending_error_reports.append((attempt.attempt_id, outcome, usage, error_class))

    def _on_attempt_success(attempt: ResolvedAttempt) -> None:
        response.headers["X-Correlation-ID"] = attempt.attempt_id
        if rate_limit_info:
            for key, value in rate_limit_headers(rate_limit_info).items():
                response.headers[key] = value

    try:
        return await run_platform_attempts(
            route=route,
            attempts=attempts,
            base_request_fields=base_request_fields,
            run_attempt=_run_attempt,
            extract_usage=adapter.extract_usage,
            classify_error=_classify_upstream_error,
            report_attempt_outcome=_report_attempt_outcome,
            on_success=_on_attempt_success,
            max_tool_iterations=tool_ctx.max_tool_iterations,
        )
    except HTTPException:
        # An error response drops the queued BackgroundTasks, so send the
        # per-attempt error reports inline before propagating. The background
        # copies never run on this path, so there is no double-report. This
        # branch only catches HTTPException (what the runner raises on the
        # all-failed path); a CancelledError propagates without doing reporting
        # I/O during teardown.
        await _flush_pending_usage_reports(config, pending_error_reports, route.request_id)
        raise


async def run_standalone_non_stream(
    *,
    adapter: FormatAdapter[ResultT, Any],
    ctx: RequestContext,
    tool_ctx: ToolContext,
    call_kwargs: dict[str, Any],
    provider: Any,
    model: str,
) -> ResultT:
    """Standalone-mode non-streaming dispatch with reservation settlement.

    Success writes the usage log (per the adapter's no-usage policy) and
    reconciles the reservation against actual cost; every failure path refunds
    the reservation before mapping the error to the format's wire envelope.
    """
    try:
        result = await dispatch_non_stream(adapter=adapter, tool_ctx=tool_ctx, call_kwargs=call_kwargs)
        if ctx.db is not None:
            usage_data = adapter.extract_usage(result)
            actual_cost: float | None = None
            if usage_data is not None or adapter.log_success_without_usage:
                actual_cost = await log_usage(
                    db=ctx.db,
                    log_writer=ctx.log_writer,
                    api_key_id=ctx.api_key_id,
                    model=model,
                    provider=provider,
                    endpoint=adapter.endpoint,
                    user_id=ctx.user_id,
                    usage_override=usage_data,
                )
            if ctx.reservation is not None:
                await reconcile_reservation(ctx.db, ctx.reservation, actual_cost or 0.0)
        return result
    except HTTPException:
        await release_reservation(ctx)
        raise
    except MaxToolIterationsExceeded as e:
        # Gateway-owned cap, not an upstream provider failure. 422 lets
        # callers distinguish a runaway tool loop from a real outage.
        logger.warning("Tool loop iteration cap hit (standalone): cap=%d", tool_ctx.max_tool_iterations)
        await _log_failure_and_refund(ctx, adapter, provider, model, str(e))
        raise adapter.error(422, str(e), ErrorKind.INVALID_REQUEST) from e
    except SandboxNotReachableError as e:
        # Sandbox is gateway-side infra, not an LLM provider. Clearer detail
        # so operators don't chase a provider outage that's really the
        # sandbox container being down.
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, e)
        await release_reservation(ctx)
        raise adapter.error(502, SANDBOX_UNREACHABLE_DETAIL, ErrorKind.API) from e
    except WebSearchNotReachableError as e:
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, e)
        await release_reservation(ctx)
        raise adapter.error(502, WEB_SEARCH_UNREACHABLE_DETAIL, ErrorKind.API) from e
    except Exception as e:
        await _log_failure_and_refund(ctx, adapter, provider, model, str(e))
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise adapter.provider_error(e) from e
