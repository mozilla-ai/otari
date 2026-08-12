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
* Every gateway-side rejection that reaches a known user records an error row
  via :func:`log_gateway_rejection`, so refused traffic is visible in the
  activity log and countable by the dashboard's failure count instead of
  vanishing. That function documents which rejections deliberately do not log.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Generic, Literal, NamedTuple, NoReturn, Protocol, TypeVar
from urllib.parse import ParseResult, urlparse

from any_llm import LLMProvider
from any_llm.exceptions import AnyLLMError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionUsage,
)
from fastapi import BackgroundTasks, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import verify_api_key_or_master_key
from gateway.api.routes._attempts import walk_attempts
from gateway.api.routes._helpers import apply_input_guardrails, resolve_user_id
from gateway.api.routes._platform import (
    _DEFAULT_STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS,
    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS,
    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP,
    _STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS_KEY,
    _STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY,
    _STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY,
    ResolvedAttempt,
    ResolvedRoute,
    _classify_upstream_error,
    _extract_platform_user_token,
    _report_platform_usage,
    _resolve_platform_code_execution,
    _resolve_platform_credentials,
    _resolve_platform_mcp_servers,
    _resolve_platform_web_search,
    is_provider_billing_error,
    redact_upstream_message,
    run_platform_attempts,
    upstream_error_message,
    upstream_exception_chain,
    upstream_exception_shape,
)
from gateway.api.routes._platform import (
    default_attempt_kwargs as default_attempt_kwargs,  # explicit re-export for the route modules
)
from gateway.api.routes._tools import (
    _build_web_search_backend,
    _extract_code_execution_tool,
    _extract_web_search_tool,
    _resolve_sandbox_purpose_hint,
    _web_search_intercept_enabled,
    declares_native_web_search,
)
from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.core.usage import cache_read_tokens_of, cache_write_1h_tokens_of, cache_write_tokens_of
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.metrics import record_abandoned_attempt, record_cost, record_tokens
from gateway.model_labeling import relabel_model
from gateway.models.entities import ModelPricing, UsageLog
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import McpServerConfig
from gateway.rate_limit import RateLimitInfo, check_rate_limit
from gateway.services.budget_service import (
    ReservationHandle,
    estimate_cost,
    get_budget_state,
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
from gateway.services.metered_pricing import calculate_metered_cost
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.policy_store import resolve_effective_policy
from gateway.services.pricing_service import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    find_model_pricing,
    gateway_tool_pricing_key,
    no_pricing_error_detail,
    price_tool_calls,
    pricing_required_but_missing,
)
from gateway.services.provider_kwargs import ResolvedProvider, resolve_provider_selector
from gateway.services.routing import (
    BudgetState,
    CompiledPlan,
    NoEligibleCandidatesError,
    compile_policy,
    needs_budget_state,
    selection_consults_router,
)
from gateway.services.routing.decide import RoutingSignal, decide_ordering
from gateway.services.sandbox_backend import (
    CODE_EXECUTION_TOOL_NAME,
    SandboxBackend,
    SandboxNotReachableError,
)
from gateway.services.tool_usage import (
    MAX_TOOL_NAMES,
    OVERFLOW_TOOL_NAME,
    TOOL_METER_NAMESPACE,
    ToolUsageTally,
)
from gateway.services.url_safety import UnsafeURLError, validate_mcp_url
from gateway.services.web_search_backend import WEB_SEARCH_TOOL_NAME, WebSearchNotReachableError
from gateway.streaming import (
    StreamFormat,
    StreamingAttemptFailure,
    iterate_streaming_attempts,
    streaming_generator,
)
from gateway.types.attempt import Attempt

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
# Fixed details for the provider failures that are the gateway's fault rather
# than the caller's. These never embed the upstream message: a rejected
# credential or an exhausted account is where provider internals concentrate,
# and it is not the caller's problem to debug (see classify_provider_error and
# test_error_detail_leakage). The two caller-fault details below are fallbacks,
# used only when the provider gave us no message to pass on.
PROVIDER_BAD_REQUEST_DETAIL = "The provider rejected the request as invalid (check the model name and parameters)"
PROVIDER_MODEL_NOT_FOUND_DETAIL = "The requested model was not found on the provider"
PROVIDER_CREDENTIALS_DETAIL = "The provider rejected the gateway's credentials"
PROVIDER_BILLING_DETAIL = (
    "The upstream provider account is out of credit or over its billing limit. "
    "Top up the provider account, or route this model to a provider that has credit."
)
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
SANDBOX_NOT_ENABLED_DETAIL = "code execution is not enabled for this workspace"
MALFORMED_CODE_EXEC_POLICY_DETAIL = "Authorization service returned a malformed code-execution policy"
SANDBOX_UNREACHABLE_DETAIL = (
    "code_execution sandbox unreachable. Check the sandbox URL in the dashboard's "
    "Tools settings, or OTARI_SANDBOX_URL, and that the container is running."
)
WEB_SEARCH_UNREACHABLE_DETAIL = (
    "web_search backend unreachable. Check the search URL in the dashboard's Tools "
    "settings, or OTARI_WEB_SEARCH_URL, and that the backend is running."
)
UNPRICED_TOOL_DETAIL_TEMPLATE = (
    "The gateway tool '{tool}' has no pricing, and this gateway runs with "
    "require_pricing enabled, so it will not run work it cannot bill. Set a "
    "per-request price for model_key '{key}' (POST /v1/pricing, or the dashboard's "
    "Tools & Guardrails screen), or set require_pricing to false to serve it unpriced."
)


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


class _PendingUsageReport(NamedTuple):
    attempt_id: str
    outcome: str
    usage: Any
    error_class: str | None
    is_final_attempt: bool


def _is_unsupported_feature_error(exc: BaseException) -> bool:
    """True when any-llm refused a request feature its backend cannot express.

    Unwraps ``original_exception`` as well: once
    ``ANY_LLM_UNIFIED_EXCEPTIONS=1`` becomes the default, the raw
    ``NotImplementedError`` arrives wrapped in a generic ``ProviderError`` and a
    check against ``exc`` alone would stop matching.
    """
    return any(isinstance(candidate, NotImplementedError) for candidate in upstream_exception_chain(exc))


def _caller_fault_detail(exc: BaseException, fallback: str) -> str:
    """The detail for a rejection that is the caller's request to fix.

    Returns the provider's own message, redacted and length-capped, because the
    provider is the only party that knows what it objected to.

    Falls back to ``fallback`` when what is left says nothing. Some SDKs
    stringify a failure as bare punctuation or the status code itself, and
    "404" is not an explanation the caller did not already have from the
    status. Requiring one letter is a low bar deliberately: it rejects the
    empty cases without second-guessing a provider that wrote a real sentence.
    A message made entirely of redaction placeholders is empty for this
    purpose, too.
    """
    redacted = redact_upstream_message(upstream_error_message(exc))
    explanatory = redacted.replace("[redacted]", "")
    if not any(char.isalpha() for char in explanatory):
        return fallback
    return redacted


def classify_provider_error(exc: BaseException) -> ProviderErrorMapping | None:
    """Map an upstream provider exception to a safe, specific (status, detail).

    Returns ``None`` when the failure carries no signal we can safely act on, so
    the caller falls back to its existing generic provider-error response. The
    mapping is intentionally conservative, classifying only the cases a caller
    can act on and leaving everything else (including provider 5xx and
    connection errors) to the generic 502.

    Detail text splits on whose fault the failure is. When the provider rejected
    the caller's request (400/422/404), the provider's own message is passed
    through, redacted and length-capped, because it is the only description of
    what was actually wrong and no fixed string we write can substitute for it.
    When the failure is the gateway's own (a rejected credential, an exhausted
    provider account, a 5xx), the detail stays a fixed string: those carry no
    remedy the caller could apply, and are where a raw message is most likely to
    name the operator's credentials or topology.

    Timeout detection (including the OpenAI/Anthropic SDKs' own
    ``APITimeoutError``, and a duck-typed fallback for other provider SDKs) is
    shared with the hybrid-mode fallback classifier via
    :func:`upstream_exception_shape`, so both stay in sync.
    """
    kind, status_code = upstream_exception_shape(exc)
    if kind == "timeout":
        return ProviderErrorMapping(status.HTTP_504_GATEWAY_TIMEOUT, PROVIDER_TIMEOUT_DETAIL)
    # any-llm raises NotImplementedError when a request asks a provider for
    # something its backend cannot express: context_management/betas against a
    # provider with no native Anthropic Messages API is the case that surfaced
    # this (#530). It carries no HTTP status, so it would otherwise fall through
    # to the generic 502/500 and tell the caller a guaranteed-permanent failure
    # was a transient one. The exception type is the whole signal here, so this
    # needs no probe into any-llm's wording, and the message it carries already
    # names the unsupported feature.
    if _is_unsupported_feature_error(exc):
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _caller_fault_detail(exc, PROVIDER_BAD_REQUEST_DETAIL))
    if status_code is None:
        return None
    # Account billing exhaustion, which several providers report as a 400/422
    # rather than the 402 the condition deserves (and DeepSeek does report as a
    # 402). Like a rejected credential this is a gateway-side account
    # fault rather than anything wrong with the caller's request, so it surfaces as
    # a 502 and never as a client-facing 400: telling the caller to "check the model
    # name and parameters" for an empty wallet sends operators debugging the wrong
    # thing. Checked ahead of the status branches so it wins over both the generic
    # bad-request detail and the 402 fall-through to a bare 502.
    if is_provider_billing_error(exc):
        return ProviderErrorMapping(status.HTTP_502_BAD_GATEWAY, PROVIDER_BILLING_DETAIL)
    if status_code in (400, 422):
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _caller_fault_detail(exc, PROVIDER_BAD_REQUEST_DETAIL))
    if status_code == 404:
        return ProviderErrorMapping(
            status.HTTP_404_NOT_FOUND, _caller_fault_detail(exc, PROVIDER_MODEL_NOT_FOUND_DETAIL)
        )
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


def failure_status_code(exc: BaseException) -> int:
    """The HTTP status to record on the usage log for an upstream failure.

    Prefers the status the provider actually returned, which is deliberately not
    always the status the caller saw: an upstream 401/403 surfaces to the caller
    as a generic 502 (a provider rejecting the gateway's credentials is a
    gateway-config fault, and the response must not say so), but the log keeps
    the 401 so "how much of my error rate is my own misconfiguration" stays
    answerable. When the provider returned no status at all (timeout,
    unreachable), records the gateway's own classification instead, so an error
    row still carries a code to group on.

    The tool-loop cap is checked first because it is the gateway's own limit, not
    an upstream failure: it carries no HTTP status of its own, so it would
    otherwise fall through to the generic 502 and read in the taxonomy as a
    provider outage. It reaches here from the streaming path, where the cap is
    raised while the SSE body is already streaming (see ``run_tool_loop_stream``)
    and settles through ``on_error``; the non-streaming path records the same 422
    at its own ``except MaxToolIterationsExceeded``.
    """
    if isinstance(exc, MaxToolIterationsExceeded):
        return status.HTTP_422_UNPROCESSABLE_CONTENT
    _kind, status_code = upstream_exception_shape(exc)
    if status_code is not None:
        return status_code
    mapping = classify_provider_error(exc)
    return mapping.status_code if mapping is not None else status.HTTP_502_BAD_GATEWAY


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _normalized_origin(parsed: ParseResult) -> tuple[str, str | None, int | None]:
    """(scheme, host, port) with the scheme's default port filled in.

    So ``https://h`` and ``https://h:443`` compare equal (and ``http`` / ``:80``),
    rather than failing on ``None != 443`` and silently not forwarding the token.
    """
    port = parsed.port if parsed.port is not None else _DEFAULT_PORTS.get(parsed.scheme)
    return (parsed.scheme, parsed.hostname, port)


def url_targets_platform(url: str, platform_base_url: str | None) -> bool:
    """True when ``url`` is the platform itself (same origin, under its base path).

    Gates forwarding the platform token to a gateway-managed backend (web search,
    sandbox): it is only safe to hand that high-privilege credential to the
    platform — the host the gateway already trusts it with for resolve. A raw
    string prefix check is not enough: with a path-less ``PLATFORM_BASE_URL``
    (e.g. ``https://api.otari.ai``) a confusable URL like
    ``https://api.otari.ai.evil.com`` or ``https://api.otari.ai@evil.com`` would
    satisfy ``startswith`` and leak the token. So compare the parsed
    (scheme, host, port) origin exactly — with default ports normalized — and
    require the target path to sit under the base path at a ``/`` boundary.
    """
    if not platform_base_url:
        return False
    base = urlparse(platform_base_url)
    target = urlparse(url)
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
        *,
        emit_native_web_search: bool = False,
    ) -> ResultT: ...

    def open_tool_loop_stream(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        *,
        emit_native_web_search: bool = False,
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

    def local_attempt_kwargs(
        self,
        attempt: Attempt,
        base_request_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Build call kwargs for a locally resolved attempt.

        The standalone counterpart of :meth:`attempt_kwargs`. It exists as a hook
        rather than being done in the walker because the shape is format-specific:
        the responses format passes ``provider`` and ``model`` as separate keywords
        and rebuilds its Codex extra-body per provider, which has to happen for the
        candidate being tried and not for the one that failed.
        """
        ...

    def prepare_platform_call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adjust the ``run_platform_attempts``-shaped kwargs for the format's
        provider call (the responses format re-splits ``provider:model``)."""
        ...


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
        started_at: float,
        resolved_provider: ResolvedProvider | None = None,
        plan: CompiledPlan | None = None,
        estimate_inputs: "EstimateInputs | None" = None,
        request_group_id: str | None = None,
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
        # USD already written onto a failure row for gateway-run tool calls. A
        # request whose plan is exhausted still owes for the searches it ran, and
        # ``log_exhausted_plan`` writes that onto the row without settling. Recording
        # it here lets the single release site reconcile it instead of refunding,
        # which is what keeps ``users.spend`` matching the row: ``refund_reservation``
        # releases the hold *without* writing spend.
        self.tool_charge: float = 0.0
        # Monotonic clock reading taken at the very start of the handler
        # preamble; used to compute the usage log's latency_ms at settlement.
        self.started_at = started_at
        # Standalone-only: the provider selector resolved once for the
        # pricing/budget gate in `resolve_request_context`. Route handlers
        # reuse this for dispatch instead of calling `resolve_provider_selector`
        # a second time, which would redo the provider-kwargs build (and, for
        # Vertex AI instances, re-parse the service-account credentials from
        # disk and reconstruct the RSA-backed `Credentials` object) for no
        # reason. `None` in hybrid mode (no local provider resolution happens)
        # and in the rare case the selector couldn't be parsed for the gate
        # check; callers fall back to `resolve_provider_selector` themselves
        # in that case, same as before this field existed.
        self.resolved_provider = resolved_provider
        # Standalone-only: the compiled routing plan when `model` named a policy.
        # `None` for a plain model or an alias, which is what keeps the
        # single-candidate path byte-identical to what it was. The head attempt is
        # what the pricing gate and the reservation above were keyed on, so
        # settlement stays keyed on whichever attempt actually serves.
        self.plan = plan
        # Standalone-only: what the reservation estimate was computed from, kept so
        # a fallover to a differently priced candidate can reprice and top up the
        # hold rather than serving a pricier model against a cheaper model's
        # reservation. `None` when nothing was reserved.
        self.estimate_inputs = estimate_inputs
        # Ties this request's usage rows together. A routed request can write more
        # than one (the attempt that served, plus one per absorbed failure), and
        # without a shared id they would be unrelated rows in the activity log.
        # `None` for an unrouted request, which writes exactly one row.
        self.request_group_id = request_group_id


def unresolvable_model_detail(model_selector: str) -> str:
    """Human-readable 400 detail for a selector the gateway cannot resolve."""
    return (
        f"Unknown or unsupported model {model_selector!r}. Use the format 'provider:model' with a configured provider."
    )


def _raise_for_unresolvable_model(model_selector: str, exc: Exception) -> NoReturn:
    """Convert a selector-parse failure into an HTTP 400 with a helpful detail.

    resolve_provider_selector raises ValueError for an unparseable
    selector (no provider: prefix) and AnyLLMError for an unknown
    provider.  Both are client input errors; surfacing them as a bare 500 is
    confusing.
    """
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=unresolvable_model_detail(model_selector),
    ) from exc


async def resolve_dispatch_provider(
    ctx: RequestContext,
    config: GatewayConfig,
    model_selector: str,
    *,
    adapter: FormatAdapter[Any, Any],
) -> ResolvedProvider:
    """Get the ``ResolvedProvider`` for dispatch, reusing the one computed for
    the pricing/budget gate (``ctx.resolved_provider``) instead of resolving
    the selector a second time. Standalone-mode route handlers should call
    this rather than ``resolve_provider_selector`` directly.

    Falls back to a fresh ``resolve_provider_selector`` call only when
    ``ctx.resolved_provider`` is ``None``: the rare case where the gate
    check couldn't parse the selector (an unparseable selector has no
    pricing, but dispatch still needs its own resolution attempt so any-llm's
    own error surfaces instead of a stale gate-check failure).
    """
    if ctx.resolved_provider is not None:
        return ctx.resolved_provider
    try:
        return resolve_provider_selector(config, model_selector, ctx.user_id)
    except (ValueError, AnyLLMError) as exc:
        # The preamble deliberately tolerated this selector, so a reservation is
        # already held: refund it, then record the drop. The hold is not always
        # zero, so this refund is load-bearing rather than a formality. The
        # preamble carries an unresolvable selector into the pricing lookup as
        # the bare model with no provider, and find_model_pricing then keys on
        # the model alone, which is exactly the `provider:model` form stored
        # pricing rows use. An instance removed from config while its pricing row
        # survives therefore prices, reserves a real estimate, and only fails
        # here; before this refund existed the hold stayed on users.reserved
        # until the next budget reset (forever, for a budget with no period).
        #
        # Releasing here and then raising is safe only because no caller above
        # catches this 400 and releases again: refund_reservation is not
        # idempotent (_release_reserved clamps at 0, but a second call still
        # subtracts the estimate a second time, silently handing the user budget
        # they never gave back). chat.py, messages.py and responses.py all let
        # the 400 propagate. Anyone adding an outer handler around
        # resolve_dispatch_provider must not refund in it.
        await release_reservation(ctx)
        await log_gateway_rejection(
            db=ctx.db,
            log_writer=ctx.log_writer,
            api_key_id=ctx.api_key_id,
            user_id=ctx.user_id,
            model=model_selector,
            provider=None,
            endpoint=adapter.endpoint,
            detail=unresolvable_model_detail(model_selector),
            status_code=status.HTTP_400_BAD_REQUEST,
            started_at=ctx.started_at,
        )
        _raise_for_unresolvable_model(model_selector, exc)


async def _bill_vision_side_call(
    *,
    db: AsyncSession,
    log_writer: LogWriter,
    config: GatewayConfig,
    api_key_id: str | None,
    user_id: str,
    endpoint: str,
    usage: CompletionUsage,
    counts_toward_budget: bool = True,
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
    # latency_ms is intentionally left NULL: this row bills the describe model as
    # its own side-call, so the enclosing request's duration would misattribute
    # the caller's wall-clock to it.
    cost = await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=resolved.model,
        provider=resolved.instance,
        endpoint=endpoint,
        user_id=user_id,
        usage_override=usage,
        counts_toward_budget=counts_toward_budget,
    )
    # Commit the spend directly via an unreserved handle (no held estimate to
    # release): this just adds the actual cost to users.spend. When the request is
    # budget-exempt the handle carries counts_toward_budget=False, so the cost is
    # logged on its own row but never folded into users.spend.
    await reconcile_reservation(
        db,
        ReservationHandle(
            user_id=user_id,
            estimate=0.0,
            reserved=False,
            strategy=config.budget_strategy,
            counts_toward_budget=counts_toward_budget,
        ),
        cost or 0.0,
    )


@dataclass(frozen=True)
class RoutingAttribution:
    """Which policy produced a usage row, and where in its plan.

    Carried onto the row so a tier-down or a fallover is answerable with a query
    instead of a log grep. ``absorbed`` marks an attempt a policy recovered from by
    trying the next candidate; see :func:`_row_status` for why that is not an error.
    """

    policy_name: str
    selection_reason: str
    position: int
    attempt_count: int
    request_group_id: str
    absorbed: bool = False


def _row_status(*, error: str | None, attribution: RoutingAttribution | None) -> str:
    """The status to record: ``success``, ``error``, or ``absorbed``.

    A failed attempt that a policy recovered from is ``absorbed``, never ``error``.
    Every error metric in the product counts ``status == "error"`` exactly, so
    recording a recovered attempt as an error would make a working fallback chain
    report an outage: the Overview error-rate tile turns amber at 2%, and the
    activity timeline would show red where the gateway in fact did its job.
    """
    if error is None:
        return "success"
    if attribution is not None and attribution.absorbed:
        return "absorbed"
    return "error"


@dataclass(frozen=True)
class EstimateInputs:
    """The inputs a budget estimate was computed from.

    Kept on the request context so a fallover can recompute the estimate for a
    differently priced candidate. Without it, a chain that fell over to a pricier
    model would run against the cheaper model's reservation and could take spend
    past a cap the gate had already approved.
    """

    prompt_chars: int
    max_output_tokens: int | None
    default_output_tokens: int
    cache_write_ttl: Literal["5m", "1h"] | None = None


async def top_up_reservation_for_attempt(ctx: RequestContext, attempt: Attempt) -> None:
    """Grow the reservation to cover ``attempt`` before dispatching it.

    Called before every candidate after the first. A cheaper candidate is a no-op
    (``increase_reservation`` ignores a non-positive delta), so the hold only ever
    grows toward the candidate that actually serves.

    A refused top-up raises, which the walker treats as terminal: the chain stops
    rather than serving a model the caller cannot afford, and the caller's outer
    handler refunds the original hold. That is the honest failure. The alternative,
    proceeding on the cheaper hold, would quietly take spend past the cap.
    """
    if ctx.db is None or ctx.reservation is None or ctx.estimate_inputs is None:
        return
    pricing = await find_model_pricing(ctx.db, attempt.instance, attempt.model)
    # `require_pricing` is a billing safety gate: it refuses a request the gateway
    # cannot price, because it then cannot debit it. The gate at admission prices
    # only the head candidate, so without this an unpriced model that 402s when
    # named directly would serve, and log cost=null, simply by being reached as a
    # fallback. A budget-exempt request is never debited, so the gate does not
    # apply to it, matching the admission-time rule.
    if ctx.reservation.counts_toward_budget and pricing_required_but_missing(
        pricing, require_pricing=ctx.config.require_pricing
    ):
        logger.warning(
            "Fallback candidate %s:%s has no pricing and require_pricing is on; stopping the chain",
            attempt.instance,
            attempt.model,
        )
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=no_pricing_error_detail(f"{attempt.instance}:{attempt.model}"),
        )
    repriced = estimate_cost(
        pricing,
        prompt_chars=ctx.estimate_inputs.prompt_chars,
        max_output_tokens=ctx.estimate_inputs.max_output_tokens,
        default_output_tokens=ctx.estimate_inputs.default_output_tokens,
        cache_write_ttl=ctx.estimate_inputs.cache_write_ttl,
    )
    delta = repriced - ctx.reservation.estimate
    if delta <= 0:
        return
    try:
        await increase_reservation(
            ctx.db,
            ctx.reservation,
            delta,
            model=f"{attempt.instance}:{attempt.model}",
            strategy=ctx.config.budget_strategy,
        )
    except HTTPException as exc:
        logger.warning(
            "Reservation top-up refused for fallback candidate %s:%s; stopping the chain",
            attempt.instance,
            attempt.model,
        )
        raise HTTPException(status_code=exc.status_code, detail=budget_exhausted_mid_failover_detail()) from exc


def budget_exhausted_mid_failover_detail() -> str:
    """Detail for a fallover the caller's remaining budget cannot cover."""
    return (
        "Budget exhausted while failing over. The next candidate prices higher than the one that "
        "failed, and reserving the difference would exceed the remaining budget, so the chain was "
        "stopped rather than allowed to overshoot. Raise the budget, or order the policy so no "
        "on_failure entry prices above its selected candidate."
    )


def policy_in_hybrid_mode_detail(model_selector: str) -> str:
    """400 detail for a policy name used against a hybrid-mode gateway."""
    return (
        f"Routing policy {model_selector!r} cannot be used in hybrid mode. The connected platform resolves "
        "the model for every request, so a local policy name is not a model it knows. Name a concrete "
        "model, or run this gateway in standalone mode where routing policies apply."
    )


async def _compile_request_plan(
    *,
    adapter: FormatAdapter[Any, Any],
    db: AsyncSession,
    log_writer: LogWriter,
    config: GatewayConfig,
    model: str,
    user_id: str | None,
    api_key_id: str | None,
    allowlist: list[str] | None,
    endpoint: str,
    started_at: float,
    routing_signal: Callable[[], RoutingSignal] | None = None,
) -> CompiledPlan | None:
    """Compile ``model`` into a plan when it names a routing policy, else ``None``.

    Budget numbers are fetched only when a condition in the policy actually reads
    them, so a plain failover policy costs no extra query. A policy naming a
    router gets one more step: the backend ranks its candidates for this request
    and the ranking becomes the plan. That step is the only asynchronous part of
    routing, which is why it happens here and not in the compiler.

    An empty plan is a 403 whose caller-facing detail names the policy and nothing
    else (a policy exists partly to keep its targets off the wire); the enumerated
    per-candidate reasons go to the activity log, which is a master-key surface.
    """
    spec = resolve_effective_policy(config, model, user_id)
    if spec is None:
        return None

    budget = BudgetState()
    if needs_budget_state(spec) and user_id is not None:
        budget = await get_budget_state(db, user_id)

    # The signal is built here rather than by the endpoint because flattening the
    # prompt is not free on a long conversation, and only a policy with a router
    # ever reads it. Every other request pays three header lookups and nothing.
    #
    # Only asked when the router entry is the one this request would reach: a `when`
    # entry ahead of it wins outright, and ranking for a plan that discards the
    # ranking is a paid embedding call plus a scan of the user's examples, followed
    # by a log line claiming a decision the request did not use.
    router_ordering = None
    if spec.router_backend is not None and selection_consults_router(
        spec, user_id=user_id, key_id=api_key_id, budget=budget
    ):
        router_ordering = await decide_ordering(
            config,
            spec,
            policy_name=model,
            user_id=user_id,
            allowlist=allowlist,
            signal=routing_signal() if routing_signal is not None else None,
        )

    try:
        return compile_policy(
            config,
            model,
            spec,
            user_id=user_id,
            key_id=api_key_id,
            allowlist=allowlist,
            budget=budget,
            router_ordering=router_ordering,
        )
    except NoEligibleCandidatesError as exc:
        logger.warning("%s", exc.operator_detail)
        await log_gateway_rejection(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            model=model,
            provider=None,
            endpoint=endpoint,
            detail=exc.operator_detail,
            status_code=exc.status_code,
            started_at=started_at,
        )
        raise adapter.error(exc.status_code, exc.caller_detail, ErrorKind.PERMISSION) from exc


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
    estimate_cache_write_ttl: Literal["5m", "1h"] | None = None,
    routing_signal: Callable[[], RoutingSignal] | None = None,
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

    ``routing_signal`` (standalone only) builds what a policy's router backend
    reads: the prompt text plus the routing headers, in a format-neutral value the
    endpoint knows how to flatten. A factory rather than a value because only a
    policy with a router consults it. Omit it and such a policy serves its default
    target, which is the correct behavior for a surface that has no prompt.

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
    # Earliest point in the shared handler preamble; anchors the request's
    # latency_ms (measured monotonically, so it is immune to wall-clock steps).
    started_at = time.monotonic()
    hybrid_mode = config.is_hybrid_mode
    route: ResolvedRoute | None = None
    user_token: str | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    rate_limit_info: RateLimitInfo | None = None
    reservation: ReservationHandle | None = None
    resolved_provider: ResolvedProvider | None = None
    plan: CompiledPlan | None = None
    estimate_inputs: EstimateInputs | None = None

    if hybrid_mode:
        # Refuse a policy name before the resolve call rather than after. Hybrid
        # mode sends the caller's selector straight upstream (config aliases are
        # already inert here for the same reason), so a policy name would reach
        # the platform as an unknown model and come back as a confusing upstream
        # 404. Routing policies are standalone-only until the platform protocol
        # carries them.
        if model in config.policy_names():
            raise adapter.error(400, policy_in_hybrid_mode_detail(model), ErrorKind.INVALID_REQUEST)
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
        try:
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
        except HTTPException as exc:
            # Only the user/key mismatch (403) is recorded: spend always binds to
            # the key's own user, so that rejection has a user to attribute the
            # drop to. resolve_user_id's other refusals (a master key with no
            # `user` field, a key with no user) name no existing user, and
            # usage_logs.user_id is a foreign key, so they stay unlogged.
            # This row carries the raw selector and no provider, unlike the gates
            # below it: nothing has been resolved this early, and resolving a
            # selector purely to shape a log row is not worth the work on a path
            # that is refusing the request anyway.
            # This gate is the only one that fires before check_rate_limit, so
            # the write is charged to the key's own bucket and skipped once
            # throttled; see throttle_early_rejection. The response stays 403.
            if (
                exc.status_code == status.HTTP_403_FORBIDDEN
                and api_key is not None
                and not throttle_early_rejection(raw_request, str(api_key.user_id))
            ):
                await log_gateway_rejection(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    user_id=api_key.user_id,
                    model=model,
                    provider=None,
                    endpoint=adapter.endpoint,
                    detail=user_forbidden_detail,
                    status_code=exc.status_code,
                    started_at=started_at,
                )
            raise
        rate_limit_info = check_rate_limit(raw_request, user_id)

        # Tolerate an unparseable / unknown-provider selector here: the budget
        # check below and the downstream provider call surface those with
        # their own status codes. A model we can't parse simply has no pricing.
        # Pricing/budget keys on the *instance* name (``instance:model``) while
        # capability detection needs the underlying implementation, so keep both.
        gate_instance: str | None
        gate_impl: LLMProvider | None
        # Resolved before the plan rather than with the gate below, because the
        # compiler must drop candidates this caller may not use: a chain that fell
        # over to a forbidden model would be an access-control bypass. The gate
        # itself stays where it was, so a plain model name is unaffected.
        key_allowlist = await resolve_request_allowlist(db, api_key)
        # A policy name resolves to a plan rather than to one selector. The head
        # candidate is what everything below keys on (allow-list, pricing,
        # reservation), exactly as a plain model would be, so a one-candidate
        # policy behaves identically to naming its target directly.
        plan = await _compile_request_plan(
            adapter=adapter,
            db=db,
            log_writer=log_writer,
            config=config,
            model=model,
            user_id=user_id,
            api_key_id=api_key_id,
            allowlist=key_allowlist,
            endpoint=adapter.endpoint,
            started_at=started_at,
            routing_signal=routing_signal,
        )
        if plan is not None:
            head = plan.head
            gate_instance, gate_impl, gate_model = head.instance, head.provider, head.model
            resolved_provider = ResolvedProvider(
                instance=head.instance,
                provider=head.provider,
                model=head.model,
                kwargs=head.kwargs,
                alias=head.display_model,
            )
        else:
            try:
                resolved = resolve_provider_selector(config, model, user_id)
                gate_instance, gate_impl, gate_model = resolved.instance, resolved.provider, resolved.model
                # Reused by the route handler for dispatch (see `RequestContext.resolved_provider`)
                # instead of calling `resolve_provider_selector` a second time.
                resolved_provider = resolved
            except (ValueError, AnyLLMError):
                gate_instance, gate_impl, gate_model = None, None, model

        # Model access control (per-key, standalone). None = unrestricted; a
        # non-null list restricts. Fail closed: a selector we could not resolve is
        # denied under a restriction rather than dispatched unchecked. Master-key
        # callers have api_key None, so the allow-list is None and this is skipped.
        # A key with no list of its own inherits its user's default here.
        # (Resolved above, before the plan compile, which needs it.)
        if key_allowlist is not None and not (
            gate_instance is not None and is_model_allowed(key_allowlist, f"{gate_instance}:{gate_model}")
        ):
            not_allowed_detail = model_not_allowed_detail(model)
            # Nothing is reserved yet (the reservation is taken below), so there
            # is no refund to do before recording the drop.
            await log_gateway_rejection(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                user_id=user_id,
                model=gate_model,
                provider=gate_instance,
                endpoint=adapter.endpoint,
                detail=not_allowed_detail,
                status_code=status.HTTP_403_FORBIDDEN,
                started_at=started_at,
            )
            raise adapter.error(403, not_allowed_detail, ErrorKind.PERMISSION)

        gate_pricing = await find_model_pricing(db, gate_instance, gate_model)
        # Captured so a fallover can reprice against a different candidate; see
        # `top_up_reservation_for_attempt`.
        estimate_inputs = EstimateInputs(
            prompt_chars=estimate_prompt_chars,
            max_output_tokens=estimate_max_output_tokens,
            default_output_tokens=config.budget_estimate_default_output_tokens,
            cache_write_ttl=estimate_cache_write_ttl,
        )
        estimate = estimate_cost(
            gate_pricing,
            prompt_chars=estimate_inputs.prompt_chars,
            max_output_tokens=estimate_inputs.max_output_tokens,
            default_output_tokens=estimate_inputs.default_output_tokens,
            cache_write_ttl=estimate_inputs.cache_write_ttl,
        )
        # A key flagged exclude_from_budget still logs its cost but is never
        # reserved, reconciled into users.spend, or gated. Master-key callers have
        # api_key None and stay on the enforced path. The decision is threaded
        # through the reservation handle so every downstream reconcile/refund/top-up
        # site inherits it (see budget_service.reconcile_reservation).
        budget_exempt = api_key is not None and api_key.exclude_from_budget
        # Reserve first so user/blocked/budget rejections (404/403) take
        # precedence over the missing-pricing rejection (402); refund if we
        # then reject for missing pricing.
        try:
            reservation = await reserve_budget(
                db,
                user_id,
                estimate,
                model=gate_model,
                pricing_provider=gate_instance,
                strategy=config.budget_strategy,
                counts_toward_budget=not budget_exempt,
            )
        except HTTPException as exc:
            # A blocked or over-budget user is refused inside reserve_budget,
            # which reserves nothing on the paths that raise, so there is no
            # refund to do before recording the drop. The 404 for an unknown user
            # is skipped: usage_logs.user_id is a foreign key to users, so a row
            # naming a user that does not exist could not be inserted.
            if exc.status_code != status.HTTP_404_NOT_FOUND:
                await log_gateway_rejection(
                    db=db,
                    log_writer=log_writer,
                    api_key_id=api_key_id,
                    user_id=user_id,
                    model=gate_model,
                    provider=gate_instance,
                    endpoint=adapter.endpoint,
                    detail=str(exc.detail),
                    status_code=exc.status_code,
                    started_at=started_at,
                )
            raise
        # require_pricing is a budget-enforcement safety gate: it refuses a request
        # we cannot price because we then cannot debit it. A budget-exempt key is
        # never debited, so the gate does not apply: the call proceeds and logs
        # cost=null when unpriced.
        if not budget_exempt and pricing_required_but_missing(gate_pricing, require_pricing=config.require_pricing):
            await refund_reservation(db, reservation)
            no_pricing_detail = no_pricing_error_detail(model)
            # Record the rejection after the refund, so an operator who flipped
            # require_pricing on can see that live traffic is being dropped.
            await log_gateway_rejection(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                user_id=user_id,
                model=gate_model,
                provider=gate_instance,
                endpoint=adapter.endpoint,
                detail=no_pricing_detail,
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                started_at=started_at,
            )
            raise adapter.error(
                402,
                no_pricing_detail,
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
                post_chars, vision_usage = await normalize_messages(user_id, gate_impl, gate_model, gate_instance)
                # Bill the vision describe side-call before the reservation
                # top-up: its cost is already incurred by normalize_messages,
                # so a 402 from the top-up below must not skip it (the refund
                # in the except path releases only the main reservation; the
                # vision spend is committed independently and stays billed).
                if vision_usage is not None:
                    await _bill_vision_side_call(
                        db=db,
                        log_writer=log_writer,
                        config=config,
                        api_key_id=api_key_id,
                        user_id=user_id,
                        endpoint=adapter.endpoint,
                        usage=vision_usage,
                        counts_toward_budget=not budget_exempt,
                    )
                # Attachments expanded the payload, so the stored inputs must
                # follow or a later fallover would reprice against the pre-
                # normalization size.
                estimate_inputs = replace(estimate_inputs, prompt_chars=post_chars)
                post_estimate = estimate_cost(
                    gate_pricing,
                    prompt_chars=estimate_inputs.prompt_chars,
                    max_output_tokens=estimate_inputs.max_output_tokens,
                    default_output_tokens=estimate_inputs.default_output_tokens,
                    cache_write_ttl=estimate_inputs.cache_write_ttl,
                )
                await increase_reservation(
                    db,
                    reservation,
                    post_estimate - estimate,
                    model=model,
                    strategy=config.budget_strategy,
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

    # The request is authorized and about to be dispatched, so from here until the
    # response has been fully sent it is genuinely in flight and the activity log
    # can show it as such. Registered after the budget, access and model-resolution
    # gates rather than at the top of the preamble: a request refused by one of
    # those was never in progress, and it already leaves a usage row of its own. The
    # caller-facing checks that run after this (`prepare_gateway_tools`: input
    # guardrails, MCP id resolution, tool opt-ins) do list the request while they
    # run, which is honest, since each of them can make a network call of its own.
    # `model` and `provider` are the pair the
    # usage row will carry (the resolved target, not the caller's selector and not
    # the display alias), so a request does not appear to change model at the moment
    # it settles. The raw selector is the fallback for the cases that resolve
    # nothing locally: hybrid mode, and a selector the gate could not parse.
    track_request(
        raw_request,
        endpoint=adapter.endpoint,
        model=resolved_provider.model if resolved_provider else model,
        provider=resolved_provider.instance if resolved_provider else None,
        user_id=user_id,
        api_key_id=api_key_id,
        policy_name=plan.policy_name if plan else None,
    )

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
        started_at=started_at,
        resolved_provider=resolved_provider,
        plan=plan,
        estimate_inputs=estimate_inputs,
        request_group_id=str(uuid.uuid4()) if plan is not None else None,
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
        config: GatewayConfig,
    ) -> None:
        self.config = config
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
        # One tally per request, handed to whichever backend runs. It lives here
        # rather than on the backend because the streaming path tears the backend
        # down while the response is still settling (``_eager_backend_stream``'s
        # ``finally`` runs during stream exhaustion, and ``streaming_generator``
        # only awaits ``on_complete`` afterwards), and because a multi-attempt
        # request shares one tally across attempts: every executed call was paid
        # for, whether or not its attempt won.
        self.tally = ToolUsageTally()

    @property
    def tools_extracted(self) -> bool:
        return self.sandbox_tool_entry is not None or self.web_search_tool_entry is not None

    @property
    def web_search_declared_name(self) -> str | None:
        """The ``name`` the caller gave its web-search declaration, if any.

        Used to retarget a forced ``tool_choice`` onto the backend's canonical tool
        name. ``None`` when no web-search entry was extracted or it carried no name.
        """
        name = (self.web_search_tool_entry or {}).get("name")
        return name if isinstance(name, str) and name else None

    @property
    def emit_native_web_search(self) -> bool:
        """Whether this request should get Anthropic-native server-tool blocks back."""
        return self.use_web_search and declares_native_web_search(self.web_search_tool_entry)

    @property
    def intercepts_web_search(self) -> bool:
        """Whether this deployment claims the provider-named web-search keywords.

        The same two conditions :func:`prepare_gateway_tools` applies before it
        extracts one: the opt-in, *and* a backend to intercept to. Both matter to a
        caller of this property, because it is also the precondition for a
        gateway-minted server-tool block existing at all: with the toggle on but no
        backend, the keyword was forwarded and any block in the transcript came from
        the provider that ran the search (see ``routes/messages.py``).
        """
        return _web_search_intercept_enabled(self.config) and self.web_search_url is not None

    @property
    def use_tool_loop(self) -> bool:
        return bool(self.mcp_server_configs) or self.use_sandbox or self.use_web_search


async def _validate_mcp_server_urls(adapter: FormatAdapter[Any, Any], mcp_servers: list[McpServerConfig]) -> None:
    """SSRF/scheme safety check for every MCP server URL in this request.

    Covers both request-body-supplied servers and platform-resolved ones
    (``mcp_server_ids``): both land in the same merged list before this runs.
    Runs concurrently since each check does an independent DNS lookup;
    ``asyncio.gather`` (default ``return_exceptions=False``) propagates the
    first ``UnsafeURLError`` it sees as soon as it's raised. Note this does
    *not* cancel the other in-flight checks: they keep running in the
    background and are simply not awaited further; harmless here since
    ``validate_mcp_url`` has no side effects beyond a DNS lookup.

    This used to run synchronously inside a Pydantic ``model_validator`` at
    request-body-parse time (see ``McpServerConfig``/``GuardrailConfig``
    docstrings). It moved here because the DNS lookup must be awaited, and
    Pydantic validators can't await. One observable side effect: a rejected
    URL now surfaces as ``400`` (via ``adapter.error``) instead of Pydantic's
    ``422``.
    """
    try:
        await asyncio.gather(
            *(
                validate_mcp_url(server.url, has_authorization_token=bool(server.authorization_token))
                for server in mcp_servers
            )
        )
    except UnsafeURLError as exc:
        raise adapter.error(400, str(exc), ErrorKind.INVALID_REQUEST) from exc


def merge_policy_guardrails(
    ctx: RequestContext, requested: list[GuardrailConfig] | None
) -> list[GuardrailConfig] | None:
    """Combine a policy's mandated guardrails with the caller's own.

    Union by profile, with the stricter setting winning on every axis: a caller
    may *add* guardrails and may tighten one, but can never weaken what the
    operator mandated. `block` beats `monitor` for both `mode` and
    `on_unavailable`, since each is a choice between enforcing and observing.

    The operator's entry also owns the URL and the validate kwargs for a profile
    it mandates, so a caller cannot point a mandated check at a service of their
    choosing.

    Returns `None` when neither side asked for anything, which is the shape
    `apply_input_guardrails` treats as "no guardrails ran".
    """
    mandated = ctx.plan.guardrails if ctx.plan is not None else []
    if not mandated:
        return requested
    merged: dict[str, GuardrailConfig] = {}
    # Caller entries first so a mandated profile of the same name overwrites them.
    for guardrail in requested or []:
        merged[guardrail.profile] = guardrail
    for guardrail in mandated:
        caller = merged.get(guardrail.profile)
        if caller is None:
            merged[guardrail.profile] = guardrail
            continue
        merged[guardrail.profile] = guardrail.model_copy(
            update={
                "mode": "block" if "block" in (guardrail.mode, caller.mode) else "monitor",
                "on_unavailable": (
                    "block" if "block" in (guardrail.on_unavailable, caller.on_unavailable) else "monitor"
                ),
            }
        )
    return list(merged.values())


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
        # A policy's guardrails are merged in here rather than at each route, so
        # every completion endpoint enforces a mandate identically and none can
        # forget to. `guardrails` as passed is the caller's own list.
        await apply_input_guardrails(
            merge_policy_guardrails(ctx, guardrails), guardrail_text, response=response, config=ctx.config
        )

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

        if mcp_servers:
            await _validate_mcp_server_urls(adapter, mcp_servers)

        sandbox_tool_entry, tools_after_sandbox = _extract_code_execution_tool(tools)
        # Read the effective config value (dashboard override / env / YAML), falling
        # back to the env var so pure-env deployments are unchanged. A dashboard
        # override mutates ctx.config, so it hot-applies on the next request.
        sandbox_url: str | None = ctx.config.sandbox_url or otari_env("SANDBOX_URL") or None
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
        sandbox_max_iterations: int | None = None
        if use_sandbox and ctx.hybrid_mode:
            assert ctx.user_token is not None  # guaranteed by the hybrid-mode preamble
            assert sandbox_tool_entry is not None  # use_sandbox implies the entry is present
            if sandbox_url is not None and url_targets_platform(sandbox_url, ctx.config.platform.get("base_url")):
                sandbox_auth_token = ctx.user_token

            # Platform owns the per-workspace code-exec policy: 403 if the workspace
            # has it off, otherwise apply the workspace defaults (per-request values
            # win) — the default purpose hint and the loop-iteration ceiling. The
            # tools allow-list + exec timeout are re-enforced by the /v1/sandbox proxy.
            policy = await _resolve_platform_code_execution(config=ctx.config, user_token=ctx.user_token)
            # Fail closed on a malformed policy: a non-bool `enabled` is a cross-service
            # contract break, not a "disabled" signal — surface it as 502, never run.
            enabled = policy.get("enabled")
            if not isinstance(enabled, bool):
                raise adapter.error(502, MALFORMED_CODE_EXEC_POLICY_DETAIL, ErrorKind.API)
            if not enabled:
                raise adapter.error(403, SANDBOX_NOT_ENABLED_DETAIL, ErrorKind.PERMISSION)
            if not sandbox_tool_entry.get("purpose_hint") and policy.get("default_purpose_hint"):
                sandbox_tool_entry["purpose_hint"] = policy["default_purpose_hint"]
            resolved_iters = policy.get("max_iterations")
            # `bool` is an `int` subclass — exclude it so a JSON `true` isn't read as 1.
            if isinstance(resolved_iters, int) and not isinstance(resolved_iters, bool) and resolved_iters > 0:
                sandbox_max_iterations = resolved_iters

        web_search_url: str | None = ctx.config.web_search_url or otari_env("WEB_SEARCH_URL") or None
        # Interception (claiming the provider-named web_search keywords) is opt-in and
        # additionally requires a backend: without one there is nothing to intercept
        # *to*, and claiming the keyword would turn a request the provider would have
        # served into a 400. So with no backend configured, or the toggle off, a
        # provider-named keyword passes through exactly as it always has.
        intercept_web_search = _web_search_intercept_enabled(ctx.config) and web_search_url is not None
        web_search_tool_entry, remaining_user_tools = _extract_web_search_tool(
            tools_after_sandbox,
            intercept=intercept_web_search,
        )
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
                if url_targets_platform(web_search_url, ctx.config.platform.get("base_url")):
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

        # Inside the try so a rejection releases the budget reservation the
        # request already took, like every other admission failure here.
        await _require_tool_pricing(adapter, ctx, use_sandbox=use_sandbox, use_web_search=use_web_search)
    except HTTPException:
        await release_reservation(ctx)
        raise

    return ToolContext(
        config=ctx.config,
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
            # The workspace's code-exec max_iterations bounds the loop too (no-op when unset).
            sandbox_max_iterations or MAX_TOOL_ITERATIONS_CAP,
        ),
        tools_header=tools_header,
    )


async def _require_tool_pricing(
    adapter: FormatAdapter[Any, Any],
    ctx: RequestContext,
    *,
    use_sandbox: bool,
    use_web_search: bool,
) -> None:
    """Reject a request whose gateway-run tool cannot be billed.

    Same posture ``require_pricing`` already applies to an unpriced model: the
    gateway does not perform work it cannot meter, because a budget cap cannot
    restrain a charge that is never recorded. Checked here, at admission, rather
    than mid-loop, so the caller pays nothing for a request that was never going
    to settle, and next to the missing-URL 400 so both misconfigurations surface
    from the same place.

    MCP tools are deliberately not covered: their names come from a caller-supplied
    server, are unbounded, and are not something an operator can pre-price.

    A budget-exempt key is skipped for the same reason the model gate skips it: the
    gate exists because an unrecorded charge cannot be restrained by a budget, and a
    key that is never debited has no budget to protect. Serving an unpriced model but
    refusing an unpriced tool on the same key would be arbitrary.
    """
    if ctx.db is None or not ctx.config.require_pricing:
        return
    if ctx.reservation is not None and not ctx.reservation.counts_toward_budget:
        return
    tools = [CODE_EXECUTION_TOOL_NAME] if use_sandbox else []
    if use_web_search:
        tools.append(WEB_SEARCH_TOOL_NAME)
    for tool in tools:
        pricing = await find_model_pricing(ctx.db, GATEWAY_TOOL_PRICING_PROVIDER, tool, use_defaults=False)
        if pricing is None:
            key = gateway_tool_pricing_key(tool)
            detail = UNPRICED_TOOL_DETAIL_TEMPLATE.format(tool=tool, key=key)
            logger.warning("Rejecting request: gateway tool '%s' has no pricing under '%s'", tool, key)
            # Logged for the same reason the model gate logs its 402: an operator who
            # turns require_pricing on has to be able to see that live traffic is
            # being dropped, and by what.
            await log_gateway_rejection(
                db=ctx.db,
                log_writer=ctx.log_writer,
                api_key_id=ctx.api_key_id,
                user_id=ctx.user_id,
                model=key,
                provider=GATEWAY_TOOL_PRICING_PROVIDER,
                endpoint=adapter.endpoint,
                detail=detail,
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                started_at=ctx.started_at,
            )
            # Same kind the model gate uses for its own 402, so both no-pricing
            # rejections map to one wire shape per format.
            raise adapter.error(402, detail, ErrorKind.INVALID_REQUEST)


# ---------------------------------------------------------------------------
# Usage logging and reservation settlement
# ---------------------------------------------------------------------------


def _compute_cost(pricing: ModelPricing, usage_data: CompletionUsage) -> float:
    """Compute standalone cost through the threshold-aware meter calculator."""
    cost, _, _ = calculate_metered_cost(pricing, usage_data)
    return cost


def _elapsed_ms(started_at: float | None) -> int | None:
    """Milliseconds elapsed since a monotonic ``started_at`` reading.

    Returns ``None`` when no start was captured (e.g. write paths with no
    meaningful request duration), so the usage log records NULL rather than a
    misleading zero.
    """
    if started_at is None:
        return None
    return round((time.monotonic() - started_at) * 1000)


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
    status_code: int | None = None,
    cost_override: float | None = None,
    latency_ms: int | None = None,
    counts_toward_budget: bool = True,
    attribution: RoutingAttribution | None = None,
    tool_tally: ToolUsageTally | None = None,
) -> float | None:
    """Log API usage to the database and return the computed cost.

    Spend is not written here; the budget reservation reconcile path owns
    ``users.spend``. This returns the cost it computed so the caller can
    reconcile the reservation with the actual amount.

    ``tool_tally`` carries the request's gateway-run tool calls. Their cost is
    added *after* ``cost_override`` and independently of whether the model itself
    resolved pricing, because the two are separate charges: a request against an
    unpriced model can still owe for three searches, and a stream that reported no
    usage still ran the searches it ran.

    The tally is per *request*, not per attempt, so a request that failed over
    through a routing policy records every attempt's tool work on the row that
    settled it. Absorbed rows are written without a tally on purpose
    (:func:`log_absorbed_attempt` passes none): they describe an attempt that did
    not serve, and they never settle a reservation, so a charge placed there would
    be visible on the row and absent from ``users.spend``. One row owning the tool
    ledger also keeps the per-tool breakdown from counting the same search twice.

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
        status_code: HTTP status classifying the failure (see
            ``UsageLog.status_code``), or None when nothing was rejected over HTTP
        cost_override: Fixed amount to record when billing without provider usage
        latency_ms: Total server-side request duration in milliseconds, or None
            when the caller has no meaningful duration to record
        attribution: Which routing policy produced this row and where in its plan,
            or None for a request that named a plain model

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
        status=_row_status(error=error, attribution=attribution),
        error_message=error,
        status_code=status_code,
        latency_ms=latency_ms,
        counts_toward_budget=counts_toward_budget,
        policy_name=attribution.policy_name if attribution else None,
        selection_reason=attribution.selection_reason if attribution else None,
        attempt_position=attribution.position if attribution else None,
        attempt_count=attribution.attempt_count if attribution else None,
        request_group_id=attribution.request_group_id if attribution else None,
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
        usage_log.cache_write_1h_tokens = cache_write_1h_tokens_of(usage_data)

        record_tokens(
            str(provider or ""),
            model,
            usage_data.prompt_tokens,
            usage_data.completion_tokens,
        )

        pricing = await find_model_pricing(db, provider, model, as_of=usage_log.timestamp)
        if pricing:
            cost, meters, breakdown = calculate_metered_cost(pricing, usage_data)
            usage_log.cost = cost
            usage_log.billing_meters = meters
            usage_log.pricing_breakdown = breakdown
        else:
            model_ref = f"{provider}:{model}" if provider else model
            logger.warning("No pricing configured for '%s'. Usage will be tracked without cost.", model_ref)

    # When the caller bills a fixed amount without provider usage (e.g. the
    # stream-missing-usage estimate policy), record that amount on the log row
    # so usage_logs.cost stays consistent with the spend that was reconciled.
    if cost_override is not None:
        usage_log.cost = cost_override

    # Gateway-run tool calls are a separate charge from the model's tokens, so they
    # are folded in last: after the token branch (which may not have run at all) and
    # after cost_override (which replaces the token cost, not the whole bill).
    await _apply_tool_charges(db, usage_log, tool_tally)

    # Emitted once here rather than inside the pricing branch so the cost metric
    # tracks the row's total, including tool charges on an unpriced model. A priced
    # row that happens to cost 0 still reports, matching the previous behavior; only
    # a row with no cost at all (None) is skipped.
    if usage_log.cost is not None:
        record_cost(str(provider or ""), model, usage_log.cost)

    await log_writer.put(usage_log)
    return usage_log.cost


async def _apply_tool_charges(
    db: AsyncSession,
    usage_log: UsageLog,
    tally: ToolUsageTally | None,
) -> None:
    """Fold a request's gateway-run tool calls onto its usage row.

    Writes the counts under the reserved ``tools`` meter namespace, appends one
    charge line per priced tool, and adds their cost to the row's total.

    Never raises. Settlement must not turn an accounting problem into a failed
    response; the precedent is :func:`log_gateway_rejection`. A pricing lookup that
    fails still leaves the counts on the row, so the work stays visible even when
    it could not be priced.
    """
    if tally is None or tally.is_empty():
        return

    tool_meters = tally.meters()
    if tally.overflowed:
        logger.warning(
            "Tool usage tally exceeded %d distinct tool names; the remainder is recorded under '%s'.",
            MAX_TOOL_NAMES,
            OVERFLOW_TOOL_NAME,
        )

    def commit_meters() -> None:
        meters = dict(usage_log.billing_meters or {})
        meters[TOOL_METER_NAMESPACE] = tool_meters
        usage_log.billing_meters = meters

    billable = tally.billable_calls()
    if not billable:
        commit_meters()
        return
    try:
        tool_cost, lines, unpriced = await price_tool_calls(db, billable, as_of=usage_log.timestamp)
    except SQLAlchemyError:
        logger.exception("Failed to price gateway tool calls; counts recorded without cost")
        commit_meters()
        return

    # The rate is stored per row, not just the cost, so per-tool spend stays
    # aggregatable in SQL (the row's own ``cost`` mixes tokens and tools) and a
    # historical row keeps the rate it was billed at after a price change.
    for line in lines:
        tool_name = str(line["meter"]).removesuffix("_calls")
        if tool_name in tool_meters:
            tool_meters[tool_name]["unit_rate"] = float(line["unit_rate"])
    commit_meters()

    if lines:
        usage_log.pricing_breakdown = list(usage_log.pricing_breakdown or []) + lines
    if tool_cost:
        usage_log.cost = (usage_log.cost or 0.0) + tool_cost
    for tool in unpriced:
        logger.warning(
            "Gateway tool '%s' ran %d time(s) but has no pricing; recorded without cost. "
            "Price it with POST /v1/pricing using model_key '%s'.",
            tool,
            billable[tool],
            gateway_tool_pricing_key(tool),
        )


def _handle_counts_toward_budget(reservation: ReservationHandle | None) -> bool:
    """Row-level budget flag for a settled request, derived from its reservation.

    A missing reservation (hybrid, or a path that reserved nothing) is treated as
    counting: only an explicit budget-exempt handle marks the row false.
    """
    return reservation.counts_toward_budget if reservation is not None else True


async def release_reservation(ctx: RequestContext) -> None:
    """Refund the request's budget reservation, if one was taken.

    No-op in hybrid mode and for requests that reserved nothing. Use this
    before raising on any path that rejects the request after
    :func:`resolve_request_context` pre-debited the estimate; otherwise the
    held amount shrinks the user's budget until the next reset (or forever,
    for budgets without a reset period).

    When ``ctx.tool_charge`` is set, the request already ran gateway-run tool calls
    that were written onto its failure row, so the reservation is *reconciled* to
    that amount rather than refunded: a refund releases the hold without recording
    spend, which would leave the charge visible in the activity log and missing from
    the budget it should have consumed.
    """
    if ctx.db is None or ctx.reservation is None:
        return
    if ctx.tool_charge:
        await reconcile_reservation(ctx.db, ctx.reservation, ctx.tool_charge)
        return
    await refund_reservation(ctx.db, ctx.reservation)


def throttle_early_rejection(raw_request: Request, user_id: str) -> bool:
    """Charge a pre-rate-limit refusal to ``user_id``'s bucket, reporting the verdict.

    The user/key mismatch gate is the one rejection that fires *before*
    ``check_rate_limit`` on both request scaffolds (every other gate that logs,
    the allow-list, the budget, an unresolvable selector, sits after it). Logging
    it unconditionally would therefore let a valid key loop mismatched requests
    and append a usage row per request without ever being throttled: DB write
    amplification plus an inflated error count on its own key. Consuming a slot
    here makes that loop self-limiting.

    Returns True when the request is now over the limit, meaning the caller must
    skip the row (the same outcome a throttled request already gets at the gates
    below, which never run). The 429 is deliberately swallowed rather than
    raised: the mismatch keeps answering 403, because which error a client sees
    must not depend on how the gateway chose to record it.
    """
    try:
        check_rate_limit(raw_request, user_id)
    except HTTPException:
        return True
    return False


async def log_gateway_rejection(
    *,
    db: AsyncSession | None,
    log_writer: LogWriter,
    api_key_id: str | None,
    user_id: str | None,
    model: str,
    provider: str | None,
    endpoint: str,
    detail: str,
    status_code: int,
    started_at: float | None,
) -> None:
    """Record a request the gateway itself refused before any provider was called.

    Gateway-side rejections used to raise without writing anything, so an
    operator had no way to see that live traffic was being dropped: the activity
    log showed nothing and the dashboard's failure count read 0 for the duration
    of the incident. The row carries ``status="error"`` (what the count and its
    drill-down read) and no cost, so it makes the drop visible and countable
    without ever moving spend or the budget.

    Callers own the reservation: refund it before calling this, exactly as the
    pre-existing rejection sites do. Nothing here touches the budget.

    ``status_code`` is the status the caller is about to return, and it is
    required rather than optional: a rejection row without one classifies as
    ``unknown`` in the failure taxonomy (``errors_by_status_code``), which is
    indistinguishable from a pre-column historical row. It is always statically
    known here, since these are the gateway's own refusals rather than upstream
    faults, so there is nothing to infer from an exception.

    ``counts_toward_budget`` is always True. The dashboard classifies
    ``counts_toward_budget=False`` rows as imported usage and offers them for
    bulk delete and set-price, which must never happen to a row the gateway
    wrote itself. There is no cost on these rows for the flag to gate, so
    pinning it True is safe even for a key flagged ``exclude_from_budget``.

    Some rejections deliberately stay unlogged, and for three different reasons.
    An authentication failure (401) is refused before any user is known: the
    column is nullable, so a NULL-user row would insert fine, but it could not be
    attributed, acted on, or filtered, and writing one would let an
    unauthenticated caller append to the usage table. ``user_id=None`` is
    therefore a no-op here. A 404 for a user that does not exist is skipped for a
    harder reason: ``usage_logs.user_id`` is a foreign key to ``users``, so that
    row could not be inserted at all. A rate-limit rejection (429) is skipped
    because a throttle is expected, self-limiting behavior rather than dropped
    traffic; the client-driven gates that do log (a selector that no longer
    resolves, a model outside an allow-list) are neither self-limiting nor
    expected, which is why the asymmetry is deliberate.

    Every gate that does log sits behind ``check_rate_limit``, so the rows a
    single key can append are bounded by its user's RPM. The one gate that fires
    earlier, the user/key mismatch, charges the bucket itself through
    :func:`throttle_early_rejection` to keep that bound.

    Writing the row is best-effort. Every caller logs and then re-raises the
    rejection it was already going to return, so an exception escaping here
    would replace a clean 403 or 400 with a 500 and make an unhealthy log writer
    look like a broken gateway to the client. Observability must not change the
    response contract, so a failure is swallowed and reported to the gateway log
    instead. ``SingleLogWriter`` already absorbs ``SQLAlchemyError`` itself, so
    what this catches is the rest (session setup or teardown, a writer whose
    queue is gone). Nothing leaks by dropping the row: every call site refunds
    its reservation before calling this, never after.
    """
    if db is None or user_id is None:
        return
    try:
        await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint=endpoint,
            user_id=user_id,
            error=detail,
            status_code=status_code,
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=True,
        )
    except Exception:
        # Deliberately broad, and deliberately not re-raised: see the docstring.
        # asyncio.CancelledError derives from BaseException, so a cancelled
        # request still unwinds rather than being swallowed here. Logged with the
        # traceback, because a swallowed exception is the only evidence an
        # operator gets that the writer or its session is unhealthy.
        logger.exception("Failed to record gateway rejection for %s on %s", user_id, endpoint)


async def _log_failure_and_refund(
    ctx: RequestContext,
    adapter: FormatAdapter[Any, Any],
    provider: Any,
    model: str,
    error: str,
    status_code: int | None = None,
    attribution: RoutingAttribution | None = None,
    tool_tally: ToolUsageTally | None = None,
) -> None:
    """Record a request-level failure and release its reservation.

    ``attribution`` carries the routing context onto the error row. A failed
    request is precisely when an operator most needs to know which policy was
    involved and how far down its chain the request got, so leaving it off would
    blank out the attribution on the rows that matter most.

    When gateway-run tool calls happened before the failure, their cost is
    reconciled rather than refunded: ``refund_reservation`` releases the hold
    *without* writing spend, which would leave the cost visible on the row and
    absent from ``users.spend``.
    """
    if ctx.db is None:
        return
    cost = await log_usage(
        db=ctx.db,
        log_writer=ctx.log_writer,
        api_key_id=ctx.api_key_id,
        model=model,
        provider=provider,
        endpoint=adapter.endpoint,
        user_id=ctx.user_id,
        error=error,
        status_code=status_code,
        latency_ms=_elapsed_ms(ctx.started_at),
        counts_toward_budget=_handle_counts_toward_budget(ctx.reservation),
        attribution=attribution,
        tool_tally=tool_tally,
    )
    if ctx.reservation is not None:
        if cost:
            await reconcile_reservation(ctx.db, ctx.reservation, cost)
        else:
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
        async with MCPClientPool(tool_ctx.mcp_server_configs, tally=tool_ctx.tally) as pool:
            kwargs = adapter.inject_hints(call_kwargs, pool.purpose_hints(), header=tool_ctx.tools_header)
            return await adapter.run_tool_loop(kwargs, pool, tool_ctx.max_tool_iterations, on_first_response)

    if tool_ctx.use_sandbox:
        assert tool_ctx.sandbox_url is not None  # guaranteed past the missing-URL 400 in prepare_gateway_tools
        sandbox_hint = _resolve_sandbox_purpose_hint(tool_ctx.sandbox_tool_entry, tool_ctx.config)
        async with SandboxBackend(
            sandbox_url=tool_ctx.sandbox_url,
            purpose_hint=sandbox_hint,
            auth_token=tool_ctx.sandbox_auth_token,
            tally=tool_ctx.tally,
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
        config=tool_ctx.config,
        tally=tool_ctx.tally,
    ) as web_backend:
        kwargs = adapter.inject_hints(call_kwargs, web_backend.purpose_hints(), header=tool_ctx.tools_header)
        return await adapter.run_tool_loop(
            kwargs,
            web_backend,
            tool_ctx.max_tool_iterations,
            on_first_response,
            emit_native_web_search=tool_ctx.emit_native_web_search,
        )


async def _lazy_mcp_stream(
    adapter: FormatAdapter[Any, ChunkT],
    kwargs: dict[str, Any],
    configs: list[McpServerConfig],
    tool_ctx: ToolContext,
) -> AsyncIterator[ChunkT]:
    # The MCP pool is entered lazily inside the generator: a dial failure
    # surfaces once the client starts pulling events. Sandbox / web_search use
    # the eager-open path below for a pre-200 HTTP error instead.
    async with MCPClientPool(configs, tally=tool_ctx.tally) as pool:
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
        async for event in adapter.open_tool_loop_stream(
            hinted,
            backend,
            tool_ctx.max_tool_iterations,
            emit_native_web_search=tool_ctx.emit_native_web_search,
        ):
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
        sandbox_hint = _resolve_sandbox_purpose_hint(tool_ctx.sandbox_tool_entry, tool_ctx.config)
        sandbox_backend = SandboxBackend(
            sandbox_url=tool_ctx.sandbox_url,
            purpose_hint=sandbox_hint,
            auth_token=tool_ctx.sandbox_auth_token,
            tally=tool_ctx.tally,
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
        config=tool_ctx.config,
        tally=tool_ctx.tally,
    )
    await web_search_backend.__aenter__()  # may raise WebSearchNotReachableError
    return _eager_backend_stream(adapter, kwargs, web_search_backend, tool_ctx)


# ---------------------------------------------------------------------------
# Streaming settlement (the single copy of the callback bundle)
# ---------------------------------------------------------------------------

# Strong references to in-flight fire-and-forget usage-report tasks. Without
# these, asyncio only holds a weak reference and a scheduled report can be
# garbage collected before it runs; a done-callback discards each task once
# it finishes.
_USAGE_REPORT_TASKS: set[asyncio.Task[None]] = set()


def _schedule_usage_report(coro: Coroutine[Any, Any, None], correlation_id: str) -> None:
    """Run a platform usage report in the background without losing it.

    Keeps the task strongly referenced until it completes and logs a failed
    report instead of letting the exception vanish with the task object.
    """
    task = asyncio.create_task(coro)
    _USAGE_REPORT_TASKS.add(task)

    def _finalize(finished: asyncio.Task[None]) -> None:
        _USAGE_REPORT_TASKS.discard(finished)
        if finished.cancelled():
            return
        exc = finished.exception()
        if exc is not None:
            logger.warning(
                "Background platform usage report failed correlation_id=%s: %s",
                correlation_id,
                exc,
            )

    task.add_done_callback(_finalize)


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
    started_at: float | None = None,
    platform_correlation_id: str | None = None,
    platform_request_id: str | None = None,
    session_label: str | None = None,
    display_model: str | None = None,
    attribution: RoutingAttribution | None = None,
    tool_tally: ToolUsageTally | None = None,
) -> StreamingResponse:
    """Wrap an already-opened upstream stream in an SSE response.

    ``attribution`` is carried onto every usage row these callbacks write, so a
    streamed request through a routing policy is as legible after the fact as a
    non-streamed one. Without it the serving row of a streamed fallover would
    carry no ``request_group_id``, and the absorbed attempt it belongs to would be
    an orphan.

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
            _schedule_usage_report(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
                    outcome="success",
                    usage=usage_data,
                    session_label=session_label,
                    is_final_attempt=True,
                ),
                platform_correlation_id,
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
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=_handle_counts_toward_budget(reservation),
            attribution=attribution,
            tool_tally=tool_tally,
        )
        if reservation is not None:
            await reconcile_reservation(db, reservation, actual_cost or 0.0)

    async def _on_no_usage() -> None:
        # Stream completed but the provider sent no usage data. Report the
        # terminal success upstream in hybrid mode; standalone settles the
        # reservation per stream_missing_usage_policy instead of billing $0.
        if platform_active:
            assert platform_correlation_id is not None
            _schedule_usage_report(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
                    outcome="success",
                    usage=None,
                    session_label=session_label,
                    is_final_attempt=True,
                ),
                platform_correlation_id,
            )
            return
        if db is None or log_writer is None or reservation is None:
            return
        policy = config.stream_missing_usage_policy
        if policy == "allow_free":
            tool_cost = await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint=adapter.endpoint,
                user_id=user_id,
                latency_ms=_elapsed_ms(started_at),
                counts_toward_budget=reservation.counts_toward_budget,
                attribution=attribution,
                tool_tally=tool_tally,
            )
            # "Free" is about the tokens the provider never reported, not about
            # tool calls the gateway definitely ran and owes for.
            if tool_cost:
                await reconcile_reservation(db, reservation, tool_cost)
            else:
                await refund_reservation(db, reservation)
            return
        # 'estimate' and 'fail' both charge the up-front estimate; 'fail' also
        # records the request as errored. status_code stays NULL: the stream
        # itself completed (the caller got a 200), so no HTTP status classifies
        # this, and stamping one would fake a rejection that never happened.
        settled_cost = await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint=adapter.endpoint,
            user_id=user_id,
            error="stream completed without usage data" if policy == "fail" else None,
            cost_override=reservation.estimate,
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=reservation.counts_toward_budget,
            attribution=attribution,
            tool_tally=tool_tally,
        )
        # The estimate covers the unreported tokens; log_usage adds any tool cost on
        # top of it, so reconcile against the row's total rather than the estimate.
        await reconcile_reservation(db, reservation, settled_cost or reservation.estimate)

    async def _on_error(exc: BaseException) -> None:
        if platform_active:
            assert platform_correlation_id is not None
            _schedule_usage_report(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
                    outcome="error",
                    usage=None,
                    session_label=session_label,
                    is_final_attempt=True,
                ),
                platform_correlation_id,
            )
            return
        if db is None or log_writer is None:
            return
        failed_cost = await log_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            endpoint=adapter.endpoint,
            user_id=user_id,
            error=str(exc),
            status_code=failure_status_code(exc),
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=_handle_counts_toward_budget(reservation),
            attribution=attribution,
            tool_tally=tool_tally,
        )
        if reservation is not None:
            # A stream that died after running searches still owes for them, and a
            # refund would release the hold without recording that spend. This is
            # also where the streaming tool-iteration cap lands, since the cap is
            # raised inside the generator.
            if failed_cost:
                await reconcile_reservation(db, reservation, failed_cost)
            else:
                await refund_reservation(db, reservation)

    async def _on_incomplete() -> None:
        # Client disconnected mid-stream: release the reservation.
        #
        # Tool work already done is still owed. Without this, disconnecting after the
        # searches have run is an unlimited supply of unbilled, unrecorded searches,
        # which is the abuse this metering exists to close. A row is written only when
        # there was tool work, so an abandoned stream that ran no tools keeps its
        # existing behavior of leaving no trace.
        if db is None or reservation is None:
            return
        if log_writer is not None and tool_tally is not None and not tool_tally.is_empty():
            abandoned_cost = await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint=adapter.endpoint,
                user_id=user_id,
                error="client disconnected before the stream completed",
                latency_ms=_elapsed_ms(started_at),
                counts_toward_budget=_handle_counts_toward_budget(reservation),
                tool_tally=tool_tally,
            )
            if abandoned_cost:
                await reconcile_reservation(db, reservation, abandoned_cost)
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
            display_model=display_model,
            keepalive_interval_seconds=config.streaming_keepalive_interval_ms / 1000,
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


def stream_final_attempt_extra_seconds(config: GatewayConfig) -> float:
    """Extra first-chunk grace granted only to the sole/final streaming attempt.

    Added on top of the per-attempt failover budget for the terminal attempt,
    which has no next entry in the routing policy to fall over to. Keeps that
    attempt's wait bounded while not converting a slow-but-valid first token into
    a timeout. Mode-agnostic (applies on top of the plain or tool-loop budget).
    """
    return (
        int(
            config.platform.get(
                _STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS_KEY,
                _DEFAULT_STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS,
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
    session_label: str | None = None,
    display_model: str | None = None,
    base_request_fields: dict[str, Any] | None = None,
) -> StreamingResponse:
    """Open a single-attempt stream and wrap it with settlement callbacks.

    Pre-stream failures settle here: gateway-side backend failures map to a
    502 with a backend-specific detail (clearer than a fake provider outage),
    provider failures go through the adapter's error mapping, and in both
    cases any budget reservation is refunded before the error surfaces.

    With a multi-candidate ``ctx.plan``, the *open* is what walks the candidates.
    That is the honest boundary for streaming failover: nothing has been flushed
    to the client yet, so trying the next provider is transparent. Once the
    stream is open, a failure mid-body cannot fall over, because the client has
    already received part of a response from a different model; those errors
    propagate, exactly as they do today.

    Deliberately not included: falling over because the first *chunk* was slow.
    That needs a peek-with-deadline around the body, and the deadline it would
    have to apply is the one that has never applied to standalone streams (see
    the first-chunk regression test). Open-time failover covers the common
    provider blip (connection refused, 429, 5xx on connect) without touching
    that behavior.
    """
    try:
        if ctx.plan is not None and len(ctx.plan.attempts) > 1 and base_request_fields is not None:

            async def _open_candidate(
                attempt: Attempt,
                attempt_kwargs: dict[str, Any],
                mark_locked_in: Callable[[], None],
            ) -> AsyncIterator[ChunkT]:
                if attempt.position > 1:
                    await top_up_reservation_for_attempt(ctx, attempt)
                return await open_stream(adapter=adapter, tool_ctx=tool_ctx, call_kwargs=attempt_kwargs)

            async def _absorbed(attempt: Attempt, exc: BaseException, _total: int) -> None:
                await log_absorbed_attempt(ctx, adapter, attempt, exc)

            # The walk reports which candidate it stopped on, so the failure row
            # names the provider that actually failed rather than the end of the plan.
            stopped_on: list[Attempt] = []

            try:
                chosen, stream = await walk_attempts(
                    attempts=ctx.plan.attempts,
                    base_request_fields=base_request_fields,
                    run_attempt=_open_candidate,
                    max_tool_iterations=tool_ctx.max_tool_iterations,
                    policy_name=ctx.plan.policy_name,
                    build_kwargs=adapter.local_attempt_kwargs,
                    on_absorbed=_absorbed,
                    on_terminal=stopped_on.append,
                )
            except HTTPException as exhausted:
                await log_exhausted_plan(
                    ctx, adapter, exhausted, stopped_on[0] if stopped_on else None, tool_tally=tool_ctx.tally
                )
                raise
            provider, model, display_model = chosen.instance, chosen.model, chosen.display_model
            stream_attribution = _attribution_for(ctx, chosen)
        else:
            stream = await open_stream(adapter=adapter, tool_ctx=tool_ctx, call_kwargs=call_kwargs)
            # A single-candidate policy still names a policy and a reason, and
            # both belong on the row.
            stream_attribution = _attribution_for(ctx, ctx.plan.head) if ctx.plan is not None else None
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
        await _log_failure_and_refund(
            ctx, adapter, provider, model, str(exc), failure_status_code(exc), attribution=_failure_attribution(ctx)
        )
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
        started_at=ctx.started_at,
        platform_correlation_id=platform_correlation_id,
        platform_request_id=platform_request_id,
        session_label=session_label,
        display_model=display_model,
        attribution=stream_attribution,
        tool_tally=tool_ctx.tally,
    )


async def _flush_pending_usage_reports(
    config: GatewayConfig,
    pending_error_reports: list[_PendingUsageReport],
    request_id: str,
    session_label: str | None = None,
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
                    _report_platform_usage(
                        config,
                        report.attempt_id,
                        report.outcome,
                        report.usage,
                        report.error_class,
                        session_label,
                        is_final_attempt=report.is_final_attempt,
                    )
                    for report in pending_error_reports
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
    session_label: str | None = None,
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
    final_attempt_extra = stream_final_attempt_extra_seconds(config)

    backend_stack = AsyncExitStack()
    pool_for_loop: Any = None
    try:
        if tool_ctx.mcp_server_configs:
            pool_for_loop = await backend_stack.enter_async_context(
                MCPClientPool(tool_ctx.mcp_server_configs, tally=tool_ctx.tally)
            )
        elif tool_ctx.use_sandbox:
            assert tool_ctx.sandbox_url is not None  # guaranteed past the missing-URL 400 in prepare_gateway_tools
            sandbox_hint = _resolve_sandbox_purpose_hint(tool_ctx.sandbox_tool_entry, tool_ctx.config)
            pool_for_loop = await backend_stack.enter_async_context(
                SandboxBackend(
                    sandbox_url=tool_ctx.sandbox_url,
                    purpose_hint=sandbox_hint,
                    auth_token=tool_ctx.sandbox_auth_token,
                    tally=tool_ctx.tally,
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
                    config=tool_ctx.config,
                    tally=tool_ctx.tally,
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
        return adapter.open_tool_loop_stream(
            kwargs,
            pool_for_loop,
            tool_ctx.max_tool_iterations,
            emit_native_web_search=tool_ctx.emit_native_web_search,
        )

    # See run_platform_non_stream: BackgroundTasks only run after a successful
    # response, so if every attempt fails before its first chunk the queued
    # reports are dropped with the terminal 502/504. Keep the background task
    # for the success path (it flushes once the SSE response completes), but
    # also stash the error reports so they can be flushed inline on the
    # all-failed path below.
    pending_error_reports: list[_PendingUsageReport] = []

    async def _on_attempt_failed(attempt: ResolvedAttempt, failure: StreamingAttemptFailure) -> None:
        background_tasks.add_task(
            _report_platform_usage,
            config,
            attempt.attempt_id,
            "error",
            None,
            failure.error_class,
            session_label,
            is_final_attempt=failure.is_final_attempt,
        )
        pending_error_reports.append(
            _PendingUsageReport(
                attempt_id=attempt.attempt_id,
                outcome="error",
                usage=None,
                error_class=failure.error_class,
                is_final_attempt=failure.is_final_attempt,
            )
        )
        record_abandoned_attempt(attempt.provider, attempt.model, failure.reason, attempt.position)
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
            final_attempt_extra_seconds=final_attempt_extra,
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
                await _flush_pending_usage_reports(config, pending_error_reports, route.request_id, session_label)
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
        session_label=session_label,
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


def raise_all_streaming_attempts_failed(
    adapter: FormatAdapter[Any, Any],
    exc: Exception,
    route: ResolvedRoute,
) -> NoReturn:
    """Map a terminal :func:`run_streaming_with_fallback` failure (no attempt
    yielded a first chunk) onto the format's wire error.

    Gateway-side backend failures (sandbox / web_search eager-open) get a 502
    with a backend-specific detail so operators don't chase a fake provider
    outage. A single attempt preserves its classified provider error. Once a
    multi-attempt route is exhausted, it surfaces the aggregate result: 504
    when the last failure was a timeout, otherwise 502.
    """
    if isinstance(exc, SandboxNotReachableError):
        logger.error("Sandbox unreachable request_id=%s: %s", route.request_id, exc)
        raise adapter.error(502, SANDBOX_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    if isinstance(exc, WebSearchNotReachableError):
        logger.error("Web search backend unreachable request_id=%s: %s", route.request_id, exc)
        raise adapter.error(502, WEB_SEARCH_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    logger.error("All streaming attempts failed request_id=%s: %s", route.request_id, exc)
    if len(route.attempts) <= 1:
        raise adapter.provider_error(exc) from exc
    kind, _ = upstream_exception_shape(exc)
    if kind == "timeout":
        raise adapter.error(504, ALL_PROVIDERS_TIMED_OUT_DETAIL, ErrorKind.API) from exc
    raise adapter.error(502, ALL_PROVIDERS_FAILED_DETAIL, ErrorKind.API) from exc


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
    session_label: str | None = None,
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
    pending_error_reports: list[_PendingUsageReport] = []

    def _report_attempt_outcome(
        attempt: ResolvedAttempt,
        outcome: str,
        usage: Any,
        error_class: str | None,
        is_final_attempt: bool,
    ) -> None:
        background_tasks.add_task(
            _report_platform_usage,
            config,
            attempt.attempt_id,
            outcome,
            usage,
            error_class,
            session_label,
            is_final_attempt=is_final_attempt,
        )
        if outcome != "success":
            pending_error_reports.append(
                _PendingUsageReport(
                    attempt_id=attempt.attempt_id,
                    outcome=outcome,
                    usage=usage,
                    error_class=error_class,
                    is_final_attempt=is_final_attempt,
                )
            )

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
    except SandboxNotReachableError as exc:
        # The sandbox is part of the gateway's own infra, not the LLM
        # provider; a distinct status stops operators chasing a "provider
        # outage" that is actually the sandbox container being down. 502
        # keeps "upstream dependency failed" semantics. Runs through the
        # inline flush below because the error response drops the queued
        # BackgroundTasks for any earlier attempts' reports.
        logger.error("Sandbox unreachable request_id=%s: %s", route.request_id, exc)
        await _flush_pending_usage_reports(config, pending_error_reports, route.request_id, session_label)
        raise adapter.error(502, SANDBOX_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    except WebSearchNotReachableError as exc:
        logger.error("Web search backend unreachable request_id=%s: %s", route.request_id, exc)
        await _flush_pending_usage_reports(config, pending_error_reports, route.request_id, session_label)
        raise adapter.error(502, WEB_SEARCH_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    except HTTPException:
        # An error response drops the queued BackgroundTasks, so send the
        # per-attempt error reports inline before propagating. The background
        # copies never run on this path, so there is no double-report. This
        # branch only catches HTTPException (what the runner raises on the
        # all-failed path); a CancelledError propagates without doing reporting
        # I/O during teardown.
        await _flush_pending_usage_reports(config, pending_error_reports, route.request_id, session_label)
        raise


def _attribution_for(ctx: RequestContext, attempt: Attempt, *, absorbed: bool = False) -> RoutingAttribution | None:
    """Attribution for a row produced by ``attempt``, or None when unrouted."""
    if ctx.plan is None or ctx.request_group_id is None:
        return None
    return RoutingAttribution(
        policy_name=ctx.plan.policy_name,
        selection_reason=attempt.selection_reason,
        position=attempt.position,
        attempt_count=len(ctx.plan.attempts),
        request_group_id=ctx.request_group_id,
        absorbed=absorbed,
    )


def _failure_attribution(ctx: RequestContext, stopped_on: Attempt | None = None) -> RoutingAttribution | None:
    """Attribution for a request that failed outright.

    Attributed to the candidate the walk actually stopped on, which the walker
    reports through ``on_terminal``. Defaulting to the end of the plan would be
    wrong for every early stop: a 400/401/403/422 or a tool-loop lock-in on the
    first candidate ends the request there, and naming the last candidate would
    blame a provider that was never called, in a row that feeds the by-provider
    breakdown and the error taxonomy.
    """
    if ctx.plan is None:
        return None
    return _attribution_for(ctx, stopped_on or ctx.plan.attempts[-1])


async def log_exhausted_plan(
    ctx: RequestContext,
    adapter: FormatAdapter[Any, Any],
    exc: HTTPException,
    stopped_on: Attempt | None = None,
    tool_tally: ToolUsageTally | None = None,
) -> None:
    """Record the failure of a plan whose every candidate failed.

    The walker maps an exhausted chain to a final ``HTTPException``, which would
    otherwise take the caller's "already mapped, do not log" path and leave the
    request with no usage row at all: a failed request naming a policy would be
    invisible in the activity log, while the same failure on a plain model is
    recorded. This writes the row and deliberately does **not** refund, so the
    caller's existing single refund site stays the only one.

    ``tool_tally`` is what makes an exhausted plan still owe for the searches it
    ran, including the case that cannot fail over at all: once a tool loop has
    produced an assistant message the plan locks to that provider (see
    ``_attempts``), so a failure inside the loop is terminal and this is the only
    row the request gets. The charge is recorded on ``ctx`` for the caller's single
    release site to reconcile.
    """
    if ctx.db is None or ctx.plan is None:
        return
    last = stopped_on or ctx.plan.attempts[-1]
    cost = await log_usage(
        db=ctx.db,
        log_writer=ctx.log_writer,
        api_key_id=ctx.api_key_id,
        model=last.model,
        provider=last.instance,
        endpoint=adapter.endpoint,
        user_id=ctx.user_id,
        error=str(exc.detail),
        status_code=exc.status_code,
        latency_ms=_elapsed_ms(ctx.started_at),
        counts_toward_budget=_handle_counts_toward_budget(ctx.reservation),
        attribution=_failure_attribution(ctx, last),
        tool_tally=tool_tally,
    )
    ctx.tool_charge = cost or 0.0


async def log_absorbed_attempt(
    ctx: RequestContext,
    adapter: FormatAdapter[Any, Any],
    attempt: Attempt,
    exc: BaseException,
) -> None:
    """Record a failed attempt the policy recovered from.

    Written as ``status="absorbed"`` so it is visible in the activity log without
    counting toward any error metric: the request is still going to be served by a
    later candidate, and a working fallback chain must not read as an outage.
    Failures here are swallowed. Losing an audit row is bad; turning a request the
    gateway is about to serve successfully into a 500 because the audit write failed
    is worse.

    Deliberately passes no ``tool_tally``: gateway-run tool calls are billed once,
    on the row that settles the request's reservation, so this row carries the
    attempt's tokens and none of the tool ledger. See :func:`log_usage`.
    """
    if ctx.db is None:
        return
    try:
        await log_usage(
            db=ctx.db,
            log_writer=ctx.log_writer,
            api_key_id=ctx.api_key_id,
            model=attempt.model,
            provider=attempt.instance,
            endpoint=adapter.endpoint,
            user_id=ctx.user_id,
            error=str(exc),
            status_code=failure_status_code(exc),
            latency_ms=_elapsed_ms(ctx.started_at),
            counts_toward_budget=False,
            attribution=_attribution_for(ctx, attempt, absorbed=True),
        )
    except Exception:
        logger.warning(
            "Could not record absorbed attempt %d for policy %s",
            attempt.position,
            ctx.plan.policy_name if ctx.plan else "?",
            exc_info=True,
        )


async def run_standalone_non_stream(
    *,
    adapter: FormatAdapter[ResultT, Any],
    ctx: RequestContext,
    tool_ctx: ToolContext,
    call_kwargs: dict[str, Any],
    response: Response,
    provider: Any,
    model: str,
    display_model: str | None = None,
    base_request_fields: dict[str, Any] | None = None,
) -> ResultT:
    """Standalone-mode non-streaming dispatch with reservation settlement.

    Success applies the rate-limit headers to ``response``, writes the usage
    log (per the adapter's no-usage policy), and reconciles the reservation
    against actual cost; every failure path refunds the reservation before
    mapping the error to the format's wire envelope.

    ``display_model`` (a configured alias) relabels the result's ``model`` field
    before returning, so the underlying provider/model stays hidden; billing and
    logging above still key on the resolved target ``model``/``provider``.

    When ``ctx.plan`` holds more than one candidate (the caller named a routing
    policy with an ``on_failure`` chain) the dispatch walks them, and
    ``provider`` / ``model`` / ``display_model`` are rebound to whichever
    candidate actually served. Settlement below is untouched: it stays the single
    place a reservation is reconciled or refunded, now keyed on the serving
    attempt rather than on the head candidate. A single-candidate plan takes the
    original path unchanged, so a plain model, an alias, and a one-target policy
    are byte-identical here. ``base_request_fields`` is the credential-free
    request payload each candidate's kwargs are built from; without it, only the
    prebuilt ``call_kwargs`` can be dispatched and no fallover is possible.
    """
    try:
        if ctx.plan is not None and len(ctx.plan.attempts) > 1 and base_request_fields is not None:

            async def _run_candidate(
                attempt: Attempt,
                attempt_kwargs: dict[str, Any],
                mark_locked_in: Callable[[], None],
            ) -> ResultT:
                if attempt.position > 1:
                    await top_up_reservation_for_attempt(ctx, attempt)
                return await dispatch_non_stream(
                    adapter=adapter,
                    tool_ctx=tool_ctx,
                    call_kwargs=attempt_kwargs,
                    on_first_response=mark_locked_in,
                )

            async def _absorbed(attempt: Attempt, exc: BaseException, _total: int) -> None:
                await log_absorbed_attempt(ctx, adapter, attempt, exc)

            # The walk reports which candidate it stopped on, so the failure row
            # names the provider that actually failed rather than the end of the plan.
            stopped_on: list[Attempt] = []

            try:
                chosen, result = await walk_attempts(
                    attempts=ctx.plan.attempts,
                    base_request_fields=base_request_fields,
                    run_attempt=_run_candidate,
                    max_tool_iterations=tool_ctx.max_tool_iterations,
                    policy_name=ctx.plan.policy_name,
                    build_kwargs=adapter.local_attempt_kwargs,
                    on_absorbed=_absorbed,
                    on_terminal=stopped_on.append,
                )
            except HTTPException as exhausted:
                await log_exhausted_plan(
                    ctx, adapter, exhausted, stopped_on[0] if stopped_on else None, tool_tally=tool_ctx.tally
                )
                raise
            provider, model, display_model = chosen.instance, chosen.model, chosen.display_model
            attribution = _attribution_for(ctx, chosen)
        else:
            result = await dispatch_non_stream(adapter=adapter, tool_ctx=tool_ctx, call_kwargs=call_kwargs)
            # A single-candidate policy still has a name and a selection reason, and
            # both belong on the row: "served by its default target" is the answer to
            # the same question a fallover answers differently.
            attribution = _attribution_for(ctx, ctx.plan.head) if ctx.plan is not None else None
        if ctx.rate_limit_info:
            for key, value in rate_limit_headers(ctx.rate_limit_info).items():
                response.headers[key] = value
        if ctx.db is not None:
            usage_data = adapter.extract_usage(result)
            actual_cost: float | None = None
            # A request whose provider reported no usage still owes for the tool
            # calls it ran, so a non-empty tally forces the row that
            # ``log_success_without_usage = False`` would otherwise suppress.
            if usage_data is not None or adapter.log_success_without_usage or not tool_ctx.tally.is_empty():
                actual_cost = await log_usage(
                    db=ctx.db,
                    log_writer=ctx.log_writer,
                    api_key_id=ctx.api_key_id,
                    model=model,
                    provider=provider,
                    endpoint=adapter.endpoint,
                    user_id=ctx.user_id,
                    usage_override=usage_data,
                    latency_ms=_elapsed_ms(ctx.started_at),
                    counts_toward_budget=_handle_counts_toward_budget(ctx.reservation),
                    attribution=attribution,
                    tool_tally=tool_ctx.tally,
                )
            if ctx.reservation is not None:
                await reconcile_reservation(ctx.db, ctx.reservation, actual_cost or 0.0)
        if display_model is not None:
            relabel_model(result, display_model)
        return result
    except HTTPException:
        await release_reservation(ctx)
        raise
    except MaxToolIterationsExceeded as e:
        # Gateway-owned cap, not an upstream provider failure. 422 lets
        # callers distinguish a runaway tool loop from a real outage.
        logger.warning("Tool loop iteration cap hit (standalone): cap=%d", tool_ctx.max_tool_iterations)
        await _log_failure_and_refund(
            ctx,
            adapter,
            provider,
            model,
            str(e),
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            attribution=_failure_attribution(ctx),
            tool_tally=tool_ctx.tally,
        )
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
        await _log_failure_and_refund(
            ctx,
            adapter,
            provider,
            model,
            str(e),
            failure_status_code(e),
            attribution=_failure_attribution(ctx),
            tool_tally=tool_ctx.tally,
        )
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise adapter.provider_error(e) from e
