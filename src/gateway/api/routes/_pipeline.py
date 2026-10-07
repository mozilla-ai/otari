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
import contextlib
import hashlib
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, StrEnum, auto
from typing import Any, Generic, Literal, NamedTuple, NoReturn, ParamSpec, Protocol, TypeVar, assert_never
from urllib.parse import ParseResult, urlparse

from any_llm import LLMProvider
from any_llm.exceptions import AnyLLMError, ContextLengthExceededError, InvalidRequestError, UnsupportedParameterError
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CompletionUsage,
)
from any_llm.types.messages import MessagesParams
from any_llm.types.responses import ResponsesParams
from fastapi import BackgroundTasks, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    ToolPorts,
    extract_credential_token,
    get_budget_service,
    verify_api_key_or_master_key,
)
from gateway.api.routes._attempts import AdmitAttempt, CandidateCannotServe, PrepareKwargs, walk_attempts
from gateway.api.routes._helpers import apply_input_guardrails, resolve_user_id
from gateway.api.routes._idempotency import (
    IDEMPOTENCY_KEY_IN_FLIGHT_DETAIL,
    IDEMPOTENCY_KEY_REUSED_DETAIL,
    INVALID_IDEMPOTENCY_KEY_DETAIL,
    IdempotencyGuard,
    IdempotentReplay,
)
from gateway.api.routes._platform import (
    _DEFAULT_STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS,
    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS,
    _DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP,
    _STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS_KEY,
    _STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY,
    _STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY,
    ResolvedAttempt,
    ResolvedRoute,
    SettledCost,
    _classify_upstream_error,
    _report_platform_usage,
    _resolve_platform_credentials,
    get_shared_rejection,
    is_provider_billing_error,
    record_abandoned_attempt,
    run_platform_attempts,
    upstream_error_message,
    upstream_exception_chain,
    upstream_exception_shape,
    upstream_retry_after,
    with_attempt_id,
)
from gateway.api.routes._platform import (
    default_attempt_kwargs as default_attempt_kwargs,  # explicit re-export for the route modules
)
from gateway.api.routes._tools import _build_web_retrieval_backend, _resolve_sandbox_purpose_hint
from gateway.core.config import ATTEMPT_ID_HEADER, END_USER_BUDGET_HEADER, REQUEST_ID_HEADER, GatewayConfig
from gateway.core.database import DATABASE_ERRORS, release_session
from gateway.core.env import otari_env
from gateway.core.error_codes import (
    CONTEXT_LENGTH_EXCEEDED,
    INVALID_MODEL,
    MODEL_NOT_ALLOWED,
    PRICING_REQUIRED,
    UPSTREAM_RATE_LIMITED,
    error_code_of,
    error_headers,
)
from gateway.core.metered_pricing import calculate_metered_cost, quantize_cost
from gateway.core.provider_params import SENSITIVE_PARAM_FIELDS
from gateway.core.unit_of_work import UnitOfWork
from gateway.core.usage import (
    cache_read_tokens_of,
    cache_tokens_in_prompt_of,
    cache_write_1h_tokens_of,
    cache_write_tokens_of,
    provider_latency_ms_of,
    reasoning_tokens_of,
)
from gateway.exceptions import TenancyError
from gateway.exceptions.control_plane_exceptions import ControlPlaneError
from gateway.exceptions.tools_exceptions import (
    CodeExecutionContainerBusyError,
    CodeExecutionDeclarationError,
    CodeExecutionPolicyResolutionFailedError,
    CodeExecutionPolicyResolutionFailure,
    CodeExecutionRefusedError,
    McpServerConfigurationError,
    McpServerDeclarationError,
    McpServerResolutionFailedError,
    WebAccessRefusedError,
    WebSearchInterceptedError,
    WebSearchPolicyResolutionFailedError,
    WebSearchPolicyResolutionFailure,
    WebToolDeclarationError,
    WorkspaceMcpServerNotFoundError,
    WorkspaceWebSearchDomainsExcludedError,
)
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.metrics import REGISTRY, Histogram
from gateway.metrics import Counter as PrometheusCounter
from gateway.model_labeling import relabel_model
from gateway.models.api_keys import APIKey
from gateway.models.guardrails import GuardrailConfig
from gateway.models.mcp import McpServerConfig
from gateway.models.money import to_usd
from gateway.models.pricing import PriceSource
from gateway.models.tools import CodeExecutor, ResolvedCodeExecutionPolicy, WebSearchCredential
from gateway.models.usage import PRICING_REFERENCE_MAX_LENGTH, UsageLog
from gateway.ports.code_execution_policy_port import CodeExecutionPolicyPort, CodeExecutionPolicyScope
from gateway.ports.code_execution_port import CodeExecutionPort
from gateway.ports.mcp_server_port import McpServerPort, McpServerScope
from gateway.ports.model_provider_port import HostedAccessDeniedError, ModelProviderPort
from gateway.ports.web_search_policy_port import WebSearchPolicyPort, WebSearchPolicyScope
from gateway.rate_limit import (
    BudgetMinuteLimits,
    RateLimitGrant,
    RateLimitInfo,
    admit_rate_limit_rules,
    check_rate_limit,
)
from gateway.services.budgets import (
    ZERO,
    BudgetScopeRequest,
    ReservationHandle,
    estimate_cost,
    estimate_tokens,
    get_budget_state,
    increase_reservation,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.code_execution import (
    ContainerLease,
    SandboxContainerRegistry,
)
from gateway.services.files import ProviderFile, SandboxFileBridge, produced_files_for
from gateway.services.guardrails import InProcessGuardrail
from gateway.services.inference import (
    BlockedCaller,
    Claimed,
    InvalidKey,
    KeyReused,
    Replay,
    StillInFlight,
    UnknownCaller,
)
from gateway.services.log_writer import LogWriter
from gateway.services.mcp_client import MCPClientPool
from gateway.services.mcp_loop import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    MAX_TOOL_ITERATIONS_CAP,
    MaxToolIterationsExceeded,
    ToolBackend,
)
from gateway.services.mcp_stateless import failure_class
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.policy_store import resolve_effective_policy
from gateway.services.pricing_service import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    find_model_pricing,
    gateway_tool_pricing_key,
    no_pricing_error_detail,
    price_tool_calls,
    pricing_required_but_missing,
    resolve_model_pricing,
)
from gateway.services.provider_kwargs import (
    ResolvedProvider,
    credential_ladder_exhausted,
    provider_key,
    resolve_provider_selector,
)
from gateway.services.providers import owned_endpoint_http_client
from gateway.services.routing import (
    BudgetState,
    CompiledPlan,
    NoEligibleCandidatesError,
    RoutingSignal,
    compile_catalog_plan,
    compile_policy,
    decide_ordering,
    needs_budget_state,
    selection_consults_router,
)
from gateway.services.sandbox_backend import (
    CODE_EXECUTION_TOOL_NAME,
    DEFAULT_EXEC_TIMEOUT_S,
    SandboxBackend,
    SandboxNotReachableError,
    SandboxSessionGoneError,
    SandboxUnavailableError,
)
from gateway.services.secret_box import SecretBoxUnavailableError, SecretDecryptionError
from gateway.services.tenancy.org_provider_key_service import cached_org_model_restriction
from gateway.services.tenancy.organization_guardrail_runner import handle as guardrail_handle
from gateway.services.tenancy.organization_guardrail_service import (
    ResolvedOrganizationGuardrail,
    resolve_organization_guardrails,
)
from gateway.services.tool_usage import (
    MAX_TOOL_NAMES,
    OVERFLOW_TOOL_NAME,
    TOOL_METER_NAMESPACE,
    ToolUsageTally,
)
from gateway.services.tools import (
    CONTAINER_GONE_DETAIL_TEMPLATE,
    AdmittedCodeExecution,
    DeclaredWebTools,
    Dialect,
    ToolUseBudget,
    admit_code_execution,
    admit_mcp_servers,
    admit_web_access,
    check_web_tools_alone,
    claim_web_declarations,
    declares_code_execution,
    extract_web_tools,
    native_rendering,
    read_web_search_max_uses,
    web_search_intercept_enabled,
)
from gateway.services.upstream_redaction import redact_upstream_message
from gateway.services.web_retrieval_backend import (
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    WebRetrievalBackend,
    WebRetrievalCounter,
    WebSearchNotReachableError,
)
from gateway.services.web_retrieval_policy import DomainPolicy
from gateway.services.workspace_scope import (
    organization_for_workspace_id,
    resolve_workspace_id,
    workspace_for_key_id,
)
from gateway.streaming import (
    StreamFormat,
    StreamingAttemptFailure,
    iterate_streaming_attempts,
    merge_stream_usage,
    streaming_generator,
)
from gateway.types.attempt import Attempt
from gateway.types.normalization_target import NormalizationTarget
from gateway.types.session_principal import SessionPrincipal

ResultT = TypeVar("ResultT")
ChunkT = TypeVar("ChunkT")
BackendT = TypeVar("BackendT")
_P = ParamSpec("_P")

TOKENS = PrometheusCounter(
    "gateway_tokens",
    "Total number of tokens processed",
    ["provider", "model", "type"],
    registry=REGISTRY,
)

REQUEST_COST_DOLLARS = Histogram(
    "gateway_request_cost_dollars",
    "Request cost in USD",
    ["provider", "model"],
    registry=REGISTRY,
)

INLINE_COST_SETTLEMENTS = PrometheusCounter(
    "gateway_inline_cost_settlements",
    "Inline cost settlement outcomes on the inference response path",
    ["outcome"],
    registry=REGISTRY,
)


def record_tokens(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record token usage metrics."""
    if prompt_tokens:
        TOKENS.labels(provider=provider, model=model, type="input").inc(prompt_tokens)
    if completion_tokens:
        TOKENS.labels(provider=provider, model=model, type="output").inc(completion_tokens)


def record_cost(provider: str, model: str, cost: float) -> None:
    """Record request cost."""
    REQUEST_COST_DOLLARS.labels(provider=provider, model=model).observe(cost)


def record_inline_cost_settlement(outcome: str) -> None:
    """Record an attached, unattached, or timed-out inline settlement."""
    INLINE_COST_SETTLEMENTS.labels(outcome=outcome).inc()


# ---------------------------------------------------------------------------
# Shared wire-level detail strings. These are client-visible API contract
# values; do not edit them without a deprecation plan.
# ---------------------------------------------------------------------------
DB_UNAVAILABLE_DETAIL = "Database session unavailable"
API_KEY_VALIDATION_FAILED_DETAIL = "API key validation failed"
API_KEY_NO_USER_DETAIL = "API key has no associated user"
NO_RESOLVABLE_PROVIDER_DETAIL = "Authorization service returned no resolvable provider"
PROVIDER_ERROR_DETAIL = "LLM provider error"
PROVIDER_TIMEOUT_DETAIL = "LLM provider timeout"
# Fixed details for the provider failures that are the gateway's fault rather
# than the caller's. These never embed the upstream message: a rejected
# credential or an exhausted account is where provider internals concentrate,
# and it is not the caller's problem to debug (see classify_provider_error and
# test_error_detail_leakage). The bad-request, model-not-found, and rate-limited
# details below are fallbacks, used only when the provider gave us no message.
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
ALL_PROVIDERS_RATE_LIMITED_DETAIL = "All upstream providers rate-limited this request"
ORGANIZATION_GUARDRAILS_UNRESOLVABLE_DETAIL = "Organization guardrails could not be resolved for this request"
ORGANIZATION_GUARDRAIL_CREDENTIAL_UNREADABLE_DETAIL = (
    "A configured organization guardrail's credential could not be read"
)
# The bound ``ModelProviderPort`` adapter answered with an upstream any-llm has
# no implementation for, so there is nothing to dispatch against. Deliberately
# says nothing about hosted inference or about which adapter answered: that is a
# defect in this build, not something a caller can act on.
HOSTED_CREDENTIAL_UNUSABLE_DETAIL = "No upstream provider is available to serve this model"
SANDBOX_UNREACHABLE_DETAIL = (
    "code_execution sandbox unreachable. Check the sandbox URL in the dashboard's "
    "Tools settings, or OTARI_SANDBOX_URL, and that the container is running."
)
SANDBOX_UNAVAILABLE_DETAIL = "code_execution sandbox temporarily unavailable. Retry later."
WEB_SEARCH_UNREACHABLE_DETAIL = (
    "web_search backend unreachable. Check the search URL in the dashboard's Tools "
    "settings, or OTARI_WEB_SEARCH_URL, and that the backend is running."
)
UNPRICED_TOOL_DETAIL_TEMPLATE = (
    "The gateway tool '{tool}' has no pricing, and this gateway runs with "
    "require_pricing enabled, so it will not run work it cannot bill. Set a "
    "per-request price for model_key '{key}' (POST /api/v1/pricing, or the dashboard's "
    "Tools & Guardrails screen), or set require_pricing to false to serve it unpriced."
)


class ErrorKind(Enum):
    """Coarse error category, for a dialect that names one on the wire.

    The set covers every category a dialect distinguishes, so an error can say what it is.
    """

    API = auto()
    AUTHENTICATION = auto()
    INVALID_REQUEST = auto()
    NOT_FOUND = auto()
    PERMISSION = auto()
    RATE_LIMIT = auto()


# An error flattened into an ``HTTPException`` no longer carries its kind, so a
# status stands in for one here.
_STATUS_ERROR_KINDS = {
    status.HTTP_400_BAD_REQUEST: ErrorKind.INVALID_REQUEST,
    status.HTTP_401_UNAUTHORIZED: ErrorKind.AUTHENTICATION,
    status.HTTP_403_FORBIDDEN: ErrorKind.PERMISSION,
    status.HTTP_404_NOT_FOUND: ErrorKind.NOT_FOUND,
    status.HTTP_429_TOO_MANY_REQUESTS: ErrorKind.RATE_LIMIT,
}


def error_kind_for_status(status_code: int) -> ErrorKind:
    """The kind a bare status implies, falling back to ``API``."""
    return _STATUS_ERROR_KINDS.get(status_code, ErrorKind.API)


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

    Newer any-llm releases use ``UnsupportedParameterError`` for typed provider
    capability checks, while older feature checks still raise
    ``NotImplementedError``. Unwrap ``original_exception`` as well so either
    signal survives the unified-exception wrapper.
    """
    unsupported_types = (NotImplementedError, UnsupportedParameterError)
    return any(isinstance(candidate, unsupported_types) for candidate in upstream_exception_chain(exc))


def _unsupported_feature_detail(exc: BaseException) -> str:
    """Return one actionable reason for a locally rejected request feature."""
    for candidate in upstream_exception_chain(exc):
        if isinstance(candidate, UnsupportedParameterError):
            # AnyLLMError.__str__ prefixes the provider to ``message``. Feeding
            # both through upstream_error_message would repeat the same reason.
            return _redacted_upstream_detail(candidate.message, PROVIDER_BAD_REQUEST_DETAIL)
    return _upstream_message_detail(exc, PROVIDER_BAD_REQUEST_DETAIL)


def _redacted_upstream_detail(message: str, fallback: str) -> str:
    """Return redacted explanatory text, or a fixed detail when none remains."""
    redacted = redact_upstream_message(message)
    explanatory = redacted.replace("[redacted]", "")
    if not any(char.isalpha() for char in explanatory):
        return fallback
    return redacted


_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '([^']+)'")

# The params a caller can actually put in a request body, across the three
# formats that share this classifier. Derived from any-llm's ``*Params`` for the
# same reason the request schemas are (see ``_schema_derive``): a param any-llm
# grows is picked up here without an edit, and nothing else can pass the gate.
#
# Derived the same way means derived *minus* ``SENSITIVE_PARAM_FIELDS``, not
# merely from the same models. ``derive_request_base`` skips those names because
# a future any-llm version could add a credential or provider-selection field to
# a typed ``*Params``, and exposing one as caller-settable would let a request
# override the operator's value (#160). A name the schema refuses to accept is by
# definition a name no caller sent, so it must not pass the caller-fault gate
# either: the two definitions of "settable by a caller" are one definition, and
# spelling it twice is how they drift.
_FORWARDED_PARAMS: frozenset[str] = frozenset(
    (set(CompletionParams.model_fields) | set(MessagesParams.model_fields) | set(ResponsesParams.model_fields))
    - SENSITIVE_PARAM_FIELDS
)


def _provider_rejected_param_detail(param: str) -> str:
    """Detail for a request param the provider serving this model cannot express."""
    return f"The provider serving this model does not accept the '{param}' parameter"


def _rejected_param(exc: BaseException) -> str | None:
    """The request param a provider SDK refused as an unknown keyword, if any.

    any-llm hands a provider's SDK the params its ``*Params`` model declares, so
    a param that model carries but the SDK's method does not take arrives as
    ``TypeError: ...create() got an unexpected keyword argument 'x'``. That is
    every OpenAI-only chat param against a provider that never grew one
    (``seed``, ``n``, the penalties, the logprobs family, against Anthropic).

    It carries no HTTP status, so it would otherwise fall through to the generic
    502 and report a permanent, caller-fixable mismatch between a model and a
    param as an upstream outage. Only the param name is returned: the SDK's
    method name is an internal detail and never reaches the client (see
    ``test_error_detail_leakage``).

    Two gates keep that from blaming a caller for someone else's fault, because
    Python raises this same wording for any bad keyword and this classifier is
    reached from ``except Exception`` arms that wrap more than the provider call:

    * The name has to be one a request body can carry (:data:`_FORWARDED_PARAMS`).
      An operator's ``client_args`` typo (``timeoutt``) and a gateway-internal
      signature drift after an SDK bump (a tool backend's ``image``) are neither,
      so they keep the generic 502 and stay on the error-rate panel as the
      upstream-or-gateway failures they are.
    * The failing callable must not be a constructor. ``client_args`` is
      operator-owned and reaches a provider *client's* ``__init__``, so a key
      there that happens to collide with a real param name (``client_args={"seed":
      1}``) would otherwise pass the first gate and read as the caller's ``seed``.

    Both gates key on the same interpreter wording the match already depends on,
    so neither adds a new assumption about how CPython phrases the error.
    """
    for candidate in upstream_exception_chain(exc):
        if not isinstance(candidate, TypeError):
            continue
        message = str(candidate)
        if "__init__" in message:
            continue
        match = _UNEXPECTED_KWARG.search(message)
        if match and match.group(1) in _FORWARDED_PARAMS:
            return match.group(1)
    return None


def _upstream_message_detail(exc: BaseException, fallback: str) -> str:
    """The detail for a failure the caller can act on.

    Returns the provider's own message, redacted and length-capped, because the
    provider is the only party that knows what it objected to.

    A message the provider put on an exception wins over the joined chain.
    An SDK's string form can repeat that message, add its own labels, or append the response body.
    A wrapper's message is its original's string form, so a message deeper in the chain wins over it.

    Falls back to ``fallback`` when what is left says nothing. Some SDKs
    stringify a failure as bare punctuation or the status code itself, and
    "404" is not an explanation the caller did not already have from the
    status. Requiring one letter is a low bar deliberately: it rejects the
    empty cases without second-guessing a provider that wrote a real sentence.
    A message made entirely of redaction placeholders is empty for this
    purpose, too.
    """
    wrapper_message: str | None = None
    for candidate in upstream_exception_chain(exc):
        message = _carried_message(candidate)
        if message is None:
            continue
        if getattr(candidate, "original_exception", None) is not None:
            wrapper_message = wrapper_message or message
        elif detail := _redacted_upstream_detail(message, ""):
            return detail
    if wrapper_message is not None and (detail := _redacted_upstream_detail(wrapper_message, "")):
        return detail
    return _redacted_upstream_detail(upstream_error_message(exc), fallback)


def _carried_message(exc: BaseException) -> str | None:
    """The message the provider put on one exception, without text the SDK adds, or ``None``."""
    message = getattr(exc, "message", None)
    if not isinstance(message, str):
        response = getattr(exc, "response", None)
        error = response.get("Error") if isinstance(response, Mapping) else None
        message = error.get("Message") if isinstance(error, Mapping) else None
    return message if isinstance(message, str) else None


def classify_provider_error(exc: BaseException) -> ProviderErrorMapping | None:
    """Map an upstream provider exception to a safe, specific (status, detail).

    Returns ``None`` when the failure carries no signal we can safely act on, so
    the caller falls back to its existing generic provider-error response. The
    mapping is intentionally conservative, classifying only the cases a caller
    can act on and leaving everything else (including provider 5xx and
    connection errors) to the generic 502.

    Detail text splits on whether the caller can act on the failure. A rejection
    of the caller's request (400/422/404) and a rate limit (429) pass the
    provider's own message through, redacted and length-capped, because it is
    the only description of what was actually wrong. When the failure is the
    gateway's own (a rejected credential, an exhausted provider account, a 5xx),
    the detail stays a fixed string: those carry no remedy the caller could
    apply, and are where a raw message is most likely to name the operator's
    credentials or topology.

    Timeout detection (including the OpenAI/Anthropic SDKs' own
    ``APITimeoutError``, and a duck-typed fallback for other provider SDKs) is
    shared with the hybrid-mode fallback classifier via
    :func:`upstream_exception_shape`, so both stay in sync.
    """
    kind, status_code = upstream_exception_shape(exc)
    if kind == "timeout":
        return ProviderErrorMapping(status.HTTP_504_GATEWAY_TIMEOUT, PROVIDER_TIMEOUT_DETAIL)
    # any-llm raises UnsupportedParameterError for typed provider-capability
    # checks and still uses NotImplementedError for older feature checks such as
    # context_management/betas (#530). Neither carries an HTTP status, so either
    # would otherwise fall through to a generic 502/500 and tell the caller a
    # guaranteed-permanent failure was transient. The exception type is the
    # whole signal here, and its message names the unsupported feature.
    if _is_unsupported_feature_error(exc):
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _unsupported_feature_detail(exc))
    # The same shape one layer down: a param any-llm forwards that the resolved
    # provider's SDK has no parameter for.
    if (param := _rejected_param(exc)) is not None:
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _provider_rejected_param_detail(param))
    if status_code is None:
        # A status-less InvalidRequestError means the provider (gemini, bedrock,
        # anthropic) rejected the request shape before assigning an HTTP status.
        # Mirror the UnsupportedParameterError / NotImplementedError branches: the
        # exception type is the whole signal, and the provider's own message names
        # what was wrong. Gated on the failure's own status rather than each
        # link's, so a chain that does carry one (401, 429, 5xx) keeps its fixed
        # detail instead of echoing the upstream text as a client 400.
        if any(isinstance(candidate, InvalidRequestError) for candidate in upstream_exception_chain(exc)):
            return ProviderErrorMapping(
                status.HTTP_400_BAD_REQUEST, _upstream_message_detail(exc, PROVIDER_BAD_REQUEST_DETAIL)
            )
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
        return ProviderErrorMapping(
            status.HTTP_400_BAD_REQUEST, _upstream_message_detail(exc, PROVIDER_BAD_REQUEST_DETAIL)
        )
    if status_code == 404:
        return ProviderErrorMapping(
            status.HTTP_404_NOT_FOUND, _upstream_message_detail(exc, PROVIDER_MODEL_NOT_FOUND_DETAIL)
        )
    # A provider rejecting the gateway's credentials is a gateway-config fault,
    # not the caller's: surface it as a 502, never as a client-facing 401/403.
    if status_code in (401, 403):
        return ProviderErrorMapping(status.HTTP_502_BAD_GATEWAY, PROVIDER_CREDENTIALS_DETAIL)
    # A 429 is not the caller's request to fix but is theirs to act on, and only
    # the provider's text names the exhausted quota and the retry window. Still
    # drops the upstream Retry-After: the (status, detail) pair cannot carry it,
    # and on a streaming failure the headers are already flushed.
    if status_code == 429:
        return ProviderErrorMapping(
            status.HTTP_429_TOO_MANY_REQUESTS, _upstream_message_detail(exc, PROVIDER_RATE_LIMITED_DETAIL)
        )
    return None


def provider_error_headers(exc: BaseException, status_code: int) -> dict[str, str] | None:
    """Response headers for a classified provider failure, or ``None``.

    On a 429, ``Otari-Error-Code: upstream_rate_limited`` (so a caller can tell
    the provider's limit from the gateway's own) and the upstream ``Retry-After``,
    the one piece of a provider's rate-limit response that its message body cannot
    always carry. Restricted to the 429: on the statuses that surface as a
    fixed-detail 502 the header would describe the gateway's own upstream
    account, which is not the caller's to read.
    """
    if status_code == status.HTTP_400_BAD_REQUEST and _is_context_length_error(exc):
        return error_headers(CONTEXT_LENGTH_EXCEEDED)
    if status_code != status.HTTP_429_TOO_MANY_REQUESTS:
        return None
    headers = error_headers(UPSTREAM_RATE_LIMITED)
    retry_after = upstream_retry_after(exc)
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return headers


def _is_context_length_error(exc: BaseException) -> bool:
    """Whether any-llm classified the failure as a prompt too long for the model."""
    return any(isinstance(current, ContextLengthExceededError) for current in upstream_exception_chain(exc))


def refusal_code(exc: BaseException) -> str | None:
    """The ``Otari-Error-Code`` an exception ending a request stands for, or None.

    Read by the stream error events, which go out after the headers that would
    otherwise carry it.
    """
    if isinstance(exc, HTTPException):
        return error_code_of(exc.headers)
    mapping = classify_provider_error(exc)
    if mapping is None:
        return None
    return error_code_of(provider_error_headers(exc, mapping.status_code))


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

    name: Dialect
    endpoint: str
    stream_format: StreamFormat

    def error(
        self,
        status_code: int,
        message: str,
        kind: ErrorKind = ErrorKind.API,
        headers: dict[str, str] | None = None,
    ) -> HTTPException:
        """Build the format's wire error for ``status_code`` / ``message``."""
        ...

    def provider_error(self, exc: BaseException) -> HTTPException:
        """Map a single-attempt upstream failure to the format's wire error."""
        ...

    def stream_error_payload(self, exc: BaseException) -> str:
        """Render the SSE error event for a failure after the stream committed.

        The status line is already on the wire by then, so this event is the only
        place the caller learns what kind of failure ended the stream.
        """
        ...

    def format_chunk(self, chunk: ChunkT) -> str: ...

    def extract_stream_usage(self, chunk: ChunkT) -> CompletionUsage | None: ...

    def extract_usage(self, result: ResultT) -> CompletionUsage | None: ...

    def attach_cost(self, value: ResultT | ChunkT, settlement: SettledCost) -> bool:
        """Attach settlement to an existing usage carrier; return whether one existed."""
        ...

    def is_stream_cost_carrier(self, chunk: ChunkT) -> bool:
        """Whether this chunk is the terminal usage object that should carry cost."""
        ...

    # When True (chat, responses) a successful non-streaming call without
    # provider usage data still writes a usage-log row; messages skips the row.
    log_success_without_usage: bool

    async def call_provider(self, kwargs: dict[str, Any]) -> ResultT: ...

    async def open_provider_stream(self, kwargs: dict[str, Any]) -> AsyncIterator[ChunkT]: ...

    def prepare_stream_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        require_usage: bool = False,
    ) -> dict[str, Any]:
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
        native_tools: frozenset[str] = frozenset(),
        use_budget: ToolUseBudget | None = None,
    ) -> ResultT: ...

    def open_tool_loop_stream(
        self,
        kwargs: dict[str, Any],
        pool: ToolBackend,
        max_iterations: int,
        *,
        native_tools: frozenset[str] = frozenset(),
        use_budget: ToolUseBudget | None = None,
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


_DOMAIN_ERROR_KINDS = {
    status.HTTP_403_FORBIDDEN: ErrorKind.PERMISSION,
    status.HTTP_404_NOT_FOUND: ErrorKind.NOT_FOUND,
}


def domain_error(adapter: FormatAdapter[Any, Any], exc: TenancyError) -> HTTPException:
    """``exc`` in the error envelope of ``adapter``'s dialect.

    A 4xx message is written for the caller and is sent as it is.
    A 5xx message describes the deployment, so it goes to the log instead.
    """
    if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        logger.error("Request failed: %s", exc.message)
        return adapter.error(exc.status_code, "Internal server error", ErrorKind.API)
    return adapter.error(
        exc.status_code,
        exc.message,
        _DOMAIN_ERROR_KINDS.get(exc.status_code, ErrorKind.INVALID_REQUEST),
        headers=error_headers(exc.error_code) if exc.error_code is not None else None,
    )


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
        uow: UnitOfWork | None,
        log_writer: LogWriter,
        hybrid_mode: bool,
        route: ResolvedRoute | None,
        user_token: str | None,
        api_key_id: str | None,
        user_id: str | None,
        rate_limit_info: RateLimitInfo | None,
        reservation: ReservationHandle | None,
        started_at: float,
        workspace_id: uuid.UUID | None = None,
        resolved_provider: ResolvedProvider | None = None,
        plan: CompiledPlan | None = None,
        estimate_inputs: "EstimateInputs | None" = None,
        request_group_id: str | None = None,
        organization_id: uuid.UUID | None = None,
        code_execution_policy: ResolvedCodeExecutionPolicy | None = None,
        code_execution_policy_loaded: bool = False,
        request_id: str | None = None,
        rate_limit_grant: RateLimitGrant | None = None,
        end_user_budget_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        # Sent to the client as ``Otari-Request-ID``: the platform's id in hybrid
        # mode, one minted by this gateway in standalone.
        self.request_id = request_id
        self.db = db
        # The Unit of Work the route resolved, over the session the route holds.
        # It is `None` where the request has none, which is every hybrid request.
        self.uow = uow
        self.log_writer = log_writer
        self.hybrid_mode = hybrid_mode
        self.route = route
        self.user_token = user_token
        self.api_key_id = api_key_id
        # The organization whose rate overrides price this request, resolved once
        # in the preamble rather than at each pricing lookup. Resolving per lookup
        # would be cheap for a keyed request (both hops memoize on immutable
        # columns) and not for a master-key one, whose workspace comes from
        # ``default_workspace_id``, which is deliberately never memoized and is
        # two indexed reads every time. A fallover repricing each candidate would
        # pay that per candidate.
        self.organization_id = organization_id
        self.user_id = user_id
        # The caller's request tags, written onto every usage row the request
        # writes once this context exists.
        self.tags = tags
        # Standalone-only, `None` in hybrid mode: the workspace this request
        # bills to, resolved once in the preamble (`resolve_workspace_id`) and
        # reused here for the rare fallback resolve in `resolve_dispatch_provider`,
        # so a request whose gate-check selector was unparseable still gets
        # organization-scoped provider keys on its real dispatch attempt.
        self.workspace_id = workspace_id
        self.rate_limit_info = rate_limit_info
        # What the ``rate_limits`` rules hold for this request, settled where its reservation is.
        self.rate_limit_grant = rate_limit_grant
        self.reservation = reservation
        # USD already written onto a failure row for gateway-run tool calls. A
        # request whose plan is exhausted still owes for the searches it ran, and
        # ``log_exhausted_plan`` writes that onto the row without settling. Recording
        # it here lets the single release site reconcile it instead of refunding,
        # which is what keeps ``users.spend`` matching the row: ``refund_reservation``
        # releases the hold *without* writing spend.
        self.tool_charge: Decimal = Decimal(0)
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
        # Standalone-only: the workspace's code-execution policy, read in the
        # preamble when the request declares code execution on a deployment with
        # a sandbox, so the staging decision and admission read one row once.
        # ``loaded`` tells "no row" apart from "not consulted".
        self.code_execution_policy = code_execution_policy
        self.code_execution_policy_loaded = code_execution_policy_loaded
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
        # Equal to `request_id` when routed; `None` for an unrouted request, whose
        # rows take `request_id` directly because no attribution carries it.
        self.request_group_id = request_group_id
        # The budget of the end user a service key named, which a streamed
        # response has to carry itself (see ``build_streaming_response``).
        self.end_user_budget_id = end_user_budget_id


def _end_user_headers(ctx: RequestContext) -> dict[str, str]:
    """``Otari-End-User-Budget`` for a request that billed an end user on a budget."""
    return {END_USER_BUDGET_HEADER: ctx.end_user_budget_id} if ctx.end_user_budget_id else {}


def scope_prompt_cache_key(request_fields: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
    """Bind a caller-supplied prompt cache key to its authenticated scope.

    Provider prompt caches can be shared by every tenant using one upstream
    account. Including the resolved identity in the routing key prevents two
    callers from deliberately choosing the same provider cache namespace and
    using cached-token counts as an exact-prefix oracle.
    """
    caller_key = request_fields.get("prompt_cache_key")
    if not isinstance(caller_key, str):
        return request_fields

    scope: tuple[str, str] | None = None
    if ctx.hybrid_mode:
        if ctx.route is not None and ctx.route.user_id:
            scope = ("user", ctx.route.user_id)
        elif ctx.route is not None and ctx.route.workspace_id:
            # Older platform peers may omit user_id. The workspace still keeps
            # their cache-routing key out of every other tenant's namespace.
            scope = ("workspace", ctx.route.workspace_id)
    elif ctx.user_id:
        scope = ("user", ctx.user_id)

    if scope is None:
        # Never forward an unscoped caller-controlled key when the peer did not
        # provide an authenticated identity. Provider-side automatic caching
        # still works without the routing hint.
        request_fields.pop("prompt_cache_key", None)
        return request_fields

    payload = json.dumps(
        ["otari-prompt-cache-v1", scope[0], scope[1], caller_key],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    request_fields["prompt_cache_key"] = hashlib.sha256(payload.encode()).hexdigest()
    return request_fields


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
        headers=error_headers(INVALID_MODEL),
    ) from exc


async def resolve_dispatch_provider(
    ctx: RequestContext,
    config: GatewayConfig,
    model_selector: str,
    *,
    adapter: FormatAdapter[Any, Any],
    model_provider: ModelProviderPort,
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

    Whichever of the two produced it, the result then passes through
    :func:`_serve_from_hosted_credential`, which is where ``model_provider`` is
    asked to serve a candidate no stored credential could. That is the last rung
    and never displaces an earlier one; see that function. The caller's own
    endpoint skips it, holding its credential already, and is given the one HTTP
    client allowed to dial it.
    """
    if ctx.resolved_provider is not None:
        return await _prepare_for_dispatch(ctx, ctx.resolved_provider, adapter=adapter, port=model_provider)
    try:
        resolved = resolve_provider_selector(
            config, model_selector, ctx.user_id, workspace_id=ctx.workspace_id, owned_endpoints=True
        )
    except (ValueError, AnyLLMError) as exc:
        # A reservation is already held for this selector, so it is released before the rejection is recorded.
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
            request_id=ctx.request_id,
            tags=ctx.tags,
        )
        _raise_for_unresolvable_model(model_selector, exc)
    return await _prepare_for_dispatch(ctx, resolved, adapter=adapter, port=model_provider)


async def _prepare_for_dispatch(
    ctx: RequestContext,
    resolved: ResolvedProvider,
    *,
    adapter: FormatAdapter[Any, Any],
    port: ModelProviderPort,
) -> ResolvedProvider:
    if resolved.owned_endpoint is not None:
        # The client re-checks and pins the endpoint's address on every request
        # and follows no redirect; see ``services/providers``.
        client_args = {"http_client": owned_endpoint_http_client()}
        return replace(resolved, kwargs={**resolved.kwargs, "client_args": client_args})
    return await _serve_from_hosted_credential(ctx, resolved, adapter=adapter, port=port)


async def _warn_if_hosted_upstream_is_unpriced(
    ctx: RequestContext,
    resolved: ResolvedProvider,
    upstream: str,
) -> None:
    """Record that a re-keyed request is about to settle free, if it is.

    The pricing and budget gates ran in the preamble against the name the caller
    asked for, and settlement reprices on the instance the request actually
    served under. So an overlay returning a ``response_provider`` it has not
    priced settles the request at ``cost=NULL`` and releases the whole hold,
    which is ``require_pricing`` bypassed rather than enforced. Refusing here is
    deliberately not done (see the caller), but the routed-chain hazard this
    resembles at least has a guard, and an operator should not have to infer this
    one from a usage report. The settlement warning alone does not distinguish
    it: an unpriced re-key looks exactly like an ordinary unpriced model there.

    Best-effort, like the rejection log: observability must not be able to fail a
    request that is otherwise about to succeed.
    """
    if ctx.db is None:
        return
    try:
        pricing = await find_model_pricing(
            ctx.db,
            upstream,
            resolved.model,
            organization_id=ctx.organization_id,
        )
    except SQLAlchemyError:
        logger.warning(
            "Could not check pricing for hosted upstream %s:%s; it may settle at no cost",
            upstream,
            resolved.model,
        )
        return
    if pricing is None:
        logger.warning(
            "ModelProviderPort re-keyed %s:%s onto %s:%s, which resolves no pricing: this "
            "request will settle at no cost and release its whole budget hold. Price the "
            "upstreams this build's adapter returns.",
            resolved.instance,
            resolved.model,
            upstream,
            resolved.model,
        )


async def _serve_from_hosted_credential(
    ctx: RequestContext,
    resolved: ResolvedProvider,
    *,
    adapter: FormatAdapter[Any, Any],
    port: ModelProviderPort,
) -> ResolvedProvider:
    """Ask ``ModelProviderPort`` to serve a candidate no stored credential could.

    The last rung of the credential ladder and only the last. An organization's
    own key, a stored provider instance and a ``config.yml`` entry are all
    resolved upstream of here (``services/provider_kwargs.py``), and a candidate
    any of them served comes back untouched, so BYO precedence is unchanged: the
    port is asked when the ladder is exhausted and never before it. The core
    adapter answers ``None`` for every candidate, which is what makes a build
    with no overlay behave exactly as it did, rung for rung.

    On a resolved credential the dispatch is re-keyed onto
    ``response_provider``, the upstream that usage and telemetry name (which the
    port documents as possibly differing from the public name the caller asked
    for), and carries that credential alone: the ladder produced nothing to
    merge with.
    """
    if ctx.organization_id is None:
        # The port keys its access decision on the organization alone, so a
        # request with no organization (no workspace resolved for it) has
        # nothing to ask about.
        return resolved
    if ctx.plan is not None and len(ctx.plan.attempts) > 1:
        # A multi-candidate plan dispatches from ``ctx.plan.attempts``, whose
        # kwargs the routing compiler built; what this function returns is not
        # what those candidates are called with. Asking anyway would meter a
        # resolve that never serves, so a routed chain keeps the ladder's own
        # answer for now. Reaching the port from the compiler is separate work:
        # the compiler is synchronous and documented as doing no I/O.
        return resolved
    if not credential_ladder_exhausted(resolved.provider, resolved.kwargs):
        return resolved

    try:
        credential = await port.resolve_hosted_credential(
            organization_id=ctx.organization_id,
            workspace_id=ctx.workspace_id,
            # The instance rather than the implementation: that is the name
            # pricing, budgets and usage key on, and for a bare selector it is
            # the one the caller wrote. (An alias resolves to its target first,
            # so an aliased request names the target's instance here, never the
            # alias.) ``response_provider`` comes back naming whichever upstream
            # actually serves it.
            provider=resolved.instance,
            model=resolved.model,
        )
    except HostedAccessDeniedError as exc:
        # The port owns this refusal precisely so a caller need not name the
        # adapter that raised it, and the adapter's own wording is internal, so
        # the response carries the same detail an organization-scoped model
        # restriction already uses. A reservation is held by this point, so it
        # is released here exactly as the unresolvable-selector branch above
        # does; nothing outside this function refunds for it.
        logger.info(
            "Hosted inference refused for %s:%s workspace=%s: %s",
            resolved.instance,
            resolved.model,
            exc.workspace_id,
            exc,
        )
        # Named to the caller the way every other ``model_not_allowed_detail``
        # site names it: the selector they wrote. For an alias that is the alias,
        # because keeping its target out of what a caller can see is the whole
        # point of one (``docs/models.md``, "Target-hiding"); the log line above
        # carries the resolved target for the operator.
        denied_detail = model_not_allowed_detail(resolved.alias or f"{resolved.instance}:{resolved.model}")
        await release_reservation(ctx)
        await log_gateway_rejection(
            db=ctx.db,
            log_writer=ctx.log_writer,
            api_key_id=ctx.api_key_id,
            user_id=ctx.user_id,
            model=resolved.model,
            provider=resolved.instance,
            endpoint=adapter.endpoint,
            detail=denied_detail,
            status_code=status.HTTP_403_FORBIDDEN,
            started_at=ctx.started_at,
            request_id=ctx.request_id,
            tags=ctx.tags,
        )
        raise adapter.error(403, denied_detail, ErrorKind.PERMISSION) from exc
    except Exception as exc:
        # Everything an adapter can fail at that is not its own refusal. Resolving
        # a hosted credential is expected to be a network call (that is what an
        # overlay binds this port to do), so a timeout, a connection reset or a
        # database error is ordinary here rather than exotic. Nothing above this
        # would catch one: all three routes call ``resolve_dispatch_provider``
        # outside any ``try`` that takes a bare exception, and ``gateway.main``
        # registers handlers only for ``TenancyError`` and
        # ``RequestValidationError``. Without this the reservation the preamble
        # took stays on ``users.reserved`` until the budget resets, which for a
        # budget with no period is forever.
        #
        # ``Exception`` and not ``BaseException``: ``asyncio.CancelledError`` is a
        # disconnected client rather than a failed lookup, and swallowing it would
        # both mislabel the outcome and mark a cancelled task as handled.
        logger.error(
            "ModelProviderPort failed to resolve a credential for %s:%s: %s",
            resolved.instance,
            resolved.model,
            exc,
            exc_info=True,
        )
        await release_reservation(ctx)
        await log_gateway_rejection(
            db=ctx.db,
            log_writer=ctx.log_writer,
            api_key_id=ctx.api_key_id,
            user_id=ctx.user_id,
            model=resolved.model,
            provider=resolved.instance,
            endpoint=adapter.endpoint,
            detail=HOSTED_CREDENTIAL_UNUSABLE_DETAIL,
            status_code=status.HTTP_502_BAD_GATEWAY,
            started_at=ctx.started_at,
            request_id=ctx.request_id,
            tags=ctx.tags,
        )
        # The same detail the unusable-``response_provider`` branch below returns:
        # from the caller's side both are "this build could not put an upstream
        # behind your request", and which internal step failed is not theirs.
        raise adapter.error(502, HOSTED_CREDENTIAL_UNUSABLE_DETAIL, ErrorKind.API) from exc

    if credential is None:
        # This build has no hosted path for the candidate. The candidate is
        # genuinely unserved, which is what it already was, so it goes to
        # any-llm uncredentialed and fails there exactly as it does today.
        return resolved

    try:
        upstream = LLMProvider(credential.response_provider)
    except ValueError as exc:
        # The adapter named an upstream any-llm does not implement, so nothing
        # can be dispatched. That is a defect in this build rather than caller
        # input: log it in full, refund, and surface a non-leaky 502.
        logger.error(
            "ModelProviderPort returned unknown response_provider %r for %s:%s",
            credential.response_provider,
            resolved.instance,
            resolved.model,
        )
        await release_reservation(ctx)
        # Logged like the 400 and 403 refusals beside it: an operator watching
        # the activity log during an overlay rollout would otherwise see dropped
        # traffic as nothing at all.
        await log_gateway_rejection(
            db=ctx.db,
            log_writer=ctx.log_writer,
            api_key_id=ctx.api_key_id,
            user_id=ctx.user_id,
            model=resolved.model,
            provider=resolved.instance,
            endpoint=adapter.endpoint,
            detail=HOSTED_CREDENTIAL_UNUSABLE_DETAIL,
            status_code=status.HTTP_502_BAD_GATEWAY,
            started_at=ctx.started_at,
            request_id=ctx.request_id,
            tags=ctx.tags,
        )
        raise adapter.error(502, HOSTED_CREDENTIAL_UNUSABLE_DETAIL, ErrorKind.API) from exc

    # Re-keying moves what settlement prices on. The pricing and budget gates
    # ran in the preamble against the name the caller asked for, and settlement
    # reprices on ``instance``, so an overlay owes a pricing row for every
    # ``response_provider`` it returns: without one the row's cost is NULL and
    # the request settles free, which is `require_pricing` bypassed rather than
    # enforced. A routed fallover to a differently priced candidate has an
    # explicit guard for the same hazard (`top_up_reservation_for_attempt`
    # refuses an unpriced candidate with a 402); this path has none, because the
    # substitution is the adapter's decision rather than a plan's and refusing it
    # here would make a build's own fleet unusable until it were priced. An
    # overlay binding this port owns that.
    await _warn_if_hosted_upstream_is_unpriced(ctx, resolved, credential.response_provider)

    kwargs: dict[str, Any] = {"api_key": credential.api_key}
    if credential.api_base is not None:
        kwargs["api_base"] = credential.api_base
    return ResolvedProvider(
        instance=credential.response_provider,
        provider=upstream,
        model=resolved.model,
        kwargs=kwargs,
        alias=resolved.alias,
    )


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
    request_id: str | None = None,
    tags: dict[str, str] | None = None,
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
    # the caller's wall-clock to it. ``GET /usage/requests/{id}`` relies on that
    # NULL to tell this row from the one that settles the request.
    cost = await log_usage(
        db=db,
        log_writer=log_writer,
        api_key_id=api_key_id,
        model=resolved.model,
        provider=resolved.instance,
        provider_type=resolved.provider.value,
        endpoint=endpoint,
        user_id=user_id,
        usage_override=usage,
        counts_toward_budget=counts_toward_budget,
        request_id=request_id,
        tags=tags,
    )
    # Commit the spend directly via an unreserved handle (no held estimate to
    # release): this just adds the actual cost to users.spend. When the request is
    # budget-exempt the handle carries counts_toward_budget=False, so the cost is
    # logged on its own row but never folded into users.spend.
    await reconcile_reservation(
        db,
        ReservationHandle(
            user_id=user_id,
            estimate=ZERO,
            reserved=False,
            strategy=config.budget_strategy,
            counts_toward_budget=counts_toward_budget,
        ),
        cost or Decimal(0),
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
    pricing = await find_model_pricing(
        ctx.db,
        attempt.instance,
        attempt.model,
        organization_id=ctx.organization_id,
    )
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
            headers=error_headers(PRICING_REQUIRED),
        )
    repriced = estimate_cost(
        pricing,
        prompt_chars=ctx.estimate_inputs.prompt_chars,
        max_output_tokens=ctx.estimate_inputs.max_output_tokens,
        default_output_tokens=ctx.estimate_inputs.default_output_tokens,
        cache_write_ttl=ctx.estimate_inputs.cache_write_ttl,
    )
    delta = repriced - ctx.reservation.estimate
    if delta <= ZERO:
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
        raise HTTPException(
            status_code=exc.status_code, detail=budget_exhausted_mid_failover_detail(), headers=exc.headers
        ) from exc


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
    workspace_id: uuid.UUID | None = None,
    request_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> CompiledPlan | None:
    """Compile ``model`` into a plan when it names a policy or a catalog ID with several offerings, else ``None``.

    Budget numbers are fetched only when a condition in the policy actually reads
    them, so a plain failover policy costs no extra query. A policy naming a
    router gets one more step: the backend ranks its candidates for this request
    and the ranking becomes the plan. That step is the only asynchronous part of
    routing, which is why it happens here and not in the compiler.

    An empty plan is a 403 whose caller-facing detail names the policy and nothing
    else (a policy exists partly to keep its targets off the wire); the enumerated
    per-candidate reasons go to the activity log, which is a master-key surface.
    """
    spec = resolve_effective_policy(config, model, user_id, workspace_id=workspace_id)
    if spec is None:
        return compile_catalog_plan(config, model, user_id=user_id, allowlist=allowlist, workspace_id=workspace_id)

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
            workspace_id=workspace_id,
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
            workspace_id=workspace_id,
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
            request_id=request_id,
            tags=tags,
        )
        raise adapter.error(exc.status_code, exc.caller_detail, ErrorKind.PERMISSION) from exc


def _names_end_user(api_key: APIKey | None, user_id_from_request: str | None) -> bool:
    """Whether a request names an end user of its service key: anyone but the key's own user."""
    return bool(
        api_key is not None
        and api_key.is_service_key
        and api_key.user_id
        and user_id_from_request
        and user_id_from_request != api_key.user_id
    )


async def _resolve_keyed_user_id(
    *,
    adapter: FormatAdapter[Any, Any],
    db: AsyncSession,
    uow: UnitOfWork | None,
    log_writer: LogWriter,
    config: GatewayConfig,
    raw_request: Request,
    api_key: APIKey | None,
    api_key_id: str | None,
    is_master_key: bool,
    user_id_from_request: str | None,
    model: str,
    master_key_user_required_detail: str,
    user_forbidden_detail: str,
    started_at: float,
    request_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """The billed user for a key- or master-key-authenticated request, and the budget of an end user it named.

    :func:`resolve_user_id` with this endpoint's error shapes, plus the one
    rejection row it owes. Split out of :func:`resolve_request_context` so the
    two ways into that preamble read as two branches rather than one branch
    wrapped around thirty lines of logging.

    A service key naming anyone but its own user names one of its owner's end
    users, which is found or created here rather than checked as a mismatch, on
    the budget ``Otari-End-User-Budget`` names when it is new.
    """
    if api_key is not None and user_id_from_request and _names_end_user(api_key, user_id_from_request):
        if uow is None:
            raise adapter.error(500, DB_UNAVAILABLE_DETAIL, ErrorKind.API)
        try:
            end_user = await get_budget_service(uow, db).resolve_end_user(
                api_key=api_key,
                external_id=user_id_from_request,
                requested_budget_id=raw_request.headers.get(END_USER_BUDGET_HEADER) or None,
            )
        except TenancyError as exc:
            raise domain_error(adapter, exc) from exc
        return end_user.user_id, end_user.budget_id
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
        # after it: nothing has been resolved this early, and resolving a
        # selector purely to shape a log row is not worth the work on a path
        # that is refusing the request anyway.
        # This gate is the only one that fires before check_rate_limit, so
        # the write is charged to the key's own bucket and skipped once
        # throttled; see throttle_early_rejection. The response stays 403.
        if (
            exc.status_code == status.HTTP_403_FORBIDDEN
            and api_key is not None
            and not await throttle_early_rejection(raw_request, str(api_key.user_id))
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
                request_id=request_id,
                tags=tags,
            )
        raise
    return user_id, None


async def _admit_idempotent(
    guard: IdempotencyGuard, adapter: FormatAdapter[Any, Any], *, user_id: str, api_key_id: str | None
) -> None:
    """Claim the request's ``Idempotency-Key``, before anything is reserved against its budget.

    Raises :class:`IdempotentReplay` when the key already holds this request's
    response, which the route returns as it is.
    """
    match await guard.admit(endpoint=adapter.endpoint, user_id=user_id, api_key_id=api_key_id):
        case None | Claimed():
            return
        case Replay() as replay:
            raise IdempotentReplay(replay)
        case InvalidKey():
            raise adapter.error(400, INVALID_IDEMPOTENCY_KEY_DETAIL, ErrorKind.INVALID_REQUEST)
        case KeyReused():
            raise adapter.error(422, IDEMPOTENCY_KEY_REUSED_DETAIL, ErrorKind.INVALID_REQUEST)
        case StillInFlight():
            raise adapter.error(
                409, IDEMPOTENCY_KEY_IN_FLIGHT_DETAIL, ErrorKind.INVALID_REQUEST, headers={"Retry-After": "5"}
            )
        case UnknownCaller():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User '{user_id}' not found")
        case BlockedCaller():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"User '{user_id}' is blocked")


async def _budget_minute_limits(
    adapter: FormatAdapter[Any, Any],
    uow: UnitOfWork | None,
    db: AsyncSession,
    user_id: str,
    strategy: str | None,
) -> BudgetMinuteLimits | None:
    """The per-minute limits of the user's own budget, admitted alongside ``rate_limits``."""
    if uow is None:
        raise adapter.error(500, DB_UNAVAILABLE_DETAIL, ErrorKind.API)
    return await get_budget_service(uow, db).minute_limits(user_id, strategy=strategy)


async def resolve_request_context(
    *,
    adapter: FormatAdapter[Any, Any],
    raw_request: Request,
    response: Response,
    db: AsyncSession | None,
    uow: UnitOfWork | None,
    config: GatewayConfig,
    log_writer: LogWriter,
    model: str,
    user_id_from_request: str | None,
    estimate_prompt_chars: int,
    estimate_max_output_tokens: int | None,
    master_key_user_required_detail: str,
    user_forbidden_detail: str,
    code_execution_policies: CodeExecutionPolicyPort,
    estimate_cache_write_ttl: Literal["5m", "1h"] | None = None,
    session_principal: SessionPrincipal | None = None,
    routing_signal: Callable[[], RoutingSignal] | None = None,
    normalize_messages: Callable[[NormalizationTarget], Awaitable[tuple[int, CompletionUsage | None]]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    idempotency: IdempotencyGuard | None = None,
    tags: dict[str, str] | None = None,
) -> RequestContext:
    """Run the shared handler preamble up to (and including) budget pre-debit.

    Hybrid mode: extract the caller's bearer token and resolve the routing
    plan against the platform; no local DB state is touched.

    Standalone mode: validate the API key, resolve the billed user, check the
    rate limit, then reserve the estimated cost. The reservation is taken
    before the missing-pricing gate so user/blocked/budget rejections
    (404/403) take precedence over the 402; it is refunded if the request is
    then rejected for missing pricing.

    ``session_principal`` (standalone only) replaces that credential step for a
    route that authenticated and authorized the caller itself, which today is
    the Playground's own completions endpoint. See :class:`SessionPrincipal` for
    what the route owes before it may build one; everything after the step it
    substitutes, the rate limit, the plan compile, both allow-list gates,
    pricing and the reservation, runs exactly as it does for a keyed request.

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
    # Stays None in hybrid mode, where there is no local tenancy to price
    # against: the platform owns rates there and this gateway never reads
    # ``organization_model_pricing``.
    organization_id: uuid.UUID | None = None
    user_id: str | None = None
    # The budget of the end user a service key named, echoed in ``Otari-End-User-Budget``.
    end_user_budget_id: str | None = None
    workspace_id: uuid.UUID | None = None
    code_execution_policy: ResolvedCodeExecutionPolicy | None = None
    code_execution_policy_loaded = False
    rate_limit_info: RateLimitInfo | None = None
    reservation: ReservationHandle | None = None
    resolved_provider: ResolvedProvider | None = None
    plan: CompiledPlan | None = None
    estimate_inputs: EstimateInputs | None = None
    rate_limit_grant: RateLimitGrant | None = None
    request_id: str

    if hybrid_mode:
        # Refuse a policy name before the resolve call rather than after. Hybrid
        # mode sends the caller's selector straight upstream (config aliases are
        # already inert here for the same reason), so a policy name would reach
        # the platform as an unknown model and come back as a confusing upstream
        # 404. What is standalone-only is this gateway's own policies, the ones
        # under `routing.policies` in config.yml. Hybrid mode still routes: the
        # platform owns the decision there, and its resolve response carries the
        # outcome as an ordered ``attempts`` list plus ``fallback_enabled`` for
        # ``run_platform_attempts`` to walk.
        if model in config.policy_names():
            raise adapter.error(400, policy_in_hybrid_mode_detail(model), ErrorKind.INVALID_REQUEST)
        user_token = extract_credential_token(raw_request)
        start_time = time.perf_counter()
        route = await _resolve_platform_credentials(
            config=config,
            user_token=user_token,
            model_selector=model,
        )
        resolve_latency_ms = (time.perf_counter() - start_time) * 1000
        request_id = route.request_id
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "Platform resolve succeeded request_id=%s attempts=%d fallback_enabled=%s resolve_latency_ms=%.2f",
            route.request_id,
            len(route.attempts),
            route.fallback_enabled,
            resolve_latency_ms,
        )
    else:
        request_id = str(uuid.uuid4())
        response.headers[REQUEST_ID_HEADER] = request_id
        if db is None:
            raise adapter.error(500, DB_UNAVAILABLE_DETAIL, ErrorKind.API)
        # No session cookie is consulted *here*, and that is the point: this
        # plane calls a provider with somebody's credentials and writes a usage
        # row against somebody's budget. ``is_master_key`` below sends both
        # through the deployment's *default* workspace, so honoring a cookie
        # here let any signed-in member of any organization spend the default
        # organization's BYO credential and bill it (otari-ai#1880). A
        # completion is authorized by an API key, the deployment's own master
        # key, or a ``SessionPrincipal`` a route built after resolving the
        # caller's own user and proving their membership of the workspace it
        # names, which is the work this rule exists to force rather than a way
        # around it.
        api_key: APIKey | None = None
        is_master_key = False
        names_end_user = False
        if session_principal is not None:
            workspace_id = session_principal.workspace_id
            user_id = session_principal.user_id
            key_allowlist = session_principal.allowed_models
        else:
            api_key, is_master_key = await verify_api_key_or_master_key(raw_request, db, config)
            api_key_id = api_key.id if api_key else None
            # Zero I/O for a keyed request: `api_key.workspace_id` is already an
            # in-memory attribute on the row just loaded. Only a master-key request
            # pays a lookup, which `resolve_workspace_id` itself accepts as
            # operator traffic (see its docstring). Used below to resolve
            # organization-scoped provider keys for a bare provider selector; an
            # instance-addressed one never consults it (`provider_kwargs.py`). A
            # master-key call therefore only ever reaches the *default* workspace's
            # organization's keys, not every organization the deployment holds
            # (`workspace_scope.py`'s docstring).
            workspace_id = await resolve_workspace_id(db, api_key)
            # Limited before the end user is resolved, because resolving one may
            # create it: the limit is what bounds how fast a service key can add
            # end users, and it is its owner's, shared by all of them.
            names_end_user = _names_end_user(api_key, user_id_from_request)
            if names_end_user and api_key is not None:
                rate_limit_info = await check_rate_limit(raw_request, str(api_key.user_id))
            user_id, end_user_budget_id = await _resolve_keyed_user_id(
                adapter=adapter,
                db=db,
                uow=uow,
                log_writer=log_writer,
                config=config,
                raw_request=raw_request,
                api_key=api_key,
                api_key_id=api_key_id,
                is_master_key=is_master_key,
                user_id_from_request=user_id_from_request,
                model=model,
                master_key_user_required_detail=master_key_user_required_detail,
                user_forbidden_detail=user_forbidden_detail,
                started_at=started_at,
                request_id=request_id,
                tags=tags,
            )
            if end_user_budget_id is not None:
                response.headers[END_USER_BUDGET_HEADER] = end_user_budget_id
            # Resolved before the plan rather than with the gate below, because the
            # compiler must drop candidates this caller may not use: a chain that fell
            # over to a forbidden model would be an access-control bypass. The gate
            # itself stays where it was, so a plain model name is unaffected.
            key_allowlist = await resolve_request_allowlist(db, api_key)
        if not names_end_user:
            rate_limit_info = await check_rate_limit(raw_request, user_id)

        # Tolerate an unparseable / unknown-provider selector here: the budget
        # check below and the downstream provider call surface those with
        # their own status codes. A model we can't parse simply has no pricing.
        # Pricing/budget keys on the *instance* name (``instance:model``) while
        # capability detection needs the underlying implementation, so keep both.
        gate_instance: str | None
        gate_impl: LLMProvider | None
        # A policy name, or a catalog ID with several offerings, resolves to a plan.
        # Everything below keys on the plan's head candidate, as it would on a plain model.
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
            workspace_id=workspace_id,
            request_id=request_id,
            tags=tags,
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
                resolved = resolve_provider_selector(
                    config, model, user_id, workspace_id=workspace_id, owned_endpoints=True
                )
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
        # A key with no list of its own inherits its user's default here, and a
        # ``SessionPrincipal`` carries that same user default (it holds no key to
        # narrow it with). (Resolved above, before the plan compile, which needs it.)
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
                request_id=request_id,
                tags=tags,
            )
            raise adapter.error(403, not_allowed_detail, ErrorKind.PERMISSION, headers=error_headers(MODEL_NOT_ALLOWED))

        # Organization-scoped model restriction (otari#643): the org key
        # resolved for this workspace+provider may narrow which models it
        # serves. Applies only when the selector did not name a configured
        # instance, the same condition `provider_kwargs.get_provider_kwargs`
        # uses to decide whether to consult the organization overlay at all;
        # an instance-addressed selector never reaches an organization key,
        # so it is never subject to this restriction either.
        if (
            workspace_id is not None
            and gate_instance is not None
            and gate_impl is not None
            and gate_instance not in config.providers
            and (resolved_provider is None or resolved_provider.owned_endpoint is None)
        ):
            org_allowlist = cached_org_model_restriction(workspace_id, gate_impl.value)
            if org_allowlist is not None and gate_model not in org_allowlist:
                not_allowed_detail = model_not_allowed_detail(model)
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
                    request_id=request_id,
                    tags=tags,
                )
                raise adapter.error(
                    403, not_allowed_detail, ErrorKind.PERMISSION, headers=error_headers(MODEL_NOT_ALLOWED)
                )

        if idempotency is not None and session_principal is None:
            try:
                await _admit_idempotent(idempotency, adapter, user_id=user_id, api_key_id=api_key_id)
            except HTTPException as exc:
                # Only a blocked user is logged: usage_logs.user_id is a foreign key, so an unknown user cannot be.
                if exc.status_code == status.HTTP_403_FORBIDDEN:
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
                        request_id=request_id,
                        tags=tags,
                    )
                raise

        # Derived from the workspace already resolved above, not via
        # `organization_for_key_id` (which would re-derive the same workspace
        # from `api_key_id` internally): a master-key request's workspace
        # lookup is deliberately never memoized, so re-deriving it here would
        # pay that cost twice for one request.
        organization_id = await organization_for_workspace_id(db, workspace_id)
        gate_pricing = await find_model_pricing(
            db,
            gate_instance,
            gate_model,
            organization_id=organization_id,
        )
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
        # The same two figures the estimate is priced from, summed, for the token
        # ceiling's hold. A budget with no token cap ignores it.
        estimated_tokens = estimate_tokens(
            prompt_chars=estimate_inputs.prompt_chars,
            max_output_tokens=estimate_inputs.max_output_tokens,
            default_output_tokens=estimate_inputs.default_output_tokens,
        )
        # A key flagged exclude_from_budget logs its cost and is never reserved, reconciled into users.spend, or gated.
        # A master-key caller has no API key and stays on the enforced path. A request to the caller's own endpoint
        # is exempt too, its owner paying the upstream; a routing plan never resolves one, so no fallover leaves it.
        budget_exempt = (api_key is not None and api_key.exclude_from_budget) or (
            resolved_provider is not None and resolved_provider.owned_endpoint is not None
        )
        # Before the reservation, so a request the rules refuse holds no budget to refund.
        rate_limit_grant = await admit_rate_limit_rules(
            raw_request,
            key_id=api_key.id if api_key is not None else None,
            user_id=user_id,
            estimated_tokens=estimated_tokens,
            budget_limits=None
            if budget_exempt
            else await _budget_minute_limits(adapter, uow, db, user_id, config.budget_strategy),
        )
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
                estimated_tokens=estimated_tokens,
                strategy=config.budget_strategy,
                counts_toward_budget=not budget_exempt,
                # The tenancy-scoped ceilings resolve from the key's workspace and
                # the identity behind it, and from the provider this attempt is
                # about to call. A fallover to a different provider keeps the
                # ceilings resolved here: repricing changes the amount held, not
                # which caps the request was admitted against.
                scope=BudgetScopeRequest(api_key=api_key, provider_instance=gate_instance),
                # Already resolved for the pricing gate above, so the free-model
                # check reads the same rate the estimate was built from.
                organization_id=organization_id,
                # The completion path is the one reserve site with the config
                # object to hand, so it is the one that can honor a deployment's
                # own TTL; the batch, search and pass-through sites take the
                # module default.
                reservation_ttl_sec=config.budget_reservation_ttl_sec,
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
                    request_id=request_id,
                    tags=tags,
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
                request_id=request_id,
                tags=tags,
            )
            raise adapter.error(
                402,
                no_pricing_detail,
                ErrorKind.INVALID_REQUEST,
                headers=error_headers(PRICING_REQUIRED),
            )

        # Resolve uploaded attachments only once the request is authorized
        # (user exists, not blocked, within budget, pricing OK). Done after the
        # budget gate so a blocked/over-budget user can't trigger extraction or
        # vision side-calls. Attachments may expand the prompt (extracted
        # document text, image captions), so top up the reservation to the
        # post-normalization size; the top-up rejects if it no longer fits.
        # Refund on any failure in this setup phase, which the downstream
        # provider-call settlement does not cover.
        # The workspace's code-execution policy, read once here so one decision
        # says whether an attachment is staged for the sandbox and, at admission,
        # who runs the code. Only a request declaring code execution on a
        # deployment with a sandbox has anything to decide. The estimate is
        # already reserved, so a read that fails releases it before propagating.
        if workspace_id is not None and config.sandbox_configured() and declares_code_execution(tools):
            try:
                code_execution_policy = await code_execution_policies.resolve(
                    CodeExecutionPolicyScope(workspace_id=workspace_id, user_token=None)
                )
            except Exception:
                await refund_reservation(db, reservation)
                raise
            code_execution_policy_loaded = True

        if normalize_messages is not None:
            try:
                # The caller's own workspace, not the resolved one: the key's for
                # a keyed request, the session's for a Playground one, and None
                # only for the master key, which has no key and resolves to the
                # default workspace, where narrowing would hide an operator's own
                # file references. The files service reads None as "every
                # workspace", matching the /api/v1/files routes.
                post_chars, vision_usage = await normalize_messages(
                    NormalizationTarget(
                        user_id=user_id,
                        provider=gate_impl,
                        model=gate_model,
                        instance=gate_instance,
                        file_workspace_id=_caller_workspace_id(api_key, session_principal),
                        credential_workspace_id=workspace_id,
                        workspace_executor=(
                            code_execution_policy.executor if code_execution_policy is not None else None
                        ),
                    )
                )
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
                        request_id=request_id,
                        tags=tags,
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
                post_estimated_tokens = estimate_tokens(
                    prompt_chars=estimate_inputs.prompt_chars,
                    max_output_tokens=estimate_inputs.max_output_tokens,
                    default_output_tokens=estimate_inputs.default_output_tokens,
                )
                await increase_reservation(
                    db,
                    reservation,
                    post_estimate - estimate,
                    additional_tokens=post_estimated_tokens - estimated_tokens,
                    model=model,
                    strategy=config.budget_strategy,
                )
            except HTTPException:
                await refund_reservation(db, reservation)
                raise
            except TenancyError as exc:
                await refund_reservation(db, reservation)
                raise domain_error(adapter, exc) from exc
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

    if rate_limit_grant is not None:
        rate_limit_grant.hand_over()
    return RequestContext(
        config=config,
        db=db,
        uow=uow,
        log_writer=log_writer,
        hybrid_mode=hybrid_mode,
        code_execution_policy=code_execution_policy,
        code_execution_policy_loaded=code_execution_policy_loaded,
        route=route,
        user_token=user_token,
        api_key_id=api_key_id,
        user_id=user_id,
        rate_limit_info=rate_limit_info,
        rate_limit_grant=rate_limit_grant,
        reservation=reservation,
        started_at=started_at,
        workspace_id=workspace_id,
        resolved_provider=resolved_provider,
        plan=plan,
        estimate_inputs=estimate_inputs,
        request_group_id=request_id if plan is not None else None,
        organization_id=organization_id,
        request_id=request_id,
        end_user_budget_id=end_user_budget_id,
        tags=tags,
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
        code_execution_port: CodeExecutionPort | None,
        sandbox_exec_timeout_s: int | None = None,
        sandbox_session_image: str | None = None,
        sandbox_allowed_tools: frozenset[str] | None = None,
        code_execution_executor: CodeExecutor | None = None,
        sandbox_container_lease: ContainerLease | None = None,
        sandbox_containers: SandboxContainerRegistry | None = None,
        use_web_search: bool,
        web_search_tool_entry: dict[str, Any] | None,
        web_search_url: str | None,
        web_search_auth_token: str | None,
        web_search_credential: WebSearchCredential | None = None,
        remaining_user_tools: list[dict[str, Any]] | None,
        max_tool_iterations: int,
        tools_header: str | None,
        config: GatewayConfig,
        use_web_fetch: bool = False,
        web_fetch_tool_entry: dict[str, Any] | None = None,
        web_fetch_policy: DomainPolicy | None = None,
        sandbox_files: SandboxFileBridge | None = None,
    ) -> None:
        self.config = config
        self.mcp_server_configs = mcp_server_configs
        self.use_sandbox = use_sandbox
        # The uploads a sandbox session is seeded with, and the store the outputs
        # of Otari's or a provider's sandbox land in. None in hybrid mode and
        # when files are disabled.
        self.sandbox_files = sandbox_files
        self.sandbox_tool_entry = sandbox_tool_entry
        # The adapter that runs this request's code, resolved from the container
        # by the route. None outside a request, where nothing opens a sandbox.
        self.code_execution_port = code_execution_port
        # The execution budget one sandbox call gets. A workspace policy may only
        # lower it, so the deployment's default is the ceiling rather than a value
        # a policy replaces.
        self.sandbox_timeout_s = min(DEFAULT_EXEC_TIMEOUT_S, float(sandbox_exec_timeout_s or DEFAULT_EXEC_TIMEOUT_S))
        # The image the sandbox session is leased against (the workspace's pin,
        # else the deployment's, else nothing) and the tool kinds the backend may
        # expose (None for "whatever it serves"). Both are resolved at admission,
        # where the request's session is live, and read again at dispatch.
        self.sandbox_session_image = sandbox_session_image
        self.sandbox_allowed_tools = sandbox_allowed_tools
        # The lease the request asked to resume, already checked against the
        # caller at admission, and the registry that holds the session past the
        # request. ``container_lease`` is what the request ends up holding, set
        # by the backend when its session opens, and is what the response reports.
        self.sandbox_container_lease = sandbox_container_lease
        self.sandbox_containers = sandbox_containers
        self.container_lease: ContainerLease | None = None
        # Who was decided to run the request's code-execution declaration, when it
        # made one: ``OTARI`` or ``PROVIDER``, never ``AUTO``. ``None`` when the
        # request declared no code execution at all.
        self.code_execution_executor = code_execution_executor
        self.use_web_search = use_web_search
        self.web_search_tool_entry = web_search_tool_entry
        self.web_search_url = web_search_url
        self.web_search_auth_token = web_search_auth_token
        # The workspace's own search key, which its searches use in place of the deployment's.
        self.web_search_credential = web_search_credential
        self.use_web_fetch = use_web_fetch
        self.web_fetch_tool_entry = web_fetch_tool_entry
        self.web_fetch_policy = web_fetch_policy or DomainPolicy()
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
        # Successful Search calls spend the caller's cap across all routing attempts.
        cap = self.max_web_search_uses
        self.use_budget = ToolUseBudget(WEB_SEARCH_TOOL_NAME, cap) if cap is not None else None
        # Search and Fetch share a separate attempted-call cap across routing attempts.
        self.web_retrieval_counter = WebRetrievalCounter()

    def build_sandbox_backend(self) -> SandboxBackend:
        """The one place a ``SandboxBackend`` is constructed for this request.

        Three call sites open one (non-streaming dispatch, the eager-open stream,
        and the fallback-walking stream), and each needs every value resolved at
        admission. Spelling the argument list once is not tidiness: it was
        spelled three times, a fourth field was added to two of them, and the
        third silently kept sending the old session body.
        """
        assert self.code_execution_port is not None  # use_sandbox implies the route resolved one
        return SandboxBackend(
            port=self.code_execution_port,
            purpose_hint=_resolve_sandbox_purpose_hint(self.sandbox_tool_entry, self.config),
            timeout_s=self.sandbox_timeout_s,
            # The tool loop's iteration cap, which is what the backend sizes the
            # session lease from: it bounds rounds rather than calls, so the
            # backend widens it (``_CALLS_PER_ROUND_ALLOWANCE``) rather than
            # taking it as the number of executions a request can make.
            max_executions=self.max_tool_iterations,
            image=self.sandbox_session_image,
            allowed_tools=self.sandbox_allowed_tools,
            tally=self.tally,
            files=self.sandbox_files,
            files_base_url=self.sandbox_files.base_url if self.sandbox_files is not None else None,
            container=self.sandbox_container_lease,
            containers=self.sandbox_containers,
            on_lease=self._note_container_lease,
        )

    def _note_container_lease(self, lease: ContainerLease) -> None:
        self.container_lease = lease

    async def forget_container(self) -> None:
        """Drop the lease this request tried to resume, once the provider said it is gone."""
        if self.sandbox_containers is None or self.sandbox_container_lease is None:
            return
        try:
            await self.sandbox_containers.forget(self.sandbox_container_lease.container_id)
        except Exception:  # noqa: BLE001 - the next resume refuses it anyway
            logger.warning("could not drop gone container %s", self.sandbox_container_lease.container_id, exc_info=True)

    @property
    def tools_extracted(self) -> bool:
        return (
            self.sandbox_tool_entry is not None
            or self.web_search_tool_entry is not None
            or self.web_fetch_tool_entry is not None
        )

    @property
    def web_search_declared_name(self) -> str | None:
        """The ``name`` the caller gave its web-search declaration, if any.

        Used to retarget a forced ``tool_choice`` onto the backend's canonical tool
        name. ``None`` when no web-search entry was extracted or it carried no name.
        """
        name = (self.web_search_tool_entry or {}).get("name")
        return name if isinstance(name, str) and name else None

    @property
    def declared_gateway_tools(self) -> dict[str, dict[str, Any] | None]:
        """Each built-in tool this request runs, by name, with the caller's declaration of it."""
        declared = {
            WEB_SEARCH_TOOL_NAME: (self.web_search_tool_entry, self.use_web_search),
            WEB_FETCH_TOOL_NAME: (self.web_fetch_tool_entry, self.use_web_fetch),
            CODE_EXECUTION_TOOL_NAME: (self.sandbox_tool_entry, self.use_sandbox),
        }
        return {name: entry for name, (entry, in_use) in declared.items() if in_use}

    def native_tools(self, dialect: Dialect) -> frozenset[str]:
        """The built-in tools this request announces in ``dialect``'s own server-tool vocabulary.

        A caller who declared a tool in a provider's words is owed that provider's
        items back, and each tool's registry entry decides whether its declaration
        asks for them.
        """
        return frozenset(
            name
            for name, entry in self.declared_gateway_tools.items()
            if (rendering := native_rendering(name, dialect)) is not None and rendering.declared(entry)
        )

    @property
    def max_web_search_uses(self) -> int | None:
        """The web-search use cap, when the caller supplied one.

        Not gated on :meth:`native_tools`: the cap bounds what the request is billed
        for, so it is honored on every declaration shape and in every wire format.
        Only the *refusal* is format-specific, an Anthropic
        ``max_uses_exceeded`` result block where the caller can read one and a plain
        tool error everywhere else.

        ``0`` is a cap of zero searches, not the absence of one. Reading it as
        "uncapped" would answer a spend limit with unlimited spend, which is the one
        direction this must not fail; a caller who meant "do not search" is better
        served by every search being refused than by a bill.

        Invalid values are rejected rather than treated as uncapped, so a malformed
        spend control cannot fail open. Rejection belongs to ``prepare_gateway_tools``,
        which is the one place that builds a ``ToolContext`` and the only one holding
        the ``adapter`` that knows the caller's error envelope. This property is a
        plain read of a value already validated there; the ``ValueError`` it would
        raise on an unvalidated entry is a programming error, not a request one.
        """
        return read_web_search_max_uses(self.web_search_tool_entry)

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
        return web_search_intercept_enabled(self.config) and self.config.web_search_configured()

    @property
    def use_tool_loop(self) -> bool:
        return bool(self.mcp_server_configs) or self.use_sandbox or self.use_web_search or self.use_web_fetch

    def build_web_retrieval_backend(self) -> WebRetrievalBackend:
        """Build the request's shared Search and Fetch backend."""
        return _build_web_retrieval_backend(
            base_url=self.web_search_url,
            search_tool_entry=self.web_search_tool_entry,
            fetch_tool_entry=self.web_fetch_tool_entry,
            fetch_policy=self.web_fetch_policy,
            counter=self.web_retrieval_counter,
            auth_token=self.web_search_auth_token,
            credential=self.web_search_credential,
            config=self.config,
            tally=self.tally,
        )


def _overlay_mandate(merged: dict[str, GuardrailConfig], mandated: Iterable[GuardrailConfig]) -> None:
    """Fold one mandated layer over the effective guardrail set, in place.

    Union by profile, with the stricter setting winning on every axis: a layer
    below may *add* guardrails and may tighten one, but can never weaken what a
    layer above mandated. `block` beats `monitor` for both `mode` and
    `on_unavailable`, since each is a choice between enforcing and observing.

    The mandating entry also owns the URL and the validate kwargs for a profile
    it names, so a caller cannot point a mandated check at a service of their
    choosing.
    """
    for guardrail in mandated:
        below = merged.get(guardrail.profile)
        if below is None:
            merged[guardrail.profile] = guardrail
            continue
        merged[guardrail.profile] = guardrail.model_copy(
            update={
                "mode": "block" if "block" in (guardrail.mode, below.mode) else "monitor",
                "on_unavailable": (
                    "block" if "block" in (guardrail.on_unavailable, below.on_unavailable) else "monitor"
                ),
            }
        )


@dataclass(frozen=True)
class EffectiveGuardrails:
    """The guardrails a request runs, and what the runner needs to know about them.

    Four fields rather than a list, because three of the four answer questions
    the list cannot: which entries carry a credential, which came from a layer
    the caller does not control, and which are run by this process rather than
    sent anywhere. The second decides how a URL that fails its safety check is
    reported, so it has to survive the merge rather than be re-derived from a
    config the merge has already flattened.

    ``in_process`` maps a profile to the definition that serves it, and is its
    own field for the reason ``mandated`` is: after the merge such an entry's
    ``url`` is ``None``, which is exactly what a remote entry falling back to the
    deployment's guardrails service looks like.
    """

    configs: list[GuardrailConfig] | None
    credentials: dict[str, str]
    mandated: frozenset[str]
    in_process: dict[str, uuid.UUID]


def merge_guardrail_layers(
    ctx: RequestContext,
    requested: list[GuardrailConfig] | None,
    organization: Sequence[ResolvedOrganizationGuardrail],
) -> EffectiveGuardrails:
    """The effective guardrails for this request, and the credentials they need.

    Three layers fold in one order, each able to add a check or tighten one and
    none able to weaken what is already there: the caller's own request, then
    what the caller's organization mandates for this workspace (otari#654), then
    what the deployment's routing policy mandates. The operator's layer is last
    because it is the outermost one: where a policy and an organization name the
    same profile, the operator's entry owns the endpoint the check is sent to.

    That last point is also why a profile the policy layer claims loses its
    organization credential here, and its definition with it. The credential was
    stored for the endpoint the organization named; once the policy's URL has
    replaced it, sending the secret on would be sending it somewhere it was
    never meant for. A definition goes the same way for the same reason: an
    operator who named a URL meant the check to go there, not to be answered
    here.

    Returns the caller's own list unchanged, `None` included, when no layer
    mandated anything, alongside an empty credential map, an empty mandated set
    and no in-process entries: that is the shape `apply_input_guardrails` treats
    as "no guardrails ran", and it is what keeps a deployment that configures
    nothing behaving exactly as it did.
    """
    policy = ctx.plan.guardrails if ctx.plan is not None else []
    if not organization and not policy:
        return EffectiveGuardrails(requested, {}, frozenset(), {})

    # Caller entries first, so a mandating layer of the same profile overwrites them.
    merged: dict[str, GuardrailConfig] = {guardrail.profile: guardrail for guardrail in requested or []}
    credentials: dict[str, str] = {}
    mandated: set[str] = set()
    in_process: dict[str, uuid.UUID] = {}
    for entry in organization:
        _overlay_mandate(merged, (entry.config,))
        mandated.add(entry.config.profile)
        if entry.credential:
            credentials[entry.config.profile] = entry.credential
        if entry.definition_id is not None:
            in_process[entry.config.profile] = entry.definition_id
    if policy:
        _overlay_mandate(merged, policy)
        for guardrail in policy:
            mandated.add(guardrail.profile)
            credentials.pop(guardrail.profile, None)
            in_process.pop(guardrail.profile, None)
    return EffectiveGuardrails(list(merged.values()), credentials, frozenset(mandated), in_process)


def _in_process_guardrails(ctx: RequestContext, effective: EffectiveGuardrails) -> dict[str, InProcessGuardrail | None]:
    """The guardrails this worker already holds for the profiles the merge marked.

    One dictionary lookup per profile, with nothing awaited and nothing built:
    the runner holds what it holds, and a definition it does not is ``None``
    here. That value is what makes the profile unevaluable rather than a check
    the deployment's guardrails service is asked for, which it has never heard
    of.

    An empty map where the organization is unknown is the true answer rather
    than a fallback: an in-process entry can only come from
    :func:`_resolve_organization_guardrails`, which refuses the request before
    resolving anything when the organization is missing.
    """
    organization_id = ctx.organization_id
    if organization_id is None:
        return {}
    return {
        profile: guardrail_handle(organization_id, definition_id)
        for profile, definition_id in effective.in_process.items()
    }


async def _resolve_organization_guardrails(
    adapter: FormatAdapter[Any, Any], ctx: RequestContext
) -> list[ResolvedOrganizationGuardrail]:
    """The guardrails the request's organization mandates for its workspace.

    Standalone only. Hybrid mode's tenancy lives on the platform, which has no
    guardrail resolve endpoint of its own (its guardrail enforcement was
    reachable only through its own completion route), so a hybrid request is
    checked exactly as it was before this plane existed.

    One read per request rather than a cached overlay, per the seam #655 settled
    and #678 wrote down. Unlike the MCP and code-execution resolves beside it,
    this one is unconditional: those run only when a request opts into the
    feature, and a mandate that only ran when the caller asked for it would not
    be a mandate. The cost is one indexed query on a table an organization edits
    by hand.

    All three of the standalone preconditions fail closed together, and
    ``organization_id`` belongs with the other two rather than beside them.
    ``workspace.organization_id`` is not nullable, so
    ``organization_for_workspace_id`` answers ``None`` only when the workspace
    row itself is missing; such a request still carries a non-``None``
    ``ctx.workspace_id``, so treating that case as "no organization plane to
    consult" would skip every mandate silently on the one input that proves the
    tenancy could not be resolved. All three are invariants today, which is why
    they refuse rather than fall through: what this guards is an *enforcement*
    decision, and the day one of them stops holding is the day a request its
    organization requires a blocking guardrail on would otherwise be served
    unchecked.
    """
    if ctx.hybrid_mode:
        return []
    if ctx.db is None or ctx.workspace_id is None or ctx.organization_id is None:
        raise adapter.error(500, ORGANIZATION_GUARDRAILS_UNRESOLVABLE_DETAIL, ErrorKind.API)
    try:
        return await resolve_organization_guardrails(
            ctx.db, organization_id=ctx.organization_id, workspace_id=ctx.workspace_id
        )
    except (SecretBoxUnavailableError, SecretDecryptionError) as exc:
        # The operator's problem, not the caller's, and the underlying message
        # names the environment variable, so it stays in the log.
        logger.error(
            "Organization guardrail credential could not be decrypted for organization %s: %s",
            ctx.organization_id,
            exc,
        )
        raise adapter.error(500, ORGANIZATION_GUARDRAIL_CREDENTIAL_UNREADABLE_DETAIL, ErrorKind.API) from exc


def _policy_failure_status(reason: WebSearchPolicyResolutionFailure) -> int:
    """The HTTP status a failed web search policy resolution renders as."""
    match reason:
        case WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE:
            return status.HTTP_502_BAD_GATEWAY
        case WebSearchPolicyResolutionFailure.NO_CALLER_CREDENTIAL | WebSearchPolicyResolutionFailure.NO_WORKSPACE:
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        case WebSearchPolicyResolutionFailure.STORED_POLICY_INVALID:
            return status.HTTP_503_SERVICE_UNAVAILABLE
        case _:
            assert_never(reason)


@dataclass(frozen=True, kw_only=True)
class DeclaredTools:
    """The tools, servers and guardrails one request declared, in terms every wire format shares."""

    code_execution_header: str | None = None
    container_id: str | None = None
    guardrail_text: str
    guardrails: list[GuardrailConfig] | None
    max_tool_iterations: int | None
    mcp_server_ids: list[uuid.UUID] | None
    mcp_servers: list[McpServerConfig] | None
    tools: list[dict[str, Any]] | None
    tools_header: str | None
    web_search_header: str | None = None


@dataclass(frozen=True, kw_only=True)
class ToolBackends:
    """What runs the tools a request may use."""

    ports: ToolPorts
    sandbox_containers: SandboxContainerRegistry | None = None
    sandbox_files: SandboxFileBridge | None = None


async def prepare_gateway_tools(
    *,
    adapter: FormatAdapter[Any, Any],
    ctx: RequestContext,
    response: Response,
    declared: DeclaredTools,
    backends: ToolBackends,
) -> ToolContext:
    """Admit the gateway-run tools one request declared, handling the reservation release on a refusal.

    The steps run in a fixed order, so a request that breaks two rules always gets the same refusal.
    MCP servers, the sandbox and the web tools cannot be combined in one request.
    Every backend URL comes from the deployment and never from the request.

    NOTE: callers must not release the budget reservation after a refusal from here.
    """
    try:
        claim_web_search = _admit_web_declarations(adapter, ctx, declared)
        await _admit_guardrails(adapter, ctx, response, declared)
        mcp_servers = await _admit_mcp_servers(adapter, ctx, declared, backends.ports.mcp_server)
        code = await _admit_code_execution(adapter, ctx, declared, backends, mcp_servers_declared=bool(mcp_servers))
        web = _extract_web_tools(
            adapter, ctx, code, claim_web_search=claim_web_search, mcp_servers_declared=bool(mcp_servers)
        )
        web_access = await _admit_web_access(adapter, ctx, web, backends.ports.web_search_policy)
        await _require_tool_pricing(
            adapter,
            ctx,
            use_sandbox=code.use_sandbox,
            use_web_search=web.search_tool_entry is not None,
            use_web_fetch=web.fetch_tool_entry is not None,
        )
    except ControlPlaneError as exc:
        # A peer's refusal of the web search, code execution or MCP resolve is a domain error,
        # so without this it would answer in `main.py`'s shape rather than this route's.
        # Not `domain_error`: that one is right for the tenancy family it is named for, and
        # would drop a 429's `Retry-After` and read the status as an invalid request. Both are
        # the peer's answer, which `_control_plane_error_handler` keeps whole for the same reason.
        await release_reservation(ctx)
        retry_after = getattr(exc, "retry_after", None)
        raise adapter.error(
            exc.status_code,
            exc.message,
            error_kind_for_status(exc.status_code),
            {"Retry-After": retry_after} if retry_after else None,
        ) from exc
    except TenancyError as exc:
        # The admission steps render their own refusals; this keeps the hold from leaking
        # should a new one escape them. After `ControlPlaneError`, which is a `TenancyError`.
        await release_reservation(ctx)
        raise domain_error(adapter, exc) from exc
    except HTTPException:
        await release_reservation(ctx)
        raise
    except DATABASE_ERRORS:
        # A database failure is not an HTTPException, so the reservation is released here too.
        # The rollback comes first, because a failed statement leaves the session unusable for the release.
        if ctx.db is not None:
            with contextlib.suppress(*DATABASE_ERRORS):
                await ctx.db.rollback()
        with contextlib.suppress(*DATABASE_ERRORS):
            await release_reservation(ctx)
        raise

    # Gotcha: the connection must not be held across the provider call that follows.
    await release_session(ctx.db)

    return ToolContext(
        config=ctx.config,
        mcp_server_configs=mcp_servers,
        use_sandbox=code.use_sandbox,
        sandbox_tool_entry=code.tool_entry,
        code_execution_port=backends.ports.code_execution,
        sandbox_exec_timeout_s=code.exec_timeout_s,
        sandbox_session_image=code.session_image,
        sandbox_allowed_tools=code.allowed_tools,
        code_execution_executor=code.executor,
        sandbox_container_lease=code.container_lease,
        sandbox_containers=code.containers,
        use_web_search=web.search_tool_entry is not None,
        web_search_tool_entry=web_access.search_tool_entry,
        web_search_url=web_access.search_url,
        web_search_auth_token=web_access.search_auth_token,
        web_search_credential=web_access.search_credential,
        use_web_fetch=web.fetch_tool_entry is not None,
        web_fetch_tool_entry=web.fetch_tool_entry,
        web_fetch_policy=web_access.fetch_policy,
        remaining_user_tools=web.remaining_user_tools,
        max_tool_iterations=min(
            declared.max_tool_iterations or DEFAULT_MAX_TOOL_ITERATIONS,
            MAX_TOOL_ITERATIONS_CAP,
            # The workspace's code-exec max_iterations bounds the loop too (no-op when unset).
            code.max_iterations or MAX_TOOL_ITERATIONS_CAP,
        ),
        tools_header=declared.tools_header,
        sandbox_files=backends.sandbox_files,
    )


async def _admit_guardrails(
    adapter: FormatAdapter[Any, Any], ctx: RequestContext, response: Response, declared: DeclaredTools
) -> None:
    """Run the request's input guardrails, with its organization's and its policy's merged in.

    A ``block`` flag refuses the request, and a ``monitor`` flag annotates ``response``.
    """
    # Merged here rather than in each route, so no completion endpoint can skip a mandate.
    effective = merge_guardrail_layers(ctx, declared.guardrails, await _resolve_organization_guardrails(adapter, ctx))
    await apply_input_guardrails(
        effective.configs,
        declared.guardrail_text,
        response=response,
        config=ctx.config,
        credentials=effective.credentials,
        mandated=effective.mandated,
        in_process=_in_process_guardrails(ctx, effective),
    )


async def _admit_mcp_servers(
    adapter: FormatAdapter[Any, Any], ctx: RequestContext, declared: DeclaredTools, port: McpServerPort
) -> list[McpServerConfig] | None:
    """The MCP servers the request may reach: its own, then the stored ones its IDs name."""
    scope = McpServerScope(workspace_id=ctx.workspace_id, user_token=ctx.user_token)
    try:
        return await admit_mcp_servers(declared.mcp_servers, declared.mcp_server_ids, port=port, scope=scope)
    except (McpServerDeclarationError, WorkspaceMcpServerNotFoundError) as exc:
        raise domain_error(adapter, exc) from exc
    except (McpServerResolutionFailedError, McpServerConfigurationError) as exc:
        # Not `domain_error`, which would hide a 5xx's fixed message behind a generic one.
        raise adapter.error(exc.status_code, exc.message, ErrorKind.API) from exc


def _admit_web_declarations(adapter: FormatAdapter[Any, Any], ctx: RequestContext, declared: DeclaredTools) -> bool:
    """Refuse an ambiguous managed web declaration, and say whether the gateway claims a provider's search keyword."""
    try:
        return claim_web_declarations(
            declared.tools,
            web_search_header=declared.web_search_header,
            config=ctx.config,
            providers=_candidate_provider_names(ctx),
            dialect=adapter.name,
        )
    except (WebToolDeclarationError, WebSearchInterceptedError) as exc:
        raise domain_error(adapter, exc) from exc


def _extract_web_tools(
    adapter: FormatAdapter[Any, Any],
    ctx: RequestContext,
    code: AdmittedCodeExecution,
    *,
    claim_web_search: bool,
    mcp_servers_declared: bool,
) -> DeclaredWebTools:
    """Take the managed web tools out of what code execution left, refusing one this deployment cannot serve."""
    try:
        web = extract_web_tools(code.tools_after_sandbox, config=ctx.config, claim_web_search=claim_web_search)
        check_web_tools_alone(web, use_sandbox=code.use_sandbox, mcp_servers_declared=mcp_servers_declared)
    except WebToolDeclarationError as exc:
        raise domain_error(adapter, exc) from exc
    return web


async def _admit_code_execution(
    adapter: FormatAdapter[Any, Any],
    ctx: RequestContext,
    declared: DeclaredTools,
    backends: ToolBackends,
    *,
    mcp_servers_declared: bool,
) -> AdmittedCodeExecution:
    """Decide who runs the request's code, and the sandbox it runs in when the gateway does."""
    try:
        return await admit_code_execution(
            declared.tools,
            container_id=declared.container_id,
            code_execution_header=declared.code_execution_header,
            config=ctx.config,
            dispatch_provider=_dispatch_provider_name(ctx),
            dialect=adapter.name,
            resolve_policy=lambda: _resolve_code_execution_policy(adapter, ctx, backends.ports.code_execution_policy),
            container_registry=backends.sandbox_containers,
            mcp_servers_declared=mcp_servers_declared,
        )
    except (CodeExecutionDeclarationError, CodeExecutionRefusedError, CodeExecutionContainerBusyError) as exc:
        raise domain_error(adapter, exc) from exc


@dataclass(frozen=True)
class _AdmittedWebAccess:
    """The web access a request may use, and where its searches go."""

    fetch_policy: DomainPolicy
    search_auth_token: str | None
    search_tool_entry: dict[str, Any] | None
    search_url: str | None
    search_credential: WebSearchCredential | None = None


async def _admit_web_access(
    adapter: FormatAdapter[Any, Any], ctx: RequestContext, web: DeclaredWebTools, port: WebSearchPolicyPort
) -> _AdmittedWebAccess:
    """Narrow the declared web tools to what the workspace's web search policy permits."""
    search_url: str | None = ctx.config.web_search_url or otari_env("WEB_SEARCH_URL") or None
    if not web.declared_any:
        return _AdmittedWebAccess(
            fetch_policy=DomainPolicy(), search_auth_token=None, search_tool_entry=None, search_url=search_url
        )
    # Forwarded to the search backend as `X-Gateway-Token`, and only where
    # that backend is the control plane, which authenticates the gateway.
    # A deployment without a platform token forwards none.
    search_auth_token: str | None = None
    if (
        web.search_tool_entry is not None
        and search_url is not None
        and url_targets_platform(search_url, ctx.config.platform.get("base_url"))
    ):
        search_auth_token = ctx.config.platform_token
    scope = WebSearchPolicyScope(workspace_id=ctx.workspace_id, user_token=ctx.user_token)
    try:
        grant = await admit_web_access(web, port=port, scope=scope, config=ctx.config)
    except WebSearchPolicyResolutionFailedError as exc:
        raise adapter.error(_policy_failure_status(exc.reason), exc.message, ErrorKind.API) from exc
    except (WebAccessRefusedError, WorkspaceWebSearchDomainsExcludedError) as exc:
        raise adapter.error(403, exc.message, ErrorKind.PERMISSION) from exc
    return _AdmittedWebAccess(
        fetch_policy=grant.fetch_policy,
        search_auth_token=search_auth_token,
        search_tool_entry=grant.search_tool_entry,
        search_url=search_url,
        search_credential=grant.search_credential,
    )


def _caller_workspace_id(api_key: APIKey | None, session_principal: SessionPrincipal | None) -> uuid.UUID | None:
    """The workspace a file reference is resolved in: the key's, else the session's, else every one.

    ``None`` is the master key alone. A Playground request has no key but does
    have a workspace it proved membership of, and a member's file uploaded
    through a key in another of their workspaces must not resolve here.
    """
    if api_key is not None:
        return api_key.workspace_id
    if session_principal is not None:
        return session_principal.workspace_id
    return None


def _dispatch_provider_name(ctx: RequestContext) -> str | None:
    """The any-llm provider the request's first attempt dispatches to, if known.

    Standalone resolved it in the preamble; hybrid has it on the platform's first
    attempt. ``None`` when neither could say, which the executor reads as "not
    natively served", the answer that brings the code here rather than forwarding
    a declaration nobody may honor.
    """
    if ctx.resolved_provider is not None:
        return ctx.resolved_provider.provider.value
    if ctx.route is not None and ctx.route.attempts:
        return ctx.route.attempts[0].provider
    return None


def _candidate_provider_names(ctx: RequestContext) -> list[str | None]:
    """The any-llm provider of every candidate the request may dispatch to, head first.

    A standalone routing plan's attempts or the platform's, which the dispatch
    walks on failure; otherwise just :func:`_dispatch_provider_name`.
    """
    if ctx.plan is not None and len(ctx.plan.attempts) > 1:
        return [attempt.provider.value for attempt in ctx.plan.attempts]
    if ctx.route is not None and len(ctx.route.attempts) > 1:
        return [attempt.provider for attempt in ctx.route.attempts]
    return [_dispatch_provider_name(ctx)]


async def _resolve_code_execution_policy(
    adapter: FormatAdapter[Any, Any],
    ctx: RequestContext,
    port: CodeExecutionPolicyPort,
) -> ResolvedCodeExecutionPolicy | None:
    """The workspace's code execution policy: the preamble's read where it made one, else the port's answer."""
    if ctx.code_execution_policy_loaded:
        return ctx.code_execution_policy
    scope = CodeExecutionPolicyScope(workspace_id=ctx.workspace_id, user_token=ctx.user_token)
    try:
        return await port.resolve(scope)
    except CodeExecutionPolicyResolutionFailedError as exc:
        raise adapter.error(_code_execution_policy_failure_status(exc.reason), exc.message, ErrorKind.API) from exc


def _code_execution_policy_failure_status(reason: CodeExecutionPolicyResolutionFailure) -> int:
    """The HTTP status a failed code execution policy resolution renders as."""
    match reason:
        case CodeExecutionPolicyResolutionFailure.ANSWER_UNREADABLE:
            return status.HTTP_502_BAD_GATEWAY
        case (
            CodeExecutionPolicyResolutionFailure.NO_CALLER_CREDENTIAL
            | CodeExecutionPolicyResolutionFailure.NO_WORKSPACE
        ):
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        case _:
            assert_never(reason)


def _implementation_for(ctx: RequestContext, instance: str) -> LLMProvider | None:
    """The any-llm provider behind the ``instance`` that served the request, if known."""
    if ctx.resolved_provider is not None and ctx.resolved_provider.instance == instance:
        return ctx.resolved_provider.provider
    candidates = [attempt.provider for attempt in ctx.plan.attempts if attempt.instance == instance] if ctx.plan else []
    for name in [*candidates, instance]:
        try:
            return LLMProvider(name)
        except ValueError:
            continue
    return None


async def _copy_provider_files(
    ctx: RequestContext, files_bridge: SandboxFileBridge | None, files: list[ProviderFile], *, instance: Any
) -> None:
    """Copy the files a provider's own sandbox produced for this request into ``/v1/files``.

    ``instance`` is the configured entry that served, whose credential is the one that can read the files.
    """
    if files_bridge is None or not files or not isinstance(instance, str):
        return
    provider = _implementation_for(ctx, instance)
    if provider is None:
        return
    await files_bridge.copy_provider_files(files, provider=provider.value, provider_instance=instance)


async def _copying_produced_files(
    stream: AsyncIterator[ChunkT], dialect: str, copy: Callable[[list[ProviderFile]], Awaitable[None]]
) -> AsyncIterator[ChunkT]:
    """Forward ``stream`` unchanged, copying the provider-held files an event cites before that event goes on.

    So the caller never sees a file ID before Otari holds the file's bytes.
    """
    async for chunk in stream:
        if files := produced_files_for(dialect, chunk):
            await copy(files)
        yield chunk


async def _require_tool_pricing(
    adapter: FormatAdapter[Any, Any],
    ctx: RequestContext,
    *,
    use_sandbox: bool,
    use_web_search: bool,
    use_web_fetch: bool = False,
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
    if use_web_fetch:
        tools.append(WEB_FETCH_TOOL_NAME)
    for tool in tools:
        pricing = await find_model_pricing(
            ctx.db,
            GATEWAY_TOOL_PRICING_PROVIDER,
            tool,
            use_defaults=False,
            organization_id=ctx.organization_id,
        )
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
                request_id=ctx.request_id,
                tags=ctx.tags,
            )
            # Same kind the model gate uses for its own 402, so both no-pricing
            # rejections map to one wire shape per format.
            raise adapter.error(402, detail, ErrorKind.INVALID_REQUEST)


# ---------------------------------------------------------------------------
# Usage logging and reservation settlement
# ---------------------------------------------------------------------------


def _elapsed_ms(started_at: float | None) -> int | None:
    """Milliseconds elapsed since a monotonic ``started_at`` reading.

    Returns ``None`` when no start was captured (e.g. write paths with no
    meaningful request duration), so the usage log records NULL rather than a
    misleading zero.
    """
    if started_at is None:
        return None
    return round((time.monotonic() - started_at) * 1000)


def _ttft_ms(started_at: float | None, first_chunk_at: float | None) -> int | None:
    """Milliseconds between ``started_at`` and the first streamed chunk.

    Unlike ``_elapsed_ms`` this is not measured against "now": time-to-first-token
    is fixed the moment the first chunk arrives, and every settlement callback
    (on_complete, on_no_usage, on_error, on_incomplete) fires after the stream has
    finished, when "now" is the wrong end of the interval. None when no chunk ever
    arrived (a stream that failed before yielding anything has no TTFT to record).
    """
    if started_at is None or first_chunk_at is None:
        return None
    return round((first_chunk_at - started_at) * 1000)


def _stored_error_message(error: str | None) -> str | None:
    """The error text a usage row may carry.

    ``UsageLog.error_message`` is read by every member of the workspace, including
    a viewer, through ``/api/v1/organizations/me/usage``, so the upstream text is
    put through the same redaction the HTTP detail gets before it is persisted.
    Redacting at write time rather than on the read path is what covers every
    reader of the column at once. A gateway-generated reason passes unchanged.
    """
    if error is None:
        return None
    return _redacted_upstream_detail(error, PROVIDER_ERROR_DETAIL)


# Per-process memory of when each unpriced model was last warned about, so a busy
# unpriced model logs once per interval rather than once per request. Bounded:
# a full map drops expired entries, then the oldest if none had expired.
UNPRICED_WARNING_INTERVAL_S = 3600.0
_UNPRICED_WARNING_MAX_MODELS = 1024
_unpriced_warned_at: dict[str, float] = {}


def _evict_unpriced_warnings(now: float) -> None:
    """Make room in the full throttle map, keeping every live suppression it can."""
    expired = [ref for ref, at in _unpriced_warned_at.items() if now - at >= UNPRICED_WARNING_INTERVAL_S]
    for ref in expired:
        del _unpriced_warned_at[ref]
    if len(_unpriced_warned_at) >= _UNPRICED_WARNING_MAX_MODELS:
        del _unpriced_warned_at[min(_unpriced_warned_at, key=_unpriced_warned_at.__getitem__)]


def _warn_unpriced_model(model_ref: str) -> None:
    """Warn that ``model_ref`` settled with no price, at most once per interval per model."""
    now = time.monotonic()
    last = _unpriced_warned_at.get(model_ref)
    if last is not None and now - last < UNPRICED_WARNING_INTERVAL_S:
        return
    if last is None and len(_unpriced_warned_at) >= _UNPRICED_WARNING_MAX_MODELS:
        _evict_unpriced_warnings(now)
    _unpriced_warned_at[model_ref] = now
    logger.warning(
        "No pricing configured for '%s'. Its tokens are recorded without cost and responses carry no inline cost; "
        "set a price for this model. Repeats for it are suppressed for %d minutes.",
        # codeql[py/clear-text-logging-sensitive-data]
        model_ref,
        int(UNPRICED_WARNING_INTERVAL_S // 60),
    )


class LoggedUsage(NamedTuple):
    """What :func:`record_usage` wrote: the row's total cost and its rate's source.

    ``pricing_source`` is set only when the model's own tokens were priced, so a
    row whose cost is tool charges alone (an unpriced model that still ran a
    search) reports a cost with no source.
    """

    cost: Decimal | None
    pricing_source: PriceSource | None


async def record_usage(
    db: AsyncSession,
    log_writer: LogWriter,
    api_key_id: str | None,
    model: str,
    provider: str | None,
    endpoint: str,
    user_id: str | None = None,
    response: ChatCompletion | AsyncIterator[ChatCompletionChunk] | None = None,
    usage_override: CompletionUsage | None = None,
    provider_type: str | None = None,
    error: str | None = None,
    status_code: int | None = None,
    cost_override: Decimal | float | None = None,
    latency_ms: int | None = None,
    ttft_ms: int | None = None,
    counts_toward_budget: bool = True,
    attribution: RoutingAttribution | None = None,
    tool_tally: ToolUsageTally | None = None,
    workspace_id: uuid.UUID | None = None,
    request_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> LoggedUsage:
    """Log API usage to the database and return the computed cost and its source.

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
        provider_type: any-llm implementation backing ``provider``, which keys
            the provider-reported latency lookup; falls back to ``provider``.
        error: Error message (if failed)
        status_code: HTTP status classifying the failure (see
            ``UsageLog.status_code``), or None when nothing was rejected over HTTP
        cost_override: Fixed amount to record when billing without provider usage
        latency_ms: Total server-side request duration in milliseconds, or None
            when the caller has no meaningful duration to record
        ttft_ms: Milliseconds from request start to the first streamed chunk, or
            None for a non-streaming request or a stream that never yielded one
        attribution: Which routing policy produced this row and where in its plan,
            or None for a request that named a plain model
        workspace_id: The workspace already resolved for this request (from
            ``resolve_workspace_id``/``RequestContext.workspace_id``), reused
            instead of re-deriving it. Every ``ctx``-bearing caller has this for
            free; only callers with no request context in scope (a gateway
            rejection logged before one exists, the vision side-call billed
            during normalization) omit it and pay ``workspace_for_key_id``'s own
            lookup, which is cheap for a keyed request (memoized on
            ``api_key_id``) and the same un-memoized cost a master-key request
            already pays once elsewhere -- passing it explicitly here is what
            avoids paying that twice on the same request.
        request_id: The ``Otari-Request-ID`` the caller was sent, stored as the
            row's ``request_group_id`` when ``attribution`` carries none, so every
            row a request writes is findable by the id the caller holds.
        tags: The caller's request tags (``RequestContext.tags``), stored as-is.

    Returns:
        The computed cost for this request, or None when usage/pricing is absent,
        with the source of the model rate that priced it.

    """
    pricing_source: PriceSource | None = None
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id if workspace_id is not None else await workspace_for_key_id(db, api_key_id),
        api_key_id=api_key_id,
        user_id=user_id,
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status=_row_status(error=error, attribution=attribution),
        error_message=_stored_error_message(error),
        status_code=status_code,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        counts_toward_budget=counts_toward_budget,
        policy_name=attribution.policy_name if attribution else None,
        selection_reason=attribution.selection_reason if attribution else None,
        attempt_position=attribution.position if attribution else None,
        attempt_count=attribution.attempt_count if attribution else None,
        request_group_id=attribution.request_group_id if attribution else request_id,
        tags=tags,
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
        usage_log.reasoning_tokens = reasoning_tokens_of(usage_data)
        # Which convention those cache counts were reported under, recorded rather
        # than left to be inferred from the numbers later (mozilla-ai/otari#690).
        usage_log.cache_tokens_in_prompt = cache_tokens_in_prompt_of(usage_data)
        usage_log.provider_latency_ms = provider_latency_ms_of(usage_data, provider_type or provider)

        record_tokens(
            str(provider or ""),
            model,
            usage_data.prompt_tokens,
            usage_data.completion_tokens,
        )

        # The organization comes off the key, via the workspace already resolved
        # for the row above, so a settled cost uses the same rate the admission
        # gate estimated against. Both lookups are memoized on immutable columns,
        # so this is dictionary reads rather than queries after the first request
        # on a key.
        resolved = await resolve_model_pricing(
            db,
            provider,
            model,
            as_of=usage_log.timestamp,
            organization_id=await organization_for_workspace_id(db, usage_log.workspace_id),
        )
        if resolved is not None:
            cost, meters, breakdown = calculate_metered_cost(resolved.pricing, usage_data)
            pricing_source = resolved.source
            usage_log.cost = cost
            usage_log.billing_meters = meters
            usage_log.pricing_breakdown = breakdown
            usage_log.pricing_source = resolved.source
            # A model key is unbounded where the column is not, and an oversized
            # value would fail the row's insert on PostgreSQL rather than lose a
            # label, so a key past the limit is recorded as unknown.
            reference = resolved.reference
            if reference is not None and len(reference) > PRICING_REFERENCE_MAX_LENGTH:
                reference = None
            usage_log.pricing_reference = reference
            usage_log.pricing_effective_at = resolved.effective_at
        else:
            _warn_unpriced_model(f"{provider}:{model}" if provider else model)

    # When the caller bills a fixed amount without provider usage (e.g. the
    # stream-missing-usage estimate policy), record that amount on the log row
    # so usage_logs.cost stays consistent with the spend that was reconciled.
    # No rate entry produced that amount, so the row names none.
    if cost_override is not None:
        usage_log.cost = to_usd(cost_override)
        usage_log.pricing_source = None
        usage_log.pricing_reference = None
        usage_log.pricing_effective_at = None

    # Gateway-run tool calls are a separate charge from the model's tokens, so they
    # are folded in last: after the token branch (which may not have run at all) and
    # after cost_override (which replaces the token cost, not the whole bill).
    await _apply_tool_charges(db, usage_log, tool_tally)
    if usage_log.cost is not None:
        usage_log.calculated_at = usage_log.timestamp

    # Emitted once here rather than inside the pricing branch so the cost metric
    # tracks the row's total, including tool charges on an unpriced model. A priced
    # row that happens to cost 0 still reports, matching the previous behavior; only
    # a row with no cost at all (None) is skipped.
    if usage_log.cost is not None:
        # The metrics registry speaks float; the row keeps the exact amount.
        record_cost(str(provider or ""), model, float(usage_log.cost))

    await log_writer.put(usage_log)
    return LoggedUsage(usage_log.cost, pricing_source)


def _cost_only(
    record: Callable[_P, Coroutine[Any, Any, LoggedUsage]],
) -> Callable[_P, Coroutine[Any, Any, Decimal | None]]:
    async def cost_only(*args: _P.args, **kwargs: _P.kwargs) -> Decimal | None:
        return (await record(*args, **kwargs)).cost

    return cost_only


log_usage = _cost_only(record_usage)
"""Log API usage to the database and return the computed cost (see :func:`record_usage`)."""


def standalone_settlement(logged: LoggedUsage) -> SettledCost | None:
    """The inline cost a standalone response carries for ``logged``, if any.

    Mirrors the hybrid rule: a priced result carries both fields, anything else
    (unpriced model, fixed-amount estimate, no usage) carries neither.
    """
    if logged.cost is None or logged.pricing_source is None:
        return None
    return SettledCost(cost_usd=f"{quantize_cost(logged.cost):.6f}", pricing_source=logged.pricing_source)


def _attach_standalone_cost(adapter: FormatAdapter[ResultT, Any], result: ResultT, logged: LoggedUsage) -> None:
    """Put a priced non-streaming result's cost on its usage object, never failing the response."""
    settlement = standalone_settlement(logged)
    if settlement is None:
        return
    try:
        attached = adapter.attach_cost(result, settlement)
    except Exception as exc:
        logger.warning("Failed to attach standalone inline cost: %s", exc)
        attached = False
    record_inline_cost_settlement("attached" if attached else "unattached")


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
        tool_cost, lines, unpriced = await price_tool_calls(
            db,
            billable,
            as_of=usage_log.timestamp,
            organization_id=await organization_for_workspace_id(db, usage_log.workspace_id),
        )
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
        usage_log.cost = (usage_log.cost or Decimal(0)) + tool_cost
    for tool in unpriced:
        logger.warning(
            "Gateway tool '%s' ran %d time(s) but has no pricing; recorded without cost. "
            "Price it with POST /api/v1/pricing using model_key '%s'.",
            tool,
            billable[tool],
            gateway_tool_pricing_key(tool),
        )


def _settled_tokens(usage: CompletionUsage | None) -> int:
    """The measured total a token ceiling settles against, or zero when unreported."""
    return max(int(usage.total_tokens or 0), 0) if usage is not None else 0


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
    held amount shrinks the user's budget until the reservation sweep reclaims
    it, which is the only thing that ever gives it back (the budget reset zeroes
    spend and leaves the hold in place).

    When ``ctx.tool_charge`` is set, the request already ran gateway-run tool calls
    that were written onto its failure row, so the reservation is *reconciled* to
    that amount rather than refunded: a refund releases the hold without recording
    spend, which would leave the charge visible in the activity log and missing from
    the budget it should have consumed.
    """
    if ctx.rate_limit_grant is not None:
        await ctx.rate_limit_grant.settle(0)
    if ctx.db is None or ctx.reservation is None:
        return
    if ctx.tool_charge:
        await reconcile_reservation(ctx.db, ctx.reservation, ctx.tool_charge)
        return
    await refund_reservation(ctx.db, ctx.reservation)


async def throttle_early_rejection(raw_request: Request, user_id: str) -> bool:
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
        await check_rate_limit(raw_request, user_id)
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
    request_id: str | None = None,
    tags: dict[str, str] | None = None,
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
            request_id=request_id,
            tags=tags,
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
    if ctx.rate_limit_grant is not None:
        await ctx.rate_limit_grant.settle(0)
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
        workspace_id=ctx.workspace_id,
        request_id=ctx.request_id,
        tags=ctx.tags,
    )
    if ctx.reservation is not None:
        if cost:
            await reconcile_reservation(ctx.db, ctx.reservation, cost)
        else:
            await refund_reservation(ctx.db, ctx.reservation)


# ---------------------------------------------------------------------------
# Backend dispatch (the single copy of the mcp / sandbox / web_search ladder)
# ---------------------------------------------------------------------------


def _loop_options(tool_ctx: ToolContext) -> dict[str, Any]:
    """Tool-loop kwargs this request carries, omitting the ones it never set.

    Presence-encoded rather than passed as ``None``, for the reason
    ``on_first_response`` is: a test fake mirrors the call shape a given request
    actually produces, so a kwarg appearing at all is itself the signal.

    The budget itself travels, not the number it was built from: every attempt of
    one request draws on the same one.
    """
    options: dict[str, Any] = {}
    if tool_ctx.use_budget is not None:
        options["use_budget"] = tool_ctx.use_budget
    return options


def _container_loop_option(adapter: FormatAdapter[Any, Any], backend: Any) -> dict[str, Any]:
    """The lease the Messages loop reports on its response, presence-encoded like the rest.

    Only the Messages dialect has a field for it, Anthropic's ``container``. The
    Responses items already carry the id, and every dialect gets the headers.
    """
    lease = getattr(backend, "lease", None)
    # An isinstance check rather than a None check: a test double stands in for
    # the backend here, and an attribute it never defined is not a lease.
    if not isinstance(lease, ContainerLease) or adapter.name != "messages":
        return {}
    return {"container": lease}


def _container_headers(lease: ContainerLease | None) -> dict[str, str]:
    """The response headers that name the held sandbox, for dialects with no field for it."""
    if lease is None:
        return {}
    return {
        "Otari-Container-Id": lease.container_id,
        "Otari-Container-Expires-At": lease.expires_at.isoformat(),
    }


def _native_loop_options(adapter: FormatAdapter[Any, Any], tool_ctx: ToolContext) -> dict[str, Any]:
    """Native-emission loop kwargs, presence-encoded like :func:`_loop_options`.

    The set travels only when it names something, so a request owed nothing in this
    adapter's vocabulary passes no kwarg at all and a format with no vocabulary of its
    own never sees one.
    """
    native_tools = tool_ctx.native_tools(adapter.name)
    return {"native_tools": native_tools} if native_tools else {}


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
        async with tool_ctx.build_sandbox_backend() as backend:
            kwargs = adapter.inject_hints(call_kwargs, backend.purpose_hints(), header=tool_ctx.tools_header)
            return await adapter.run_tool_loop(
                kwargs,
                backend,
                tool_ctx.max_tool_iterations,
                on_first_response,
                **_native_loop_options(adapter, tool_ctx),
                **_container_loop_option(adapter, backend),
            )

    assert tool_ctx.use_web_search or tool_ctx.use_web_fetch
    async with tool_ctx.build_web_retrieval_backend() as web_backend:
        kwargs = adapter.inject_hints(call_kwargs, web_backend.purpose_hints(), header=tool_ctx.tools_header)
        return await adapter.run_tool_loop(
            kwargs,
            web_backend,
            tool_ctx.max_tool_iterations,
            on_first_response,
            **_native_loop_options(adapter, tool_ctx),
            **_loop_options(tool_ctx),
        )


class _ToolBackendKind(StrEnum):
    """The kind of tool backend a streamed request holds open, as its log lines name it."""

    MCP = "MCP"
    SANDBOX = "sandbox"
    WEB_RETRIEVAL = "web retrieval"

    @classmethod
    def of(cls, tool_ctx: ToolContext) -> _ToolBackendKind:
        if tool_ctx.mcp_server_configs:
            return cls.MCP
        return cls.SANDBOX if tool_ctx.use_sandbox else cls.WEB_RETRIEVAL


async def _close_tool_backend(closing: Awaitable[object], kind: _ToolBackendKind) -> None:
    """Await a tool backend's close, logging an ordinary failure rather than raising it.

    A failed close must not replace or cut off what the stream produced, but a cancellation still propagates.
    """
    try:
        await closing
    except BaseExceptionGroup as group:
        ordinary, fatal = group.split(Exception)
        if ordinary is not None:
            logger.warning("The %s tool backend failed to close: %s", kind, failure_class(ordinary))
        if fatal is not None:
            raise fatal from None
    except Exception as exc:
        logger.warning("The %s tool backend failed to close: %s", kind, failure_class(exc))


@contextlib.asynccontextmanager
async def _held_tool_backend(
    backend: AbstractAsyncContextManager[BackendT], kind: _ToolBackendKind
) -> AsyncIterator[BackendT]:
    """Enter ``backend`` for the block, and close it through :func:`_close_tool_backend`."""
    entered = await backend.__aenter__()
    try:
        yield entered
    except BaseException as exc:
        await _close_tool_backend(backend.__aexit__(type(exc), exc, exc.__traceback__), kind)
        raise
    await _close_tool_backend(backend.__aexit__(None, None, None), kind)


async def _lazy_mcp_stream(
    adapter: FormatAdapter[Any, ChunkT],
    kwargs: dict[str, Any],
    configs: list[McpServerConfig],
    tool_ctx: ToolContext,
) -> AsyncIterator[ChunkT]:
    # The MCP pool is entered lazily inside the generator: a dial failure
    # surfaces once the client starts pulling events. Sandbox / web_search use
    # the eager-open path below for a pre-200 HTTP error instead.
    async with _held_tool_backend(MCPClientPool(configs, tally=tool_ctx.tally), _ToolBackendKind.MCP) as pool:
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
    # owns the matching ``__aexit__`` once the stream finishes or errors, and
    # hands it the error so a backend can abandon work nobody will read.
    kind = _ToolBackendKind.of(tool_ctx)
    try:
        hinted = adapter.inject_hints(kwargs, backend.purpose_hints(), header=tool_ctx.tools_header)
        async for event in adapter.open_tool_loop_stream(
            hinted,
            backend,
            tool_ctx.max_tool_iterations,
            **_native_loop_options(adapter, tool_ctx),
            **_loop_options(tool_ctx),
            **(_container_loop_option(adapter, backend) if tool_ctx.use_sandbox else {}),
        ):
            yield event
    except BaseException as exc:
        await _close_tool_backend(backend.__aexit__(type(exc), exc, exc.__traceback__), kind)
        raise
    await _close_tool_backend(backend.__aexit__(None, None, None), kind)


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
        sandbox_backend = tool_ctx.build_sandbox_backend()
        await sandbox_backend.__aenter__()  # may raise SandboxNotReachableError
        return _eager_backend_stream(adapter, kwargs, sandbox_backend, tool_ctx)

    assert tool_ctx.use_web_search or tool_ctx.use_web_fetch
    web_search_backend = tool_ctx.build_web_retrieval_backend()
    await web_search_backend.__aenter__()  # may raise WebSearchNotReachableError
    return _eager_backend_stream(adapter, kwargs, web_search_backend, tool_ctx)


# ---------------------------------------------------------------------------
# Streaming settlement (the single copy of the callback bundle)
# ---------------------------------------------------------------------------

# Strong references to in-flight usage-report tasks. Reports awaited for inline
# cost remain tracked after a timeout or caller cancellation, so accounting can
# still complete in the background.
_USAGE_REPORT_TASKS: set[asyncio.Task[SettledCost | None]] = set()


def _start_usage_report(
    coro: Coroutine[Any, Any, SettledCost | None],
    correlation_id: str,
) -> asyncio.Task[SettledCost | None]:
    task = asyncio.create_task(coro)
    _USAGE_REPORT_TASKS.add(task)

    def _finalize(finished: asyncio.Task[SettledCost | None]) -> None:
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
    return task


def _schedule_usage_report(
    coro: Coroutine[Any, Any, SettledCost | None],
    correlation_id: str,
) -> None:
    """Run a platform usage report in the background without losing it."""
    _start_usage_report(coro, correlation_id)


def _inline_settlement_timeout_seconds(config: GatewayConfig) -> float:
    return int(config.platform.get("usage_inline_timeout_ms", 1500)) / 1000


async def _await_usage_report(
    coro: Coroutine[Any, Any, SettledCost | None],
    correlation_id: str,
    config: GatewayConfig,
) -> SettledCost | None:
    """Await one report under the inline budget without cancelling accounting."""
    timeout_seconds = _inline_settlement_timeout_seconds(config)
    task = _start_usage_report(coro, correlation_id)
    try:
        settlement = await asyncio.wait_for(
            asyncio.shield(task),
            timeout=timeout_seconds,
        )
    except asyncio.CancelledError:
        raise
    except (asyncio.TimeoutError, TimeoutError):
        logger.info(
            "Inline settlement budget expired; responding without cost correlation_id=%s",
            correlation_id,
        )
        record_inline_cost_settlement("timeout")
        return None
    except Exception:
        # The tracked task finalizer logs the failure without exposing details.
        record_inline_cost_settlement("unattached")
        return None
    if settlement is None:
        record_inline_cost_settlement("unattached")
    return settlement


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
    request_id: str | None = None,
    session_label: str | None = None,
    display_model: str | None = None,
    attribution: RoutingAttribution | None = None,
    tool_tally: ToolUsageTally | None = None,
    workspace_id: uuid.UUID | None = None,
    extra_headers: dict[str, str] | None = None,
    rate_limit_grant: RateLimitGrant | None = None,
    tags: dict[str, str] | None = None,
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

    A stream that fails or is abandoned after the provider reported usage (an
    Anthropic ``message_start`` carries the input tokens) still owes for those
    tokens, so ``on_error`` and ``on_incomplete`` price what was reported rather
    than recording the request as free. Every row carries ``request_id`` as its
    ``request_group_id``, which is what lets a caller whose stream never delivered
    a cost look it up by ``Otari-Request-ID`` afterwards.
    """
    platform_active = platform_correlation_id is not None
    # Both modes settle before the terminal suffix so its usage object can carry
    # the cost: hybrid from the platform's report, standalone from its own row.
    settles_inline = platform_active or (db is not None and log_writer is not None)
    # Provider-latency lookup is keyed by implementation, not instance name.
    provider_type = config.provider_instance_type(provider_key(provider))

    first_chunk_at: float | None = None
    # Usage the provider reported before the stream ended, whether or not it ended
    # cleanly. The generator hands only a completed stream's usage to a callback.
    reported_usage: CompletionUsage | None = None

    def _on_first_chunk() -> None:
        nonlocal first_chunk_at
        first_chunk_at = time.monotonic()

    def _extract_usage(chunk: ChunkT) -> CompletionUsage | None:
        nonlocal reported_usage
        chunk_usage = adapter.extract_stream_usage(chunk)
        if chunk_usage:
            reported_usage = chunk_usage if reported_usage is None else merge_stream_usage(reported_usage, chunk_usage)
        return chunk_usage

    async def _on_complete(usage_data: CompletionUsage) -> SettledCost | None:
        if rate_limit_grant is not None:
            await rate_limit_grant.settle(_settled_tokens(usage_data))
        if platform_active:
            assert platform_correlation_id is not None
            return await _await_usage_report(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
                    outcome="success",
                    usage=usage_data,
                    session_label=session_label,
                    ttft_ms=_ttft_ms(started_at, first_chunk_at),
                    is_final_attempt=True,
                ),
                platform_correlation_id,
                config,
            )
        if db is None or log_writer is None:
            return None
        logged = await record_usage(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            model=model,
            provider=provider,
            provider_type=provider_type,
            endpoint=adapter.endpoint,
            user_id=user_id,
            usage_override=usage_data,
            latency_ms=_elapsed_ms(started_at),
            ttft_ms=_ttft_ms(started_at, first_chunk_at),
            counts_toward_budget=_handle_counts_toward_budget(reservation),
            attribution=attribution,
            tool_tally=tool_tally,
            workspace_id=workspace_id,
            request_id=request_id,
            tags=tags,
        )
        if reservation is not None:
            await reconcile_reservation(
                db, reservation, logged.cost or Decimal(0), actual_tokens=_settled_tokens(usage_data)
            )
        return standalone_settlement(logged)

    async def _on_no_usage() -> None:
        # Stream completed but the provider sent no usage data. Report the
        # terminal success upstream in hybrid mode; standalone settles the
        # reservation per stream_missing_usage_policy instead of billing $0.
        if platform_active:
            assert platform_correlation_id is not None
            settlement = await _await_usage_report(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
                    outcome="success",
                    usage=None,
                    session_label=session_label,
                    ttft_ms=_ttft_ms(started_at, first_chunk_at),
                    is_final_attempt=True,
                ),
                platform_correlation_id,
                config,
            )
            if settlement is not None:
                record_inline_cost_settlement("unattached")
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
                ttft_ms=_ttft_ms(started_at, first_chunk_at),
                counts_toward_budget=reservation.counts_toward_budget,
                attribution=attribution,
                tool_tally=tool_tally,
                workspace_id=workspace_id,
                request_id=request_id,
                tags=tags,
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
            ttft_ms=_ttft_ms(started_at, first_chunk_at),
            counts_toward_budget=reservation.counts_toward_budget,
            attribution=attribution,
            tool_tally=tool_tally,
            workspace_id=workspace_id,
            request_id=request_id,
            tags=tags,
        )
        # The estimate covers the unreported tokens; log_usage adds any tool cost on
        # top of it, so reconcile against the row's total rather than the estimate.
        # The token axis settles at its estimate for the same reason: the provider
        # reported no count, and the alternative records zero tokens for a request
        # that certainly used some.
        await reconcile_reservation(
            db,
            reservation,
            settled_cost or reservation.estimate,
            actual_tokens=reservation.token_estimate,
        )

    async def _on_error(exc: BaseException) -> None:
        if rate_limit_grant is not None:
            await rate_limit_grant.settle(_settled_tokens(reported_usage))
        if platform_active:
            assert platform_correlation_id is not None
            _schedule_usage_report(
                _report_platform_usage(
                    config=config,
                    correlation_id=platform_correlation_id,
                    outcome="error",
                    usage=None,
                    session_label=session_label,
                    ttft_ms=_ttft_ms(started_at, first_chunk_at),
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
            usage_override=reported_usage,
            error=str(exc),
            status_code=failure_status_code(exc),
            latency_ms=_elapsed_ms(started_at),
            ttft_ms=_ttft_ms(started_at, first_chunk_at),
            counts_toward_budget=_handle_counts_toward_budget(reservation),
            attribution=attribution,
            tool_tally=tool_tally,
            workspace_id=workspace_id,
            request_id=request_id,
            tags=tags,
        )
        if reservation is not None:
            # A stream that died after reporting tokens or running searches still
            # owes for them, and a refund would release the hold without recording
            # that spend. This is also where the streaming tool-iteration cap
            # lands, since the cap is raised inside the generator.
            if failed_cost:
                await reconcile_reservation(db, reservation, failed_cost, actual_tokens=_settled_tokens(reported_usage))
            else:
                await refund_reservation(db, reservation)

    async def _on_incomplete() -> None:
        # Client disconnected mid-stream: release the reservation.
        #
        # Tokens the provider already reported and tool work already done are still
        # owed. Without this, disconnecting after the searches have run is an
        # unlimited supply of unbilled, unrecorded searches, which is the abuse this
        # metering exists to close. A row is written only when there is something to
        # bill, so an abandoned stream that reported nothing and ran no tools leaves
        # no trace.
        if rate_limit_grant is not None:
            await rate_limit_grant.settle(_settled_tokens(reported_usage))
        if db is None or reservation is None:
            return
        has_tool_work = tool_tally is not None and not tool_tally.is_empty()
        if log_writer is not None and (has_tool_work or reported_usage is not None):
            abandoned_cost = await log_usage(
                db=db,
                log_writer=log_writer,
                api_key_id=api_key_id,
                model=model,
                provider=provider,
                endpoint=adapter.endpoint,
                user_id=user_id,
                usage_override=reported_usage,
                error="client disconnected before the stream completed",
                latency_ms=_elapsed_ms(started_at),
                ttft_ms=_ttft_ms(started_at, first_chunk_at),
                counts_toward_budget=_handle_counts_toward_budget(reservation),
                tool_tally=tool_tally,
                workspace_id=workspace_id,
                request_id=request_id,
                tags=tags,
            )
            if abandoned_cost:
                await reconcile_reservation(
                    db, reservation, abandoned_cost, actual_tokens=_settled_tokens(reported_usage)
                )
                return
        await refund_reservation(db, reservation)

    def _attach_inline_cost(value: Any, settlement: SettledCost) -> bool:
        if value is None:
            record_inline_cost_settlement("unattached")
            return False
        attached = adapter.attach_cost(value, settlement)
        record_inline_cost_settlement("attached" if attached else "unattached")
        return attached

    # StreamingResponse builds its own response object, so headers we want on
    # the wire have to be passed in here; assigning to the dependency-injected
    # ``Response`` object does not propagate to streaming responses.
    headers: dict[str, str] = dict(rate_limit_headers(rate_limit_info)) if rate_limit_info else {}
    if platform_correlation_id:
        headers[ATTEMPT_ID_HEADER] = platform_correlation_id
    if request_id:
        headers[REQUEST_ID_HEADER] = request_id
    if extra_headers:
        headers.update(extra_headers)

    return StreamingResponse(
        streaming_generator(
            stream=stream,
            format_chunk=adapter.format_chunk,
            extract_usage=_extract_usage,
            fmt=adapter.stream_format,
            on_complete=_on_complete,
            on_error=_on_error,
            label=f"{provider}:{model}",
            on_no_usage=_on_no_usage,
            on_incomplete=_on_incomplete,
            display_model=display_model,
            keepalive_interval_seconds=config.streaming_keepalive_interval_ms / 1000,
            settle_before_done=settles_inline,
            is_cost_carrier=adapter.is_stream_cost_carrier if settles_inline else None,
            attach_settlement=_attach_inline_cost if settles_inline else None,
            on_first_chunk=_on_first_chunk,
            error_payload=adapter.stream_error_payload,
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


def stream_final_attempt_extra_seconds(
    config: GatewayConfig,
    *,
    tool_mode: bool = False,
    has_forwarded_tools: bool = False,
) -> float:
    """Extra first-chunk grace granted only to the sole/final streaming attempt.

    Added on top of the per-attempt failover budget for the terminal attempt,
    which has no next entry in the routing policy to fall over to. A request that
    forwards provider-native tools but does not run a gateway-managed tool loop
    keeps the plain failover budget on non-final attempts, and on the final one
    is raised to the tool-loop base before the configured grace is added, so
    tool-heavy agents get the relaxed criterion only where there is nowhere left
    to fail over. The grace stays additive in every mode: an operator who
    configures it always buys that much more time.
    """
    configured_extra = (
        int(
            config.platform.get(
                _STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS_KEY,
                _DEFAULT_STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS,
            )
        )
        / 1000
    )
    if tool_mode or not has_forwarded_tools:
        return configured_extra

    plain_budget = stream_first_chunk_timeout_seconds(config, tool_mode=False)
    tool_loop_budget = stream_first_chunk_timeout_seconds(config, tool_mode=True)
    return configured_extra + max(0.0, tool_loop_budget - plain_budget)


# ---------------------------------------------------------------------------
# Shared request runners
# ---------------------------------------------------------------------------


async def _prepared(
    adapter: FormatAdapter[Any, Any], prepare_kwargs: PrepareKwargs | None, instance: str, call_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """What the only candidate of a request with no fallover is sent.

    Raises:
        HTTPException: the candidate cannot serve the request, or it was refused.
    """
    if prepare_kwargs is None:
        return call_kwargs
    try:
        return await prepare_kwargs(instance, call_kwargs)
    except CandidateCannotServe as exc:
        raise exc.refusal from exc
    except TenancyError as exc:
        raise domain_error(adapter, exc) from exc


def _model_admission(ctx: RequestContext) -> AdmitAttempt | None:
    """Admits each candidate of a walk under the ``per: model`` rate limits, skipping a full one."""
    grant = ctx.rate_limit_grant
    if grant is None:
        return None

    async def _admit(attempt: Attempt) -> Callable[[bool], Awaitable[None]] | None:
        try:
            hold = await grant.admit_model(attempt.instance, attempt.model, name_model=False)
        except HTTPException as refusal:
            raise CandidateCannotServe(refusal) from refusal
        return hold.drop if hold is not None else None

    return _admit


async def _admitted_and_prepared(
    ctx: RequestContext,
    adapter: FormatAdapter[Any, Any],
    prepare_kwargs: PrepareKwargs | None,
    instance: str,
    model: str,
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Admit the only candidate of a request under the ``per: model`` rate limits, then prepare its kwargs.

    Admitted first, so a full model is not prepared for; a candidate that then
    cannot be prepared keeps nothing it was admitted with.

    Raises:
        HTTPException: 429 when a rule naming the model is full.
    """
    hold = None
    if ctx.rate_limit_grant is not None:
        # A policy's target is not the caller's to see; a model the caller named is.
        hold = await ctx.rate_limit_grant.admit_model(instance, model, name_model=ctx.plan is None)
    try:
        return await _prepared(adapter, prepare_kwargs, instance, call_kwargs)
    except BaseException:
        if hold is not None:
            await hold.drop(sent=False)
        raise


async def run_single_attempt_stream(
    *,
    adapter: FormatAdapter[Any, ChunkT],
    ctx: RequestContext,
    tool_ctx: ToolContext,
    call_kwargs: dict[str, Any],
    provider: Any,
    model: str,
    platform_correlation_id: str | None = None,
    session_label: str | None = None,
    display_model: str | None = None,
    base_request_fields: dict[str, Any] | None = None,
    prepare_kwargs: PrepareKwargs | None = None,
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

            async def _skipped(attempt: Attempt, refusal: HTTPException) -> None:
                await log_skipped_attempt(ctx, adapter, attempt, refusal)

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
                    prepare_kwargs=prepare_kwargs,
                    admit_attempt=_model_admission(ctx),
                    on_absorbed=_absorbed,
                    on_skipped=_skipped,
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
            call_kwargs = await _admitted_and_prepared(ctx, adapter, prepare_kwargs, provider, model, call_kwargs)
            stream = await open_stream(adapter=adapter, tool_ctx=tool_ctx, call_kwargs=call_kwargs)
            # A single-candidate policy still names a policy and a reason, and
            # both belong on the row.
            stream_attribution = _attribution_for(ctx, ctx.plan.head) if ctx.plan is not None else None
    except HTTPException:
        await release_reservation(ctx)
        raise
    except SandboxNotReachableError as exc:
        # codeql[py/clear-text-logging-sensitive-data]
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, exc)
        await release_reservation(ctx)
        if isinstance(exc, SandboxSessionGoneError):
            await tool_ctx.forget_container()
        raise _sandbox_error(adapter, exc, tool_ctx=tool_ctx) from exc
    except WebSearchNotReachableError as exc:
        # codeql[py/clear-text-logging-sensitive-data]
        logger.error("Web search backend unreachable for %s:%s: %s", provider, model, exc)
        await release_reservation(ctx)
        raise adapter.error(502, WEB_SEARCH_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    except Exception as exc:
        await _log_failure_and_refund(
            ctx, adapter, provider, model, str(exc), failure_status_code(exc), attribution=_failure_attribution(ctx)
        )
        # codeql[py/clear-text-logging-sensitive-data]
        logger.error("Stream creation failed for %s:%s: %s", provider, model, exc)
        raise adapter.provider_error(exc) from exc

    files_bridge = tool_ctx.sandbox_files
    if files_bridge is not None:

        async def _copy(files: list[ProviderFile]) -> None:
            await _copy_provider_files(ctx, files_bridge, files, instance=provider)

        stream = _copying_produced_files(stream, adapter.name, _copy)

    return build_streaming_response(
        adapter=adapter,
        stream=stream,
        provider=provider,
        model=model,
        config=ctx.config,
        db=ctx.db,
        extra_headers=_container_headers(tool_ctx.container_lease) | _end_user_headers(ctx),
        log_writer=ctx.log_writer,
        api_key_id=ctx.api_key_id,
        user_id=ctx.user_id,
        rate_limit_info=ctx.rate_limit_info,
        reservation=ctx.reservation,
        rate_limit_grant=ctx.rate_limit_grant,
        started_at=ctx.started_at,
        workspace_id=ctx.workspace_id,
        platform_correlation_id=platform_correlation_id,
        request_id=ctx.request_id,
        session_label=session_label,
        display_model=display_model,
        attribution=stream_attribution,
        tool_tally=tool_ctx.tally,
        tags=ctx.tags,
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
    started_at: float,
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
    final_attempt_extra = stream_final_attempt_extra_seconds(
        config,
        tool_mode=tool_mode,
        has_forwarded_tools=bool(tool_ctx.remaining_user_tools),
    )

    # Only tool backends go on this stack, because a failure to close it is logged rather than raised.
    backend_stack = AsyncExitStack()
    pool_for_loop: Any = None
    try:
        if tool_ctx.mcp_server_configs:
            pool_for_loop = await backend_stack.enter_async_context(
                MCPClientPool(tool_ctx.mcp_server_configs, tally=tool_ctx.tally)
            )
        elif tool_ctx.use_sandbox:
            pool_for_loop = await backend_stack.enter_async_context(tool_ctx.build_sandbox_backend())
        elif tool_ctx.use_web_search or tool_ctx.use_web_fetch:
            pool_for_loop = await backend_stack.enter_async_context(tool_ctx.build_web_retrieval_backend())
    except BaseException:
        # Eager-open failure (e.g. SandboxNotReachableError): propagate so the
        # route handler maps it to the existing HTTP status. Nothing to clean
        # up on the stack yet because the entry failed.
        await backend_stack.aclose()
        raise

    async def _build_for_attempt(attempt: ResolvedAttempt) -> AsyncIterator[ChunkT]:
        completion_kwargs = adapter.prepare_stream_kwargs(
            adapter.attempt_kwargs(attempt, base_request_fields),
            require_usage=True,
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
            **_native_loop_options(adapter, tool_ctx),
            **_loop_options(tool_ctx),
        )

    # See run_platform_non_stream: BackgroundTasks only run after a successful
    # response, so if every attempt fails before its first chunk the queued
    # reports are dropped with the terminal 502/504. Keep the background task
    # for the success path (it flushes once the SSE response completes), but
    # also stash the error reports so they can be flushed inline on the
    # all-failed path below.
    pending_error_reports: list[_PendingUsageReport] = []
    attempt_errors: list[BaseException] = []

    async def _on_attempt_failed(attempt: ResolvedAttempt, failure: StreamingAttemptFailure) -> None:
        attempt_errors.append(failure.exception)
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
            await backend_stack.__aexit__(type(exc), exc, exc.__traceback__)
        if not isinstance(exc, Exception):
            raise
        # Only this frame knows which attempt was tried last, so the terminal
        # error is mapped here.
        last_attempt_id = pending_error_reports[-1].attempt_id if pending_error_reports else None
        if (rejection := get_shared_rejection(attempt_errors)) is not None:
            kind = error_kind_for_status(rejection.status_code)
            shared = adapter.error(rejection.status_code, str(rejection.detail), kind, dict(rejection.headers or {}))
            raise with_attempt_id(shared, last_attempt_id) from exc
        raise_all_streaming_attempts_failed(adapter, exc, route, last_attempt_id)

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
        stream_to_return = _stream_with_stack_cleanup(stream, backend_stack, _ToolBackendKind.of(tool_ctx))

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
        request_id=route.request_id,
        session_label=session_label,
        started_at=started_at,
    )


async def _stream_with_stack_cleanup(
    stream: AsyncIterator[ChunkT],
    backend_stack: AsyncExitStack,
    kind: _ToolBackendKind,
) -> AsyncIterator[ChunkT]:
    try:
        async for chunk in stream:
            yield chunk
    except BaseException as exc:
        await _close_tool_backend(backend_stack.__aexit__(type(exc), exc, exc.__traceback__), kind)
        raise
    await _close_tool_backend(backend_stack.aclose(), kind)


def _sandbox_error(
    adapter: FormatAdapter[Any, Any], exc: SandboxNotReachableError, *, tool_ctx: ToolContext | None = None
) -> HTTPException:
    if isinstance(exc, SandboxSessionGoneError) and tool_ctx is not None and tool_ctx.sandbox_container_lease:
        # The provider was asked for the container the caller named and no
        # longer has it: the caller's id is stale, which is a request to fix,
        # not a backend outage to retry.
        container_id = tool_ctx.sandbox_container_lease.container_id
        detail = CONTAINER_GONE_DETAIL_TEMPLATE.format(container_id=container_id)
        return adapter.error(400, detail, ErrorKind.INVALID_REQUEST)
    if isinstance(exc, SandboxUnavailableError):
        headers = {"Retry-After": exc.retry_after} if exc.retry_after is not None else None
        return adapter.error(503, SANDBOX_UNAVAILABLE_DETAIL, ErrorKind.API, headers)
    return adapter.error(502, SANDBOX_UNREACHABLE_DETAIL, ErrorKind.API)


def raise_all_streaming_attempts_failed(
    adapter: FormatAdapter[Any, Any],
    exc: Exception,
    route: ResolvedRoute,
    last_attempt_id: str | None = None,
) -> NoReturn:
    """Map pre-stream failures, preserving sandbox retry hints and provider status.

    ``last_attempt_id`` names the attempt that failed last, which the hybrid
    protocol owes the caller on total failure.
    """
    if isinstance(exc, SandboxNotReachableError):
        logger.error("Sandbox unreachable request_id=%s: %s", route.request_id, exc)
        raise _sandbox_error(adapter, exc) from exc
    if isinstance(exc, WebSearchNotReachableError):
        logger.error("Web search backend unreachable request_id=%s: %s", route.request_id, exc)
        raise adapter.error(502, WEB_SEARCH_UNREACHABLE_DETAIL, ErrorKind.API) from exc
    logger.error("All streaming attempts failed request_id=%s: %s", route.request_id, exc)
    if len(route.attempts) <= 1:
        raise with_attempt_id(adapter.provider_error(exc), last_attempt_id) from exc
    kind, status_code = upstream_exception_shape(exc)
    if kind == "timeout":
        timed_out = adapter.error(504, ALL_PROVIDERS_TIMED_OUT_DETAIL, ErrorKind.API)
        raise with_attempt_id(timed_out, last_attempt_id) from exc
    if status_code == 429:
        rate_limited = adapter.error(
            429,
            ALL_PROVIDERS_RATE_LIMITED_DETAIL,
            ErrorKind.RATE_LIMIT,
            provider_error_headers(exc, 429),
        )
        raise with_attempt_id(rate_limited, last_attempt_id) from exc
    all_failed = adapter.error(502, ALL_PROVIDERS_FAILED_DETAIL, ErrorKind.API)
    raise with_attempt_id(all_failed, last_attempt_id) from exc


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
    successful_report: tuple[ResolvedAttempt, Any] | None = None

    def _report_attempt_outcome(
        attempt: ResolvedAttempt,
        outcome: str,
        usage: Any,
        error_class: str | None,
        is_final_attempt: bool,
    ) -> None:
        nonlocal successful_report
        if outcome == "success":
            successful_report = (attempt, usage)
            return
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
        response.headers[ATTEMPT_ID_HEADER] = attempt.attempt_id
        if rate_limit_info:
            for key, value in rate_limit_headers(rate_limit_info).items():
                response.headers[key] = value

    try:
        result = await run_platform_attempts(
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
        # Error responses drop queued tasks; flush earlier attempt reports.
        logger.error("Sandbox unreachable request_id=%s: %s", route.request_id, exc)
        await _flush_pending_usage_reports(config, pending_error_reports, route.request_id, session_label)
        raise _sandbox_error(adapter, exc) from exc
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

    if successful_report is None:
        return result
    attempt, usage = successful_report
    settlement = await _await_usage_report(
        _report_platform_usage(
            config=config,
            correlation_id=attempt.attempt_id,
            outcome="success",
            usage=usage,
            session_label=session_label,
            is_final_attempt=True,
        ),
        attempt.attempt_id,
        config,
    )
    if settlement is not None:
        try:
            attached = adapter.attach_cost(result, settlement)
        except Exception as exc:
            logger.warning(
                "Failed to attach inline settlement correlation_id=%s: %s",
                attempt.attempt_id,
                exc,
            )
            record_inline_cost_settlement("unattached")
        else:
            record_inline_cost_settlement("attached" if attached else "unattached")
    return result


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
        workspace_id=ctx.workspace_id,
        tags=ctx.tags,
    )
    ctx.tool_charge = cost or Decimal(0)


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
    await _log_absorbed_row(ctx, adapter, attempt, str(exc), failure_status_code(exc))


# Starts the error of an absorbed row for a candidate that was skipped rather than
# called, which is how the activity log tells "skipped" from "failed".
SKIPPED_ATTEMPT_PREFIX = "Skipped: "


async def log_skipped_attempt(
    ctx: RequestContext,
    adapter: FormatAdapter[Any, Any],
    attempt: Attempt,
    refusal: HTTPException,
) -> None:
    """Record a candidate the walk skipped, such as a model a ``per: model`` rate limit had no room on.

    An absorbed row like a recovered failure, so it counts toward no request or
    error total. Its error starts with :data:`SKIPPED_ATTEMPT_PREFIX`.
    """
    await _log_absorbed_row(ctx, adapter, attempt, f"{SKIPPED_ATTEMPT_PREFIX}{refusal.detail}", refusal.status_code)


async def _log_absorbed_row(
    ctx: RequestContext, adapter: FormatAdapter[Any, Any], attempt: Attempt, error: str, status_code: int
) -> None:
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
            error=error,
            status_code=status_code,
            latency_ms=_elapsed_ms(ctx.started_at),
            counts_toward_budget=False,
            attribution=_attribution_for(ctx, attempt, absorbed=True),
            workspace_id=ctx.workspace_id,
            tags=ctx.tags,
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
    prepare_kwargs: PrepareKwargs | None = None,
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

            async def _skipped(attempt: Attempt, refusal: HTTPException) -> None:
                await log_skipped_attempt(ctx, adapter, attempt, refusal)

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
                    prepare_kwargs=prepare_kwargs,
                    admit_attempt=_model_admission(ctx),
                    on_absorbed=_absorbed,
                    on_skipped=_skipped,
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
            call_kwargs = await _admitted_and_prepared(ctx, adapter, prepare_kwargs, provider, model, call_kwargs)
            result = await dispatch_non_stream(adapter=adapter, tool_ctx=tool_ctx, call_kwargs=call_kwargs)
            # A single-candidate policy still has a name and a selection reason, and
            # both belong on the row: "served by its default target" is the answer to
            # the same question a fallover answers differently.
            attribution = _attribution_for(ctx, ctx.plan.head) if ctx.plan is not None else None
        if ctx.rate_limit_info:
            for key, value in rate_limit_headers(ctx.rate_limit_info).items():
                response.headers[key] = value
        for key, value in _container_headers(tool_ctx.container_lease).items():
            response.headers[key] = value
        if ctx.db is not None:
            usage_data = adapter.extract_usage(result)
            if ctx.rate_limit_grant is not None:
                await ctx.rate_limit_grant.settle(_settled_tokens(usage_data))
            logged = LoggedUsage(None, None)
            # A request whose provider reported no usage still owes for the tool
            # calls it ran, so a non-empty tally forces the row that
            # ``log_success_without_usage = False`` would otherwise suppress.
            if usage_data is not None or adapter.log_success_without_usage or not tool_ctx.tally.is_empty():
                logged = await record_usage(
                    db=ctx.db,
                    log_writer=ctx.log_writer,
                    api_key_id=ctx.api_key_id,
                    model=model,
                    provider=provider,
                    provider_type=ctx.config.provider_instance_type(provider_key(provider)),
                    endpoint=adapter.endpoint,
                    user_id=ctx.user_id,
                    usage_override=usage_data,
                    latency_ms=_elapsed_ms(ctx.started_at),
                    counts_toward_budget=_handle_counts_toward_budget(ctx.reservation),
                    attribution=attribution,
                    tool_tally=tool_ctx.tally,
                    workspace_id=ctx.workspace_id,
                    request_id=ctx.request_id,
                    tags=ctx.tags,
                )
            if ctx.reservation is not None:
                await reconcile_reservation(
                    ctx.db, ctx.reservation, logged.cost or Decimal(0), actual_tokens=_settled_tokens(usage_data)
                )
            _attach_standalone_cost(adapter, result, logged)
            await _copy_provider_files(
                ctx, tool_ctx.sandbox_files, produced_files_for(adapter.name, result), instance=provider
            )
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
        # codeql[py/clear-text-logging-sensitive-data]
        logger.error("Sandbox unreachable for %s:%s: %s", provider, model, e)
        await release_reservation(ctx)
        if isinstance(e, SandboxSessionGoneError):
            await tool_ctx.forget_container()
        raise _sandbox_error(adapter, e, tool_ctx=tool_ctx) from e
    except WebSearchNotReachableError as e:
        # codeql[py/clear-text-logging-sensitive-data]
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
        # codeql[py/clear-text-logging-sensitive-data]
        logger.error("Provider call failed for %s:%s: %s", provider, model, e)
        raise adapter.provider_error(e) from e
