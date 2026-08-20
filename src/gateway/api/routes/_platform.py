"""Hybrid-mode shared infrastructure.

Holds the resolved-route Pydantic types, the generic ``run_platform_attempts``
runner that the chat / messages / responses endpoints all use for multi-attempt
fallback on the non-streaming path, and the platform-side helpers (credential
resolution, MCP server resolution, error classification, usage reporting).
The runner is format-agnostic — callers pass a per-attempt dispatcher and a
usage-extractor and the runner handles iteration, error classification,
lock-in semantics, and the terminal all-failed status mapping uniformly.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Awaitable, Callable, Iterator
from typing import Any, Literal, NamedTuple, TypeVar

import httpx
from anthropic import APIConnectionError as _AnthropicAPIConnectionError
from anthropic import APITimeoutError as _AnthropicAPITimeoutError
from any_llm import LLMProvider
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException, Request, status
from openai import APIConnectionError as _OpenAIAPIConnectionError
from openai import APITimeoutError as _OpenAIAPITimeoutError
from pydantic import BaseModel, Field, ValidationError

from gateway.core.config import GatewayConfig
from gateway.core.usage import cache_read_tokens_of, cache_write_tokens_of
from gateway.log_config import logger
from gateway.metrics import record_abandoned_attempt
from gateway.models.mcp import McpServerConfig
from gateway.services.bedrock_gateway_auth import build_bedrock_client_args
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError

T = TypeVar("T")

# Status codes returned by the platform's usage-report endpoint that the
# gateway should NOT retry. Auth / payment-required / not-found / conflict /
# unprocessable are all permanent rejection signals — retrying would just
# hammer the platform (an overdrawn or missing wallet won't recover within the
# retry window). 402 is already excluded by the >= 500 retry predicate below;
# listing it keeps the intent explicit and robust to changes in that predicate.
_USAGE_NON_RETRYABLE_STATUS_CODES = {401, 402, 404, 409, 422}

# Statuses on which the billing-message probe runs. A 402 is payment required by
# definition, so it is handled directly in ``is_provider_billing_error`` without
# depending on a provider's error text. Anthropic reports "credit balance is too
# low" as a 400 ``invalid_request_error`` and OpenAI's "billing hard limit
# reached" is a 400. 429 is deliberately excluded: OpenAI's
# ``insufficient_quota`` arrives as a 429, which is already both failover-eligible
# and surfaced to the caller as a rate limit, so backing off remains a sane client
# action and reclassifying it would only change the wording.
_BILLING_MESSAGE_CANDIDATE_STATUS_CODES = {400, 422}

# Lowercase substrings that identify account-level billing exhaustion in an
# upstream provider message. Deliberately narrow: a false positive turns a
# genuinely malformed request into wasted failover attempts against every
# provider in the route, so a phrase only earns a place here if it cannot plausibly
# describe a malformed request. Best-effort against current provider phrasing; a
# reworded message degrades safely to the generic bad-request handling rather
# than misclassifying.
_BILLING_MESSAGE_PROBES: tuple[str, ...] = (
    "credit balance is too low",  # anthropic
    "purchase credits",  # anthropic
    "plans & billing",  # anthropic
    "billing hard limit",  # openai
    "insufficient balance",  # deepseek, moonshot
    "insufficient_quota",  # openai (error code, echoed in some messages)
    "exceeded your current quota",  # openai
    "insufficient credits",  # openrouter, together
)

# Streaming first-chunk timeouts (hybrid-mode fallback). Plain LLM streams
# rarely take long to produce a first token, so a tight cap keeps failed-
# attempt latency low. Tool-loop streams may reason before emitting tokens
# or a tool_call (especially with extended thinking), so they get more
# headroom. Both are operator-tunable via ``config.platform``.
_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS = 2000
_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP = 30000
_STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY = "streaming_first_chunk_timeout_ms"
_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY = "streaming_first_chunk_timeout_ms_tool_loop"

# Extra first-chunk grace added ONLY to the sole/final attempt, on top of the
# per-attempt budget above. That budget is a failover trigger: it abandons a slow
# attempt so the next entry in the routing policy can be tried. The final attempt
# has no next entry, so a tight cap there only turns a slow-but-valid first token
# into a 504 with nothing to fall over to. Granting grace keeps the terminal wait
# bounded (a genuinely hung upstream still times out at budget + grace) while not
# failing valid slow-to-start responses. Applied on top of whichever base budget
# is in effect (plain or tool-loop). Defaults to 0 (no grace), so behavior is
# unchanged unless an operator opts in via ``config.platform`` (v1.2 will move
# these onto the routing_policy schema).
_DEFAULT_STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS = 0
_STREAM_FINAL_ATTEMPT_EXTRA_FIRST_CHUNK_TIMEOUT_MS_KEY = "streaming_final_attempt_extra_first_chunk_timeout_ms"


class ResolvedAttempt(BaseModel):
    """A single resolution attempt returned by the platform."""

    attempt_id: str
    position: int
    provider: str
    model: str
    api_base: str | None = None
    api_key: str
    managed: bool
    extra_params: dict[str, str] | None = None
    """Provider-specific extra client kwargs beyond api_key/api_base (e.g. AWS
    Bedrock's ``region_name``/``aws_access_key_id``). Sourced only from the
    trusted platform peer, never from the caller's request body. See
    ``default_attempt_kwargs``, which merges these in non-overridably."""


class SettledCost(BaseModel):
    """An attachable platform settlement: a priced cost and the basis for it."""

    cost_usd: str = Field(pattern=r"^-?\d+\.\d{6}$")
    pricing_source: str


class _UsagePricing(BaseModel):
    source: str | None


class _CompletedUsageSettlement(BaseModel):
    correlation_id: str
    status: Literal["completed"]
    outcome: Literal["success"]
    cost_usd: str = Field(pattern=r"^-?\d+\.\d{6}$")
    currency: Literal["USD"]
    usage_status: Literal["reported", "unavailable"]
    pricing: _UsagePricing


class ResolvedRoute(BaseModel):
    """The full resolution plan returned by the platform."""

    request_id: str
    fallback_enabled: bool
    attempts: list[ResolvedAttempt]
    # The platform identity that owns the presented X-User-Token, when the peer
    # supplies one (see docs/hybrid-mode-protocol.md's Extension policy). An
    # opaque string, never parsed; absent for peers that predate the field or
    # simply don't send it. This is the only per-caller identity hybrid mode
    # has, so it is what gateway-side survivals (aliases, routing memory,
    # files, batches) can key their state on in a later hybrid-mode phase.
    user_id: str | None = None
    # The tenant that owns this resolution: the workspace the presented
    # X-User-Token belongs to, and the organization above it. Both opaque
    # strings, never parsed. The platform has carried them on its resolve
    # response all along (see docs/hybrid-mode-protocol.md); keeping them here
    # is the only way anything downstream can attribute a record to a tenant,
    # because the gateway token that authenticates the call to the platform
    # carries no tenant of its own. Absent on a peer that does not send them,
    # in which case a record that needs them is dropped rather than pooled
    # into some other tenant's.
    workspace_id: str | None = None
    organization_id: str | None = None


class _AttemptFailure(NamedTuple):
    position: int
    provider: str
    model: str
    error_class: str


def build_attempt_client_args(attempt: ResolvedAttempt) -> dict[str, Any] | None:
    """Build the ``client_args`` any-llm needs to construct this attempt's
    provider client, or ``None`` when the attempt carries no ``extra_params``.

    This *must* be a separate ``client_args`` dict, not merged flat into the
    completion kwargs: any-llm's ``acompletion()`` only forwards a
    ``client_args`` mapping to the provider's client constructor
    (``AnyLLM.create(provider, api_key=..., api_base=..., **client_args)``);
    every other keyword argument goes to the completion *call* instead. A
    provider-specific credential field passed flat (e.g. Bedrock's
    ``region_name``) never reaches ``boto3.client()`` that way and is instead
    silently forwarded into the raw provider API call, which is how an
    earlier version of this forwarding path still hit ``NoRegionError``
    despite carrying the right value.

    Bedrock gets dedicated handling (see :mod:`gateway.services.bedrock_gateway_auth`)
    because it also needs its secret aliased to a different boto3 kwarg name
    and, for the bearer-token ("Bedrock API key") credential shape, a custom
    pre-built client. Every other provider's ``extra_params`` is forwarded
    as-is.
    """
    if not attempt.extra_params:
        return None
    if LLMProvider(attempt.provider) == LLMProvider.BEDROCK:
        return build_bedrock_client_args(attempt.api_key, attempt.extra_params)
    return dict(attempt.extra_params)


def default_attempt_kwargs(
    attempt: ResolvedAttempt,
    base_request_fields: dict[str, Any],
) -> dict[str, Any]:
    """Standard platform-attempt kwargs: credentials + ``provider:model`` selector.

    ``client_args`` (built from the attempt's ``extra_params``, see
    :func:`build_attempt_client_args`) is merged in *after*
    ``base_request_fields`` (and after the forced ``model`` key) so a
    provider-specific credential field the platform returns can never be
    shadowed by a same-named field in the caller's own request body,
    matching how ``api_key``/``model`` are already non-overridable.
    """
    attempt_provider = LLMProvider(attempt.provider)
    kwargs: dict[str, Any] = {"api_key": attempt.api_key}
    if attempt.api_base:
        kwargs["api_base"] = attempt.api_base
    merged = {
        **kwargs,
        **base_request_fields,
        "model": f"{attempt_provider.value}:{attempt.model}",
    }
    client_args = build_attempt_client_args(attempt)
    if client_args is not None:
        merged["client_args"] = client_args
    return merged


def _provider_failure_http_exc(exc: BaseException, *, fallback_detail: str) -> HTTPException:
    """Build the terminal HTTPException for a failed platform attempt.

    Reuses the shared provider-error classifier so platform-mode failures get
    the same specific-but-safe statuses (400/404/429/504) the standalone
    adapters return, instead of a blanket 502. Falls back to a 502 carrying
    ``fallback_detail`` when the failure has no signal we can safely surface.
    The classifier applies its caller-fault versus gateway-fault detail split,
    so caller-fault details are sanitized provider diagnostics and gateway-fault
    details remain fixed strings.
    """
    # Deferred import: _pipeline imports this module, so importing it at module
    # scope would be circular.
    from gateway.api.routes._pipeline import classify_provider_error

    mapping = classify_provider_error(exc)
    if mapping is not None:
        return HTTPException(status_code=mapping.status_code, detail=mapping.detail)
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=fallback_detail)


async def run_platform_attempts(
    *,
    route: ResolvedRoute,
    attempts: list[ResolvedAttempt],
    base_request_fields: dict[str, Any],
    run_attempt: Callable[[dict[str, Any], Callable[[], None]], Awaitable[T]],
    extract_usage: Callable[[T], Any],
    classify_error: Callable[[BaseException], tuple[bool, str]],
    report_attempt_outcome: Callable[[ResolvedAttempt, str, Any, str | None, bool], None],
    on_success: Callable[[ResolvedAttempt], None],
    max_tool_iterations: int,
) -> T:
    """Iterate ``attempts``, returning the first one that succeeds.

    ``run_attempt`` receives the per-attempt ``completion_kwargs`` (the merged
    ``api_key`` / ``api_base`` / base request fields, with ``model`` set to
    ``"{provider}:{model}"``) plus a per-attempt ``on_first_response``
    callback. Tool-loop callers thread that callback into ``mcp_tool_loop``
    (and its per-format siblings) so it fires exactly once after the first
    upstream response — locking the request in to the current attempt. The
    runner is agnostic to the response shape; ``extract_usage`` pulls the
    usage object out for reporting and ``on_success`` lets the caller mutate
    the FastAPI response object on success.

    Lock-in semantics: once ``on_first_response`` has fired on an attempt,
    any subsequent error from that attempt terminates the request instead of
    falling through. A tool-use loop's intermediate state (provider-specific
    ``tool_call`` ids / reasoning blocks) cannot be transparently replayed
    on a different provider. Pre-lock-in failures still walk the attempts
    list and fall through to the next provider. Classification only labels the
    attempt for logging and platform usage reporting.

    ``MaxToolIterationsExceeded`` is treated as a gateway-side cap hit, not
    an upstream failure — it raises a distinct 422 and does not advance to
    the next attempt.

    If every attempt fails the runner raises 504 on timeout. Otherwise a
    single-attempt failure is classified into a specific safe status
    (400/404/429/...) when the upstream error carries one (via
    ``_provider_failure_http_exc``), falling back to a generic 502; a
    multi-attempt fallthrough keeps the generic 502 "All upstream providers
    failed" rather than attributing one provider's status to the whole set.

    Callers are expected to hand a non-empty ``attempts`` list — the platform
    resolve endpoint guarantees one. Passing an empty list is a caller
    programming error (the route handler should have raised 502 with
    "no resolvable provider" already); the runner surfaces it as a 500 so the
    bug doesn't masquerade as an "all upstream providers failed" message.
    """
    if not attempts:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Internal error: run_platform_attempts received an empty attempts list — "
                "the caller should have raised a 502 'no resolvable provider' before reaching the runner"
            ),
        )

    failures: list[_AttemptFailure] = []
    last_exc: BaseException | None = None

    for index, attempt in enumerate(attempts):
        completion_kwargs = default_attempt_kwargs(attempt, base_request_fields)
        is_last_planned_attempt = index == len(attempts) - 1

        # Per-attempt lock-in flag. Flipped the moment the upstream returns
        # its first assistant message via the ``_mark_locked_in`` callback
        # that tool-loop callers thread into ``mcp_tool_loop`` /
        # ``anthropic_tool_loop`` / ``responses_tool_loop``.
        locked_in = False

        def _mark_locked_in(_pos: int = attempt.position) -> None:
            nonlocal locked_in
            locked_in = True
            logger.info(
                "Tool-loop lock-in request_id=%s position=%d provider=%s model=%s",
                route.request_id,
                _pos,
                attempt.provider,
                attempt.model,
            )

        try:
            result = await run_attempt(completion_kwargs, _mark_locked_in)
        except HTTPException:
            raise
        except MaxToolIterationsExceeded as exc:
            report_attempt_outcome(attempt, "error", None, None, True)
            logger.warning(
                "Tool loop iteration cap hit request_id=%s position=%d cap=%d",
                route.request_id,
                attempt.position,
                max_tool_iterations,
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except (SandboxNotReachableError, WebSearchNotReachableError):
            # Gateway-side backend failure, not an upstream provider error:
            # the same backend serves every attempt, so falling through cannot
            # help. Report a terminal outcome without a provider error class,
            # then propagate raw so ``run_platform_non_stream`` maps it to a
            # 502 with the backend-specific detail.
            report_attempt_outcome(attempt, "error", None, None, True)
            raise
        except asyncio.CancelledError:
            # The catch below is `BaseException` so a provider client raising
            # outside the `Exception` hierarchy still falls through to the next
            # candidate. Cancellation is the one case that must not: the caller
            # is gone, so there is nobody to serve and no provider at fault.
            # Letting the classifier see it would record an abandoned attempt
            # against a provider that answered fine, and swallowing it into an
            # HTTPException would suppress the cancellation the server is
            # waiting to unwind.
            raise
        except BaseException as exc:
            retryable, error_class = classify_error(exc)
            is_final_attempt = locked_in or not retryable or is_last_planned_attempt
            report_attempt_outcome(attempt, "error", None, error_class, is_final_attempt)
            logger.warning(
                "Provider call failed request_id=%s position=%d provider=%s model=%s "
                "error=%s retryable=%s locked_in=%s",
                route.request_id,
                attempt.position,
                attempt.provider,
                attempt.model,
                error_class,
                retryable,
                locked_in,
            )
            last_exc = exc
            # Count as abandoned-before-first-chunk only when nothing was
            # produced yet. A locked-in failure already yielded a first
            # assistant message, so it is not abandonment waste. Non-streaming
            # attempts have no separable build phase, so a classified timeout
            # maps to ``timeout`` and everything else to ``upstream_error``.
            if not locked_in:
                reason = "timeout" if error_class == "timeout" else "upstream_error"
                record_abandoned_attempt(attempt.provider, attempt.model, reason, attempt.position)
            # Locked-in: at least one tool-loop round produced an assistant
            # message on this attempt. Subsequent failures cannot be
            # transparently retried on another provider.
            if locked_in:
                raise _provider_failure_http_exc(exc, fallback_detail="LLM provider error") from exc
            if not retryable:
                raise _provider_failure_http_exc(exc, fallback_detail="LLM provider error") from exc
            failures.append(_AttemptFailure(attempt.position, attempt.provider, attempt.model, error_class))
            continue

        # Success on this attempt.
        report_attempt_outcome(attempt, "success", extract_usage(result), None, True)
        on_success(attempt)
        return result

    # All attempts exhausted after provider failures.
    logger.error(
        "All upstream attempts failed request_id=%s failures=%s",
        route.request_id,
        failures,
    )
    is_single_attempt = len(attempts) <= 1
    if last_exc is not None and upstream_exception_shape(last_exc)[0] == "timeout":
        detail = "LLM provider timeout" if is_single_attempt else "All upstream providers timed out"
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=detail,
        ) from last_exc
    # A single attempt has one identifiable upstream failure we can classify;
    # a multi-attempt fallthrough aggregates heterogeneous failures, so it keeps
    # the generic 502 rather than attributing one provider's status to the set.
    if is_single_attempt and last_exc is not None:
        raise _provider_failure_http_exc(last_exc, fallback_detail="LLM provider error") from last_exc
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="All upstream providers failed",
    ) from last_exc


# ---------- platform-side helpers ----------


def _extract_platform_user_token(request: Request) -> str:
    """Pull the user's bearer token off the ``Authorization`` header.

    Used in hybrid mode to forward the caller's identity to the platform's
    resolve endpoint. Standalone mode uses ``verify_api_key_or_master_key``
    instead.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )
    return token


def _split_model_selector(model_selector: str) -> tuple[str | None, str]:
    """Parse ``provider:model`` or ``provider/model`` into ``(provider, model)``.

    Used when calling the platform's resolve endpoint with the model selector
    from the request. Returns ``(None, model_selector)`` for bare model names.
    """
    if ":" in model_selector:
        provider, model_name = model_selector.split(":", 1)
        return provider or None, model_name
    if "/" in model_selector:
        provider, model_name = model_selector.split("/", 1)
        return provider or None, model_name
    return None, model_selector


def _platform_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _safe_detail_from_platform(response: httpx.Response, fallback: str) -> str:
    try:
        payload = response.json()
    except ValueError:
        return fallback

    detail = payload.get("detail") if isinstance(payload, dict) else None
    return detail if isinstance(detail, str) else fallback


async def _post_platform(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: float,
) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        return await client.post(url, headers=headers, json=body)


async def _post_resolve(
    config: GatewayConfig,
    *,
    user_token: str,
    path: str,
    body: dict[str, Any],
    client_error_detail: str,
) -> Any:
    """POST ``body`` to a platform resolve endpoint and return the parsed JSON.

    Owns the pieces every resolve helper shares: the base_url guard, the
    gateway/user token headers, the bounded POST, and the status-code ladder.
    A 200 returns the parsed payload; client errors (400/401/402/403/404/429)
    are forwarded with the platform's detail when it is a safe string (falling
    back to ``client_error_detail``), keeping Retry-After on a 429; timeouts,
    network errors, the platform's server-side failures, and any unexpected
    status collapse to a 502. 400 is included here (unlike 422, which stays
    collapsed) because this backend's own 400s are deliberately hand-written,
    caller-safe rejections (e.g. a Bedrock BYO key using an auth shape that
    cannot be forwarded through a gateway), not raw framework validation
    errors that might otherwise leak internal request-shape detail.
    """
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid mode is misconfigured",
        )

    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    resolve_url = _platform_url(platform_base_url, path)
    headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }

    try:
        response = await _post_platform(
            url=resolve_url,
            headers=headers,
            body=body,
            timeout_seconds=timeout_ms / 1000,
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        return response.json()

    if response.status_code in {400, 401, 402, 403, 404, 429}:
        detail = _safe_detail_from_platform(response, client_error_detail)
        response_headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            response_headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=response_headers)

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
    )


async def _resolve_platform_credentials(
    config: GatewayConfig,
    user_token: str,
    model_selector: str,
) -> ResolvedRoute:
    """Call the platform's ``/gateway/provider-keys/resolve`` to get the
    routing plan (one or more ``ResolvedAttempt`` entries).
    """
    provider, model_name = _split_model_selector(model_selector)
    resolve_body: dict[str, Any] = {"model": model_name}
    if provider:
        resolve_body["provider"] = provider

    payload = await _post_resolve(
        config,
        user_token=user_token,
        path="/gateway/provider-keys/resolve",
        body=resolve_body,
        client_error_detail="Authorization request rejected",
    )
    return _parse_resolve_payload(payload)


def _tenant_ids(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read the optional ``(workspace_id, organization_id)`` pair off a resolve
    payload, tolerating absence.

    Shared by both payload shapes, since the fields sit at the top level in
    each. A peer that omits them yields ``None`` rather than an error, so
    resolution is never blocked on a field only recording needs.
    """
    workspace_id = payload.get("workspace_id")
    organization_id = payload.get("organization_id")
    return (
        str(workspace_id) if workspace_id is not None else None,
        str(organization_id) if organization_id is not None else None,
    )


def _parse_resolve_payload(payload: dict[str, Any]) -> ResolvedRoute:
    """Build a ResolvedRoute from either the new attempts-list shape or the
    legacy single-attempt shape.

    The legacy shape lacks ``attempts``/``request_id`` and instead has the
    primary attempt's fields at the top level (``provider``, ``model``,
    ``api_key``, ``api_base``, ``managed``, ``correlation_id``). Older otari
    deployments still respond this way; we map them onto a single-attempt route
    so the rest of the gateway code never has to know.
    """
    attempts_payload = payload.get("attempts")
    if attempts_payload is not None:
        attempts = [
            ResolvedAttempt(
                attempt_id=str(att["attempt_id"]),
                position=int(att["position"]),
                provider=str(att["provider"]),
                model=str(att["model"]),
                api_base=att.get("api_base"),
                api_key=str(att["api_key"]),
                managed=bool(att.get("managed", False)),
                extra_params=att.get("extra_params"),
            )
            for att in attempts_payload
        ]
        raw_user_id = payload.get("user_id")
        workspace_id, organization_id = _tenant_ids(payload)
        return ResolvedRoute(
            request_id=str(payload["request_id"]),
            fallback_enabled=bool(payload.get("fallback_enabled", False)),
            attempts=attempts,
            user_id=str(raw_user_id) if raw_user_id is not None else None,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )

    # Legacy single-attempt shape predates user_id entirely, same treatment as
    # extra_params (see the class docstring above): no legacy mirror to read.
    # The tenant ids are the exception: they sit at the top level in both
    # shapes, so a flat-answering peer that knows its tenant still gets read.
    correlation_id = str(payload["correlation_id"])
    workspace_id, organization_id = _tenant_ids(payload)
    return ResolvedRoute(
        request_id=correlation_id,
        fallback_enabled=False,
        workspace_id=workspace_id,
        organization_id=organization_id,
        attempts=[
            ResolvedAttempt(
                attempt_id=correlation_id,
                position=0,
                provider=str(payload["provider"]),
                model=str(payload["model"]),
                api_base=payload.get("api_base"),
                api_key=str(payload["api_key"]),
                managed=bool(payload.get("managed", False)),
                extra_params=payload.get("extra_params"),
            )
        ],
    )


UpstreamErrorKind = Literal["timeout", "conn_err"]


def upstream_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception and its ``original_exception`` chain once each."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "original_exception", None)


def upstream_exception_shape(exc: BaseException) -> tuple[UpstreamErrorKind | None, int | None]:
    """Classify the *shape* of an upstream exception, independent of retry policy.

    Returns ``(kind, status_code)`` where ``kind`` is ``"timeout"``,
    ``"conn_err"``, or ``None``, and ``status_code`` is the HTTP status the
    exception carries, or ``None``. At most one of the two is set. ``kind`` is
    a ``Literal``, not a bare ``str``, so a typo at a ``kind == "..."``
    comparison site (e.g. in :func:`_classify_upstream_error`) is a mypy error
    instead of a silent no-op. Shared by :func:`_classify_upstream_error`
    (drives the hybrid-mode fallback decision) and
    :func:`gateway.api.routes._pipeline.classify_provider_error` (drives the
    final client-facing status/detail), so the two stay in sync.

    Recognizes, in order, walking ``.original_exception`` when the current
    exception carries no signal of its own:

    1. Stdlib/httpx timeout and network exceptions directly.
    2. The OpenAI and Anthropic SDKs' own ``APITimeoutError`` /
       ``APIConnectionError``. any-llm calls these SDKs directly (it does not
       wrap httpx itself unless ``ANY_LLM_UNIFIED_EXCEPTIONS=1``), and both
       SDKs catch ``httpx.TimeoutException`` / network errors internally and
       re-raise as their own types. Those wrapped types are not instances of
       any httpx exception and carry no ``status_code``/``response``, so a
       real provider timeout or "provider unreachable" would otherwise be
       unclassifiable. ``APITimeoutError`` subclasses ``APIConnectionError``
       in both SDKs, so the timeout check must run first. This covers the
       majority of any-llm providers, which reuse ``BaseOpenAIProvider`` or
       ``BaseAnthropicProvider``.
    3. An HTTP status code carried directly on the exception or on its
       attached ``.response``.
    4. A conservative duck-typed fallback, by exception class name, for the
       remaining any-llm provider SDKs that don't reuse the OpenAI/Anthropic
       base classes (e.g. cohere, mistral, groq, bedrock) and whose own
       timeout/connection exception types aren't imported here. Mirrors the
       heuristic ``any_llm.utils.exception_handler.convert_exception`` already
       uses internally. Only applies when the exception carries no status code
       at all, so it never shadows a real HTTP-status classification.
    5. If none of the above match, ``.original_exception`` (set by any-llm's
       ``AnyLLMError`` family, e.g. via ``convert_exception``) is unwrapped and
       reclassified from step 1. This keeps detection working once
       ``ANY_LLM_UNIFIED_EXCEPTIONS=1`` becomes the default: any-llm's
       ``convert_exception`` re-wraps a raw SDK timeout/connection error into
       a generic ``any_llm.exceptions.ProviderError`` (class name matches
       neither ``*TimeoutError`` nor ``*ConnectionError``, no ``status_code``),
       but always preserves the original SDK exception on
       ``original_exception``. Bounded by an ``id()``-based seen-set so a
       (pathological, self-referential) cycle terminates instead of looping.
    """
    for current in upstream_exception_chain(exc):
        if isinstance(
            current,
            (
                asyncio.TimeoutError,
                TimeoutError,
                httpx.TimeoutException,
                _OpenAIAPITimeoutError,
                _AnthropicAPITimeoutError,
            ),
        ):
            return "timeout", None
        if isinstance(current, (httpx.NetworkError, _OpenAIAPIConnectionError, _AnthropicAPIConnectionError)):
            return "conn_err", None

        status_code = getattr(current, "status_code", None)
        if status_code is None:
            resp = getattr(current, "response", None)
            if resp is not None:
                status_code = getattr(resp, "status_code", None)
        if isinstance(status_code, int):
            return None, status_code

        class_name = type(current).__name__
        if class_name.endswith("TimeoutError"):
            return "timeout", None
        if class_name.endswith("ConnectionError"):
            return "conn_err", None

    return None, None


def upstream_error_message(exc: BaseException) -> str:
    """Best-effort human-readable text from an upstream exception (and its wrapper).

    Concatenates the exception's ``message`` attribute and ``str()`` across its
    ``original_exception`` chain, so a substring probe still matches whether
    otari receives the raw SDK error today or a wrapped ``AnyLLMError`` after
    any-llm flips to unified exceptions.

    Used both for classification and, on a caller-fault rejection, as the text
    the caller sees. Run it through :func:`redact_upstream_message` before it
    reaches a response body.
    """
    parts: list[str] = []
    seen: set[str] = set()
    for candidate in upstream_exception_chain(exc):
        message = getattr(candidate, "message", None)
        candidates = (message, str(candidate)) if isinstance(message, str) else (str(candidate),)
        for part in candidates:
            # ``message`` and ``str()`` are usually the same text on a raw SDK
            # error, and a wrapper usually restringifies what it wraps. Keeping
            # both would read as a stutter now that this text can reach the
            # caller, so each distinct part is kept once, in order.
            stripped = part.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                parts.append(stripped)
    return " ".join(parts)


_REDACTION_PLACEHOLDER = "[redacted]"
# Cap on an exposed upstream message. Providers occasionally echo the offending
# request back in the error body, and a response detail is not the place for a
# few hundred KB of it.
MAX_EXPOSED_DETAIL_CHARS = 400
# Applied in order, so a URL is masked before the trailing catch-all can pick
# apart its path. Each targets a shape that carries secrets rather than meaning:
# nothing here removes text a caller needs to understand what it got wrong.
_SECRET_SHAPES: tuple[re.Pattern[str], ...] = (
    # Authorization-header credentials, whatever scheme.
    re.compile(r"(?:Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    # Provider key formats: a known prefix plus the key body. Covers OpenAI
    # (sk-, sk-proj-), Anthropic (sk-ant-), Groq (gsk_), xAI (xai-), and
    # otari's own tk_ / gw_ tokens.
    re.compile(r"\b(?:sk|pk|rk|gsk|xai|tk|gw)[-_][A-Za-z0-9._-]{8,}", re.IGNORECASE),
    # Google API keys, which carry a fixed prefix and no separator.
    re.compile(r"\bAIza[A-Za-z0-9._-]{10,}"),
    # An explicitly named credential, as it appears in a query string or an
    # echoed request body.
    re.compile(r"\b(?:api[_-]?key|access[_-]?token|token|secret|password)\s*[=:]\s*\S+", re.IGNORECASE),
    # Upstream account identifiers. A managed-model request runs on the
    # platform's own provider account, so an error naming that account would
    # tell a workspace user whose credentials served them.
    re.compile(r"\b(?:org|proj|acct|account)[-_][A-Za-z0-9]{6,}", re.IGNORECASE),
    # Any absolute URL. A self-hosted or proxied ``api_base`` is gateway
    # topology the caller has no business learning, and a credential is
    # sometimes embedded in one.
    re.compile(r"\bhttps?://\S+", re.IGNORECASE),
    # Azure OpenAI and Mistral issue prefixless 32-character API keys.
    re.compile(r"\b[A-Za-z0-9]{32}\b"),
    # Catch-all for key material with no recognizable prefix. Long unbroken
    # alphanumeric runs are tokens, not prose.
    re.compile(r"[A-Za-z0-9_-]{40,}"),
)

# Upstream APIs sometimes reflect the request that failed. Such an echo can
# contain prompt text, tool arguments, or gateway-generated context, none of
# which belongs in a client-facing error. Parameter paths such as
# ``messages.0.content`` stay useful, while field/value pairs, including common
# validation-error spellings, and JSON payloads are rejected as a whole and make
# the caller-fault classifier use its fallback.
_PAYLOAD_ECHO = re.compile(
    r"(?:[\"']?(?:messages|input|prompt|tools?|tool_calls|response|request(?:_body)?|body|content|input_value)[\"']?\s*[:=]|\b(?:messages|input|tools?|tool_calls)(?:\.\d+|\[[^]]+\])+\.(?:content|input_value)\s*:\s*(?:(?:input_)?value\s*=|[\"'{[]))",
    re.IGNORECASE,
)


def redact_upstream_message(message: str) -> str:
    """Make an upstream provider message safe to return to the caller.

    The gateway calls providers with the *operator's* credentials, so an
    upstream message is not automatically the caller's to read: it can carry the
    gateway's own key, a self-hosted ``api_base``, or other topology. This masks
    those shapes and caps the length, leaving the part that says what was
    actually wrong with the request.

    Redaction is the second line rather than the only one. Statuses where
    secrets concentrate (a rejected credential, a 5xx) never reach here at all,
    because :func:`gateway.api.routes._pipeline.classify_provider_error` keeps a
    fixed detail for them.
    """
    redacted = message.strip()
    if _PAYLOAD_ECHO.search(redacted):
        return ""
    for pattern in _SECRET_SHAPES:
        redacted = pattern.sub(_REDACTION_PLACEHOLDER, redacted)
    redacted = " ".join(redacted.split())
    if len(redacted) > MAX_EXPOSED_DETAIL_CHARS:
        redacted = redacted[: MAX_EXPOSED_DETAIL_CHARS - len("...")].rstrip() + "..."
    return redacted


def is_provider_billing_error(exc: BaseException) -> bool:
    """True when a 4xx is the provider saying "this account is out of money".

    Providers disagree on the status for account billing exhaustion. Anthropic
    returns a 400 ``invalid_request_error`` carrying "Your credit balance is too
    low to access the Anthropic API"; OpenAI has a 400 "billing hard limit
    reached"; DeepSeek uses 402 "Insufficient Balance". A 400 normally means the
    request itself is malformed, so these would otherwise be reported to the
    caller as "check the model name and parameters" and, worse, skip failover on
    the assumption that every provider would reject the same request. The
    account balance is per-provider, so the next attempt in the route can very
    plausibly succeed.

    A 402 is always a billing error: HTTP defines it as payment required, even
    when a provider SDK discards the response reason and body. The 400/422 probe
    stays deliberately narrow so error reports do not falsely label malformed
    requests as account billing exhaustion.
    """
    _kind, status_code = upstream_exception_shape(exc)
    if status_code == 402:
        return True
    if status_code not in _BILLING_MESSAGE_CANDIDATE_STATUS_CODES:
        return False
    message = upstream_error_message(exc).lower()
    return any(probe in message for probe in _BILLING_MESSAGE_PROBES)


def _classify_upstream_error(exc: BaseException) -> tuple[bool, str]:
    """Classify an upstream provider error for observability.

    All provider failures that reach an attempt walker before lock-in advance to
    the next candidate. The boolean stays in the return shape for the generic
    streaming iterator, but is always ``True`` here. Gateway-side failures,
    cancellations, and tool loops that already produced an assistant response are
    handled by the walker before this classifier runs.
    """
    kind, status_code = upstream_exception_shape(exc)
    if kind is not None:
        return True, kind

    if isinstance(status_code, int):
        # Keep account exhaustion distinguishable in logs and in reports to the
        # platform, even though every provider error now advances the plan.
        if is_provider_billing_error(exc):
            return True, f"http_{status_code}_billing"
        return True, f"http_{status_code}"

    return True, "unknown"


async def _resolve_platform_mcp_servers(
    config: GatewayConfig,
    user_token: str,
    mcp_server_ids: list[uuid.UUID],
) -> list[McpServerConfig]:
    """Swap workspace-scoped MCP server ids for inline configs by calling the platform."""
    payload = await _post_resolve(
        config,
        user_token=user_token,
        path="/gateway/mcp-servers/resolve",
        body={"mcp_server_ids": [str(uid) for uid in mcp_server_ids]},
        client_error_detail="MCP server resolution failed",
    )
    return [
        McpServerConfig(
            name=s["name"],
            url=s["url"],
            authorization_token=s.get("authorization_token"),
            purpose_hint=s.get("purpose_hint"),
            allowed_tools=s.get("allowed_tools"),
        )
        for s in payload.get("servers", [])
    ]


async def _resolve_platform_web_search(
    config: GatewayConfig,
    user_token: str,
) -> dict[str, Any]:
    """Resolve the workspace's web-search policy via the platform.

    POSTs an empty body to `/gateway/web-search/resolve` (via `_post_resolve`,
    which owns the shared guard/headers/status-code ladder) and returns the
    parsed JSON dict on 200 (``{enabled, provider, max_results, purpose_hint,
    allowed_domains, blocked_domains, provider_options}``).
    """
    payload = await _post_resolve(
        config,
        user_token=user_token,
        path="/gateway/web-search/resolve",
        body={},
        client_error_detail="Web search resolution failed",
    )
    return payload if isinstance(payload, dict) else {}


async def _resolve_platform_code_execution(
    config: GatewayConfig,
    user_token: str,
) -> dict[str, Any]:
    """Resolve the workspace's code-execution policy via the platform.

    POSTs an empty body to `/gateway/code-execution/resolve` (via
    `_post_resolve`, which owns the shared guard/headers/status-code ladder)
    and returns the parsed JSON dict on 200 (``{enabled, tools,
    default_purpose_hint, max_iterations, exec_timeout_s}``, soft limits
    already clamped to operator ceilings platform-side).
    """
    payload = await _post_resolve(
        config,
        user_token=user_token,
        path="/gateway/code-execution/resolve",
        body={},
        client_error_detail="Code execution resolution failed",
    )
    return payload if isinstance(payload, dict) else {}


async def _report_platform_usage(
    config: GatewayConfig,
    correlation_id: str,
    outcome: str,
    usage: CompletionUsage | None,
    error_class: str | None = None,
    session_label: str | None = None,
    *,
    is_final_attempt: bool,
) -> SettledCost | None:
    """POST a usage record back to the platform with bounded retries.

    ``is_final_attempt`` tells the platform that no later planned fallback will
    run. Failures are swallowed after ``max_retries`` so they don't impact the
    user's response path. Non-retryable status codes (auth /
    payment-required / not-found / conflict / unprocessable) short-circuit the
    retry loop.
    """
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        return None

    timeout_ms = int(config.platform.get("usage_timeout_ms", 5000))
    max_retries = int(config.platform.get("usage_max_retries", 3))
    usage_url = _platform_url(platform_base_url, "/gateway/usage")
    headers = {"X-Gateway-Token": config.platform_token or ""}

    payload: dict[str, Any] = {
        "correlation_id": correlation_id,
        "status": outcome,
        "is_final_attempt": is_final_attempt,
    }
    # Forward the caller's session label so the platform can attribute this
    # attempt's spend. Blank is treated as absent; the body field already caps
    # length, so the platform never has to truncate.
    normalized_label = (session_label or "").strip()
    if normalized_label:
        payload["session_label"] = normalized_label
    if outcome == "success":
        if usage is not None:
            payload["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cache_read_tokens": cache_read_tokens_of(usage),
                "cache_write_tokens": cache_write_tokens_of(usage),
            }
    elif error_class is not None:
        payload["error_class"] = error_class

    delay_seconds = 0.25
    for attempt in range(1, max_retries + 1):
        should_retry = False
        try:
            response = await _post_platform(
                url=usage_url,
                headers=headers,
                body=payload,
                timeout_seconds=timeout_ms / 1000,
            )
            if response.status_code in {202, 204}:
                return None
            if response.status_code == 200:
                if outcome != "success":
                    return None
                try:
                    completed = _CompletedUsageSettlement.model_validate(response.json())
                except (ValueError, ValidationError):
                    logger.warning(
                        "Platform usage report returned an invalid completed body correlation_id=%s",
                        correlation_id,
                    )
                    return None
                if completed.correlation_id != correlation_id:
                    logger.warning(
                        "Platform usage report correlation mismatch expected=%s received=%s",
                        correlation_id,
                        completed.correlation_id,
                    )
                    return None
                if completed.usage_status != "reported" or completed.pricing.source is None:
                    # Settlement succeeded but no pricing source was applied, so
                    # ``cost_usd`` is a placeholder zero rather than a priced amount.
                    # Inlining it would read as "this request was free".
                    logger.debug(
                        "Platform settlement is not attachable correlation_id=%s usage_status=%s priced=%s",
                        correlation_id,
                        completed.usage_status,
                        completed.pricing.source is not None,
                    )
                    return None
                return SettledCost(
                    cost_usd=completed.cost_usd,
                    pricing_source=completed.pricing.source,
                )
            if response.status_code in _USAGE_NON_RETRYABLE_STATUS_CODES or response.status_code == 410:
                return None
            should_retry = response.status_code >= 500
        except (httpx.TimeoutException, httpx.NetworkError):
            should_retry = True

        if not should_retry or attempt == max_retries:
            return None

        await asyncio.sleep(delay_seconds)
        delay_seconds *= 2

    return None
