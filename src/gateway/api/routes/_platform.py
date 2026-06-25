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
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple, TypeVar

import httpx
from any_llm import LLMProvider
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from gateway.core.config import GatewayConfig
from gateway.core.usage import cache_read_tokens_of, cache_write_tokens_of
from gateway.log_config import logger
from gateway.models.mcp import McpServerConfig
from gateway.services.mcp_loop import MaxToolIterationsExceeded

T = TypeVar("T")

# Status codes returned by the platform's usage-report endpoint that the
# gateway should NOT retry. Auth / payment-required / not-found / conflict /
# unprocessable are all permanent rejection signals — retrying would just
# hammer the platform (an overdrawn or missing wallet won't recover within the
# retry window). 402 is already excluded by the >= 500 retry predicate below;
# listing it keeps the intent explicit and robust to changes in that predicate.
_USAGE_NON_RETRYABLE_STATUS_CODES = {401, 402, 404, 409, 422}

# Status codes that cause the gateway to move on to the next attempt in a
# multi-attempt route. 401/403 are included because users configure
# multi-attempt routing policies on the platform precisely to handle
# credential outages — when they've opted in, an auth failure on one provider
# should fall through to the next, not surface to the client. Single-attempt
# requests still see auth errors directly because there's nothing to fall
# back to.
#
# 404/405/409/410 mean "this model or endpoint is unavailable at THIS provider"
# (deprecated, renamed, retired, or never offered). Recovering from a model
# being retired is one of the main reasons users configure fallback, so these
# fall through to the next entry rather than failing the whole request. They are
# distinct from 400/422, which mean the request itself is malformed and would be
# rejected by every provider, so falling through on those just wastes attempts.
_FALLBACK_RETRYABLE_STATUS_CODES = {401, 403, 404, 405, 408, 409, 410, 429, 500, 502, 503, 504}
_FALLBACK_NON_RETRYABLE_STATUS_CODES = {400, 422}

# Streaming first-chunk timeouts (hybrid-mode fallback). Plain LLM streams
# rarely take long to produce a first token, so a tight cap keeps failed-
# attempt latency low. Tool-loop streams may reason before emitting tokens
# or a tool_call (especially with extended thinking), so they get more
# headroom. Both are operator-tunable via ``config.platform``.
_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS = 2000
_DEFAULT_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP = 30000
_STREAM_FIRST_CHUNK_TIMEOUT_MS_KEY = "streaming_first_chunk_timeout_ms"
_STREAM_FIRST_CHUNK_TIMEOUT_MS_TOOL_LOOP_KEY = "streaming_first_chunk_timeout_ms_tool_loop"


class ResolvedAttempt(BaseModel):
    """A single resolution attempt returned by the platform."""

    attempt_id: str
    position: int
    provider: str
    model: str
    api_base: str | None = None
    api_key: str
    managed: bool


class ResolvedRoute(BaseModel):
    """The full resolution plan returned by the platform."""

    request_id: str
    fallback_enabled: bool
    attempts: list[ResolvedAttempt]


class _AttemptFailure(NamedTuple):
    position: int
    provider: str
    model: str
    error_class: str


def _provider_failure_http_exc(exc: BaseException, *, fallback_detail: str) -> HTTPException:
    """Build the terminal HTTPException for a failed platform attempt.

    Reuses the shared provider-error classifier so platform-mode failures get
    the same specific-but-safe statuses (400/404/429/504) the standalone
    adapters return, instead of a blanket 502. Falls back to a 502 carrying
    ``fallback_detail`` when the failure has no signal we can safely surface.
    The classified detail is always a fixed string, so it cannot leak the raw
    upstream message.
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
    report_attempt_outcome: Callable[[ResolvedAttempt, str, Any, str | None], None],
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
    list and fall through to the next provider per the retryable
    classification.

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

    for attempt in attempts:
        attempt_provider = LLMProvider(attempt.provider)
        attempt_kwargs: dict[str, Any] = {"api_key": attempt.api_key}
        if attempt.api_base:
            attempt_kwargs["api_base"] = attempt.api_base

        completion_kwargs = {
            **attempt_kwargs,
            **base_request_fields,
            "model": f"{attempt_provider.value}:{attempt.model}",
        }

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
        except BaseException as exc:
            retryable, error_class = classify_error(exc)
            report_attempt_outcome(attempt, "error", None, error_class)
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
        report_attempt_outcome(attempt, "success", extract_usage(result), None)
        on_success(attempt)
        return result

    # All attempts exhausted with retryable errors.
    logger.error(
        "All upstream attempts failed request_id=%s failures=%s",
        route.request_id,
        failures,
    )
    is_single_attempt = len(attempts) <= 1
    if last_exc is not None and isinstance(last_exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
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


async def _resolve_platform_credentials(
    config: GatewayConfig,
    user_token: str,
    model_selector: str,
) -> ResolvedRoute:
    """Call the platform's ``/gateway/provider-keys/resolve`` to get the
    routing plan (one or more ``ResolvedAttempt`` entries).
    """
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid mode is misconfigured",
        )

    provider, model_name = _split_model_selector(model_selector)
    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    resolve_url = _platform_url(platform_base_url, "/gateway/provider-keys/resolve")
    resolve_headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }
    resolve_body: dict[str, Any] = {"model": model_name}
    if provider:
        resolve_body["provider"] = provider

    try:
        response = await _post_platform(
            url=resolve_url,
            headers=resolve_headers,
            body=resolve_body,
            timeout_seconds=timeout_ms / 1000,
        )
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
        return _parse_resolve_payload(payload)

    if response.status_code in {401, 402, 403, 404, 429}:
        detail = _safe_detail_from_platform(response, "Authorization request rejected")
        headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=headers)

    if response.status_code == 422 or response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
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
            )
            for att in attempts_payload
        ]
        return ResolvedRoute(
            request_id=str(payload["request_id"]),
            fallback_enabled=bool(payload.get("fallback_enabled", False)),
            attempts=attempts,
        )

    correlation_id = str(payload["correlation_id"])
    return ResolvedRoute(
        request_id=correlation_id,
        fallback_enabled=False,
        attempts=[
            ResolvedAttempt(
                attempt_id=correlation_id,
                position=0,
                provider=str(payload["provider"]),
                model=str(payload["model"]),
                api_base=payload.get("api_base"),
                api_key=str(payload["api_key"]),
                managed=bool(payload.get("managed", False)),
            )
        ],
    )


def _classify_upstream_error(exc: BaseException) -> tuple[bool, str]:
    """Classify an upstream provider error.

    Returns ``(retryable, error_class)``. ``error_class`` is a short string used
    for logging and reporting back to the platform. Streaming-only failures still
    pass through this classifier; the caller decides whether to actually retry.
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return True, "timeout"
    if isinstance(exc, httpx.NetworkError):
        return True, "conn_err"

    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status_code = getattr(resp, "status_code", None)

    if isinstance(status_code, int):
        if status_code in _FALLBACK_NON_RETRYABLE_STATUS_CODES:
            return False, f"http_{status_code}"
        if status_code in _FALLBACK_RETRYABLE_STATUS_CODES or 500 <= status_code <= 599:
            return True, f"http_{status_code}"
        return False, f"http_{status_code}"

    return False, "unknown"


async def _resolve_platform_mcp_servers(
    config: GatewayConfig,
    user_token: str,
    mcp_server_ids: list[uuid.UUID],
) -> list[McpServerConfig]:
    """Swap workspace-scoped MCP server ids for inline configs by calling the platform."""
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid mode is misconfigured",
        )

    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    resolve_url = _platform_url(platform_base_url, "/gateway/mcp-servers/resolve")
    headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }
    body: dict[str, Any] = {"mcp_server_ids": [str(uid) for uid in mcp_server_ids]}

    try:
        response = await _post_platform(url=resolve_url, headers=headers, body=body, timeout_seconds=timeout_ms / 1000)
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
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

    # Mirror the status-code semantics of `_resolve_platform_credentials`:
    # client errors (auth/quota/not-found/rate-limit) are forwarded so the
    # caller sees the real status (and can honour Retry-After on 429), while
    # the platform's server-side or unexpected responses collapse to 502.
    if response.status_code in {401, 402, 403, 404, 429}:
        detail = _safe_detail_from_platform(response, "MCP server resolution failed")
        response_headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            response_headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=response_headers)

    if response.status_code == 422 or response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
    )


async def _resolve_platform_web_search(
    config: GatewayConfig,
    user_token: str,
) -> dict[str, Any]:
    """Resolve the workspace's web-search policy via the platform.

    Mirrors `_resolve_platform_mcp_servers`: same base_url guard, timeout,
    headers, `_post_platform` call, and status-code handling. POSTs an empty
    body to `/gateway/web-search/resolve` and returns the parsed JSON dict on
    200 (``{enabled, provider, max_results, purpose_hint, allowed_domains,
    blocked_domains, provider_options}``). Client errors (auth/quota/not-found/
    rate-limit) forward verbatim; the platform's server-side or unexpected
    responses collapse to 502.
    """
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Hybrid mode is misconfigured",
        )

    timeout_ms = int(config.platform.get("resolve_timeout_ms", 5000))
    resolve_url = _platform_url(platform_base_url, "/gateway/web-search/resolve")
    headers = {
        "X-Gateway-Token": config.platform_token or "",
        "X-User-Token": user_token,
    }
    body: dict[str, Any] = {}

    try:
        response = await _post_platform(url=resolve_url, headers=headers, body=body, timeout_seconds=timeout_ms / 1000)
    except (httpx.TimeoutException, httpx.NetworkError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        ) from None

    if response.status_code == 200:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    # Mirror the status-code semantics of `_resolve_platform_mcp_servers`:
    # client errors (auth/quota/not-found/rate-limit) are forwarded so the
    # caller sees the real status (and can honour Retry-After on 429), while
    # the platform's server-side or unexpected responses collapse to 502.
    if response.status_code in {401, 402, 403, 404, 429}:
        detail = _safe_detail_from_platform(response, "Web search resolution failed")
        response_headers: dict[str, str] | None = None
        if response.status_code == 429 and response.headers.get("Retry-After"):
            response_headers = {"Retry-After": response.headers["Retry-After"]}
        raise HTTPException(status_code=response.status_code, detail=detail, headers=response_headers)

    if response.status_code == 422 or response.status_code >= 500:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Authorization service unavailable",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Authorization service unavailable",
    )


async def _report_platform_usage(
    config: GatewayConfig,
    correlation_id: str,
    outcome: str,
    usage: CompletionUsage | None,
    error_class: str | None = None,
) -> None:
    """POST a usage record back to the platform with bounded retries.

    Best-effort — failures are swallowed after ``max_retries`` so they don't
    impact the user's response path. Non-retryable status codes (auth /
    payment-required / not-found / conflict / unprocessable) short-circuit the
    retry loop.
    """
    platform_base_url = config.platform.get("base_url")
    if not platform_base_url:
        return

    timeout_ms = int(config.platform.get("usage_timeout_ms", 5000))
    max_retries = int(config.platform.get("usage_max_retries", 3))
    usage_url = _platform_url(platform_base_url, "/gateway/usage")
    headers = {"X-Gateway-Token": config.platform_token or ""}

    payload: dict[str, Any] = {"correlation_id": correlation_id, "status": outcome}
    if outcome == "success":
        token_usage = usage or CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        payload["usage"] = {
            "prompt_tokens": token_usage.prompt_tokens,
            "completion_tokens": token_usage.completion_tokens,
            "total_tokens": token_usage.total_tokens,
            "cache_read_tokens": cache_read_tokens_of(token_usage),
            "cache_write_tokens": cache_write_tokens_of(token_usage),
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
            if response.status_code == 204:
                return
            if response.status_code in _USAGE_NON_RETRYABLE_STATUS_CODES:
                return
            should_retry = response.status_code >= 500
        except (httpx.TimeoutException, httpx.NetworkError):
            should_retry = True

        if not should_retry or attempt == max_retries:
            return

        await asyncio.sleep(delay_seconds)
        delay_seconds *= 2
