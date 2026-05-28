"""Platform-mode shared infrastructure.

Holds the resolved-route Pydantic types and the generic ``run_platform_attempts``
runner that the chat / messages / responses endpoints all use for multi-attempt
fallback on the non-streaming path. The runner is format-agnostic — callers
pass a per-attempt dispatcher and a usage-extractor and the runner handles
iteration, error classification, lock-in semantics, and the terminal
all-failed status mapping uniformly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, NamedTuple, TypeVar

import httpx
from any_llm import LLMProvider
from fastapi import HTTPException, status
from pydantic import BaseModel

from gateway.log_config import logger
from gateway.services.mcp_loop import MaxToolIterationsExceeded

T = TypeVar("T")


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

    If every attempt fails the runner raises 504 (on timeout) or 502
    (otherwise), with the detail string distinguishing single-attempt from
    multi-attempt outcomes.

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
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="LLM provider error",
                ) from exc
            if not retryable:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="LLM provider error",
                ) from exc
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
    detail = "LLM provider error" if is_single_attempt else "All upstream providers failed"
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=detail,
    ) from last_exc
