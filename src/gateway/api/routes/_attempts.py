"""Walk an ordered list of attempts, returning the first that succeeds.

This is the executor half of routing: something upstream decides *which*
candidates to try and in what order (a local routing policy, or the platform's
resolve response in hybrid mode), and this module tries them. It makes no
selection decisions of its own, which is the property that keeps the data plane
free of routing logic (see ARCHITECTURE.md).

It deliberately does **not** settle the budget. A reservation is per request,
not per attempt, so the caller reserves once, calls this, and reconciles or
refunds once against the attempt that actually served the request. Settling in
here would double-charge a chain.

Relationship to :func:`gateway.api.routes._platform.run_platform_attempts`: that
one is the hybrid-mode walker, coupled to the platform's ``ResolvedAttempt``
shape and its per-attempt upstream reporting. This one is credential-agnostic
(:class:`gateway.types.attempt.Attempt` carries an opaque ``kwargs`` dict), so a
locally resolved attempt with no API key at all works. The two share the error
classification and the terminal-status mapping, so a failure looks the same to a
caller whichever walker produced it. Hybrid stays on its own walker for now;
moving it here is a follow-up guarded by the settlement characterization tests in
``tests/unit/test_pipeline_settlement.py``.

Import direction: this module imports ``_platform`` for the shared classifier and
status mapping, and ``_pipeline`` imports this one. ``_platform`` already defers
its own import of ``_pipeline`` to break a pre-existing cycle, so this adds no
new edge.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NamedTuple, TypeVar

from fastapi import HTTPException, status

from gateway.api.routes._platform import (
    _provider_failure_http_exc,
    is_provider_billing_error,
    upstream_exception_shape,
)
from gateway.log_config import logger
from gateway.metrics import record_abandoned_attempt
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError
from gateway.types.attempt import Attempt

T = TypeVar("T")

# Statuses worth trying the next candidate for, when the credentials are the
# operator's own. Deliberately narrower than the hybrid set: 401/403 are absent.
#
# The hybrid walker retries them because a workspace's upstream key can be
# rotated or revoked out from under the platform, so the next attempt may hold a
# working credential. A standalone operator configured every provider in this
# gateway's own config, so a 401 means *they* have a broken key: failing over
# would move that traffic (and its spend) to another provider and hide the
# misconfiguration behind a working response. Fail loudly instead.
_LOCAL_RETRYABLE_STATUS_CODES = {404, 405, 408, 409, 410, 429, 500, 502, 503, 504}
_LOCAL_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 422}

ALL_ATTEMPTS_FAILED_DETAIL = "All upstream providers failed"
ALL_ATTEMPTS_TIMED_OUT_DETAIL = "All upstream providers timed out"
# Fixed, non-leaky: an empty plan is a gateway bug, and the message must not
# describe internals to the caller. The compiler raises a specific 403/400 for
# every *expected* way a plan can end up empty, so reaching this is a defect.
EMPTY_PLAN_DETAIL = "Routing produced no candidate to try"


class AttemptFailure(NamedTuple):
    """One failed attempt, for the exhaustion log line."""

    position: int
    instance: str
    model: str
    error_class: str


def classify_local_attempt_error(exc: BaseException) -> tuple[bool, str]:
    """Classify a locally credentialed provider failure as ``(retryable, class)``.

    Mirrors the hybrid classifier's shape so logs and reported error classes are
    comparable, but applies :data:`_LOCAL_RETRYABLE_STATUS_CODES`. Transport
    failures (timeout, connection error) are retryable in both.
    """
    kind, status_code = upstream_exception_shape(exc)
    if kind is not None:
        return True, kind

    if isinstance(status_code, int):
        # Checked first, as in the hybrid classifier: a provider reporting
        # account billing exhaustion as a 400/402/422 is an account condition on
        # *this* provider, not a malformed request, so the next candidate is
        # worth trying even though the bare status is non-retryable.
        if is_provider_billing_error(exc):
            return True, f"http_{status_code}_billing"
        if status_code in _LOCAL_NON_RETRYABLE_STATUS_CODES:
            return False, f"http_{status_code}"
        if status_code in _LOCAL_RETRYABLE_STATUS_CODES or 500 <= status_code <= 599:
            return True, f"http_{status_code}"
        return False, f"http_{status_code}"

    return False, "unknown"


async def walk_attempts(
    *,
    attempts: Sequence[Attempt],
    base_request_fields: dict[str, Any],
    run_attempt: Callable[[Attempt, dict[str, Any], Callable[[], None]], Awaitable[T]],
    max_tool_iterations: int,
    policy_name: str | None = None,
    classify_error: Callable[[BaseException], tuple[bool, str]] = classify_local_attempt_error,
    build_kwargs: Callable[[Attempt, dict[str, Any]], dict[str, Any]] | None = None,
    on_absorbed: Callable[[Attempt, BaseException, int], Awaitable[None]] | None = None,
    on_terminal: Callable[[Attempt], None] | None = None,
) -> tuple[Attempt, T]:
    """Try each attempt in order; return ``(chosen, result)`` for the first success.

    ``run_attempt`` receives the attempt, its merged call kwargs, and a
    ``mark_locked_in`` callback that tool-loop callers fire once the upstream has
    produced its first assistant message.

    ``on_terminal`` is called with the candidate the walk actually stopped on,
    before the terminal error is raised. The caller cannot infer it: only a
    retryable exhaustion reaches the last candidate, while a non-retryable status,
    a tool-loop lock-in, a gateway-side refusal for one candidate (a refused
    reservation top-up, an unpriced fallback) or the tool-iteration cap all stop
    the walk early, and attributing the failure to the end of the plan would name
    a provider that was never called.

    ``on_absorbed`` is awaited for each attempt the walk *recovers* from, i.e. a
    retryable failure with another candidate left to try. It exists so the caller can
    record the failure without it counting as a request error, since the request
    itself is still going to be served. It is not called for a terminal failure: that
    one is the request's outcome and the caller logs it as such.

    ``build_kwargs`` builds each candidate's call kwargs, defaulting to
    :meth:`Attempt.call_kwargs`. Formats whose provider call takes a different
    shape pass their own (the responses format splits ``provider`` from ``model``
    and rebuilds its Codex extra-body per provider), so the transformation happens
    for the candidate being tried rather than for the one that failed.

    Lock-in semantics, matching the hybrid walker: once ``mark_locked_in`` has
    fired, a later failure on that attempt terminates the request instead of
    falling through, because a tool-use loop's intermediate state (provider
    specific tool-call ids, reasoning blocks) cannot be replayed on a different
    provider.

    Failures that are not the provider's fault do not advance the plan:
    ``MaxToolIterationsExceeded`` is a gateway-side cap (422), and an unreachable
    sandbox or web-search backend is gateway-side infrastructure that serves
    every attempt equally, so trying the next candidate cannot help.

    On exhaustion: 504 when the last failure was a timeout, the classified status
    when there was only one candidate (so a single-candidate policy answers
    exactly as naming that model directly would), and a generic 502 for a
    multi-candidate fallthrough, which aggregates heterogeneous failures and must
    not attribute one provider's status to the whole plan.
    """
    if not attempts:
        logger.error("Attempt plan was empty policy=%s", policy_name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=EMPTY_PLAN_DETAIL)

    make_kwargs = build_kwargs or (lambda attempt, fields: attempt.call_kwargs(fields))
    failures: list[AttemptFailure] = []
    last_exc: BaseException | None = None

    for attempt in attempts:
        locked_in = False

        def _mark_locked_in(_attempt: Attempt = attempt) -> None:
            nonlocal locked_in
            locked_in = True
            logger.info(
                "Tool-loop lock-in policy=%s position=%d instance=%s model=%s",
                policy_name,
                _attempt.position,
                _attempt.instance,
                _attempt.model,
            )

        try:
            result = await run_attempt(attempt, make_kwargs(attempt, base_request_fields), _mark_locked_in)
        except HTTPException:
            # A gateway-side refusal for this candidate (a refused reservation
            # top-up, an unpriced fallback under `require_pricing`), not a provider
            # failure to try the next one for. Report the candidate it happened on:
            # the caller cannot infer it, and defaulting to the end of the plan
            # would name a provider that was never called.
            if on_terminal is not None:
                on_terminal(attempt)
            raise
        except MaxToolIterationsExceeded as exc:
            logger.warning(
                "Tool loop iteration cap hit policy=%s position=%d cap=%d",
                policy_name,
                attempt.position,
                max_tool_iterations,
            )
            if on_terminal is not None:
                on_terminal(attempt)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        except (SandboxNotReachableError, WebSearchNotReachableError):
            raise
        except asyncio.CancelledError:
            # The catch below is `BaseException` so a provider client raising
            # outside the `Exception` hierarchy still falls through to the next
            # candidate. Cancellation is the one case that must not: the caller is
            # gone, so there is nobody to serve and no provider at fault. Letting
            # the classifier see it would record an `upstream_error` attempt
            # against a provider that answered fine, and swallowing it into an
            # HTTPException would suppress the cancellation the server is waiting
            # to unwind.
            raise
        except BaseException as exc:
            retryable, error_class = classify_error(exc)
            logger.warning(
                "Attempt failed policy=%s position=%d instance=%s model=%s error=%s retryable=%s locked_in=%s",
                policy_name,
                attempt.position,
                attempt.instance,
                attempt.model,
                error_class,
                retryable,
                locked_in,
            )
            last_exc = exc
            if not locked_in:
                reason = "timeout" if error_class == "timeout" else "upstream_error"
                record_abandoned_attempt(attempt.instance, attempt.model, reason, attempt.position)
            if locked_in or not retryable:
                if on_terminal is not None:
                    on_terminal(attempt)
                raise _provider_failure_http_exc(exc, fallback_detail="LLM provider error") from exc
            failures.append(AttemptFailure(attempt.position, attempt.instance, attempt.model, error_class))
            # Only a failure with somewhere left to go is "absorbed"; the last one is
            # the request's own outcome and is logged by the caller as an error.
            if on_absorbed is not None and attempt.position < len(attempts):
                await on_absorbed(attempt, exc, len(attempts))
            continue

        if failures:
            logger.info(
                "Attempt succeeded after %d failure(s) policy=%s position=%d instance=%s model=%s",
                len(failures),
                policy_name,
                attempt.position,
                attempt.instance,
                attempt.model,
            )
        return attempt, result

    logger.error("All attempts failed policy=%s failures=%s", policy_name, failures)
    # Exhaustion did reach the end of the plan, so the last candidate is the one
    # that failed last.
    if on_terminal is not None:
        on_terminal(attempts[-1])
    single = len(attempts) <= 1
    if last_exc is not None and upstream_exception_shape(last_exc)[0] == "timeout":
        detail = "LLM provider timeout" if single else ALL_ATTEMPTS_TIMED_OUT_DETAIL
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail) from last_exc
    if single and last_exc is not None:
        raise _provider_failure_http_exc(last_exc, fallback_detail="LLM provider error") from last_exc
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=ALL_ATTEMPTS_FAILED_DETAIL) from last_exc
