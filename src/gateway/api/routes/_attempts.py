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
    record_abandoned_attempt,
    upstream_exception_shape,
)
from gateway.log_config import logger
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_retrieval_backend import WebSearchNotReachableError
from gateway.types.attempt import Attempt

T = TypeVar("T")

ALL_ATTEMPTS_FAILED_DETAIL = "All upstream providers failed"
ALL_ATTEMPTS_TIMED_OUT_DETAIL = "All upstream providers timed out"
# Fixed, non-leaky: an empty plan is a gateway bug, and the message must not
# describe internals to the caller. The compiler raises a specific 403/400 for
# every *expected* way a plan can end up empty, so reaching this is a defect.
EMPTY_PLAN_DETAIL = "Routing produced no candidate to try"


# Turns one candidate's call kwargs into what it is sent. It takes the
# candidate's instance and its kwargs as they would be dispatched.
PrepareKwargs = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class CandidateCannotServe(Exception):
    """The candidate cannot serve this request, which says nothing about its provider.

    ``refusal`` is what the request is answered with when no candidate can.
    """

    def __init__(self, refusal: HTTPException) -> None:
        super().__init__(refusal.detail)
        self.refusal = refusal


class AttemptFailure(NamedTuple):
    """One failed attempt, for the exhaustion log line."""

    position: int
    instance: str
    model: str
    error_class: str


def classify_local_attempt_error(exc: BaseException) -> tuple[bool, str]:
    """Classify a locally credentialed provider failure as ``(True, class)``.

    The walker advances on every provider failure before lock-in. The boolean
    remains for its generic callback shape; classifications only label logs and
    usage rows. Gateway-side failures are caught before this function runs.
    """
    kind, status_code = upstream_exception_shape(exc)
    if kind is not None:
        return True, kind

    if isinstance(status_code, int):
        if is_provider_billing_error(exc):
            return True, f"http_{status_code}_billing"
        return True, f"http_{status_code}"

    return True, "unknown"


async def walk_attempts(
    *,
    attempts: Sequence[Attempt],
    base_request_fields: dict[str, Any],
    run_attempt: Callable[[Attempt, dict[str, Any], Callable[[], None]], Awaitable[T]],
    max_tool_iterations: int,
    policy_name: str | None = None,
    classify_error: Callable[[BaseException], tuple[bool, str]] = classify_local_attempt_error,
    build_kwargs: Callable[[Attempt, dict[str, Any]], dict[str, Any]] | None = None,
    prepare_kwargs: PrepareKwargs | None = None,
    on_absorbed: Callable[[Attempt, BaseException, int], Awaitable[None]] | None = None,
    on_terminal: Callable[[Attempt], None] | None = None,
) -> tuple[Attempt, T]:
    """Try each attempt in order; return ``(chosen, result)`` for the first success.

    ``run_attempt`` receives the attempt, its merged call kwargs, and a
    ``mark_locked_in`` callback that tool-loop callers fire once the upstream has
    produced its first assistant message.

    ``on_terminal`` is called with the candidate the walk actually stopped on,
    before the terminal error is raised. The caller cannot infer it: exhaustion
    reaches the last candidate, while a tool-loop lock-in, a gateway-side refusal
    for one candidate (a refused reservation top-up, an unpriced fallback), or the
    tool-iteration cap can stop the walk early. In those cases, attributing the
    failure to the end of the plan would name a provider that was never called.

    ``on_absorbed`` is awaited for each provider failure the walk *recovers* from,
    once the next candidate is about to be sent the request. It exists so the
    caller can record the failure without it counting as a request error, since
    the request itself is still going to be served. It is not called for a terminal failure: that
    one is the request's outcome and the caller logs it as such.

    ``build_kwargs`` builds each candidate's call kwargs, defaulting to
    :meth:`Attempt.call_kwargs`. Formats whose provider call takes a different
    shape pass their own (the responses format splits ``provider`` from ``model``
    and rebuilds its Codex extra-body per provider), so the transformation happens
    for the candidate being tried rather than for the one that failed.

    ``prepare_kwargs`` then turns those kwargs into what the candidate is sent,
    for work that depends on the account a candidate's credential reaches.
    It may raise :class:`CandidateCannotServe`, which skips the candidate without
    counting it as a provider failure and without reordering the plan. When no
    candidate is left, the request is answered with the last one's refusal.

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
    when only one candidate was called (so a single-candidate policy answers
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
    cannot_serve: CandidateCannotServe | None = None
    # A failure is absorbed only once another candidate is sent the request.
    unabsorbed: tuple[Attempt, BaseException] | None = None
    last_failed: Attempt | None = None

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
            call_kwargs = make_kwargs(attempt, base_request_fields)
            if prepare_kwargs is not None:
                call_kwargs = await prepare_kwargs(attempt.instance, call_kwargs)
            if unabsorbed is not None and on_absorbed is not None:
                await on_absorbed(*unabsorbed, len(attempts))
            unabsorbed = None
            result = await run_attempt(attempt, call_kwargs, _mark_locked_in)
        except CandidateCannotServe as exc:
            logger.info(
                "Candidate cannot serve the request policy=%s position=%d instance=%s model=%s reason=%s",
                policy_name,
                attempt.position,
                attempt.instance,
                attempt.model,
                exc,
            )
            cannot_serve = exc
            continue
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
            last_failed = attempt
            if not locked_in:
                reason = "timeout" if error_class == "timeout" else "upstream_error"
                record_abandoned_attempt(attempt.instance, attempt.model, reason, attempt.position)
            if locked_in or not retryable:
                if on_terminal is not None:
                    on_terminal(attempt)
                raise _provider_failure_http_exc(exc, fallback_detail="LLM provider error") from exc
            failures.append(AttemptFailure(attempt.position, attempt.instance, attempt.model, error_class))
            unabsorbed = (attempt, exc)
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

    if cannot_serve is not None and not failures:
        if on_terminal is not None:
            on_terminal(attempts[-1])
        raise cannot_serve.refusal

    logger.error("All attempts failed policy=%s failures=%s", policy_name, failures)
    # A candidate that could not serve was never called, so the last one that
    # failed is the one to name, and the status rule counts only the ones called.
    if on_terminal is not None:
        on_terminal(last_failed or attempts[-1])
    single = len(failures) <= 1
    if last_exc is not None and upstream_exception_shape(last_exc)[0] == "timeout":
        detail = "LLM provider timeout" if single else ALL_ATTEMPTS_TIMED_OUT_DETAIL
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail) from last_exc
    if single and last_exc is not None:
        raise _provider_failure_http_exc(last_exc, fallback_detail="LLM provider error") from last_exc
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=ALL_ATTEMPTS_FAILED_DETAIL) from last_exc
