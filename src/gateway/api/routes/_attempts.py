"""Walk an ordered list of attempts, returning the first that succeeds.

This is the executor half of routing: something upstream decides *which*
candidates to try and in what order (a routing policy), and this module tries them. It makes no
selection decisions of its own.

It deliberately does **not** settle the budget. A reservation is per request,
not per attempt, so the caller reserves once, calls this, and reconciles or
refunds once against the attempt that actually served the request. Settling in
here would double-charge a chain.

Import direction: ``_pipeline`` imports this module, so the provider-error
classifier it needs from there is imported where it is called.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastapi import HTTPException, status

from gateway.api.routes._attempt_metrics import record_abandoned_attempt
from gateway.api.routes._helpers import with_retry_hint
from gateway.core.error_codes import (
    ALL_CANDIDATES_REJECTED,
    CONTEXT_LENGTH_EXCEEDED,
    PROVIDER_CREDENTIAL_REJECTED,
    UNSUPPORTED_CONTENT,
    error_code_of,
    error_headers,
)
from gateway.log_config import logger
from gateway.services.inference import (
    is_empty_completion,
    is_provider_billing_error,
    is_provider_credential_rejection,
    upstream_exception_shape,
)
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.provider_kwargs import (
    missing_credential,
    no_candidate_configured_detail,
)
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


def provider_failure_http_exc(
    exc: BaseException,
    *,
    fallback_detail: str,
) -> HTTPException:
    """Build the terminal HTTPException for a failed attempt.

    Uses the shared provider-error classifier for a specific-but-safe status
    (400/404/429/504), and falls back to a 502 whose detail says what happened
    upstream, or is ``fallback_detail`` when the failure has no signal we can
    safely surface.
    """
    # Deferred import: _pipeline imports this module, so importing it at module
    # scope would be circular.
    from gateway.api.routes._pipeline import classify_provider_error, provider_error_headers, provider_failure_detail

    mapping = classify_provider_error(exc)
    if mapping is not None:
        return HTTPException(
            status_code=mapping.status_code,
            detail=mapping.detail,
            headers=with_retry_hint(exc, mapping.status_code, provider_error_headers(exc, mapping.status_code)),
        )
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=provider_failure_detail(exc, fallback_detail),
        headers=with_retry_hint(exc, status.HTTP_502_BAD_GATEWAY, None),
    )


def all_attempts_rejected_detail(tried: int) -> str:
    """400 detail for a request every attempt rejected as invalid."""
    return f"No model accepted the request ({tried} attempts). Check the request parameters."


def prompt_too_long_for_every_attempt_detail(tried: int) -> str:
    """400 detail for a prompt every attempt rejected as too long."""
    return f"The prompt is too long for every model tried ({tried} attempts)."


def no_attempt_accepts_images_detail(tried: int) -> str:
    """400 detail for a request carrying an image that no attempt's model accepts."""
    return f"None of the models tried accepts image content ({tried} attempts)."


def no_attempt_served_detail(tried: int) -> str:
    """502 detail for a request no attempt could serve, because none of them knew the model."""
    return f"No model could serve the request ({tried} attempts)."


def no_credential_accepted_detail(tried: int) -> str:
    """424 detail for a request whose every attempt lacked a credential or had it rejected by the provider."""
    return f"No provider accepted the gateway's credentials for this request ({tried} attempts)."


def get_shared_rejection(errors: Sequence[BaseException]) -> HTTPException | None:
    """The answer for attempts that all rejected the request alike, or ``None`` when they failed differently.

    The detail is the gateway's own, because one provider's message does not explain why the whole plan failed.
    """
    if len(errors) < 2:
        return None
    missing = [entry for entry in map(missing_credential, errors) if entry is not None]
    if len(missing) == len(errors):
        return HTTPException(
            status_code=missing[0].status_code,
            detail=no_candidate_configured_detail(missing),
            headers=error_headers(missing[0].code),
        )
    rejected = sum(1 for error in errors if is_provider_credential_rejection(error))
    if rejected and rejected + len(missing) == len(errors):
        return HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail=no_credential_accepted_detail(len(errors)),
            headers=error_headers(PROVIDER_CREDENTIAL_REJECTED),
        )
    answers = [provider_failure_http_exc(error, fallback_detail="") for error in errors]
    statuses = {answer.status_code for answer in answers}
    if statuses == {status.HTTP_404_NOT_FOUND}:
        return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=no_attempt_served_detail(len(errors)))
    if statuses != {status.HTTP_400_BAD_REQUEST}:
        return None
    if {error_code_of(answer.headers) for answer in answers} == {CONTEXT_LENGTH_EXCEEDED}:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=prompt_too_long_for_every_attempt_detail(len(errors)),
            headers=error_headers(CONTEXT_LENGTH_EXCEEDED),
        )
    if {error_code_of(answer.headers) for answer in answers} == {UNSUPPORTED_CONTENT}:
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=no_attempt_accepts_images_detail(len(errors)),
            headers=error_headers(UNSUPPORTED_CONTENT),
        )
    # NOTE: Some 400s are the gateway's fault, such as a provider rejecting a model ID it does not serve.
    # The status cannot tell them apart, so such a plan blames the request.
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=all_attempts_rejected_detail(len(errors)),
        headers=error_headers(ALL_CANDIDATES_REJECTED),
    )


# Turns one candidate's call kwargs into what it is sent. It takes the
# candidate's instance and its kwargs as they would be dispatched.
PrepareKwargs = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

# Gives back what admitting a candidate took, once the walk moves past it; the
# argument says whether the candidate was sent the request.
DropAdmission = Callable[[bool], Awaitable[None]]
# Admits a candidate before it is sent the request, or raises CandidateCannotServe.
AdmitAttempt = Callable[[Attempt], Awaitable[DropAdmission | None]]


class CandidateCannotServe(Exception):
    """The candidate cannot serve this request, which says nothing about its provider.

    ``refusal`` is what the request is answered with when no candidate can.
    """

    def __init__(self, refusal: HTTPException) -> None:
        super().__init__(refusal.detail)
        self.refusal = refusal


@dataclass(frozen=True)
class AttemptFailure:
    """One failed attempt: what the exhaustion log line names, and the error behind it."""

    position: int
    instance: str
    model: str
    error_class: str
    error: BaseException = field(repr=False)


def classify_local_attempt_error(exc: BaseException) -> tuple[bool, str]:
    """Classify a locally credentialed provider failure as ``(True, class)``.

    The walker advances on every provider failure before lock-in. The boolean
    remains for its generic callback shape; classifications only label logs and
    usage rows. Gateway-side failures are caught before this function runs.
    """
    if missing_credential(exc) is not None:
        return True, "missing_credential"
    if is_empty_completion(exc):
        return True, "empty_completion"

    kind, status_code = upstream_exception_shape(exc)
    if kind is not None:
        return True, kind

    if isinstance(status_code, int):
        if is_provider_billing_error(exc):
            return True, f"http_{status_code}_billing"
        return True, f"http_{status_code}"

    return True, "unknown"


def _soonest_refusal(refusals: Sequence[tuple[Attempt, HTTPException]]) -> tuple[Attempt, HTTPException]:
    """The candidate and refusal a caller can retry soonest, so ``Retry-After`` is not the last candidate's wait."""

    def _retry_after(entry: tuple[Attempt, HTTPException]) -> float:
        value = (entry[1].headers or {}).get("Retry-After")
        try:
            return float(value) if value is not None else math.inf
        except ValueError:
            return math.inf

    return min(reversed(refusals), key=_retry_after)


async def walk_attempts(
    *,
    attempts: Sequence[Attempt],
    base_request_fields: dict[str, Any],
    run_attempt: Callable[[Attempt, dict[str, Any], Callable[[], None]], Awaitable[T]],
    max_tool_iterations: int,
    policy_name: str | None = None,
    classify_error: Callable[[BaseException], tuple[bool, str]] = classify_local_attempt_error,
    build_kwargs: Callable[[Attempt, dict[str, Any]], dict[str, Any]] | None = None,
    credential_attempt: Callable[[Attempt], Awaitable[Attempt]] | None = None,
    prepare_kwargs: PrepareKwargs | None = None,
    admit_attempt: AdmitAttempt | None = None,
    on_absorbed: Callable[[Attempt, BaseException, int], Awaitable[None]] | None = None,
    on_skipped: Callable[[Attempt, HTTPException], Awaitable[None]] | None = None,
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

    ``on_skipped`` is the same for a candidate that could not serve, with its
    refusal, so a model skipped because it was full is on record beside the one
    that served. A walk that ends in that refusal reports it as the outcome instead.

    ``build_kwargs`` builds each candidate's call kwargs, defaulting to
    :meth:`Attempt.call_kwargs`. Formats whose provider call takes a different
    shape pass their own (the responses format splits ``provider`` from ``model``
    and rebuilds its Codex extra-body per provider), so the transformation happens
    for the candidate being tried rather than for the one that failed.

    ``credential_attempt`` runs first, and returns the candidate as it will be
    sent: itself, or itself served on a credential the deployment owns, which
    may name another instance. It may raise :class:`CandidateCannotServe`.

    ``prepare_kwargs`` then turns those kwargs into what the candidate is sent,
    for work that depends on the account a candidate's credential reaches.
    It may raise :class:`CandidateCannotServe`, which skips the candidate without
    counting it as a provider failure and without reordering the plan. When no
    candidate is left, the request is answered with the refusal whose
    ``Retry-After`` is soonest, or the last one's when none sets it.

    ``admit_attempt`` admits the candidate before ``prepare_kwargs`` runs, so a
    full model is not prepared for (a ``per: model`` rate limit with no room
    raises :class:`CandidateCannotServe` the same way). What it returns is
    awaited when the candidate fails and the walk moves on, so a failed
    candidate does not keep what it was admitted with.

    Lock-in semantics: once ``mark_locked_in`` has
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
    A fallthrough whose candidates all rejected the request alike answers as :func:`get_shared_rejection` says.
    """
    if not attempts:
        logger.error("Attempt plan was empty policy=%s", policy_name)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=EMPTY_PLAN_DETAIL)

    make_kwargs = build_kwargs or (lambda attempt, fields: attempt.call_kwargs(fields))
    failures: list[AttemptFailure] = []
    last_exc: BaseException | None = None
    cannot_serve: CandidateCannotServe | None = None
    refusals: list[tuple[Attempt, HTTPException]] = []
    # A failure is absorbed, and a skip recorded, only once another candidate is sent the request.
    unabsorbed: tuple[Attempt, BaseException] | None = None
    unrecorded_skips: list[tuple[Attempt, HTTPException]] = []
    last_failed: Attempt | None = None

    for planned in attempts:
        # Rebound to the candidate as it is sent once ``credential_attempt`` has run.
        attempt = planned
        locked_in = False
        drop_admission: DropAdmission | None = None
        sent = False

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
            if credential_attempt is not None:
                attempt = await credential_attempt(attempt)
            call_kwargs = make_kwargs(attempt, base_request_fields)
            if admit_attempt is not None:
                drop_admission = await admit_attempt(attempt)
            if prepare_kwargs is not None:
                call_kwargs = await prepare_kwargs(attempt.instance, call_kwargs)
            if unabsorbed is not None and on_absorbed is not None:
                await on_absorbed(*unabsorbed, len(attempts))
            unabsorbed = None
            if on_skipped is not None:
                for skipped, refusal in unrecorded_skips:
                    await on_skipped(skipped, refusal)
            unrecorded_skips = []
            sent = True
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
            refusals.append((attempt, exc.refusal))
            unrecorded_skips.append((attempt, exc.refusal))
            if drop_admission is not None:
                await drop_admission(False)
            continue
        except HTTPException:
            # A gateway-side refusal for this candidate (a refused reservation
            # top-up, an unpriced fallback under `require_pricing`), not a provider
            # failure to try the next one for. Report the candidate it happened on:
            # the caller cannot infer it, and defaulting to the end of the plan
            # would name a provider that was never called.
            if drop_admission is not None and not locked_in:
                # Refused before the provider answered, so its request is not counted against the model.
                await drop_admission(False)
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
                raise provider_failure_http_exc(exc, fallback_detail="LLM provider error") from exc
            failures.append(AttemptFailure(attempt.position, attempt.instance, attempt.model, error_class, exc))
            unabsorbed = (attempt, exc)
            if drop_admission is not None:
                await drop_admission(sent)
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
        refused_on, refusal = _soonest_refusal(refusals)
        if on_terminal is not None:
            on_terminal(refused_on)
        raise refusal

    if on_skipped is not None:
        for skipped, refusal in unrecorded_skips:
            await on_skipped(skipped, refusal)

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
        raise provider_failure_http_exc(last_exc, fallback_detail="LLM provider error") from last_exc
    if (rejection := get_shared_rejection([failure.error for failure in failures])) is not None:
        raise rejection from last_exc
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=ALL_ATTEMPTS_FAILED_DETAIL) from last_exc
