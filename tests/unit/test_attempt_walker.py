"""Unit tests for the credential-agnostic attempt walker (issue #463).

Two things are being pinned. First, the walk itself: order, when it advances,
when it refuses to, and what status an exhausted plan produces. Second, the
deliberate difference from the hybrid walker: 401/403 are terminal here, because
a standalone operator owns every credential in their own config and failing over
a broken key would move that traffic to another provider and hide the
misconfiguration.
"""

import asyncio
from typing import Any, Literal, cast

import httpx
import pytest
from any_llm import LLMProvider
from fastapi import HTTPException

from gateway.api.routes._attempts import (
    ALL_ATTEMPTS_FAILED_DETAIL,
    ALL_ATTEMPTS_TIMED_OUT_DETAIL,
    EMPTY_PLAN_DETAIL,
    classify_local_attempt_error,
    walk_attempts,
)
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.types.attempt import Attempt


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://upstream")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


def _attempt(position: int, model: str, *, instance: str = "openai", **overrides: Any) -> Attempt:
    defaults: dict[str, Any] = {
        "position": position,
        "instance": instance,
        "provider": LLMProvider.OPENAI,
        "model": model,
        "kwargs": {"api_key": "sk-test"},
    }
    defaults.update(overrides)
    return Attempt(**defaults)


async def _walk(attempts: list[Attempt], behaviors: list[Any], **overrides: Any) -> tuple[Attempt, Any]:
    """Walk ``attempts`` where ``behaviors[i]`` is raised (if an exception) or
    returned for attempt ``i``.
    """
    calls: list[dict[str, Any]] = []

    async def run_attempt(attempt: Attempt, call_kwargs: dict[str, Any], mark_locked_in: Any) -> Any:
        calls.append(call_kwargs)
        behavior = behaviors[attempt.position - 1]
        if overrides.get("lock_in"):
            mark_locked_in()
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior

    chosen, result = await walk_attempts(
        attempts=attempts,
        base_request_fields={"messages": [{"role": "user", "content": "hi"}]},
        run_attempt=run_attempt,
        max_tool_iterations=10,
        policy_name=overrides.get("policy_name", "fast"),
        on_terminal=overrides.get("on_terminal"),
    )
    return chosen, result


# ---------------------------------------------------------------------------
# Call-kwargs construction
# ---------------------------------------------------------------------------


def test_call_kwargs_applies_the_attempts_selector_last() -> None:
    attempt = _attempt(1, "gpt-5-mini", kwargs={"api_key": "sk-real", "api_base": "http://x"})
    merged = attempt.call_kwargs({"messages": [], "model": "whatever-the-caller-said"})

    assert merged["model"] == "openai:gpt-5-mini"
    assert merged["api_key"] == "sk-real"
    assert merged["api_base"] == "http://x"


def test_caller_supplied_credentials_never_reach_the_merge() -> None:
    """``call_kwargs`` lets request fields override credentials by key, matching
    the hybrid helper. That is only safe because a caller cannot get an
    ``api_key`` or ``api_base`` into those fields: the request schema derives
    from any-llm's ``CompletionParams``, which has neither, and pydantic drops
    unknown fields. This pins the assumption the merge order relies on.
    """
    from gateway.api.routes.chat import ChatCompletionRequest

    request = ChatCompletionRequest.model_validate(
        {
            "model": "fast",
            "messages": [{"role": "user", "content": "hi"}],
            "api_key": "sk-attacker",
            "api_base": "http://attacker.example",
        }
    )
    dumped = request.model_dump(exclude_unset=True)

    assert "api_key" not in dumped
    assert "api_base" not in dumped


def test_dispatch_model_uses_the_implementation_not_the_instance() -> None:
    attempt = _attempt(1, "gpt-5-mini", instance="prod-openai")
    assert attempt.dispatch_model == "openai:gpt-5-mini"
    assert attempt.instance == "prod-openai"


# ---------------------------------------------------------------------------
# Walking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_attempt_success() -> None:
    chosen, result = await _walk([_attempt(1, "gpt-5-mini")], ["ok"])
    assert chosen.position == 1
    assert result == "ok"


@pytest.mark.asyncio
async def test_advances_past_a_retryable_failure() -> None:
    chosen, result = await _walk(
        [_attempt(1, "gpt-5-mini"), _attempt(2, "claude-haiku-4-5", instance="anthropic")],
        [_http_error(503), "ok"],
    )
    assert chosen.position == 2
    assert chosen.model == "claude-haiku-4-5"
    assert result == "ok"


@pytest.mark.asyncio
async def test_stops_on_a_malformed_request_without_advancing() -> None:
    """A 400 would be rejected by every provider, so falling through is waste."""
    second_ran = False

    async def run_attempt(attempt: Attempt, call_kwargs: dict[str, Any], mark_locked_in: Any) -> Any:
        nonlocal second_ran
        if attempt.position == 1:
            raise _http_error(400)
        second_ran = True
        return "ok"

    with pytest.raises(HTTPException) as exc_info:
        await walk_attempts(
            attempts=[_attempt(1, "a"), _attempt(2, "b")],
            base_request_fields={},
            run_attempt=run_attempt,
            max_tool_iterations=10,
        )

    assert second_ran is False
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_locked_in_failure_does_not_advance() -> None:
    """Once the upstream produced its first assistant message, the tool-loop
    state cannot be replayed on another provider.
    """
    with pytest.raises(HTTPException):
        await _walk(
            [_attempt(1, "a"), _attempt(2, "b")],
            [_http_error(503), "ok"],
            lock_in=True,
        )


@pytest.mark.asyncio
async def test_tool_iteration_cap_is_a_gateway_error_not_a_provider_one() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _walk([_attempt(1, "a"), _attempt(2, "b")], [MaxToolIterationsExceeded("cap hit"), "ok"])
    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_backend_unreachable_propagates_raw() -> None:
    """The same sandbox serves every attempt, so failing over cannot help, and
    the caller needs the distinct type to map it to a backend-specific 502.
    """
    with pytest.raises(SandboxNotReachableError):
        await _walk([_attempt(1, "a"), _attempt(2, "b")], [SandboxNotReachableError("down"), "ok"])


@pytest.mark.asyncio
async def test_http_exception_from_an_attempt_is_passed_through() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _walk([_attempt(1, "a"), _attempt(2, "b")], [HTTPException(status_code=402, detail="no pricing"), "ok"])
    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == "no pricing"


@pytest.mark.asyncio
async def test_a_gateway_side_refusal_reports_the_candidate_it_happened_on() -> None:
    """A refused reservation top-up (or an unpriced fallback under
    ``require_pricing``) raises an ``HTTPException`` from the middle of the plan.
    The caller writes the failure's usage row from ``on_terminal``, so without it
    the row would name the last candidate, which was never called.
    """
    stopped: list[Attempt] = []
    with pytest.raises(HTTPException):
        await _walk(
            [_attempt(1, "a"), _attempt(2, "b"), _attempt(3, "c")],
            [_http_error(503), HTTPException(status_code=402, detail="budget"), "ok"],
            on_terminal=stopped.append,
        )
    assert [attempt.position for attempt in stopped] == [2]


@pytest.mark.asyncio
async def test_the_tool_iteration_cap_reports_the_candidate_it_happened_on() -> None:
    stopped: list[Attempt] = []
    with pytest.raises(HTTPException) as exc_info:
        await _walk(
            [_attempt(1, "a"), _attempt(2, "b"), _attempt(3, "c")],
            [_http_error(503), MaxToolIterationsExceeded("cap hit"), "ok"],
            on_terminal=stopped.append,
        )
    assert exc_info.value.status_code == 422
    assert [attempt.position for attempt in stopped] == [2]


@pytest.mark.asyncio
async def test_a_cancelled_request_unwinds_instead_of_becoming_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnected caller is not a provider failure.

    ``CancelledError`` derives from ``BaseException``, so the broad catch that
    lets a non-``Exception`` provider client fall through to the next candidate
    would otherwise classify a cancellation as ``unknown``, count an abandoned
    attempt against a provider that answered fine, and convert it into a 502 that
    suppresses the cancellation.
    """
    abandoned: list[tuple[str, str, str, int]] = []
    monkeypatch.setattr(
        "gateway.api.routes._attempts.record_abandoned_attempt",
        lambda instance, model, reason, position: abandoned.append((instance, model, reason, position)),
    )

    with pytest.raises(asyncio.CancelledError):
        await _walk(
            [_attempt(1, "a"), _attempt(2, "b")],
            [asyncio.CancelledError(), "ok"],
        )
    assert abandoned == []


# ---------------------------------------------------------------------------
# Exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_attempt_exhaustion_is_a_generic_502() -> None:
    """Heterogeneous failures must not attribute one provider's status to the plan."""
    with pytest.raises(HTTPException) as exc_info:
        await _walk([_attempt(1, "a"), _attempt(2, "b")], [_http_error(503), _http_error(429)])
    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == ALL_ATTEMPTS_FAILED_DETAIL


@pytest.mark.asyncio
async def test_single_attempt_exhaustion_keeps_the_classified_status() -> None:
    """A one-candidate policy must answer exactly as naming that model directly
    would, which is what makes the compatibility story testable.
    """
    with pytest.raises(HTTPException) as exc_info:
        await _walk([_attempt(1, "a")], [_http_error(429)])
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_timeout_exhaustion_is_504() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await _walk([_attempt(1, "a"), _attempt(2, "b")], [_http_error(503), asyncio.TimeoutError()])
    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == ALL_ATTEMPTS_TIMED_OUT_DETAIL


@pytest.mark.asyncio
async def test_empty_plan_is_a_500_that_leaks_nothing() -> None:
    """The compiler raises a specific 403/400 for every expected way a plan ends
    up empty, so this is an unreachable-assertion path. Its detail must still be
    a fixed string that describes no internals.
    """

    async def run_attempt(attempt: Attempt, call_kwargs: dict[str, Any], mark_locked_in: Any) -> Any:
        raise AssertionError("must not be called")

    with pytest.raises(HTTPException) as exc_info:
        await walk_attempts(
            attempts=[], base_request_fields={}, run_attempt=run_attempt, max_tool_iterations=10
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == EMPTY_PLAN_DETAIL
    for leaked in ("walk_attempts", "caller", "Internal error"):
        assert leaked not in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Local classification: the deliberate divergence from hybrid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_terminal_for_local_credentials(status: int) -> None:
    """The hybrid walker retries these because a workspace key may have been
    rotated upstream. Locally the operator owns the key, so a 401 is their bug:
    failing over would move the spend and hide it.
    """
    retryable, error_class = classify_local_attempt_error(_http_error(status))
    assert retryable is False
    assert error_class == f"http_{status}"


@pytest.mark.parametrize("status", [404, 405, 408, 409, 410, 429, 500, 502, 503, 504])
def test_transient_and_model_gone_statuses_fall_through(status: int) -> None:
    retryable, _ = classify_local_attempt_error(_http_error(status))
    assert retryable is True


@pytest.mark.parametrize("status", [400, 422])
def test_malformed_request_statuses_are_terminal(status: int) -> None:
    retryable, _ = classify_local_attempt_error(_http_error(status))
    assert retryable is False


@pytest.mark.parametrize(
    "exc, expected",
    [(asyncio.TimeoutError(), "timeout"), (httpx.ConnectError("refused"), "conn_err")],
)
def test_transport_failures_are_retryable(exc: BaseException, expected: str) -> None:
    retryable, error_class = classify_local_attempt_error(exc)
    assert retryable is True
    assert error_class == expected


@pytest.mark.asyncio
async def test_a_401_does_not_advance_the_plan() -> None:
    second_ran = False

    async def run_attempt(attempt: Attempt, call_kwargs: dict[str, Any], mark_locked_in: Any) -> Any:
        nonlocal second_ran
        if attempt.position == 1:
            raise _http_error(401)
        second_ran = True
        return "ok"

    with pytest.raises(HTTPException):
        await walk_attempts(
            attempts=[_attempt(1, "a"), _attempt(2, "b")],
            base_request_fields={},
            run_attempt=run_attempt,
            max_tool_iterations=10,
        )
    assert second_ran is False


# ---------------------------------------------------------------------------
# Policy guardrail merge
#
# A policy's guardrails are the operator's mandate. The caller may add to them
# and may tighten one, but must never be able to weaken one, which is the whole
# reason a policy can carry guardrails at all.
# ---------------------------------------------------------------------------


def _request_context(plan: Any) -> Any:
    """A real RequestContext, so these exercise the production type."""
    import time

    from gateway.api.routes._pipeline import RequestContext
    from gateway.core.config import GatewayConfig

    return RequestContext(
        config=GatewayConfig(),
        db=None,
        # No settlement happens in these tests; the merge reads only `ctx.plan`.
        log_writer=cast(Any, None),
        hybrid_mode=False,
        route=None,
        user_token=None,
        api_key_id=None,
        user_id=None,
        rate_limit_info=None,
        reservation=None,
        started_at=time.monotonic(),
        plan=plan,
    )


def _ctx_with_guardrails(*guardrails: Any) -> Any:
    from gateway.services.routing import CompiledPlan

    return _request_context(
        CompiledPlan(policy_name="p", attempts=[_attempt(1, "m")], guardrails=list(guardrails))
    )


def _guardrail(
    profile: str,
    mode: Literal["block", "monitor"] = "block",
    on_unavailable: Literal["block", "monitor"] = "block",
) -> Any:
    from gateway.models.guardrails import GuardrailConfig

    return GuardrailConfig(profile=profile, mode=mode, on_unavailable=on_unavailable)


def test_a_mandated_guardrail_is_applied_when_the_caller_asked_for_nothing() -> None:
    from gateway.api.routes._pipeline import merge_policy_guardrails

    merged = merge_policy_guardrails(_ctx_with_guardrails(_guardrail("prompt-injection")), None)

    assert merged is not None
    assert [g.profile for g in merged] == ["prompt-injection"]
    assert merged[0].mode == "block"


def test_a_caller_cannot_weaken_a_mandated_guardrail() -> None:
    from gateway.api.routes._pipeline import merge_policy_guardrails

    merged = merge_policy_guardrails(
        _ctx_with_guardrails(_guardrail("prompt-injection", mode="block", on_unavailable="block")),
        [_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")],
    )

    assert merged is not None and len(merged) == 1
    assert merged[0].mode == "block"
    assert merged[0].on_unavailable == "block"


def test_a_caller_may_tighten_a_mandated_guardrail() -> None:
    from gateway.api.routes._pipeline import merge_policy_guardrails

    merged = merge_policy_guardrails(
        _ctx_with_guardrails(_guardrail("prompt-injection", mode="monitor", on_unavailable="monitor")),
        [_guardrail("prompt-injection", mode="block", on_unavailable="block")],
    )

    assert merged is not None
    assert merged[0].mode == "block"
    assert merged[0].on_unavailable == "block"


def test_a_caller_may_add_their_own_guardrails_alongside_a_mandate() -> None:
    from gateway.api.routes._pipeline import merge_policy_guardrails

    merged = merge_policy_guardrails(
        _ctx_with_guardrails(_guardrail("prompt-injection")),
        [_guardrail("pii", mode="monitor")],
    )

    assert merged is not None
    assert sorted(g.profile for g in merged) == ["pii", "prompt-injection"]


def test_an_unrouted_request_keeps_exactly_the_callers_guardrails() -> None:
    from gateway.api.routes._pipeline import merge_policy_guardrails

    unrouted = _request_context(None)
    caller = [_guardrail("pii", mode="monitor")]
    assert merge_policy_guardrails(unrouted, caller) is caller
    assert merge_policy_guardrails(unrouted, None) is None
