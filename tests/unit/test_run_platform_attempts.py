"""Unit tests for the ``run_platform_attempts`` runner.

The runner is exercised end-to-end through the hybrid-mode integration
tests; this file covers narrow defensive paths that are awkward to provoke
through a full request, such as the empty-attempts guard.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from gateway.api.routes import _platform
from gateway.api.routes._platform import ResolvedAttempt, ResolvedRoute, run_platform_attempts
from gateway.core.config import GatewayConfig
from gateway.metrics import REGISTRY


def _abandoned_sample(provider: str, model: str, reason: str, position: int) -> float:
    return (
        REGISTRY.get_sample_value(
            "gateway_abandoned_attempts_total",
            {"provider": provider, "model": model, "reason": reason, "position": str(position)},
        )
        or 0.0
    )


def _single_attempt(provider: str, model: str) -> ResolvedAttempt:
    return ResolvedAttempt(
        attempt_id="a0", position=0, provider=provider, model=model, api_key="k", managed=False
    )


@pytest.mark.asyncio
async def test_empty_attempts_raises_500_with_explicit_diagnostic() -> None:
    """A caller that hands the runner an empty ``attempts`` list is in a
    programming-error state — the route handler should have raised a 502
    "no resolvable provider" before reaching the runner. The runner surfaces
    the bug as a 500 with a clear message rather than falling through to the
    terminal "all upstream providers failed" path (which would carry a
    misleading ``last_exc=None``).
    """
    route = ResolvedRoute(request_id="test", fallback_enabled=False, attempts=[])

    async def _never_called(_kwargs: dict[str, Any], _on_first_response: Any) -> Any:
        raise AssertionError("run_attempt must not be called when attempts is empty")

    with pytest.raises(HTTPException) as ei:
        await run_platform_attempts(
            route=route,
            attempts=[],
            base_request_fields={},
            run_attempt=_never_called,
            extract_usage=lambda _r: None,
            classify_error=lambda _exc: (False, "unknown"),
            report_attempt_outcome=lambda *_args: None,
            on_success=lambda _attempt: None,
            max_tool_iterations=1,
        )
    assert ei.value.status_code == 500
    assert "empty attempts list" in ei.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_class", "expected_reason"),
    [("conn_err", "upstream_error"), ("timeout", "timeout")],
)
async def test_pre_lock_in_failure_counts_as_abandoned(error_class: str, expected_reason: str) -> None:
    """A non-streaming attempt that fails before any assistant message is
    abandonment waste: it is counted under ``gateway_abandoned_attempts`` with a
    reason derived from the error classification."""
    attempt = _single_attempt("openai", f"gpt-{expected_reason}")
    route = ResolvedRoute(request_id="r", fallback_enabled=False, attempts=[attempt])
    before = _abandoned_sample("openai", attempt.model, expected_reason, 0)

    async def _run_attempt(_kwargs: dict[str, Any], _on_first_response: Any) -> Any:
        raise RuntimeError("boom before any output")

    with pytest.raises(HTTPException):
        await run_platform_attempts(
            route=route,
            attempts=[attempt],
            base_request_fields={},
            run_attempt=_run_attempt,
            extract_usage=lambda _r: None,
            classify_error=lambda _exc: (False, error_class),
            report_attempt_outcome=lambda *_a: None,
            on_success=lambda _a: None,
            max_tool_iterations=1,
        )

    assert _abandoned_sample("openai", attempt.model, expected_reason, 0) - before == 1.0


@pytest.mark.asyncio
async def test_locked_in_failure_not_counted_as_abandoned() -> None:
    """A locked-in attempt already produced a first assistant message, so a
    later failure is not abandonment-before-first-chunk and must not inflate the
    counter."""
    attempt = _single_attempt("anthropic", "claude-locked")
    route = ResolvedRoute(request_id="r", fallback_enabled=False, attempts=[attempt])
    before = _abandoned_sample("anthropic", "claude-locked", "upstream_error", 0)

    async def _run_attempt(_kwargs: dict[str, Any], on_first_response: Any) -> Any:
        on_first_response()  # lock in on the first upstream response
        raise RuntimeError("failed after the first assistant message")

    with pytest.raises(HTTPException):
        await run_platform_attempts(
            route=route,
            attempts=[attempt],
            base_request_fields={},
            run_attempt=_run_attempt,
            extract_usage=lambda _r: None,
            classify_error=lambda _exc: (True, "http_500"),
            report_attempt_outcome=lambda *_a: None,
            on_success=lambda _a: None,
            max_tool_iterations=1,
        )

    assert _abandoned_sample("anthropic", "claude-locked", "upstream_error", 0) == before


@pytest.mark.asyncio
async def test_report_platform_usage_does_not_retry_on_402(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 402 from the usage-report endpoint is a permanent rejection (the org
    wallet is overdrawn or missing and won't recover within the retry window).
    The gateway must POST once and give up, never retry."""
    config = cast(
        GatewayConfig,
        SimpleNamespace(
            platform={"base_url": "http://platform", "usage_max_retries": 3},
            platform_token="gw-test",
        ),
    )

    post_mock = AsyncMock(return_value=httpx.Response(402))
    monkeypatch.setattr(_platform, "_post_platform", post_mock)
    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    await _platform._report_platform_usage(config, "corr-1", "success", None)

    assert post_mock.call_count == 1
    sleep_mock.assert_not_awaited()
    # Pin the classification itself, not just the (currently equivalent) retry
    # behaviour: 402 must stay in the non-retryable set even if the >= 500 retry
    # predicate changes.
    assert 402 in _platform._USAGE_NON_RETRYABLE_STATUS_CODES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("session_label", "expected"),
    [
        ("my-run-personas", "my-run-personas"),
        ("  spaced-out  ", "spaced-out"),  # trimmed
        (None, None),  # omitted
        ("   ", None),  # blank treated as absent
    ],
)
async def test_report_platform_usage_forwards_session_label(
    monkeypatch: pytest.MonkeyPatch,
    session_label: str | None,
    expected: str | None,
) -> None:
    """The caller's session label rides the usage report so the platform can
    attribute spend; blank/absent labels are omitted from the payload."""
    config = cast(
        GatewayConfig,
        SimpleNamespace(
            platform={"base_url": "http://platform", "usage_max_retries": 3},
            platform_token="gw-test",
        ),
    )

    post_mock = AsyncMock(return_value=httpx.Response(204))
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    await _platform._report_platform_usage(config, "corr-1", "success", None, session_label=session_label)

    body = post_mock.call_args.kwargs["body"]
    if expected is None:
        assert "session_label" not in body
    else:
        assert body["session_label"] == expected
