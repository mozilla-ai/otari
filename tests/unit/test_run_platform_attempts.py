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
from any_llm.types.completion import CompletionUsage
from fastapi import HTTPException

from gateway.api.routes import _platform
from gateway.api.routes._platform import ResolvedAttempt, ResolvedRoute, default_attempt_kwargs, run_platform_attempts
from gateway.core.config import GatewayConfig
from gateway.metrics import REGISTRY
from gateway.services.mcp_loop import MaxToolIterationsExceeded
from gateway.services.sandbox_backend import SandboxNotReachableError
from gateway.services.web_search_backend import WebSearchNotReachableError


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
async def test_cancelled_request_unwinds_instead_of_becoming_a_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnected caller is not a provider failure in hybrid mode.

    ``CancelledError`` derives from ``BaseException``, so the broad catch that
    lets a non-``Exception`` provider client fall through to the next candidate
    would otherwise classify a cancellation as ``unknown``, count an abandoned
    attempt against a provider that answered fine, and convert it into an
    HTTPException that suppresses the cancellation. The guard re-raises the
    cancellation ahead of the broad catch, mirroring the standalone walker.
    """
    abandoned: list[tuple[str, str, str, int]] = []
    monkeypatch.setattr(
        "gateway.api.routes._platform.record_abandoned_attempt",
        lambda provider, model, reason, position: abandoned.append((provider, model, reason, position)),
    )

    attempts = [_single_attempt("openai", "gpt-primary"), _single_attempt("openai", "gpt-fallback")]

    async def _run_attempt(_kwargs: dict[str, Any], _on_first_response: Any) -> Any:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await run_platform_attempts(
            route=ResolvedRoute(request_id="r", fallback_enabled=True, attempts=attempts),
            attempts=attempts,
            base_request_fields={},
            run_attempt=_run_attempt,
            extract_usage=lambda _r: None,
            classify_error=lambda _exc: (False, "unknown"),
            report_attempt_outcome=lambda *_args: None,
            on_success=lambda _attempt: None,
            max_tool_iterations=1,
        )
    assert abandoned == []


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


def test_default_attempt_kwargs_omits_extra_params_when_unset() -> None:
    """Most providers' attempts carry no ``extra_params``; the kwargs shape is
    unchanged from before the field existed."""
    attempt = _single_attempt("openai", "gpt-4o-mini")

    kwargs = default_attempt_kwargs(attempt, {"temperature": 0.2})

    assert kwargs == {
        "api_key": "k",
        "temperature": 0.2,
        "model": "openai:gpt-4o-mini",
    }


def test_default_attempt_kwargs_forwards_extra_params_as_client_args() -> None:
    """A generic (non-Bedrock) attempt's ``extra_params`` is nested under
    ``client_args``, not merged flat: any-llm's ``acompletion()`` only routes
    a ``client_args`` mapping to the provider's client constructor, so a flat
    top-level key would silently reach the completion call instead."""
    attempt = ResolvedAttempt(
        attempt_id="a0",
        position=0,
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-...",
        managed=False,
        extra_params={"some_client_kwarg": "value"},
    )

    kwargs = default_attempt_kwargs(attempt, {})

    assert kwargs["client_args"] == {"some_client_kwarg": "value"}
    assert "some_client_kwarg" not in kwargs
    assert kwargs["api_key"] == "sk-..."
    assert kwargs["model"] == "openai:gpt-4o-mini"


def test_default_attempt_kwargs_forwards_bedrock_extra_params_via_client_args() -> None:
    """A Bedrock classic-IAM-pair attempt's ``extra_params`` reaches
    ``client_args`` with the secret aliased to ``aws_secret_access_key``,
    which any-llm's Bedrock provider actually reads when building its boto3
    client (unlike a plain ``api_key``)."""
    attempt = ResolvedAttempt(
        attempt_id="a0",
        position=0,
        provider="bedrock",
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        api_key="secret-access-key",
        managed=False,
        extra_params={"region_name": "us-east-1", "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE"},
    )

    kwargs = default_attempt_kwargs(attempt, {})

    assert kwargs["client_args"] == {
        "region_name": "us-east-1",
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "secret-access-key",
    }
    assert kwargs["api_key"] == "secret-access-key"
    assert kwargs["model"] == "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_default_attempt_kwargs_client_args_not_overridable_by_request_fields() -> None:
    """``client_args`` (built from the platform-trusted ``extra_params``) must
    win over a same-named field in the caller's own request body, exactly
    like ``api_key``/``model`` already do: a request cannot smuggle its own
    client_args past the platform's resolved credentials."""
    attempt = ResolvedAttempt(
        attempt_id="a0",
        position=0,
        provider="openai",
        model="gpt-4o-mini",
        api_key="sk-...",
        managed=False,
        extra_params={"some_client_kwarg": "platform-value"},
    )

    kwargs = default_attempt_kwargs(attempt, {"client_args": {"some_client_kwarg": "attacker-controlled"}})

    assert kwargs["client_args"] == {"some_client_kwarg": "platform-value"}


@pytest.mark.asyncio
async def test_locked_in_retryable_failure_marks_attempt_final() -> None:
    attempts = [
        ResolvedAttempt(
            attempt_id="a0", position=0, provider="anthropic", model="claude-primary", api_key="bad", managed=False
        ),
        ResolvedAttempt(
            attempt_id="a1", position=1, provider="openai", model="gpt-fallback", api_key="unused", managed=False
        ),
    ]
    route = ResolvedRoute(request_id="r", fallback_enabled=True, attempts=attempts)
    reports: list[tuple[Any, ...]] = []

    async def _run_attempt(_kwargs: dict[str, Any], on_first_response: Any) -> Any:
        on_first_response()
        raise RuntimeError("failed after lock-in")

    with pytest.raises(HTTPException):
        await run_platform_attempts(
            route=route,
            attempts=attempts,
            base_request_fields={},
            run_attempt=_run_attempt,
            extract_usage=lambda _r: None,
            classify_error=lambda _exc: (True, "http_500"),
            report_attempt_outcome=lambda *args: reports.append(args),
            on_success=lambda _attempt: None,
            max_tool_iterations=1,
        )

    assert reports == [(attempts[0], "error", None, "http_500", True)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_error", "expected_error"),
    [
        (MaxToolIterationsExceeded("iteration cap reached"), HTTPException),
        (SandboxNotReachableError("sandbox unavailable"), SandboxNotReachableError),
        (WebSearchNotReachableError("web search unavailable"), WebSearchNotReachableError),
    ],
)
async def test_gateway_terminal_error_marks_current_attempt_final(
    terminal_error: BaseException,
    expected_error: type[BaseException],
) -> None:
    attempts = [
        ResolvedAttempt(
            attempt_id="a0", position=0, provider="openai", model="gpt-primary", api_key="bad", managed=False
        ),
        ResolvedAttempt(
            attempt_id="a1", position=1, provider="openai", model="gpt-fallback", api_key="unused", managed=False
        ),
    ]
    route = ResolvedRoute(request_id="r", fallback_enabled=True, attempts=attempts)
    reports: list[tuple[Any, ...]] = []

    async def _run_attempt(_kwargs: dict[str, Any], _on_first_response: Any) -> Any:
        raise terminal_error

    with pytest.raises(expected_error):
        await run_platform_attempts(
            route=route,
            attempts=attempts,
            base_request_fields={},
            run_attempt=_run_attempt,
            extract_usage=lambda _r: None,
            classify_error=lambda _exc: (False, "unused"),
            report_attempt_outcome=lambda *args: reports.append(args),
            on_success=lambda _attempt: None,
            max_tool_iterations=1,
        )

    assert reports == [(attempts[0], "error", None, None, True)]


@pytest.mark.asyncio
async def test_fallback_reports_nonfinal_error_and_final_success() -> None:
    attempts = [
        ResolvedAttempt(
            attempt_id="a0", position=0, provider="openai", model="gpt-4o", api_key="bad", managed=False
        ),
        ResolvedAttempt(
            attempt_id="a1", position=1, provider="openai", model="gpt-4o", api_key="good", managed=False
        ),
    ]
    route = ResolvedRoute(request_id="r", fallback_enabled=True, attempts=attempts)
    reports: list[tuple[Any, ...]] = []

    async def _run_attempt(kwargs: dict[str, Any], _on_first_response: Any) -> str:
        if kwargs["api_key"] == "bad":
            raise RuntimeError("primary failed")
        return "ok"

    result = await run_platform_attempts(
        route=route,
        attempts=attempts,
        base_request_fields={},
        run_attempt=_run_attempt,
        extract_usage=lambda _result: None,
        classify_error=lambda _exc: (True, "http_500"),
        report_attempt_outcome=lambda *args: reports.append(args),
        on_success=lambda _attempt: None,
        max_tool_iterations=1,
    )

    assert result == "ok"
    assert reports == [
        (attempts[0], "error", None, "http_500", False),
        (attempts[1], "success", None, None, True),
    ]


@pytest.mark.asyncio
async def test_nonretryable_error_marks_first_attempt_final() -> None:
    attempts = [
        ResolvedAttempt(
            attempt_id="a0", position=0, provider="openai", model="gpt-4o", api_key="bad", managed=False
        ),
        ResolvedAttempt(
            attempt_id="a1", position=1, provider="openai", model="gpt-4o", api_key="unused", managed=False
        ),
    ]
    route = ResolvedRoute(request_id="r", fallback_enabled=True, attempts=attempts)
    reports: list[tuple[Any, ...]] = []

    async def _run_attempt(_kwargs: dict[str, Any], _on_first_response: Any) -> str:
        raise RuntimeError("invalid request")

    with pytest.raises(HTTPException):
        await run_platform_attempts(
            route=route,
            attempts=attempts,
            base_request_fields={},
            run_attempt=_run_attempt,
            extract_usage=lambda _result: None,
            classify_error=lambda _exc: (False, "http_400"),
            report_attempt_outcome=lambda *args: reports.append(args),
            on_success=lambda _attempt: None,
            max_tool_iterations=1,
        )

    assert reports == [(attempts[0], "error", None, "http_400", True)]


@pytest.mark.asyncio
def _usage_config() -> GatewayConfig:
    return cast(
        GatewayConfig,
        SimpleNamespace(
            platform={"base_url": "http://platform", "usage_max_retries": 3},
            platform_token="gw-test",
        ),
    )


def _completed_usage_body(
    *,
    correlation_id: str = "3f1b6a1e-0000-4000-8000-000000000002",
    cost_usd: str = "0.012345",
    usage_status: str = "reported",
    pricing_source: str | None = "managed",
) -> dict[str, Any]:
    return {
        "request_id": "3f1b6a1e-0000-4000-8000-000000000001",
        "correlation_id": correlation_id,
        "status": "completed",
        "outcome": "success",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "cost_usd": cost_usd,
        "currency": "USD",
        "usage_status": usage_status,
        "usage": None,
        "pricing": {"source": pricing_source, "reference": "price-1"},
        "calculated_at": "2026-08-20T10:00:00Z",
    }


@pytest.mark.asyncio
async def test_report_platform_usage_returns_completed_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    post_mock = AsyncMock(return_value=httpx.Response(200, json=_completed_usage_body()))
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    result = await _platform._report_platform_usage(
        _usage_config(),
        "3f1b6a1e-0000-4000-8000-000000000002",
        "success",
        CompletionUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
        is_final_attempt=True,
    )

    assert result == _platform.SettledCost(cost_usd="0.012345", pricing_source="managed")


@pytest.mark.asyncio
async def test_report_platform_usage_accepts_opaque_correlation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    correlation_id = "01HX1ABCDEFGHJKMNPQRSTVWXYZ"
    post_mock = AsyncMock(
        return_value=httpx.Response(200, json=_completed_usage_body(correlation_id=correlation_id))
    )
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    result = await _platform._report_platform_usage(
        _usage_config(),
        correlation_id,
        "success",
        CompletionUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
        is_final_attempt=True,
    )

    assert result == _platform.SettledCost(cost_usd="0.012345", pricing_source="managed")


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [202, 204])
async def test_report_platform_usage_returns_none_without_completed_cost(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    post_mock = AsyncMock(return_value=httpx.Response(status_code))
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    result = await _platform._report_platform_usage(
        _usage_config(),
        "3f1b6a1e-0000-4000-8000-000000000002",
        "success",
        None,
        is_final_attempt=True,
    )

    assert result is None
    post_mock.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cost_usd", "usage_status", "pricing_source", "expected"),
    [
        ("0.000000", "reported", None, None),
        ("0.000000", "unavailable", None, None),
        (
            "0.000000",
            "reported",
            "managed",
            _platform.SettledCost(cost_usd="0.000000", pricing_source="managed"),
        ),
    ],
    ids=["unpriced", "unavailable", "priced-zero"],
)
async def test_report_platform_usage_applies_pricing_source_gate(
    monkeypatch: pytest.MonkeyPatch,
    cost_usd: str,
    usage_status: str,
    pricing_source: str | None,
    expected: _platform.SettledCost | None,
) -> None:
    monkeypatch.setattr(
        _platform,
        "_post_platform",
        AsyncMock(
            return_value=httpx.Response(
                200,
                json=_completed_usage_body(
                    cost_usd=cost_usd,
                    usage_status=usage_status,
                    pricing_source=pricing_source,
                ),
            )
        ),
    )

    result = await _platform._report_platform_usage(
        _usage_config(),
        "3f1b6a1e-0000-4000-8000-000000000002",
        "success",
        None,
        is_final_attempt=True,
    )

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"status": "completed"}),
        httpx.Response(410),
        httpx.Response(
            200,
            json=_completed_usage_body(correlation_id="3f1b6a1e-0000-4000-8000-000000000003"),
        ),
    ],
    ids=["malformed", "gone", "correlation-mismatch"],
)
async def test_report_platform_usage_ignores_non_attachable_responses(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
) -> None:
    post_mock = AsyncMock(return_value=response)
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    result = await _platform._report_platform_usage(
        _usage_config(),
        "3f1b6a1e-0000-4000-8000-000000000002",
        "success",
        None,
        is_final_attempt=True,
    )

    assert result is None
    post_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_report_platform_usage_ignores_failed_outcome_body(monkeypatch: pytest.MonkeyPatch) -> None:
    post_mock = AsyncMock(
        return_value=httpx.Response(
            200,
            json={
                "request_id": "3f1b6a1e-0000-4000-8000-000000000001",
                "status": "completed",
                "outcome": "failed",
                "cost_usd": "0.000000",
                "currency": "USD",
            },
        )
    )
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    result = await _platform._report_platform_usage(
        _usage_config(),
        "3f1b6a1e-0000-4000-8000-000000000002",
        "error",
        None,
        is_final_attempt=True,
    )

    assert result is None


@pytest.mark.asyncio
async def test_report_platform_usage_retries_then_returns_completed_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post_mock = AsyncMock(
        side_effect=[
            httpx.Response(500),
            httpx.Response(200, json=_completed_usage_body()),
        ]
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr(_platform, "_post_platform", post_mock)
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)

    result = await _platform._report_platform_usage(
        _usage_config(),
        "3f1b6a1e-0000-4000-8000-000000000002",
        "success",
        None,
        is_final_attempt=True,
    )

    assert result == _platform.SettledCost(cost_usd="0.012345", pricing_source="managed")
    assert post_mock.await_count == 2
    sleep_mock.assert_awaited_once_with(0.25)


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

    await _platform._report_platform_usage(
        config,
        "corr-1",
        "success",
        None,
        is_final_attempt=True,
    )

    assert post_mock.call_count == 1
    sleep_mock.assert_not_awaited()
    # Pin the classification itself, not just the (currently equivalent) retry
    # behavior: 402 must stay in the non-retryable set even if the >= 500 retry
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

    await _platform._report_platform_usage(
        config,
        "corr-1",
        "success",
        None,
        session_label=session_label,
        is_final_attempt=True,
    )

    body = post_mock.call_args.kwargs["body"]
    if expected is None:
        assert "session_label" not in body
    else:
        assert body["session_label"] == expected


@pytest.mark.asyncio
async def test_report_platform_usage_omits_unavailable_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    config = cast(
        GatewayConfig,
        SimpleNamespace(
            platform={"base_url": "http://platform", "usage_max_retries": 3},
            platform_token="gw-test",
        ),
    )
    post_mock = AsyncMock(return_value=httpx.Response(204))
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    await _platform._report_platform_usage(
        config,
        "corr-1",
        "success",
        None,
        is_final_attempt=True,
    )

    assert "usage" not in post_mock.call_args.kwargs["body"]


@pytest.mark.asyncio
async def test_report_platform_usage_forwards_final_attempt_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    config = cast(
        GatewayConfig,
        SimpleNamespace(
            platform={"base_url": "http://platform", "usage_max_retries": 3},
            platform_token="gw-test",
        ),
    )
    post_mock = AsyncMock(return_value=httpx.Response(204))
    monkeypatch.setattr(_platform, "_post_platform", post_mock)

    await _platform._report_platform_usage(
        config,
        "corr-1",
        "error",
        None,
        error_class="http_400",
        is_final_attempt=True,
    )

    assert post_mock.call_args.kwargs["body"]["is_final_attempt"] is True
