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
from gateway.api.routes._platform import ResolvedRoute, run_platform_attempts
from gateway.core.config import GatewayConfig


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
