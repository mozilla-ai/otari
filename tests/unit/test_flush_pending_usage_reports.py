"""Unit tests for ``_flush_pending_usage_reports``: the inline, bounded flush
of per-attempt error reports on the all-failed platform path.
"""

import asyncio
from typing import Any

import pytest

import gateway.api.routes._pipeline as _pipeline
from gateway.core.config import GatewayConfig


def test_flush_delivers_all_reports_when_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[str, str, str | None, str | None, bool]] = []

    async def fake_report(
        config: Any,
        cid: str,
        outcome: str,
        usage: Any,
        error_class: str | None,
        session_label: str | None = None,
        is_final_attempt: bool = False,
    ) -> None:
        sent.append((cid, outcome, error_class, session_label, is_final_attempt))

    monkeypatch.setattr(_pipeline, "_report_platform_usage", fake_report)
    config = GatewayConfig(platform={"base_url": "http://platform.test"})

    asyncio.run(
        _pipeline._flush_pending_usage_reports(
            config,
            [
                _pipeline._PendingUsageReport(
                    attempt_id="att-1",
                    outcome="error",
                    usage=None,
                    error_class="http_500",
                    is_final_attempt=False,
                ),
                _pipeline._PendingUsageReport(
                    attempt_id="att-2",
                    outcome="error",
                    usage=None,
                    error_class="http_429",
                    is_final_attempt=True,
                ),
            ],
            "req-1",
            "my-run-personas",
        )
    )

    # Every flushed report carries the per-request session label.
    assert sorted(sent) == [
        ("att-1", "error", "http_500", "my-run-personas", False),
        ("att-2", "error", "http_429", "my-run-personas", True),
    ]


def test_flush_is_bounded_and_does_not_wait_for_a_slow_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degraded usage endpoint must not stall the already-failing response.

    The flush is capped at ``usage_timeout_ms``; a report that hangs past that
    is cut off rather than awaited to completion.
    """
    completed: list[str] = []

    async def slow_report(
        config: Any,
        cid: str,
        outcome: str,
        usage: Any,
        error_class: str | None,
        session_label: str | None = None,
        is_final_attempt: bool = False,
    ) -> None:
        await asyncio.sleep(10)
        completed.append(cid)

    monkeypatch.setattr(_pipeline, "_report_platform_usage", slow_report)
    # 20ms cap so the test returns promptly without a wall-clock assertion.
    config = GatewayConfig(platform={"base_url": "http://platform.test", "usage_timeout_ms": 20})

    asyncio.run(
        _pipeline._flush_pending_usage_reports(
            config,
            [
                _pipeline._PendingUsageReport(
                    attempt_id="att-1",
                    outcome="error",
                    usage=None,
                    error_class="http_500",
                    is_final_attempt=True,
                )
            ],
            "req-1",
        )
    )

    # The slow report was cancelled by the bound, never running to completion.
    assert completed == []


def test_flush_is_noop_when_no_pending_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def fake_report(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(_pipeline, "_report_platform_usage", fake_report)
    config = GatewayConfig(platform={"base_url": "http://platform.test"})

    asyncio.run(_pipeline._flush_pending_usage_reports(config, [], "req-1"))

    assert called is False
