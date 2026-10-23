"""``UsageSignalPort``: what an overlay is told about each usage row, and what it cannot break.

The row is the billing record and the port is an overlay's, so the writer that
tells the port hands the row over first and swallows whatever the port does.
What crosses is a fixed set of facts: never the error text a row may carry.
"""

import dataclasses
import logging
import sys
import uuid
from collections.abc import Generator, Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.usage_signal_adapter import NullUsageSignalAdapter
from gateway.container import PortShapeError, build_container
from gateway.log_config import logger as gateway_logger
from gateway.models.usage import PLAYGROUND_USAGE_ENDPOINT, SERVED_HERE_SLUG, UsageLog
from gateway.ports.usage_signal_port import RecordedUsage, UsageSignalPort
from gateway.services.log_writer import (
    BatchLogWriter,
    SignalingLogWriter,
    SingleLogWriter,
    _recorded_usage,
    create_log_writer,
)

NO_SESSION = cast(AsyncSession, None)
_WORKSPACE = uuid.uuid4()
_T0 = datetime(2026, 10, 15, 12, tzinfo=UTC)
_SECRET_ERROR = "upstream said: sk-do-not-forward"  # pragma: allowlist secret


def _usage_row(**overrides: Any) -> UsageLog:
    fields: dict[str, Any] = {
        "id": "row-1",
        "request_group_id": "req-1",
        "timestamp": _T0,
        "workspace_id": _WORKSPACE,
        "user_id": "alice",
        "api_key_id": "key-1",
        "provider": "openai",
        "model": "gpt-5",
        "endpoint": "/v1/chat/completions",
        "status": "success",
        "status_code": None,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cost": Decimal("0.0049"),
        "latency_ms": 2900,
        "tags": {"purpose": "secret-project"},
    }
    return UsageLog(**(fields | overrides))


class _Recorder:
    def __init__(self) -> None:
        self.recorded: list[RecordedUsage] = []

    async def usage_recorded(self, usage: RecordedUsage) -> None:
        self.recorded.append(usage)


class _Raising:
    async def usage_recorded(self, usage: RecordedUsage) -> None:
        raise RuntimeError(_SECRET_ERROR)


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """The gateway logger does not propagate, so caplog sees it only with its handler attached."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def test_a_row_maps_to_the_facts_the_port_reports() -> None:
    usage = _recorded_usage(_usage_row(status="error", status_code=502, error_message=_SECRET_ERROR))

    assert usage == RecordedUsage(
        usage_id="row-1",
        request_group_id="req-1",
        timestamp=_T0,
        workspace_id=_WORKSPACE,
        user_id="alice",
        api_key_id="key-1",
        provider="openai",
        model="gpt-5",
        endpoint="/v1/chat/completions",
        source=SERVED_HERE_SLUG,
        status="error",
        status_code=502,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        cost=Decimal("0.0049"),
        latency_ms=2900,
        integration_traffic=True,
        deployment_paid=False,
    )


def test_no_error_text_tags_or_content_cross_the_seam() -> None:
    fields = {field.name for field in dataclasses.fields(RecordedUsage)}

    assert not fields & {"error_message", "tags", "source_label", "billing_meters", "pricing_breakdown"}
    usage = _recorded_usage(_usage_row(error_message=_SECRET_ERROR))
    assert _SECRET_ERROR not in repr(usage)
    assert "secret-project" not in repr(usage)


def test_a_float_cost_is_reported_as_an_exact_amount() -> None:
    assert _recorded_usage(_usage_row(cost=0.25)).cost == Decimal("0.25")
    assert _recorded_usage(_usage_row(cost=None)).cost is None


@pytest.mark.parametrize(
    ("overrides", "paid"), [({}, False), ({"deployment_paid": False}, False), ({"deployment_paid": True}, True)]
)
def test_deployment_paid_is_reported_and_false_when_unset(overrides: dict[str, Any], paid: bool) -> None:
    """A row built without the flag reads as the column default, false."""
    assert _recorded_usage(_usage_row(**overrides)).deployment_paid is paid


@pytest.mark.parametrize(
    ("overrides", "integration"),
    [
        ({}, True),
        ({"source": SERVED_HERE_SLUG}, True),
        ({"endpoint": PLAYGROUND_USAGE_ENDPOINT}, False),
        ({"source": "claude_code"}, False),
        ({"source": "claude_code", "endpoint": PLAYGROUND_USAGE_ENDPOINT}, False),
    ],
)
def test_integration_traffic_is_served_here_and_not_the_playground(
    overrides: dict[str, Any], integration: bool
) -> None:
    """A row the request path built has no source until it is flushed, which means served here."""
    usage = _recorded_usage(_usage_row(**overrides))

    assert usage.integration_traffic is integration
    assert usage.source == overrides.get("source", SERVED_HERE_SLUG)


@pytest.mark.asyncio
async def test_the_row_is_written_then_the_port_is_told_once() -> None:
    order: list[str] = []
    inner = AsyncMock()

    async def _put(log: UsageLog) -> bool:
        order.append("put")
        return True

    inner.put.side_effect = _put
    signal = AsyncMock()
    signal.usage_recorded.side_effect = lambda usage: order.append("signal")
    row = _usage_row()

    await SignalingLogWriter(inner, signal).put(row)

    inner.put.assert_awaited_once_with(row)
    signal.usage_recorded.assert_awaited_once_with(_recorded_usage(row))
    assert order == ["put", "signal"]


@pytest.mark.asyncio
async def test_a_raising_port_does_not_stop_the_row_and_logs_no_row_text(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    inner = AsyncMock()
    row = _usage_row(error_message=_SECRET_ERROR)

    await SignalingLogWriter(inner, _Raising()).put(row)

    inner.put.assert_awaited_once_with(row)
    warnings = [record for record in gateway_logs.records if record.levelno == logging.WARNING]
    assert [record.getMessage() for record in warnings] == ["Usage row not signaled (RuntimeError)"]
    assert _SECRET_ERROR not in gateway_logs.text


@pytest.mark.asyncio
async def test_a_row_the_facts_cannot_be_built_from_is_still_written(
    gateway_logs: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _broken(log: UsageLog) -> RecordedUsage:
        raise ValueError(_SECRET_ERROR)

    monkeypatch.setattr("gateway.services.log_writer._recorded_usage", _broken)
    inner = AsyncMock()
    recorder = _Recorder()

    await SignalingLogWriter(inner, recorder).put(_usage_row())

    inner.put.assert_awaited_once()
    assert recorder.recorded == []
    assert "Usage row not signaled (ValueError)" in gateway_logs.text
    assert _SECRET_ERROR not in gateway_logs.text


@pytest.mark.asyncio
async def test_a_row_the_writer_dropped_is_not_reported() -> None:
    inner = AsyncMock()
    inner.put.return_value = False
    recorder = _Recorder()

    accepted = await SignalingLogWriter(inner, recorder).put(_usage_row())

    assert accepted is False
    assert recorder.recorded == []


@pytest.mark.asyncio
async def test_the_factory_wraps_either_strategy_in_the_signal() -> None:
    for strategy, inner_type in (("single", SingleLogWriter), ("batch", BatchLogWriter)):
        writer = create_log_writer(strategy, NullUsageSignalAdapter())
        assert isinstance(writer, SignalingLogWriter)
        assert isinstance(writer._inner, inner_type)


@pytest.mark.asyncio
async def test_a_full_batch_queue_refuses_the_row() -> None:
    writer = BatchLogWriter(max_queue=1)
    assert await writer.put(_usage_row()) is True
    assert await writer.put(_usage_row()) is False


@pytest.mark.asyncio
async def test_start_and_stop_reach_the_wrapped_writer() -> None:
    inner = AsyncMock()
    writer = SignalingLogWriter(inner, NullUsageSignalAdapter())

    await writer.start()
    await writer.stop()

    inner.start.assert_awaited_once()
    inner.stop.assert_awaited_once()


def test_the_null_adapter_is_bound_by_default() -> None:
    container = build_container()

    assert isinstance(container.resolve(UsageSignalPort, NO_SESSION), NullUsageSignalAdapter)
    assert "UsageSignalPort" in container.summary


@pytest.fixture
def bootstrap_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Any]:
    """Write a bootstrap module and take it back out of ``sys.modules`` afterwards."""
    written: list[str] = []

    def write(name: str, body: str) -> None:
        (tmp_path / f"{name}.py").write_text(body)
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop(name, None)
        written.append(name)

    yield write
    for name in written:
        sys.modules.pop(name, None)


def test_an_overlay_bootstrap_rebinds_the_port(bootstrap_module: Any) -> None:
    bootstrap_module(
        "usage_signal_bootstrap",
        """
class ProbeUsageSignal:
    async def usage_recorded(self, usage):
        return None


def register(container):
    from gateway.ports.usage_signal_port import UsageSignalPort

    container.bind(UsageSignalPort, lambda session: ProbeUsageSignal())
""",
    )

    container = build_container("usage_signal_bootstrap:register")

    assert type(container.resolve(UsageSignalPort, NO_SESSION)).__name__ == "ProbeUsageSignal"
    assert container.summary == "usage_signal_bootstrap:register rebound UsageSignalPort"


def test_an_overlay_adapter_of_the_wrong_shape_fails_at_build(bootstrap_module: Any) -> None:
    """Resolved once at startup, so a stale adapter would otherwise only warn on every row."""
    bootstrap_module(
        "stale_usage_signal_bootstrap",
        """
class StaleUsageSignal:
    async def request_settled(self, usage):
        return None


def register(container):
    from gateway.ports.usage_signal_port import UsageSignalPort

    container.bind(UsageSignalPort, lambda session: StaleUsageSignal())
""",
    )

    with pytest.raises(PortShapeError, match="StaleUsageSignal, bound to UsageSignalPort, lacks usage_recorded"):
        build_container("stale_usage_signal_bootstrap:register")
