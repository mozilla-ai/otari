from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from gateway.models.usage import UsageLog
from gateway.services.log_writer import LogWriter, SignalingLogWriter, SingleLogWriter, TracingLogWriter


@pytest.mark.asyncio
async def test_single_log_writer_rolls_back_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit.side_effect = SQLAlchemyError("boom")

    @asynccontextmanager
    async def _session_cm() -> AsyncGenerator[AsyncMock, None]:
        yield session

    monkeypatch.setattr("gateway.services.log_writer.create_log_session", lambda: _session_cm())
    writer = SingleLogWriter()

    log = UsageLog(id="log", model="test-model", endpoint="/v1/test", status="success")
    await writer.put(log)

    session.rollback.assert_awaited()


@pytest.mark.asyncio
async def test_usage_is_signaled_after_commit() -> None:
    signal = AsyncMock()
    inner = AsyncMock(spec=LogWriter)
    writer = SignalingLogWriter(inner, signal)
    logs = [
        UsageLog(
            id="batch-row",
            model="test-model",
            provider="openai",
            endpoint="/v1/batches/results",
            status="success",
        )
    ]

    await writer.after_commit(logs)

    inner.after_commit.assert_awaited_once_with(logs)
    signal.usage_recorded.assert_awaited_once()


@pytest.mark.asyncio
async def test_usage_is_traced_after_commit() -> None:
    inner = AsyncMock(spec=LogWriter)
    traces = MagicMock()
    trace = MagicMock()
    traces.get.return_value = trace
    writer = TracingLogWriter(inner, traces)
    logs = [
        UsageLog(
            id="batch-row",
            request_group_id="request-group",
            model="test-model",
            provider="openai",
            endpoint="/v1/batches/results",
            status="success",
        )
    ]

    trace.record_llm_call.assert_not_called()

    await writer.after_commit(logs)

    trace.record_llm_call.assert_called_once()
    inner.after_commit.assert_awaited_once_with(logs)
