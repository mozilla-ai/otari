"""The session series on SQLite: which sessions it counts as failed.

A tool's error goes back to the model, which carries on, so a session whose only
errors are its tools' succeeded, MCP tools' included; an error of the session's
own requests, or a failed connection to an MCP server, fails it.
Both stay counted in the session's ``error_count``.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.traces import TracesRepositories
from gateway.services.traces import TraceService
from gateway.types.traces import SpanRecord, TraceFilter, TraceScope, TraceWrite


def _no_window(provider: str | None, model: str) -> int | None:
    return None


pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
_WORKSPACE = uuid.uuid4()


@pytest_asyncio.fixture
async def store() -> AsyncIterator[TraceService]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            uow = UnitOfWork(session)
            yield TraceService(uow, TracesRepositories.on(uow), _no_window)
    finally:
        await engine.dispose()


def _span(span_id: str, *, kind: str = "step", outcome: str = "ok", recovered: bool = False) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        kind=kind,
        origin="gateway",
        name=kind,
        outcome=outcome,
        recovered=recovered,
        start_time=_T0,
        end_time=_T0 + timedelta(seconds=2),
    )


def _write(trace_id: str, *spans: SpanRecord) -> TraceWrite:
    return TraceWrite(
        workspace_id=_WORKSPACE,
        trace_id=trace_id,
        user_id=None,
        api_key_id=None,
        session_source="harness",
        harness="claude-code",
        spans=spans,
    )


async def test_tool_errors_alone_do_not_fail_a_session(store: TraceService) -> None:
    await store.write(
        (
            _write(
                "recovered", _span("req-1"), _span("tool-1", kind="tool", outcome="error"), _span("llm-1", kind="llm")
            ),
            _write("tools-only", _span("req-1"), *(_span(f"t{n}", kind="tool", outcome="error") for n in range(15))),
            _write("clean", _span("req-1"), _span("tool-1", kind="tool")),
        )
    )

    series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="hour")

    assert [(point.succeeded, point.failed) for point in series] == [(3, 0)]
    page = await store.search(TraceScope.deployment(), TraceFilter(has_error=True), limit=10, offset=0)
    assert sorted((item.trace_id, item.error_count) for item in page.items) == [("recovered", 1), ("tools-only", 15)]


async def test_an_unrecovered_error_of_the_session_itself_fails_it(store: TraceService) -> None:
    await store.write(
        (
            _write("failed-request", _span("req-1", outcome="error"), _span("tool-1", kind="tool", outcome="error")),
            _write("failed-call", _span("req-1"), _span("llm-1", kind="llm", outcome="error")),
            _write("fallback", _span("req-1"), _span("llm-1", kind="llm", outcome="error", recovered=True)),
        )
    )
    # A later write to the same session grows the same totals.
    await store.write((_write("fallback", _span("tool-1", kind="tool", outcome="error")),))

    series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="hour")

    assert [(point.succeeded, point.failed) for point in series] == [(1, 2)]


async def test_an_mcp_tool_error_is_a_tools_and_a_failed_mcp_connection_is_not(store: TraceService) -> None:
    # MCPClient records a tool call that failed, in transport or with ``isError``, as a ``tool`` span.
    await store.write(
        (
            _write("mcp-tool-failed", _span("req-1"), _span("mcp-1", kind="tool", outcome="error")),
            _write("mcp-unreachable", _span("req-1"), _span("connect-1", kind="mcp_connect", outcome="error")),
        )
    )

    series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="hour")

    assert [(point.succeeded, point.failed) for point in series] == [(1, 1)]
