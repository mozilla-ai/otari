"""The database trace store keeps the port's contract on PostgreSQL.

These are the properties every ``TraceStoragePort`` adapter owes, run against the
core one: writes are idempotent per span and grow a trace's totals once, a trace
keeps one owner, no read crosses a workspace, and purge, erasure and retention
delete what they say. The store opens its own Unit of Work on the metering pool,
so the tenancy rows it points at are committed before it runs.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.trace_storage_adapter import LocalTraceStorage
from gateway.core.config import GatewayConfig
from gateway.core.database import dispose_db, init_db
from gateway.core.unit_of_work import create_log_unit_of_work
from gateway.models.users import User
from gateway.ports.trace_storage_port import SpanRecord, TraceFilter, TraceScope, TraceWrite
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository
from gateway.repositories.traces import TracesRepositories

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def store(test_config: GatewayConfig, clean_database: None) -> AsyncIterator[LocalTraceStorage]:
    init_db(test_config)
    try:
        yield LocalTraceStorage(create_log_unit_of_work, TracesRepositories.on)
    finally:
        await dispose_db()


@pytest_asyncio.fixture
async def tenants(async_db: AsyncSession) -> tuple[uuid.UUID, uuid.UUID]:
    """Two workspaces and two users, committed so the store's own sessions see them."""
    organization = await OrganizationRepository(async_db).create_organization(
        name="Acme", slug="acme-traces", created_by_user_id=None
    )
    workspaces = WorkspaceRepository(async_db)
    first = await workspaces.create_workspace(name="First", organization_id=organization.id, created_by_user_id=None)
    second = await workspaces.create_workspace(name="Second", organization_id=organization.id, created_by_user_id=None)
    async_db.add_all([User(user_id="alice"), User(user_id="bob")])
    await async_db.commit()
    return first.id, second.id


def _span(span_id: str, *, kind: str = "step", minutes: int = 0, **fields: Any) -> SpanRecord:
    start = _T0 + timedelta(minutes=minutes)
    return SpanRecord(
        span_id=span_id,
        kind=kind,
        origin="gateway",
        name=kind,
        outcome=fields.pop("outcome", "ok"),
        start_time=start,
        end_time=start + timedelta(seconds=2),
        **fields,
    )


def _write(workspace_id: uuid.UUID, trace_id: str, *spans: SpanRecord, user_id: str = "alice") -> TraceWrite:
    return TraceWrite(
        workspace_id=workspace_id,
        trace_id=trace_id,
        user_id=user_id,
        api_key_id=None,
        session_source="harness",
        harness="claude-code",
        spans=spans,
    )


async def test_a_write_creates_the_trace_and_totals_its_spans(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    result = await store.write(
        (
            _write(
                workspace,
                "session-1",
                _span("req-1"),
                _span(
                    "llm-1",
                    kind="llm",
                    parent_span_id="req-1",
                    input_tokens=100,
                    output_tokens=20,
                    cost_snapshot=Decimal("0.0049"),
                ),
                _span("tool-1", kind="tool", minutes=1, tool_name="Bash", outcome="error"),
            ),
        )
    )

    assert (result.accepted, result.duplicate, result.rejected) == (3, 0, 0)
    detail = await store.get(TraceScope.workspaces(frozenset({workspace})), "session-1", span_limit=10)
    assert detail is not None
    summary = detail.summary
    assert (summary.step_count, summary.span_count, summary.error_count) == (1, 3, 1)
    assert (summary.input_tokens, summary.output_tokens) == (100, 20)
    assert summary.cost_snapshot == Decimal("0.0049")
    assert summary.started_at == _T0
    assert summary.last_activity_at == _T0 + timedelta(minutes=1, seconds=2)
    assert [span.span_id for span in detail.spans] == ["llm-1", "req-1", "tool-1"]
    assert detail.truncated is False


async def test_a_replayed_write_is_counted_as_duplicate_and_adds_nothing(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    batch = (_write(workspace, "session-1", _span("req-1"), _span("llm-1", kind="llm", input_tokens=10)),)
    await store.write(batch)

    replay = await store.write(batch)

    assert (replay.accepted, replay.duplicate) == (0, 2)
    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)
    assert detail is not None
    assert (detail.summary.span_count, detail.summary.input_tokens) == (2, 10)


async def test_a_span_repeated_inside_one_write_is_stored_once(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants

    result = await store.write((_write(workspace, "session-1", _span("req-1"), _span("req-1")),))

    assert result.accepted == 1
    assert await store.count(TraceScope.deployment(), TraceFilter()) == 1


async def test_a_later_write_extends_the_trace(store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("req-1")),))

    await store.write((_write(workspace, "session-1", _span("req-2", minutes=30)),))

    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)
    assert detail is not None
    assert detail.summary.step_count == 2
    assert detail.summary.last_activity_at == _T0 + timedelta(minutes=30, seconds=2)


async def test_another_users_spans_never_join_a_trace(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("req-1")),))

    result = await store.write((_write(workspace, "session-1", _span("req-9"), user_id="bob"),))

    assert (result.accepted, result.rejected) == (0, 1)
    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)
    assert detail is not None
    assert [span.span_id for span in detail.spans] == ["req-1"]


async def test_the_same_ids_in_two_workspaces_stay_two_traces(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write((_write(first, "session-1", _span("req-1")), _write(second, "session-1", _span("req-1"))))

    only_first = TraceScope.workspaces(frozenset({first}))

    page = await store.search(only_first, TraceFilter(), limit=10, offset=0)
    assert [(item.workspace_id, item.trace_id) for item in page.items] == [(first, "session-1")]
    assert await store.count(only_first, TraceFilter()) == 1
    assert await store.count(TraceScope.deployment(), TraceFilter()) == 2


async def test_a_trace_in_another_workspace_reads_as_absent(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write((_write(second, "theirs", _span("req-1")),))

    assert await store.get(TraceScope.workspaces(frozenset({first})), "theirs", span_limit=10) is None
    assert await store.get(TraceScope.workspaces(frozenset()), "theirs", span_limit=10) is None


async def test_search_pages_newest_activity_first_and_filters(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write(
        tuple(
            _write(workspace, f"session-{index}", _span(f"req-{index}", minutes=index, outcome=outcome))
            for index, outcome in enumerate(["ok", "error", "ok"])
        )
    )
    scope = TraceScope.workspaces(frozenset({workspace}))

    first_page = await store.search(scope, TraceFilter(), limit=2, offset=0)
    second_page = await store.search(scope, TraceFilter(), limit=2, offset=2)
    failing = await store.search(scope, TraceFilter(has_error=True), limit=10, offset=0)
    by_prefix = await store.search(scope, TraceFilter(trace_id_prefix="session-2"), limit=10, offset=0)
    windowed = await store.count(scope, TraceFilter(start=_T0 + timedelta(minutes=1)))

    assert [item.trace_id for item in first_page.items] == ["session-2", "session-1"]
    assert first_page.has_more is True
    assert [item.trace_id for item in second_page.items] == ["session-0"]
    assert second_page.has_more is False
    assert [item.trace_id for item in failing.items] == ["session-1"]
    assert [item.trace_id for item in by_prefix.items] == ["session-2"]
    assert windowed == 2


async def test_a_prefix_search_treats_wildcards_literally(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("req-1")),))

    assert await store.count(TraceScope.deployment(), TraceFilter(trace_id_prefix="%")) == 0
    assert await store.count(TraceScope.deployment(), TraceFilter(trace_id_prefix="session_")) == 0


async def test_a_trace_with_more_spans_than_asked_for_says_it_was_cut(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", *(_span(f"req-{i}", minutes=i) for i in range(5))),))

    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=3)

    assert detail is not None
    assert len(detail.spans) == 3
    assert detail.truncated is True


async def test_purge_deletes_only_the_scopes_matching_traces(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write(
        (
            _write(first, "keep", _span("req-1")),
            _write(first, "drop", _span("req-2", outcome="error")),
            _write(second, "drop", _span("req-3", outcome="error")),
        )
    )

    removed = await store.purge(TraceScope.workspaces(frozenset({first})), TraceFilter(has_error=True))

    assert removed == 1
    assert await store.get(TraceScope.deployment(), "keep", span_limit=10) is not None
    assert await store.count(TraceScope.workspaces(frozenset({second})), TraceFilter()) == 1


async def test_purge_user_erases_every_trace_the_user_owns(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write(
        (
            _write(first, "a", _span("req-1")),
            _write(second, "b", _span("req-2")),
            _write(first, "c", _span("req-3"), user_id="bob"),
        )
    )

    assert await store.purge_user("alice") == 2
    page = await store.search(TraceScope.deployment(), TraceFilter(), limit=10, offset=0)
    assert [item.trace_id for item in page.items] == ["c"]


async def test_expire_deletes_traces_idle_since_before_the_cutoff(
    store: LocalTraceStorage, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "old", _span("req-1")), _write(workspace, "new", _span("req-2", minutes=60))))

    removed = await store.expire(_T0 + timedelta(minutes=30))

    assert removed == 1
    page = await store.search(TraceScope.deployment(), TraceFilter(), limit=10, offset=0)
    assert [item.trace_id for item in page.items] == ["new"]
