"""The trace service keeps its storage contract on PostgreSQL.

Writes are idempotent per span and grow a trace's totals once, a trace keeps one
owner, no read crosses a workspace, and purge, erasure and retention delete what
they say. The service runs on a worker Unit of Work, as the trace writer does, so
the tenancy rows it points at are committed before it runs.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.database import dispose_db, init_db
from gateway.core.unit_of_work import create_unit_of_work
from gateway.models.users import User
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository
from gateway.repositories.traces import TracesRepositories
from gateway.services.traces import TraceService
from gateway.types.traces import SpanRecord, TraceFilter, TraceScope, TraceWrite

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def store(test_config: GatewayConfig, clean_database: None) -> AsyncIterator[TraceService]:
    init_db(test_config)
    try:
        async with create_unit_of_work() as uow:
            yield TraceService(uow, TracesRepositories.on(uow))
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
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
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
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
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
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants

    result = await store.write((_write(workspace, "session-1", _span("req-1"), _span("req-1")),))

    assert result.accepted == 1
    assert await store.count(TraceScope.deployment(), TraceFilter()) == 1


async def test_a_later_write_extends_the_trace(store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("req-1")),))

    await store.write((_write(workspace, "session-1", _span("req-2", minutes=30)),))

    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)
    assert detail is not None
    assert detail.summary.step_count == 2
    assert detail.summary.last_activity_at == _T0 + timedelta(minutes=30, seconds=2)


async def test_another_users_spans_never_join_a_trace(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("req-1")),))

    result = await store.write((_write(workspace, "session-1", _span("req-9"), user_id="bob"),))

    assert (result.accepted, result.rejected) == (0, 1)
    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)
    assert detail is not None
    assert [span.span_id for span in detail.spans] == ["req-1"]


async def test_the_same_ids_in_two_workspaces_stay_two_traces(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write((_write(first, "session-1", _span("req-1")), _write(second, "session-1", _span("req-1"))))

    only_first = TraceScope.workspaces(frozenset({first}))

    page = await store.search(only_first, TraceFilter(), limit=10, offset=0)
    assert [(item.workspace_id, item.trace_id) for item in page.items] == [(first, "session-1")]
    assert await store.count(only_first, TraceFilter()) == 1
    assert await store.count(TraceScope.deployment(), TraceFilter()) == 2


async def test_get_reads_the_workspace_it_names_when_an_id_is_in_two(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write((_write(first, "session-1", _span("req-1")),))
    await store.write((_write(second, "session-1", _span("req-1"), _span("req-2", minutes=5)),))
    both = TraceScope.deployment()

    named = await store.get(both, "session-1", span_limit=10, workspace_id=first)
    unnamed = await store.get(both, "session-1", span_limit=10)

    assert named is not None and named.summary.workspace_id == first
    assert unnamed is not None and unnamed.summary.workspace_id == second, "the most recently active one"


async def test_a_trace_in_another_workspace_reads_as_absent(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write((_write(second, "theirs", _span("req-1")),))

    assert await store.get(TraceScope.workspaces(frozenset({first})), "theirs", span_limit=10) is None
    assert await store.get(TraceScope.workspaces(frozenset()), "theirs", span_limit=10) is None


async def test_search_pages_newest_activity_first_and_filters(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
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
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("req-1")),))

    assert await store.count(TraceScope.deployment(), TraceFilter(trace_id_prefix="%")) == 0
    assert await store.count(TraceScope.deployment(), TraceFilter(trace_id_prefix="session_")) == 0


async def test_a_trace_with_more_spans_than_asked_for_says_it_was_cut(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", *(_span(f"req-{i}", minutes=i) for i in range(5))),))

    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=3)

    assert detail is not None
    assert len(detail.spans) == 3
    assert detail.truncated is True


async def test_purge_user_erases_every_trace_the_user_owns(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
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
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "old", _span("req-1")), _write(workspace, "new", _span("req-2", minutes=60))))

    removed = await store.expire(_T0 + timedelta(minutes=30))

    assert removed == 1
    page = await store.search(TraceScope.deployment(), TraceFilter(), limit=10, offset=0)
    assert [item.trace_id for item in page.items] == ["new"]


async def test_a_batch_costs_the_same_statements_however_many_traces_it_holds(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    statements: list[str] = []

    def count(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        if "trace" in statement:
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", count)
    try:
        await store.write(
            tuple(
                _write(workspace, f"s{index}", _span(f"req-{index}"), _span(f"llm-{index}", kind="llm"))
                for index in range(20)
                for workspace in (first, second)
            )
        )
    finally:
        event.remove(Engine, "before_cursor_execute", count)

    assert len(statements) == 4, statements


async def test_two_writers_sharing_traces_in_opposite_order_do_not_deadlock(
    test_config: GatewayConfig, store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    ids = [f"shared-{index}" for index in range(10)]

    async def write(order: list[str], tag: str) -> None:
        async with create_unit_of_work() as uow:
            await TraceService(uow, TracesRepositories.on(uow)).write(
                tuple(_write(workspace, trace_id, _span(f"{tag}-{trace_id}")) for trace_id in order)
            )

    for attempt in range(5):
        await asyncio.gather(write(ids, f"a{attempt}"), write(ids[::-1], f"b{attempt}"))

    detail = await store.get(TraceScope.deployment(), "shared-0", span_limit=100)
    assert detail is not None and detail.summary.span_count == 10
