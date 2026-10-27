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
from sqlalchemy import event, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.database import dispose_db, init_db
from gateway.core.unit_of_work import create_unit_of_work
from gateway.exceptions.traces_exceptions import GuardrailVersionLimitError, TraceAmbiguousError
from gateway.models.traces import TraceGuardrailDefinition, TraceSpan
from gateway.models.users import User
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository
from gateway.repositories.traces import TracesRepositories
from gateway.repositories.traces.definition_repository import GuardrailDefinitionRepository
from gateway.services.secret_box import generate_secret_key
from gateway.services.traces import TraceService
from gateway.services.traces import _service as trace_service_module
from gateway.services.traces._collector import call_id_digest
from gateway.types.traces import SpanRecord, TraceFilter, TraceScope, TraceWrite
from otari_agent.domain.definitions import GateDefinition, definition_digest


def _no_window(provider: str | None, model: str) -> int | None:
    return None


pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def store(test_config: GatewayConfig, clean_database: None) -> AsyncIterator[TraceService]:
    init_db(test_config)
    try:
        async with create_unit_of_work() as uow:
            yield TraceService(uow, TracesRepositories.on(uow), _no_window)
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


def _write(
    workspace_id: uuid.UUID,
    trace_id: str,
    *spans: SpanRecord,
    user_id: str = "alice",
    session_source: str = "harness",
    name: str | None = None,
    harness: str = "claude-code",
) -> TraceWrite:
    return TraceWrite(
        workspace_id=workspace_id,
        trace_id=trace_id,
        user_id=user_id,
        api_key_id=None,
        session_source=session_source,
        harness=harness,
        name=name,
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
                    cache_read_tokens=60,
                    cache_write_tokens=30,
                    reasoning_tokens=5,
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
    llm = detail.spans[0]
    assert (llm.cache_read_tokens, llm.cache_write_tokens, llm.reasoning_tokens) == (60, 30, 5)
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


async def test_a_requests_cost_counts_once_on_its_session_and_is_not_a_span(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    priced = _write(
        workspace,
        "session-1",
        _span("usage-1", kind="llm", operation="usage", request_id="req_1", cost_snapshot=Decimal("0.25")),
        session_source="otlp",
    )
    await store.write((_write(workspace, "session-1", _span("llm-1", kind="llm", request_id="req_1")),))
    await store.write((priced,))

    replay = await store.write((priced,))

    assert replay.duplicate == 1
    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)
    assert detail is not None
    assert (detail.summary.span_count, detail.summary.cost_snapshot) == (1, Decimal("0.25"))


async def test_a_logs_only_export_creates_no_session_until_its_spans_arrive(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    priced = _write(
        workspace,
        "session-1",
        _span("usage-1", kind="llm", operation="usage", request_id="req_1", cost_snapshot=Decimal("0.25")),
        session_source="otlp",
    )

    await store.write((priced,))

    scope = TraceScope.deployment()
    assert await store.count(scope, TraceFilter()) == 0
    assert (await store.search(scope, TraceFilter(), limit=10, offset=0)).items == ()
    assert await store.series(scope, TraceFilter(), bucket="hour") == ()

    await store.write((_write(workspace, "session-1", _span("llm-1", kind="llm", request_id="req_1")),))

    [summary] = (await store.search(scope, TraceFilter(), limit=10, offset=0)).items
    assert (summary.span_count, summary.cost_snapshot) == (1, Decimal("0.25"))


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


async def test_get_reads_the_workspace_it_names_and_refuses_to_guess_between_two(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    first, second = tenants
    await store.write((_write(first, "session-1", _span("req-1")),))
    await store.write((_write(second, "session-1", _span("req-1"), _span("req-2", minutes=5)),))
    both = TraceScope.deployment()

    named = await store.get(both, "session-1", span_limit=10, workspace_id=first)

    assert named is not None and named.summary.workspace_id == first
    with pytest.raises(TraceAmbiguousError):
        await store.get(both, "session-1", span_limit=10)


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
    by_search = await store.search(scope, TraceFilter(search="session-2"), limit=10, offset=0)
    windowed = await store.count(scope, TraceFilter(start=_T0 + timedelta(minutes=1)))

    assert [item.trace_id for item in first_page.items] == ["session-2", "session-1"]
    assert first_page.has_more is True
    assert [item.trace_id for item in second_page.items] == ["session-0"]
    assert second_page.has_more is False
    assert [item.trace_id for item in failing.items] == ["session-1"]
    assert [item.trace_id for item in by_search.items] == ["session-2"]
    assert windowed == 2


async def test_a_window_lists_the_sessions_active_in_it_as_the_series_counts_them(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """A session still running past the window's end is listed, as the chart counts it."""
    workspace, _ = tenants
    await store.write(
        (
            _write(workspace, "spanning", _span("start", minutes=0), _span("later", minutes=30)),
            _write(workspace, "before", _span("only", minutes=5)),
            _write(workspace, "after", _span("only", minutes=20)),
        )
    )
    window = TraceFilter(start=_T0 + timedelta(minutes=10), end=_T0 + timedelta(minutes=20))

    page = await store.search(TraceScope.deployment(), window, limit=10, offset=0)

    assert [item.trace_id for item in page.items] == ["spanning"]
    assert await store.count(TraceScope.deployment(), window) == 1


async def test_a_search_finds_any_part_of_the_id_harness_or_user_in_any_case(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write(
        (
            _write(workspace, "s-1a2b3c", _span("req-1"), user_id="alice", harness="claude-code"),
            _write(workspace, "s-4d5e6f", _span("req-2"), user_id="bob", harness="codex"),
        )
    )
    scope = TraceScope.workspaces(frozenset({workspace}))

    async def found(text: str) -> list[str]:
        return sorted(
            item.trace_id for item in (await store.search(scope, TraceFilter(search=text), limit=10, offset=0)).items
        )

    assert await found("1A2B") == ["s-1a2b3c"]
    assert await found("CODEX") == ["s-4d5e6f"]
    assert await found("lic") == ["s-1a2b3c"]
    assert await found("s-") == ["s-1a2b3c", "s-4d5e6f"]
    assert await found("gemini") == []


async def test_a_search_treats_wildcards_literally(store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("req-1")),))

    assert await store.count(TraceScope.deployment(), TraceFilter(search="%")) == 0
    assert await store.count(TraceScope.deployment(), TraceFilter(search="session_")) == 0


async def test_a_search_finds_a_session_by_any_part_of_its_name_in_any_case_inside_the_scope_only(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, other = tenants
    await store.write(
        (
            _write(workspace, "s-aaa", _span("req-1"), session_source="client", name="hackbot-1234"),
            _write(workspace, "s-bbb", _span("req-2"), session_source="client", name="hackbot-1299"),
            _write(other, "s-ccc", _span("req-3"), session_source="client", name="hackbot-1234"),
        )
    )
    scope = TraceScope.workspaces(frozenset({workspace}))

    exact = await store.search(scope, TraceFilter(search="hackbot-1234"), limit=10, offset=0)
    prefix = await store.search(scope, TraceFilter(search="hackbot-12"), limit=10, offset=0)
    middle = await store.search(scope, TraceFilter(search="BOT-12"), limit=10, offset=0)

    assert [(item.trace_id, item.name) for item in exact.items] == [("s-aaa", "hackbot-1234")]
    assert sorted(item.trace_id for item in prefix.items) == ["s-aaa", "s-bbb"]
    assert sorted(item.trace_id for item in middle.items) == ["s-aaa", "s-bbb"]
    assert await store.count(scope, TraceFilter(search="hackbot_")) == 0
    assert await store.count(scope, TraceFilter(search="HACKBOT-1234")) == 1
    assert await store.count(scope, TraceFilter(search="S-AAA")) == 1


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

    removed = await store.expire(idle_before=_T0 + timedelta(minutes=30), started_before=_T0 - timedelta(days=1))

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
            await TraceService(uow, TracesRepositories.on(uow), _no_window).write(
                tuple(_write(workspace, trace_id, _span(f"{tag}-{trace_id}")) for trace_id in order)
            )

    for attempt in range(5):
        await asyncio.gather(write(ids, f"a{attempt}"), write(ids[::-1], f"b{attempt}"))

    detail = await store.get(TraceScope.deployment(), "shared-0", span_limit=100)
    assert detail is not None and detail.summary.span_count == 10


async def test_expire_ends_a_session_kept_alive_past_its_age(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    """A client that keeps reusing one session id does not keep one trace forever."""
    workspace, _ = tenants
    await store.write((_write(workspace, "long", _span("req-1"), _span("req-2", minutes=600)),))

    removed = await store.expire(idle_before=_T0, started_before=_T0 + timedelta(minutes=1))

    assert removed == 1
    await store.write((_write(workspace, "long", _span("req-3", minutes=700)),))
    detail = await store.get(TraceScope.deployment(), "long", span_limit=10)
    assert detail is not None and [span.span_id for span in detail.spans] == ["req-3"], "a fresh trace"


async def test_the_session_age_limit_leaves_a_request_outside_any_session_to_its_idle_limit(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write(
        (
            _write(workspace, "session", _span("req-1")),
            _write(workspace, "lone-request", _span("req-2"), session_source="none"),
        )
    )

    removed = await store.expire(idle_before=_T0 - timedelta(days=1), started_before=_T0 + timedelta(minutes=1))

    assert removed == 1
    page = await store.search(TraceScope.deployment(), TraceFilter(), limit=10, offset=0)
    assert [item.trace_id for item in page.items] == ["lone-request"]


async def test_a_series_returns_only_the_newest_buckets_however_wide_its_window(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = tenants
    monkeypatch.setattr(trace_service_module, "MAX_SERIES_BUCKETS", 2)
    await store.write(tuple(_write(workspace, f"s{hour}", _span("req", minutes=60 * hour)) for hour in range(3)))

    series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="hour")

    assert [point.bucket for point in series] == ["2026-10-07T13:00:00Z", "2026-10-07T14:00:00Z"]


async def test_a_series_never_reads_traces_older_than_its_buckets_can_reach(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _ = tenants
    monkeypatch.setattr(trace_service_module, "MAX_SERIES_BUCKETS", 2)
    await store.write(tuple(_write(workspace, f"s{hour}", _span("req", minutes=60 * hour)) for hour in (0, 2, 3)))
    statements: list[str] = []

    def capture(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
        if "GROUP BY" in statement:
            statements.append(statement)

    event.listen(Engine, "before_cursor_execute", capture)
    try:
        series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="hour")
    finally:
        event.remove(Engine, "before_cursor_execute", capture)

    assert [point.bucket for point in series] == ["2026-10-07T14:00:00Z", "2026-10-07T15:00:00Z"]
    [grouped] = statements
    assert "last_activity_at >=" in grouped


async def test_a_session_counts_in_every_bucket_it_was_active_in(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write(
        (
            _write(workspace, "long", _span("start", minutes=0), _span("later", minutes=35)),
            _write(workspace, "short", _span("only", minutes=50, outcome="error")),
        )
    )

    series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="30m")

    assert [(point.bucket, point.succeeded, point.failed) for point in series] == [
        ("2026-10-07T12:00:00Z", 1, 0),
        ("2026-10-07T12:30:00Z", 1, 1),
    ]


async def test_a_minute_series_is_aligned_to_the_minute(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "s", _span("a", minutes=3), _span("b", minutes=5)),))

    series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="1m")

    assert [point.bucket for point in series] == [f"2026-10-07T12:0{minute}:00Z" for minute in (3, 4, 5)]


async def test_a_series_fails_a_session_for_its_own_errors_and_not_its_tools(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID]
) -> None:
    workspace, _ = tenants
    await store.write(
        (
            _write(workspace, "recovered", _span("req-1"), _span("tool-1", kind="tool", outcome="error")),
            _write(workspace, "failed", _span("req-1", outcome="error"), _span("tool-1", kind="tool", outcome="error")),
        )
    )

    series = await store.series(TraceScope.deployment(), TraceFilter(), bucket="hour")

    assert [(point.succeeded, point.failed) for point in series] == [(1, 1)]
    page = await store.search(TraceScope.deployment(), TraceFilter(has_error=True), limit=10, offset=0)
    assert sorted((item.trace_id, item.error_count) for item in page.items) == [("failed", 2), ("recovered", 1)]


@pytest.mark.parametrize("legacy", ["call/" + "x" * 80, "toolu_01AbCdEf"])
async def test_a_tool_call_id_an_older_version_stored_as_sent_reads_back_as_its_digest(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID], async_db: AsyncSession, legacy: str
) -> None:
    workspace, _ = tenants
    await store.write((_write(workspace, "session-1", _span("tool-1", kind="tool", tool_name="Bash")),))
    await async_db.execute(update(TraceSpan).where(col(TraceSpan.span_id) == "tool-1").values(tool_call_id=legacy))
    await async_db.commit()

    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)

    assert detail is not None
    assert detail.spans[0].tool_call_id == call_id_digest(workspace, legacy)


async def test_what_a_span_was_about_is_stored_encrypted_and_reads_back(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID], async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    workspace, _ = tenants
    command = "cat ~/.aws/credentials"
    await store.write(
        (_write(workspace, "session-1", _span("tool-1", kind="tool", tool_name="Bash", content={"command": command})),)
    )

    sealed = (
        await async_db.execute(select(TraceSpan.content_sealed).where(col(TraceSpan.span_id) == "tool-1"))
    ).scalar_one()
    assert sealed is not None and "credentials" not in sealed
    detail = await store.get(TraceScope.deployment(), "session-1", span_limit=10)
    assert detail is not None
    assert dict(detail.spans[0].content) == {"command": command}


async def test_without_a_key_what_a_span_was_about_is_not_stored(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID], async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    workspace, _ = tenants
    await store.write(
        (_write(workspace, "session-1", _span("tool-1", kind="tool", content={"command": "cat ~/.aws/credentials"})),)
    )

    sealed = (
        await async_db.execute(select(TraceSpan.content_sealed).where(col(TraceSpan.span_id) == "tool-1"))
    ).scalar_one()
    assert sealed is None


async def test_concurrent_new_guardrail_versions_never_pass_the_workspace_cap(
    store: TraceService, tenants: tuple[uuid.UUID, uuid.UUID], async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trace_service_module, "MAX_GUARDRAIL_VERSIONS_PER_WORKSPACE", 2)
    workspace, _ = tenants

    def gates(message: str) -> tuple[GateDefinition, ...]:
        return (
            GateDefinition(
                gate_id="no-changelog",
                gate_type="path",
                enforcement="required",
                origin="repo",
                runs=("pre_tool_use.edit_target",),
                message=message,
            ),
        )

    first = gates("version 0")
    await store.record_definition(workspace, first, definition_digest(first))

    # Widen the window between the count and the insert, so writers that are not serialized all pass the count.
    count_for_workspace = GuardrailDefinitionRepository.count_for_workspace

    async def slow_count(self: GuardrailDefinitionRepository, workspace_id: uuid.UUID) -> int:
        count = await count_for_workspace(self, workspace_id)
        await asyncio.sleep(0.1)
        return count

    monkeypatch.setattr(GuardrailDefinitionRepository, "count_for_workspace", slow_count)

    async def record(index: int) -> bool:
        version = gates(f"version {index}")
        async with create_unit_of_work() as uow:
            try:
                await TraceService(uow, TracesRepositories.on(uow), _no_window).record_definition(
                    workspace, version, definition_digest(version)
                )
            except GuardrailVersionLimitError:
                return False
        return True

    stored = await asyncio.gather(*(record(index) for index in range(1, 6)))

    assert stored.count(True) == 1
    digests = (
        await async_db.execute(
            select(TraceGuardrailDefinition.digest).where(col(TraceGuardrailDefinition.workspace_id) == workspace)
        )
    ).all()
    assert len(digests) == 2
