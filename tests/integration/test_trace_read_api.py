"""The trace read API answers inside the caller's scope and nowhere else.

The operator routes read every workspace of the operator's active organization,
the rule ``/api/v1/keys`` follows. The organization routes follow the
rule organization usage follows: an owner or admin reads every workspace in
their active organization, a member the workspaces they belong to, and nobody
reads another organization's traces, whose ids answer 404 like ones that do
not exist.
"""

import asyncio
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.adapters.content_key_adapter import SECRET_KEY_WRAPPER, ContentKeyring, SecretKeyContentKeys
from gateway.api.deps import is_workspace_member, workspace_members
from gateway.api.routes.traces import _MAX_FILTER_VALUES
from gateway.core.config import API_ROOT
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.models.traces import (
    Trace,
    TraceBreakGlassRead,
    TraceContentBarrier,
    TraceContentRead,
    TraceContentShare,
    TraceGuardrailDefinition,
    TraceSpan,
)
from gateway.models.users import User as BillingUser
from gateway.repositories.traces.break_glass_repository import BreakGlassRepository
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.secret_box import generate_secret_key
from gateway.services.traces import SpanContent, TraceService, _service

from .conftest import _to_async_url
from .test_otlp import _agent_span, _exempt_key, _otlp_traces

_T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str,
    workspace_ids: tuple[uuid.UUID, ...] = (),
    operator: bool = False,
) -> str:
    user = User(email=email, full_name="X", active_organization_id=organization_id, is_superuser=operator)
    session.add(user)
    session.commit()
    session.refresh(user)
    session.add(OrganizationMember(organization_id=organization_id, user_id=user.id, role=role, status="active"))
    for workspace_id in workspace_ids:
        session.add(WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role="member", status="active"))
    token = f"otari-sess-{email}"
    session.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    session.commit()
    return token


def _trace(session: Session, workspace_id: uuid.UUID, trace_id: str, *, failed: bool = False) -> None:
    session.add(
        Trace(
            workspace_id=workspace_id,
            trace_id=trace_id,
            session_source="harness",
            harness="claude-code",
            started_at=_T0,
            last_activity_at=_T0 + timedelta(minutes=2),
            step_count=2,
            span_count=5,
            error_count=1 if failed else 0,
            failing_error_count=1 if failed else 0,
            cost_snapshot=Decimal("0.02"),
        )
    )
    session.flush()
    steps = [("req-1", True, 0), ("req-2", False, 60)]
    for span_id, opens, offset in steps:
        start = _T0 + timedelta(seconds=offset)
        session.add(
            TraceSpan(
                workspace_id=workspace_id,
                trace_id=trace_id,
                span_id=span_id,
                kind="step",
                origin="gateway",
                name="/v1/chat/completions",
                request_id=f"{trace_id}-{span_id}",
                outcome="error" if failed and span_id == "req-2" else "ok",
                opens_turn=opens,
                start_time=start,
                end_time=start + timedelta(seconds=10),
            )
        )
        session.add(
            TraceSpan(
                workspace_id=workspace_id,
                trace_id=trace_id,
                span_id=f"llm-{span_id}",
                parent_span_id=span_id,
                kind="llm",
                origin="gateway",
                name="chat",
                request_id=f"{trace_id}-{span_id}",
                outcome="ok",
                input_tokens=10,
                output_tokens=2,
                cost_snapshot=Decimal("0.01"),
                start_time=start,
                end_time=start + timedelta(seconds=9),
            )
        )
    # A client tool answered by the second step: its start is left for the read to fill.
    session.add(
        TraceSpan(
            workspace_id=workspace_id,
            trace_id=trace_id,
            span_id="call-toolu_1",
            kind="tool",
            origin="gateway",
            name="execute_tool:Bash",
            tool_name="Bash",
            tool_type="client",
            outcome="ok",
            end_time=_T0 + timedelta(seconds=60),
        )
    )
    session.commit()


@pytest.fixture
def world(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> dict[str, str]:
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK
    with db_session_factory() as session:
        alpha, beta = Organization(name="Alpha", slug="alpha-t"), Organization(name="Beta", slug="beta-t")
        session.add_all([alpha, beta])
        session.commit()
        one, two = Workspace(name="One", organization_id=alpha.id), Workspace(name="Two", organization_id=alpha.id)
        theirs = Workspace(name="Theirs", organization_id=beta.id)
        session.add_all([one, two, theirs])
        session.commit()
        _trace(session, one.id, "s-one")
        _trace(session, two.id, "s-two", failed=True)
        _trace(session, theirs.id, "s-theirs")
        return {
            "member": _identity(
                session, email="m@a.test", organization_id=alpha.id, role="member", workspace_ids=(one.id,)
            ),
            "admin": _identity(session, email="a@a.test", organization_id=alpha.id, role="admin"),
            "outsider": _identity(session, email="o@b.test", organization_id=beta.id, role="owner"),
            "operator": _identity(session, email="op@a.test", organization_id=alpha.id, role="owner", operator=True),
        }


def _operator_get(
    client: TestClient, world: dict[str, str], path: str, params: Mapping[str, str | list[str]] | None = None
) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, world["operator"])
    try:
        return client.get(f"{API_ROOT}{path}", params=params)
    finally:
        client.cookies.clear()


def _listed(client: TestClient, cookie: str) -> list[str]:
    client.cookies.set(SESSION_COOKIE_NAME, cookie)
    try:
        response = client.get(f"{API_ROOT}/organizations/me/traces")
    finally:
        client.cookies.clear()
    assert response.status_code == status.HTTP_200_OK, response.text
    return sorted(item["trace_id"] for item in response.json()["items"])


def test_a_member_reads_their_workspaces_and_an_admin_the_whole_organization(
    client: TestClient, world: dict[str, str]
) -> None:
    assert _listed(client, world["member"]) == ["s-one"]
    assert _listed(client, world["admin"]) == ["s-one", "s-two"]
    assert _listed(client, world["outsider"]) == ["s-theirs"]


def test_another_organizations_trace_is_not_found(client: TestClient, world: dict[str, str]) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, world["outsider"])
    try:
        response = client.get(f"{API_ROOT}/organizations/me/traces/s-one")
    finally:
        client.cookies.clear()

    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_the_operator_reads_every_workspace_of_their_organization_and_the_detail_has_turns(
    client: TestClient, world: dict[str, str]
) -> None:
    listed = _operator_get(client, world, "/traces").json()
    assert sorted(item["trace_id"] for item in listed["items"]) == ["s-one", "s-two"]
    assert _operator_get(client, world, "/traces/s-theirs").status_code == status.HTTP_404_NOT_FOUND

    detail = _operator_get(client, world, "/traces/s-two").json()

    [turn] = detail["turns"]
    assert (turn["state"], turn["llm_calls"], turn["tool_calls"], turn["step_ids"]) == (
        "failed",
        2,
        1,
        ["req-1", "req-2"],
    )
    assert {span["name"] for span in detail["spans"] if span["kind"] == "step"} == {"/v1/chat/completions"}
    tool = next(span for span in detail["spans"] if span["kind"] == "tool")
    assert tool["approximate"] is True
    assert tool["parent_span_id"] == "req-1"
    assert tool["start_time"].startswith("2026-10-07T12:00:10")
    assert {(span["otel_trace_id"], span["otel_span_id"]) for span in detail["spans"]} == {(None, None)}


def test_a_request_opens_the_session_it_was_traced_into_inside_the_scope_only(
    client: TestClient, world: dict[str, str]
) -> None:
    def status_and_trace(cookie: str, request_id: str) -> tuple[int, str | None]:
        client.cookies.set(SESSION_COOKIE_NAME, cookie)
        try:
            response = client.get(f"{API_ROOT}/organizations/me/traces/requests/{request_id}")
        finally:
            client.cookies.clear()
        return response.status_code, response.json().get("trace_id")

    assert status_and_trace(world["member"], "s-one-req-2") == (status.HTTP_200_OK, "s-one")
    # Another workspace of the organization, another organization, and an id never traced read alike.
    assert status_and_trace(world["member"], "s-two-req-1")[0] == status.HTTP_404_NOT_FOUND
    assert status_and_trace(world["outsider"], "s-one-req-1")[0] == status.HTTP_404_NOT_FOUND
    assert status_and_trace(world["admin"], "never-traced")[0] == status.HTTP_404_NOT_FOUND
    assert status_and_trace(world["admin"], "s-two-req-1") == (status.HTTP_200_OK, "s-two")

    found = _operator_get(client, world, "/traces/requests/s-two-req-2").json()
    assert (found["trace_id"], found["error_count"]) == ("s-two", 1)
    assert _operator_get(client, world, "/traces/requests/s-theirs-req-1").status_code == status.HTTP_404_NOT_FOUND


def test_the_series_counts_sessions_by_whether_they_failed(client: TestClient, world: dict[str, str]) -> None:
    series = _operator_get(client, world, "/traces/series", {"bucket": "hour"}).json()

    assert series["points"] == [{"bucket": "2026-10-07T12:00:00Z", "succeeded": 1, "failed": 1}]


def test_a_search_ignores_the_spaces_around_it(client: TestClient, world: dict[str, str]) -> None:
    listed = _operator_get(client, world, "/traces", {"q": "  ONE "}).json()

    assert [item["trace_id"] for item in listed["items"]] == ["s-one"]


def test_the_header_master_key_reads_the_default_organization(
    client: TestClient, master_key_header: dict[str, str], world: dict[str, str]
) -> None:
    listed = client.get(f"{API_ROOT}/traces", headers=master_key_header).json()
    assert listed["items"] == []


def test_the_operator_routes_refuse_a_member(client: TestClient, world: dict[str, str]) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, world["admin"])
    try:
        response = client.get(f"{API_ROOT}/traces")
    finally:
        client.cookies.clear()

    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_a_workspace_filter_narrows_the_scope_and_never_widens_it(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    with db_session_factory() as session:
        ids = {workspace.name: str(workspace.id) for workspace in session.scalars(select(Workspace))}

    def read(path: str, workspace: str, cookie: str | None = None) -> dict[str, Any]:
        client.cookies.set(SESSION_COOKIE_NAME, cookie or world["operator"])
        try:
            response = client.get(f"{API_ROOT}{path}", params={"workspace_id": ids[workspace]})
        finally:
            client.cookies.clear()
        assert response.status_code == status.HTTP_200_OK, response.text
        return dict(response.json())

    assert [item["trace_id"] for item in read("/traces", "Two")["items"]] == ["s-two"]
    assert read("/traces/count", "Theirs")["count"] == 0
    assert read("/traces/series", "Two")["points"] == [{"bucket": "2026-10-07T12:00:00Z", "succeeded": 0, "failed": 1}]

    admin = "/organizations/me/traces"
    assert [item["trace_id"] for item in read(admin, "Two", world["admin"])["items"]] == ["s-two"]
    # A workspace outside the caller's scope yields nothing rather than another tenant's sessions.
    assert read(admin, "Theirs", world["admin"])["items"] == []
    assert read(f"{admin}/count", "Two", world["member"])["count"] == 0
    assert read(f"{admin}/series", "Theirs", world["member"])["points"] == []


@pytest.mark.parametrize(
    ("sort", "descending"),
    [
        ("last_activity", ["s-short", "s-long", "s-costly"]),
        ("latency", ["s-long", "s-costly", "s-short"]),
        ("tokens", ["s-short", "s-costly", "s-long"]),
        ("cost", ["s-costly", "s-short", "s-long"]),
        ("errors", ["s-long", "s-short", "s-costly"]),
    ],
)
def test_the_list_sorts_by_the_column_asked_for_across_pages(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    sort: str,
    descending: list[str],
) -> None:
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace).where(col(Workspace.name) == "One")).one()
        for trace_id, started, last, tokens, cost, errors in (
            ("s-short", 60, 61, (900, 100), "0.05", 1),
            ("s-long", 0, 50, (10, 10), "0.01", 3),
            ("s-costly", 10, 40, (300, 200), "0.90", 0),
        ):
            session.add(
                Trace(
                    workspace_id=workspace.id,
                    trace_id=trace_id,
                    session_source="client",
                    started_at=_T0 + timedelta(days=1, minutes=started),
                    last_activity_at=_T0 + timedelta(days=1, minutes=last),
                    input_tokens=tokens[0],
                    output_tokens=tokens[1],
                    cost_snapshot=Decimal(cost),
                    error_count=errors,
                    span_count=1,
                )
            )
        session.commit()
        workspace_id = str(workspace.id)

    def listed(order: str) -> list[str]:
        # Two pages of two, so the order holds across an offset.
        pages = [
            _operator_get(
                client,
                world,
                "/traces",
                {"workspace_id": workspace_id, "sort": sort, "order": order, "limit": "2", "skip": str(skip)},
            )
            for skip in (0, 2)
        ]
        assert all(page.status_code == status.HTTP_200_OK for page in pages), pages[0].text
        return [item["trace_id"] for page in pages for item in page.json()["items"] if item["trace_id"] != "s-one"]

    assert listed("desc") == descending
    assert listed("asc") == descending[::-1]


@pytest.mark.parametrize("sort", ["latency", "tokens", "cost", "errors"])
def test_sessions_that_tie_on_the_sort_page_once_each(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    sort: str,
) -> None:
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace).where(col(Workspace.name) == "One")).one()
        # Equal on every sort key, and two pairs equal on last activity too, so only the key breaks those ties.
        for index in range(6):
            session.add(
                Trace(
                    workspace_id=workspace.id,
                    trace_id=f"s-tie-{index}",
                    session_source="client",
                    started_at=_T0,
                    last_activity_at=_T0 + timedelta(minutes=2 + index // 2),
                    input_tokens=5,
                    output_tokens=5,
                    cost_snapshot=Decimal("0.02"),
                    error_count=0,
                    span_count=1,
                )
            )
        session.commit()
        workspace_id = str(workspace.id)

    for order in ("asc", "desc"):
        seen = [
            item["trace_id"]
            for skip in range(7)
            for item in _operator_get(
                client,
                world,
                "/traces",
                {"workspace_id": workspace_id, "sort": sort, "order": order, "limit": "1", "skip": str(skip)},
            ).json()["items"]
        ]
        assert sorted(seen) == sorted([*(f"s-tie-{index}" for index in range(6)), "s-one"])


def test_the_list_refuses_a_sort_it_does_not_offer(client: TestClient, world: dict[str, str]) -> None:
    for params in ({"sort": "name"}, {"order": "up"}):
        assert _operator_get(client, world, "/traces", params).status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_the_detail_reads_the_workspace_it_names(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    with db_session_factory() as session:
        ids = {workspace.name: str(workspace.id) for workspace in session.scalars(select(Workspace))}

    def status_of(workspace: str) -> int:
        return int(_operator_get(client, world, "/traces/s-two", {"workspace_id": ids[workspace]}).status_code)

    assert status_of("Two") == status.HTTP_200_OK
    assert status_of("One") == status.HTTP_404_NOT_FOUND


def test_an_id_in_two_workspaces_read_without_naming_one_is_a_conflict(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    with db_session_factory() as session:
        one = session.scalars(select(Workspace).where(col(Workspace.name) == "One")).one()
        _trace(session, one.id, "s-two")
        session.commit()

    response = _operator_get(client, world, "/traces/s-two")

    assert response.status_code == status.HTTP_409_CONFLICT


def test_a_span_received_over_otlp_reads_back_with_the_otel_ids_it_was_sent(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    headers = _exempt_key(client, master_key_header)
    span = _agent_span("5b8efff798038103d269b633813fc60c", "eee19b7ec3c1b174")
    assert client.post("/otlp/v1/traces", json=_otlp_traces(span), headers=headers).status_code == status.HTTP_200_OK

    deadline = time.monotonic() + 10
    while not (items := client.get(f"{API_ROOT}/traces", headers=master_key_header).json()["items"]):
        assert time.monotonic() < deadline, "the exported span was never written"
        time.sleep(0.1)
    [trace] = items
    detail = client.get(f"{API_ROOT}/traces/{trace['trace_id']}", headers=master_key_header).json()

    [read] = detail["spans"]
    assert (read["otel_trace_id"], read["otel_span_id"]) == ("5b8efff798038103d269b633813fc60c", "eee19b7ec3c1b174")


def test_the_detail_counts_each_gate_an_agents_hook_reported_and_names_the_ones_that_never_ran(
    client: TestClient, world: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    with db_session_factory() as session:
        workspace_id = session.scalars(select(Trace.workspace_id).where(col(Trace.trace_id) == "s-one")).one()
        gate = {"gate_type": "path", "enforcement": "required", "origin": "repo", "runs": ["pre_tool_use.edit_target"]}
        session.add(
            TraceGuardrailDefinition(
                workspace_id=workspace_id,
                digest="0123456789abcdef",
                gates=[
                    {**gate, "gate_id": "no-changelog", "message": "CHANGELOG.md is generated."},
                    {**gate, "gate_id": "no-secret-reads", "message": ""},
                ],
            )
        )
        for index, (outcome, blocked) in enumerate([("pass", False), ("pass", False), ("fail", True)]):
            session.add(
                TraceSpan(
                    workspace_id=workspace_id,
                    trace_id="s-one",
                    span_id=f"gate-{index}",
                    kind="guardrail",
                    origin="otlp",
                    name="gate:no-changelog",
                    outcome="blocked" if blocked else "ok",
                    start_time=_T0 + timedelta(seconds=index),
                    end_time=_T0 + timedelta(seconds=index),
                    attributes={
                        "otari.gate.id": "no-changelog",
                        "otari.gate.outcome": outcome,
                        "otari.gate.blocked": blocked,
                        "otari.policy.digest": "0123456789abcdef",
                    },
                )
            )
        session.commit()

    detail = _operator_get(client, world, "/traces/s-one").json()

    assert detail["gates"] == [
        {
            "gate_id": "no-changelog",
            "passed": 2,
            "blocked": 1,
            "flagged": 0,
            "refused": 0,
            "undecided": 0,
            "span_id": "gate-2",
            "gate_type": "path",
            "enforcement": "required",
            "message": "CHANGELOG.md is generated.",
            "runs": ["pre_tool_use.edit_target"],
            "origin": "repo",
        },
        {
            "gate_id": "no-secret-reads",
            "passed": 0,
            "blocked": 0,
            "flagged": 0,
            "refused": 0,
            "undecided": 0,
            "span_id": None,
            "gate_type": "path",
            "enforcement": "required",
            "message": None,
            "runs": ["pre_tool_use.edit_target"],
            "origin": "repo",
        },
    ]
    assert _operator_get(client, world, "/traces/s-two").json()["gates"] == []


def _sealed(content: dict[str, str], *, workspace_id: uuid.UUID, trace_id: str, span_id: str) -> str | None:
    keys = ContentKeyring(SECRET_KEY_WRAPPER, {SECRET_KEY_WRAPPER: SecretKeyContentKeys()})
    return asyncio.run(SpanContent(keys).seal(content, workspace_id=workspace_id, trace_id=trace_id, span_id=span_id))


def _seal_tool_call(db_session_factory: Callable[[], Session], command: str = "pytest -x") -> None:
    with db_session_factory() as session:
        span = session.scalars(
            select(TraceSpan).where(col(TraceSpan.trace_id) == "s-one", col(TraceSpan.span_id) == "call-toolu_1")
        ).one()
        span.content_sealed = _sealed(
            {"command": command}, workspace_id=span.workspace_id, trace_id=span.trace_id, span_id=span.span_id
        )
        session.commit()


def _give_s_one_to(db_session_factory: Callable[[], Session], email: str) -> None:
    with db_session_factory() as session:
        owner = session.scalars(select(User).where(col(User.email) == email)).one()
        trace = session.scalars(select(Trace).where(col(Trace.trace_id) == "s-one")).one()
        session.add(BillingUser(user_id=str(owner.id)))
        session.flush()
        trace.user_id = str(owner.id)
        session.commit()


def _detail(client: TestClient, cookie: str) -> dict[str, Any]:
    response = _as(client, cookie, "GET", "/organizations/me/traces/s-one")
    assert response.status_code == status.HTTP_200_OK, response.text
    detail: dict[str, Any] = response.json()
    return detail


def _tool_call(detail: dict[str, Any]) -> dict[str, Any]:
    span: dict[str, Any] = next(span for span in detail["spans"] if span["span_id"] == "call-toolu_1")
    return span


def test_a_sessions_content_opens_for_its_own_user_and_is_private_to_everyone_else(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")

    own = _detail(client, world["member"])
    assert (_tool_call(own)["content"], own["content_access"]) == ({"command": "pytest -x"}, "owner")

    for reader in ("admin", "operator"):
        detail = (
            _detail(client, world[reader])
            if reader == "admin"
            else _operator_get(client, world, "/traces/s-one").json()
        )
        tool = _tool_call(detail)
        assert (tool["content"], tool["content_private"], detail["content_access"]) == ({}, True, "none")
        request = next(span for span in detail["spans"] if span["span_id"] == "req-1")
        assert request["content_private"] is False


def test_a_session_no_member_owns_opens_for_nobody(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content kept before the rule, on a session no person owns, opens for no admin and no operator."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)

    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True
    assert _tool_call(_detail(client, world["member"]))["content_private"] is True
    assert _tool_call(_operator_get(client, world, "/traces/s-one").json())["content_private"] is True


def test_a_share_ends_for_good_once_its_owner_leaves_the_workspace(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    share = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    ).json()
    assert _tool_call(_detail(client, world["admin"]))["content"] == {"command": "pytest -x"}

    with db_session_factory() as session:
        member = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()
        for row in session.scalars(select(WorkspaceMember).where(col(WorkspaceMember.user_id) == member.id)):
            session.delete(row)
        session.commit()

    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True
    with db_session_factory() as session:
        stored = session.get(TraceContentShare, uuid.UUID(share["id"]))
        assert stored is not None and stored.revoked_at is not None
        member = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()
        workspace_id = session.scalars(select(col(Workspace.id)).where(col(Workspace.name) == "One")).one()
        session.add(WorkspaceMember(workspace_id=workspace_id, user_id=member.id, role="member", status="active"))
        session.commit()
    # Back in the workspace, the ended share stays ended.
    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True


def test_a_share_lets_its_audience_read_until_its_owner_ends_it_and_records_each_read(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    path = "/organizations/me/traces/s-one/shares"

    refused = _as(client, world["admin"], "POST", path, json={"audience": "organization_admins"})
    assert refused.status_code == status.HTTP_403_FORBIDDEN
    created = _as(client, world["member"], "POST", path, json={"audience": "organization_admins", "note": "flaky test"})
    assert created.status_code == status.HTTP_201_CREATED, created.text
    share = created.json()
    assert (share["audience"], share["note"], share["active"]) == ("organization_admins", "flaky test", True)

    shared = _detail(client, world["admin"])
    assert _tool_call(shared)["content"] == {"command": "pytest -x"}
    assert (shared["content_access"], shared["content_share"]["note"]) == ("share", "flaky test")
    # This operator also owns the organization, so a share with its owners and admins reaches them there too.
    assert _tool_call(_operator_get(client, world, "/traces/s-one").json())["content"] == {"command": "pytest -x"}

    listed = _as(client, world["member"], "GET", path)
    assert listed.status_code == status.HTTP_200_OK, listed.text
    [entry] = listed.json()["items"]
    # Each reader through the share is recorded: the operator, who owns the organization, then the admin.
    assert sorted(read["reader_email"] for read in entry["reads"]) == ["a@a.test", "op@a.test"]
    assert _as(client, world["admin"], "GET", path).status_code == status.HTTP_403_FORBIDDEN

    ended = _as(client, world["member"], "DELETE", f"{path}/{share['id']}")
    assert ended.status_code == status.HTTP_204_NO_CONTENT
    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True
    [entry] = _as(client, world["member"], "GET", path).json()["items"]
    assert entry["active"] is False and entry["revoked_at"] is not None
    missing = _as(client, world["member"], "DELETE", f"{path}/{uuid.uuid4()}")
    assert missing.status_code == status.HTTP_404_NOT_FOUND


def test_a_share_with_the_operator_opens_for_the_operator(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")

    created = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "deployment_operator"},
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text

    detail = _operator_get(client, world, "/traces/s-one").json()
    assert (_tool_call(detail)["content"], detail["content_access"]) == ({"command": "pytest -x"}, "share")
    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True


def test_a_share_lasts_until_it_is_ended(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    created = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    )
    with db_session_factory() as session:
        share = session.get(TraceContentShare, uuid.UUID(created.json()["id"]))
        assert share is not None
        share.created_at = datetime.now(UTC) - timedelta(days=365)
        session.commit()

    assert _tool_call(_detail(client, world["admin"]))["content"] == {"command": "pytest -x"}


def test_the_record_of_a_read_through_a_share_outlives_the_session(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    created = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text
    assert _tool_call(_detail(client, world["admin"]))["content"] == {"command": "pytest -x"}

    with db_session_factory() as session:
        trace = session.scalars(select(Trace).where(col(Trace.trace_id) == "s-one")).one()
        workspace_id, owner = trace.workspace_id, trace.user_id
        session.delete(trace)
        session.commit()

    with db_session_factory() as session:
        assert session.get(TraceContentShare, uuid.UUID(created.json()["id"])) is None
        [read] = session.scalars(select(TraceContentRead)).all()
        assert read.share_id is None
        assert (read.workspace_id, read.trace_id, read.session_user_id) == (workspace_id, "s-one", owner)
        assert (read.audience, read.reader_email) == ("organization_admins", "a@a.test")


def _workspace_id(db_session_factory: Callable[[], Session], name: str) -> uuid.UUID:
    with db_session_factory() as session:
        workspace_id: uuid.UUID = session.scalars(select(col(Workspace.id)).where(col(Workspace.name) == name)).one()
        return workspace_id


def _as(client: TestClient, cookie: str, method: str, path: str, **kwargs: Any) -> Any:
    client.cookies.set(SESSION_COOKIE_NAME, cookie)
    try:
        return client.request(method, f"{API_ROOT}{path}", **kwargs)
    finally:
        client.cookies.clear()


def test_a_workspace_keeps_no_content_until_its_managers_turn_it_on(
    client: TestClient, world: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    workspace_id = _workspace_id(db_session_factory, "One")

    shown = _as(client, world["admin"], "GET", f"/workspaces/{workspace_id}")
    assert shown.json()["trace_content_enabled"] is False
    refused = _as(client, world["member"], "PATCH", f"/workspaces/{workspace_id}", json={"trace_content_enabled": True})
    assert refused.status_code == status.HTTP_403_FORBIDDEN
    turned_on = _as(
        client, world["admin"], "PATCH", f"/workspaces/{workspace_id}", json={"trace_content_enabled": True}
    )
    assert turned_on.status_code == status.HTTP_200_OK, turned_on.text
    assert turned_on.json()["trace_content_enabled"] is True
    # A null leaves the switch where it is rather than failing the update.
    renamed = _as(
        client,
        world["admin"],
        "PATCH",
        f"/workspaces/{workspace_id}",
        json={"name": "Uno", "trace_content_enabled": None},
    )
    assert renamed.status_code == status.HTTP_200_OK, renamed.text
    assert renamed.json()["trace_content_enabled"] is True

    # Turning it off also ends the content generation, so nothing already queued is stored.
    turned_off = _as(
        client, world["admin"], "PATCH", f"/workspaces/{workspace_id}", json={"trace_content_enabled": False}
    )
    assert turned_off.status_code == status.HTTP_200_OK, turned_off.text
    with db_session_factory() as session:
        barrier = session.get(TraceContentBarrier, workspace_id)
        assert barrier is not None and barrier.generation == 2


def test_a_workspaces_managers_delete_its_stored_content_and_keep_its_sessions(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    workspace_id = _workspace_id(db_session_factory, "One")
    with db_session_factory() as session:
        span = session.scalars(select(TraceSpan).where(col(TraceSpan.span_id) == "call-toolu_1")).first()
        assert span is not None
        span.content_sealed = _sealed(
            {"command": "pytest -x"}, workspace_id=span.workspace_id, trace_id=span.trace_id, span_id=span.span_id
        )
        session.commit()

    refused = _as(client, world["member"], "DELETE", f"/workspaces/{workspace_id}/trace-content")
    assert refused.status_code == status.HTTP_403_FORBIDDEN
    deleted = _as(client, world["admin"], "DELETE", f"/workspaces/{workspace_id}/trace-content")
    assert deleted.status_code == status.HTTP_200_OK, deleted.text

    with db_session_factory() as session:
        held = session.scalars(
            select(TraceSpan.span_id).where(
                col(TraceSpan.workspace_id) == workspace_id, col(TraceSpan.content_sealed).is_not(None)
            )
        ).all()
        kept = session.scalars(select(TraceSpan.span_id).where(col(TraceSpan.workspace_id) == workspace_id)).all()
    assert held == [] and kept


def test_the_content_policy_says_whether_the_deployment_keeps_content_and_only_to_a_signed_in_caller(
    client: TestClient, world: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = cast(FastAPI, client.app).state.config
    path = "/trace-content/policy"
    assert client.get(f"{API_ROOT}{path}").status_code == status.HTTP_401_UNAUTHORIZED
    assert "trace_content_available" not in client.get(f"{API_ROOT}/bootstrap").json()

    policy = _as(client, world["member"], "GET", path).json()
    assert policy == {
        "content_available": False,
        "platform_break_glass": False,
        "retention_days": config.trace_retention_days,
        "session_max_age_days": config.trace_session_max_age_days,
    }

    monkeypatch.setattr(config, "trace_content_enabled", True)
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    # Allowed, but with no key to seal it with nothing would be kept, so a workspace is not offered it.
    assert _as(client, world["member"], "GET", path).json()["content_available"] is False
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    assert _as(client, world["member"], "GET", path).json()["content_available"] is True


def test_a_share_records_a_polling_reader_once_and_pages_a_long_history(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    path = "/organizations/me/traces/s-one/shares"
    share = _as(client, world["member"], "POST", path, json={"audience": "organization_admins"}).json()

    # A page polling the session reads it again and again; that is one read.
    for _ in range(3):
        _detail(client, world["admin"])
    [entry] = _as(client, world["member"], "GET", path).json()["items"]
    assert (entry["read_count"], len(entry["reads"])) == (1, 1)

    with db_session_factory() as session:
        trace = session.scalars(select(Trace).where(col(Trace.trace_id) == "s-one")).one()
        session.add_all(
            TraceContentRead(
                share_id=uuid.UUID(share["id"]),
                workspace_id=trace.workspace_id,
                trace_id=trace.trace_id,
                session_user_id=trace.user_id,
                audience="organization_admins",
                reader_email=f"r{n}@a.test",
                read_at=datetime.now(UTC) - timedelta(days=1, minutes=n),
            )
            for n in range(30)
        )
        session.commit()

    [entry] = _as(client, world["member"], "GET", path).json()["items"]
    assert (entry["read_count"], len(entry["reads"])) == (31, 10)
    first = _as(client, world["member"], "GET", f"{path}/{share['id']}/reads", params={"limit": 20}).json()
    rest = _as(client, world["member"], "GET", f"{path}/{share['id']}/reads", params={"limit": 20, "skip": 20}).json()
    assert (len(first["items"]), first["has_more"]) == (20, True)
    assert (len(rest["items"]), rest["has_more"]) == (11, False)
    assert _as(client, world["admin"], "GET", f"{path}/{share['id']}/reads").status_code == status.HTTP_403_FORBIDDEN


def test_removing_a_member_from_the_workspace_ends_their_shares_at_once(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    share = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    ).json()
    workspace_id = _workspace_id(db_session_factory, "One")
    with db_session_factory() as session:
        member = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()

    removed = _as(client, world["admin"], "DELETE", f"/workspaces/{workspace_id}/members/{member.id}")
    assert removed.status_code == status.HTTP_200_OK, removed.text

    with db_session_factory() as session:
        stored = session.get(TraceContentShare, uuid.UUID(share["id"]))
        assert stored is not None and stored.revoked_at is not None


def test_a_running_share_is_on_the_first_page_however_many_ended_ones_follow_it(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    path = "/organizations/me/traces/s-one/shares"
    running = _as(client, world["member"], "POST", path, json={"audience": "organization_admins"}).json()
    with db_session_factory() as session:
        owner = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()
        workspace_id = _workspace_id(db_session_factory, "One")
        now = datetime.now(UTC)
        session.add_all(
            TraceContentShare(
                workspace_id=workspace_id,
                trace_id="s-one",
                audience="organization_admins",
                shared_by_user_id=owner.id,
                created_at=now + timedelta(minutes=n + 1),
                revoked_at=now,
            )
            for n in range(25)
        )
        session.commit()

    page = _as(client, world["member"], "GET", path).json()

    assert page["items"][0]["id"] == running["id"] and page["items"][0]["active"] is True
    assert len(page["items"]) == 20 and page["has_more"] is True


_REASON = "HR investigation 2026-17, approved by legal"


def test_an_owner_opens_a_private_session_by_break_glass_and_it_is_recorded_first(
    client: TestClient,
    master_key_header: dict[str, str],
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    path = "/organizations/me/traces/s-one/break-glass"

    # Admins and members are not owners, and a reason has to say something.
    assert _as(client, world["admin"], "POST", path, json={"reason": _REASON}).status_code == 403
    assert _as(client, world["member"], "POST", path, json={"reason": _REASON}).status_code == 403
    assert _as(client, world["operator"], "POST", path, json={"reason": "why"}).status_code == 422
    assert _as(client, world["operator"], "POST", path, json={"reason": " " * 20}).status_code == 422

    # A header master key names nobody for the record, so it opens nothing.
    unnamed = client.post(f"{API_ROOT}{path}", json={"reason": _REASON}, headers=master_key_header)
    assert unnamed.status_code == status.HTTP_403_FORBIDDEN and "master key" in unnamed.json()["detail"], unnamed.text

    opened = _as(client, world["operator"], "POST", path, json={"reason": _REASON})
    assert opened.status_code == status.HTTP_200_OK, opened.text
    assert opened.json()["content_access"] == "break_glass"
    assert _tool_call(opened.json())["content"] == {"command": "pytest -x"}

    log = _as(client, world["operator"], "GET", "/organizations/me/traces/audit/break-glass").json()
    [entry] = log["items"]
    assert (entry["kind"], entry["trace_id"], entry["reason"], entry["reader_email"]) == (
        "organization",
        "s-one",
        _REASON,
        "op@a.test",
    )
    assert _as(client, world["admin"], "GET", "/organizations/me/traces/audit/break-glass").status_code == 403
    # An owner's reasons stay with the organization's owners: the operator's log lists only the operator's openings.
    assert _operator_get(client, world, "/traces/audit/break-glass").json()["items"] == []
    # The session's owner is not told: their own view of the session is unchanged.
    assert _detail(client, world["member"])["content_access"] == "owner"


def test_break_glass_opens_a_departed_members_session(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    with db_session_factory() as session:
        member = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()
        for row in session.scalars(select(WorkspaceMember).where(col(WorkspaceMember.user_id) == member.id)):
            session.delete(row)
        session.commit()

    opened = _as(
        client, world["operator"], "POST", "/organizations/me/traces/s-one/break-glass", json={"reason": _REASON}
    )

    assert _tool_call(opened.json())["content"] == {"command": "pytest -x"}


def test_nothing_opens_when_the_break_glass_record_cannot_be_written(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    opened: list[str] = []
    real_open = SpanContent.open

    async def tracked_open(self: SpanContent, *args: Any, **kwargs: Any) -> dict[str, str]:
        opened.append("opened")
        return await real_open(self, *args, **kwargs)

    async def refuse(self: Any, **values: object) -> None:
        raise RuntimeError("the audit table is unavailable")

    monkeypatch.setattr(SpanContent, "open", tracked_open)
    monkeypatch.setattr(BreakGlassRepository, "record", refuse)

    with pytest.raises(RuntimeError):
        _as(client, world["operator"], "POST", "/organizations/me/traces/s-one/break-glass", json={"reason": _REASON})
    assert opened == []


def test_platform_break_glass_is_off_by_default_and_reaches_any_organization_when_on(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with db_session_factory() as session:
        span = session.scalars(select(TraceSpan).where(col(TraceSpan.trace_id) == "s-theirs")).first()
        assert span is not None
        span.content_sealed = _sealed(
            {"command": "make deploy"}, workspace_id=span.workspace_id, trace_id=span.trace_id, span_id=span.span_id
        )
        theirs = span.workspace_id
        session.commit()
    body = {"reason": "Court order 4411 from the Dublin District Court", "reference": "LEGAL-4411"}
    path = f"/traces/s-theirs/break-glass?workspace_id={theirs}"

    assert _as(client, world["operator"], "POST", path, json=body).status_code == status.HTTP_404_NOT_FOUND

    monkeypatch.setattr(cast(FastAPI, client.app).state.config, "trace_platform_break_glass_enabled", True)
    assert _as(client, world["operator"], "POST", path, json={"reason": body["reason"]}).status_code == 422
    opened = _as(client, world["operator"], "POST", path, json=body)
    assert opened.status_code == status.HTTP_200_OK, opened.text
    contents = [span["content"] for span in opened.json()["spans"] if span["content"]]
    assert contents == [{"command": "make deploy"}]

    [entry] = _operator_get(client, world, "/traces/audit/break-glass").json()["items"]
    assert (entry["kind"], entry["reference"]) == ("deployment", "LEGAL-4411")
    # Not on the organization owners' own log: theirs lists owners' openings.
    assert _as(client, world["outsider"], "GET", "/organizations/me/traces/audit/break-glass").json()["items"] == []
    assert _as(client, world["admin"], "POST", path, json=body).status_code == status.HTTP_403_FORBIDDEN


def test_removing_a_member_from_the_organization_ends_their_shares_so_a_return_revives_none(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    share = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    ).json()
    with db_session_factory() as session:
        member = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()
        membership = session.scalars(
            select(OrganizationMember).where(col(OrganizationMember.user_id) == member.id)
        ).one()
        membership_id = membership.id

    removed = _as(client, world["admin"], "DELETE", f"/organizations/me/members/{membership_id}")
    assert removed.status_code == status.HTTP_200_OK, removed.text

    with db_session_factory() as session:
        stored = session.get(TraceContentShare, uuid.UUID(share["id"]))
        assert stored is not None and stored.revoked_at is not None
        # Back in the organization, the share stays ended.
        returned = session.get(OrganizationMember, membership_id)
        assert returned is not None
        returned.status = "active"
        session.commit()
    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True


def _admin_shares_s_one(
    client: TestClient, world: dict[str, str], db_session_factory: Callable[[], Session]
) -> tuple[uuid.UUID, uuid.UUID]:
    """The organization's admin owns s-one and shares it with the admins; their membership and the share."""
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "a@a.test")
    share = _as(
        client,
        world["admin"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    )
    assert share.status_code == status.HTTP_201_CREATED, share.text
    with db_session_factory() as session:
        admin = session.scalars(select(User).where(col(User.email) == "a@a.test")).one()
        membership = session.scalars(
            select(OrganizationMember).where(col(OrganizationMember.user_id) == admin.id)
        ).one()
        return membership.id, uuid.UUID(share.json()["id"])


def _set_role(client: TestClient, world: dict[str, str], membership_id: uuid.UUID, role: str) -> None:
    changed = _as(client, world["operator"], "PATCH", f"/organizations/me/members/{membership_id}", json={"role": role})
    assert changed.status_code == status.HTTP_200_OK, changed.text


def test_losing_the_admin_role_that_was_the_only_way_into_a_workspace_ends_its_shares_for_good(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    membership_id, share_id = _admin_shares_s_one(client, world, db_session_factory)

    # Demoted and promoted back with no read in between, so nothing ends the share lazily.
    _set_role(client, world, membership_id, "member")
    with db_session_factory() as session:
        stored = session.get(TraceContentShare, share_id)
        assert stored is not None and stored.revoked_at is not None
    _set_role(client, world, membership_id, "admin")

    tool = _tool_call(_operator_get(client, world, "/traces/s-one").json())
    assert (tool["content"], tool["content_private"]) == ({}, True)


def test_losing_the_admin_role_keeps_shares_in_a_workspace_its_owner_is_a_member_of(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    membership_id, share_id = _admin_shares_s_one(client, world, db_session_factory)
    with db_session_factory() as session:
        trace = session.scalars(select(Trace).where(col(Trace.trace_id) == "s-one")).one()
        admin = session.scalars(select(User).where(col(User.email) == "a@a.test")).one()
        session.add(WorkspaceMember(workspace_id=trace.workspace_id, user_id=admin.id, role="member", status="active"))
        session.commit()

    _set_role(client, world, membership_id, "member")

    with db_session_factory() as session:
        stored = session.get(TraceContentShare, share_id)
        assert stored is not None and stored.revoked_at is None
    assert _tool_call(_operator_get(client, world, "/traces/s-one").json())["content"] == {"command": "pytest -x"}


def test_a_share_made_after_its_owner_was_removed_mid_request_is_refused(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scope was resolved while they were a member; the removal lands before the share is written."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    real_owned = TraceService._owned

    async def owned_then_removed(self: TraceService, *args: Any, **kwargs: Any) -> Any:
        row = await real_owned(self, *args, **kwargs)
        with db_session_factory() as session:
            member = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()
            for membership in session.scalars(select(WorkspaceMember).where(col(WorkspaceMember.user_id) == member.id)):
                session.delete(membership)
            session.commit()
        return row

    monkeypatch.setattr(TraceService, "_owned", owned_then_removed)

    refused = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    )

    assert refused.status_code == status.HTTP_403_FORBIDDEN, refused.text
    with db_session_factory() as session:
        assert session.scalars(select(TraceContentShare)).all() == []


def test_an_operator_who_owns_the_organization_reads_a_share_with_its_owners_on_the_operator_routes(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dashboard reads through the operator's routes for an operator, who may be an owner as well."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    assert _tool_call(_operator_get(client, world, "/traces/s-one").json())["content_private"] is True

    created = _as(
        client,
        world["member"],
        "POST",
        "/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
    )
    assert created.status_code == status.HTTP_201_CREATED, created.text

    detail = _operator_get(client, world, "/traces/s-one").json()
    assert (_tool_call(detail)["content"], detail["content_access"]) == ({"command": "pytest -x"}, "share")


def _share_s_one(client: TestClient, world: dict[str, str], audience: str = "organization_admins") -> dict[str, Any]:
    created = _as(client, world["member"], "POST", "/organizations/me/traces/s-one/shares", json={"audience": audience})
    assert created.status_code == status.HTTP_201_CREATED, created.text
    share: dict[str, Any] = created.json()
    return share


def _user_id(db_session_factory: Callable[[], Session], email: str) -> uuid.UUID:
    with db_session_factory() as session:
        user_id: uuid.UUID = session.scalars(select(col(User.id)).where(col(User.email) == email)).one()
        return user_id


def test_deactivating_an_account_ends_its_shares_and_reactivating_brings_none_back(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    share = _share_s_one(client, world)
    assert _tool_call(_detail(client, world["admin"]))["content"] == {"command": "pytest -x"}
    member_id = _user_id(db_session_factory, "m@a.test")

    deactivated = _as(client, world["operator"], "PATCH", f"/admin/users/{member_id}", json={"is_active": False})
    assert deactivated.status_code == status.HTTP_200_OK, deactivated.text
    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True
    with db_session_factory() as session:
        stored = session.get(TraceContentShare, uuid.UUID(share["id"]))
        assert stored is not None and stored.revoked_at is not None

    reactivated = _as(client, world["operator"], "PATCH", f"/admin/users/{member_id}", json={"is_active": True})
    assert reactivated.status_code == status.HTTP_200_OK, reactivated.text
    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True


def test_a_deactivated_account_is_no_member_whose_content_is_kept_or_shared(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deactivated some other way than the admin page, its shares still open for nobody."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    _share_s_one(client, world)
    with db_session_factory() as session:
        member = session.scalars(select(User).where(col(User.email) == "m@a.test")).one()
        member.is_active = False
        session.commit()

    assert _tool_call(_detail(client, world["admin"]))["content_private"] is True


def test_a_header_master_key_reads_no_content_even_of_the_account_it_stands_for(
    client: TestClient,
    master_key_header: dict[str, str],
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with db_session_factory() as session:
        # The organization the master key acts in: the one the world fixture did not make.
        default_organization = session.scalars(
            select(col(Organization.id)).where(col(Organization.slug).not_in(["alpha-t", "beta-t"]))
        ).one()
        workspace = session.scalars(select(Workspace).where(col(Workspace.name) == "One")).one()
        workspace.organization_id = default_organization
        bootstrap = session.scalars(
            select(OrganizationMember).where(
                col(OrganizationMember.organization_id) == default_organization,
                col(OrganizationMember.role) == "owner",
            )
        ).first()
        assert bootstrap is not None
        # The bootstrap account already bills as itself, so its session is given to it directly.
        trace = session.scalars(select(Trace).where(col(Trace.trace_id) == "s-one")).one()
        trace.user_id = str(bootstrap.user_id)
        session.commit()
    _seal_tool_call(db_session_factory)

    for path in ("/traces/s-one", "/organizations/me/traces/s-one"):
        response = client.get(f"{API_ROOT}{path}", headers=master_key_header)
        assert response.status_code == status.HTTP_200_OK, response.text
        detail = response.json()
        assert (_tool_call(detail)["content_private"], detail["content_access"]) == (True, "none")
    shared = client.post(
        f"{API_ROOT}/organizations/me/traces/s-one/shares",
        json={"audience": "organization_admins"},
        headers=master_key_header,
    )
    assert shared.status_code == status.HTTP_403_FORBIDDEN


def test_a_share_of_one_session_is_not_found_through_another(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    with db_session_factory() as session:
        _trace(session, _workspace_id(db_session_factory, "One"), "s-own")
        trace = session.scalars(select(Trace).where(col(Trace.trace_id) == "s-own")).one()
        trace.user_id = str(_user_id(db_session_factory, "m@a.test"))
        session.commit()
    share = _share_s_one(client, world)

    other = "/organizations/me/traces/s-own/shares"
    assert _as(client, world["member"], "GET", f"{other}/{share['id']}/reads").status_code == 404
    assert _as(client, world["member"], "DELETE", f"{other}/{share['id']}").status_code == 404
    assert _as(client, world["member"], "GET", other).json()["items"] == []
    assert _tool_call(_detail(client, world["admin"]))["content"] == {"command": "pytest -x"}


def test_a_workspace_admin_or_viewer_reads_no_share_and_breaks_no_glass(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    _share_s_one(client, world)
    one = _workspace_id(db_session_factory, "One")
    cookies = {}
    with db_session_factory() as session:
        alpha = session.get(Workspace, one).organization_id  # type: ignore[union-attr]
        for role in ("admin", "viewer"):
            email = f"ws-{role}@a.test"
            cookies[role] = _identity(session, email=email, organization_id=alpha, role="member")
            user = session.scalars(select(User).where(col(User.email) == email)).one()
            session.add(WorkspaceMember(workspace_id=one, user_id=user.id, role=role, status="active"))
        session.commit()

    for role, cookie in cookies.items():
        tool = _tool_call(_detail(client, cookie))
        assert tool["content_private"] is True, role
        opened = _as(client, cookie, "POST", "/organizations/me/traces/s-one/break-glass", json={"reason": _REASON})
        assert opened.status_code == status.HTTP_403_FORBIDDEN, role
    # The workspace's admin manages its switch; its viewer does not.
    toggle = {"trace_content_enabled": True}
    assert _as(client, cookies["admin"], "PATCH", f"/workspaces/{one}", json=toggle).status_code == 200
    assert _as(client, cookies["viewer"], "PATCH", f"/workspaces/{one}", json=toggle).status_code == 403
    assert _as(client, cookies["viewer"], "DELETE", f"/workspaces/{one}/trace-content").status_code == 403


def test_an_owner_of_another_organization_cannot_break_glass_into_this_one(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _seal_tool_call(db_session_factory)
    _give_s_one_to(db_session_factory, "m@a.test")
    one = _workspace_id(db_session_factory, "One")

    opened = _as(
        client,
        world["outsider"],
        "POST",
        "/organizations/me/traces/s-one/break-glass",
        params={"workspace_id": str(one)},
        json={"reason": _REASON},
    )

    assert opened.status_code == status.HTTP_404_NOT_FOUND
    with db_session_factory() as session:
        assert session.scalars(select(TraceBreakGlassRead)).all() == []


def test_deleting_stored_content_clears_it_a_batch_at_a_time(
    client: TestClient,
    world: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(_service, "CLEAR_CONTENT_BATCH", 2)
    workspace_id = _workspace_id(db_session_factory, "One")
    with db_session_factory() as session:
        for span in session.scalars(select(TraceSpan).where(col(TraceSpan.workspace_id) == workspace_id)):
            span.content_sealed = _sealed(
                {"command": "ls"}, workspace_id=span.workspace_id, trace_id=span.trace_id, span_id=span.span_id
            )
        session.commit()

    deleted = _as(client, world["admin"], "DELETE", f"/workspaces/{workspace_id}/trace-content")

    assert deleted.status_code == status.HTTP_200_OK, deleted.text
    with db_session_factory() as session:
        held = session.scalars(
            select(TraceSpan.span_id).where(
                col(TraceSpan.workspace_id) == workspace_id, col(TraceSpan.content_sealed).is_not(None)
            )
        ).all()
    assert held == []


def test_the_batched_membership_check_agrees_with_asking_one_pair_at_a_time(
    world: dict[str, str], db_session_factory: Callable[[], Session], postgres_url: str
) -> None:
    with db_session_factory() as session:
        ids = {user.email: str(user.id) for user in session.scalars(select(User))}
        workspaces = {workspace.name: workspace.id for workspace in session.scalars(select(Workspace))}
        one_row = session.get(Workspace, workspaces["One"])
        assert one_row is not None
        gone = User(email="gone@a.test", full_name="X", is_active=False, active_organization_id=one_row.organization_id)
        suspended = User(email="s@a.test", full_name="X", active_organization_id=one_row.organization_id)
        session.add_all([gone, suspended])
        session.commit()
        session.add_all(
            [
                OrganizationMember(
                    organization_id=one_row.organization_id, user_id=gone.id, role="owner", status="active"
                ),
                OrganizationMember(
                    organization_id=one_row.organization_id, user_id=suspended.id, role="member", status="active"
                ),
                WorkspaceMember(workspace_id=one_row.id, user_id=suspended.id, role="member", status="suspended"),
            ]
        )
        session.commit()
        ids |= {"gone@a.test": str(gone.id), "s@a.test": str(suspended.id)}

    one, two, theirs = workspaces["One"], workspaces["Two"], workspaces["Theirs"]
    pairs: set[tuple[uuid.UUID, str | None]] = {
        (one, ids["m@a.test"]),
        (two, ids["m@a.test"]),
        (one, ids["a@a.test"]),
        (two, ids["a@a.test"]),
        (one, ids["o@b.test"]),
        (theirs, ids["o@b.test"]),
        (one, ids["gone@a.test"]),
        (one, ids["s@a.test"]),
        (one, ids["m@a.test"].upper()),
        (one, None),
        (one, "billing-user"),
        (uuid.uuid4(), ids["a@a.test"]),
        (one, str(uuid.uuid4())),
    }

    async def ask() -> tuple[set[tuple[uuid.UUID, str | None]], set[tuple[uuid.UUID, str | None]]]:
        engine = create_async_engine(_to_async_url(postgres_url))
        try:
            async with async_sessionmaker(engine)() as db:
                batched = await workspace_members(db, pairs)
                singly = {pair for pair in pairs if await is_workspace_member(db, *pair)}
                return batched, singly
        finally:
            await engine.dispose()

    batched, singly = asyncio.run(ask())

    assert (
        batched
        == singly
        == {
            (one, ids["m@a.test"]),
            (one, ids["a@a.test"]),
            (two, ids["a@a.test"]),
            (theirs, ids["o@b.test"]),
        }
    )


def test_a_model_filter_keeps_the_sessions_with_a_call_on_any_of_the_models(
    client: TestClient, world: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    models = {"s-one": "model-a", "s-two": "model-b", "s-theirs": "model-a"}
    with db_session_factory() as session:
        sibling = session.scalars(select(Workspace).where(col(Workspace.name) == "Two")).one()
        # A sibling workspace's session reuses the id "s-one" on a model of its own.
        _trace(session, sibling.id, "s-one")
        for span in session.scalars(select(TraceSpan).where(col(TraceSpan.kind) == "llm")):
            span.model = (
                "model-x" if span.workspace_id == sibling.id and span.trace_id == "s-one" else models[span.trace_id]
            )
        session.commit()
        sibling_id = str(sibling.id)

    def operator(path: str, *values: str) -> Any:
        response = _operator_get(client, world, path, {"model": list(values)})
        assert response.status_code == status.HTTP_200_OK, response.text
        return response.json()

    assert [item["trace_id"] for item in operator("/traces", "model-a")["items"]] == ["s-one"]
    assert operator("/traces/count", "model-b")["count"] == 1
    assert operator("/traces/series", "model-b")["points"] == [
        {"bucket": "2026-10-07T12:00:00Z", "succeeded": 0, "failed": 1}
    ]
    assert operator("/traces/count", "model-none")["count"] == 0
    assert operator("/traces/series", "model-none")["points"] == []
    assert sorted(item["trace_id"] for item in operator("/traces", "model-a", "model-b")["items"]) == [
        "s-one",
        "s-two",
    ]
    assert operator("/traces/count", "model-a", "model-b")["count"] == 2
    # The probe is keyed on the workspace too: a shared trace id never lends its spans across workspaces.
    shared = operator("/traces", "model-x")["items"]
    assert [(item["trace_id"], item["workspace_id"]) for item in shared] == [("s-one", sibling_id)]
    assert operator("/traces/count", "model-x")["count"] == 1
    assert [item["workspace_id"] for item in operator("/traces", "model-a")["items"]] != [sibling_id]
    too_many = [f"model-{index}" for index in range(_MAX_FILTER_VALUES + 1)]
    response = _operator_get(client, world, "/traces", {"model": too_many})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def organization(cookie: str, path: str = "") -> Any:
        client.cookies.set(SESSION_COOKIE_NAME, cookie)
        try:
            response = client.get(f"{API_ROOT}/organizations/me/traces{path}", params={"model": ["model-a"]})
        finally:
            client.cookies.clear()
        assert response.status_code == status.HTTP_200_OK, response.text
        return response.json()

    # The filter narrows inside the scope: model-a is also on another organization's session.
    assert [item["trace_id"] for item in organization(world["admin"])["items"]] == ["s-one"]
    assert organization(world["admin"], "/count")["count"] == 1
    assert organization(world["member"], "/count")["count"] == 1
    assert organization(world["outsider"], "/count")["count"] == 1
    assert [item["trace_id"] for item in organization(world["outsider"])["items"]] == ["s-theirs"]
    assert len(organization(world["admin"], "/series")["points"]) == 1

    def shared_in_organization(cookie: str) -> list[str]:
        client.cookies.set(SESSION_COOKIE_NAME, cookie)
        try:
            response = client.get(f"{API_ROOT}/organizations/me/traces", params={"model": ["model-x"]})
        finally:
            client.cookies.clear()
        assert response.status_code == status.HTTP_200_OK, response.text
        return [item["workspace_id"] for item in response.json()["items"]]

    assert shared_in_organization(world["admin"]) == [sibling_id]
    assert shared_in_organization(world["member"]) == []
    assert shared_in_organization(world["outsider"]) == []
