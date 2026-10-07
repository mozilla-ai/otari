"""The trace read API answers inside the caller's scope and nowhere else.

The operator routes read every workspace. The organization routes follow the
rule organization usage follows: an owner or admin reads every workspace in
their active organization, a member the workspaces they belong to, and nobody
reads another organization's traces, whose ids answer 404 like ones that do
not exist.
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.models.traces import Trace, TraceSpan
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

_T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)


def _identity(
    session: Session, *, email: str, organization_id: uuid.UUID, role: str, workspace_ids: tuple[uuid.UUID, ...] = ()
) -> str:
    user = User(email=email, full_name="X", active_organization_id=organization_id)
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
                name="step",
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
        }


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


def test_the_operator_reads_every_workspace_and_the_detail_has_turns(
    client: TestClient, master_key_header: dict[str, str], world: dict[str, str]
) -> None:
    listed = client.get(f"{API_ROOT}/traces", headers=master_key_header).json()
    assert sorted(item["trace_id"] for item in listed["items"]) == ["s-one", "s-theirs", "s-two"]

    detail = client.get(f"{API_ROOT}/traces/s-two", headers=master_key_header).json()

    [turn] = detail["turns"]
    assert (turn["state"], turn["llm_calls"], turn["tool_calls"], turn["step_ids"]) == (
        "failed",
        2,
        1,
        ["req-1", "req-2"],
    )
    tool = next(span for span in detail["spans"] if span["kind"] == "tool")
    assert tool["approximate"] is True
    assert tool["parent_span_id"] == "req-1"
    assert tool["start_time"].startswith("2026-10-07T12:00:10")


def test_the_series_counts_sessions_by_whether_they_failed(
    client: TestClient, master_key_header: dict[str, str], world: dict[str, str]
) -> None:
    series = client.get(f"{API_ROOT}/traces/series?bucket=hour", headers=master_key_header).json()

    assert series["points"] == [{"bucket": "2026-10-07T12:00:00Z", "succeeded": 2, "failed": 1}]


def test_the_operator_routes_refuse_a_member(client: TestClient, world: dict[str, str]) -> None:
    client.cookies.set(SESSION_COOKIE_NAME, world["admin"])
    try:
        response = client.get(f"{API_ROOT}/traces")
    finally:
        client.cookies.clear()

    assert response.status_code == status.HTTP_403_FORBIDDEN
