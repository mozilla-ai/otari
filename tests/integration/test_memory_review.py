"""The review queue: what the consolidator holds for an admin, and what approving or discarding it does.

Agents propose over MCP and one consolidation pass runs as the worker would, so what is asserted is what the
queue lists, what the next pass makes of an approved proposal, and who may act on it.
"""

import uuid
from collections.abc import Callable, Generator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.memory import MemoryEntry, MemoryProposal
from gateway.models.tenancy import DashboardSession, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.memory import MemoryModels, ModelUse
from gateway.services.memory._models import Decision
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client
from .test_memory_guardrails import INJECTION, _Guardrails, _mandate
from .test_memory_guardrails import _empty_runner as _empty_runner  # noqa: F401 - the autouse fixture, used by name
from .test_memory_mcp import DEPLOY, _call, _consolidate, _embed, _key
from .test_memory_mcp import memory_client as memory_client  # noqa: F401 - the fixture, used by name
from .test_memory_tool_loop import _switches

# Alike to DEPLOY in every word the test embedding counts, so the consolidator compares the two.
NEVER_FROM_MAIN = "Deploy the api service from a release branch, never from main"


def _workspace_id(db_session_factory: Callable[[], Session]) -> uuid.UUID:
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace)).first()
        assert workspace is not None
        return workspace.id


def _queue_url(workspace_id: uuid.UUID | str) -> str:
    return f"{API_ROOT}/workspaces/{workspace_id}/memory/review"


def _queue(client: TestClient, headers: dict[str, str], workspace_id: uuid.UUID) -> list[dict[str, Any]]:
    response = client.get(_queue_url(workspace_id), headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == len(body["data"])
    return list(body["data"])


def _propose(client: TestClient, headers: dict[str, str], text: str, **arguments: Any) -> None:
    proposed = _call(client, headers, "memory_propose", text=text, kind=arguments.pop("kind", "procedure"), **arguments)
    assert proposed["isError"] is False, proposed


def _remembered(client: TestClient, headers: dict[str, str], query: str) -> list[dict[str, Any]]:
    found = _call(client, headers, "memory_search", query=query, limit=10)
    return list(found["structuredContent"]["results"])


def _proposals(db_session_factory: Callable[[], Session]) -> list[tuple[str, str, str | None]]:
    with db_session_factory() as session:
        return [(p.scope, p.state, p.review_reason) for p in session.scalars(select(MemoryProposal))]


def test_the_queue_lists_held_shared_proposals_opened_and_approving_one_keeps_it_as_reviewed(
    memory_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _switches(memory_client, db_session_factory, master_key_header, write_policy="review")
    _propose(memory_client, headers, DEPLOY)
    assert _consolidate(memory_client) == 1

    [held] = _queue(memory_client, master_key_header, workspace_id)
    assert (held["text"], held["kind"], held["reason"], held["trust"], held["entry"]) == (
        DEPLOY,
        "procedure",
        "policy",
        "agent",
        None,
    )
    created = datetime.fromisoformat(held["created_at"])
    assert datetime.fromisoformat(held["expires_at"]) - created == timedelta(hours=24)
    assert created > datetime.now(UTC) - timedelta(minutes=5)
    memory = memory_client.get(f"{API_ROOT}/workspaces/{workspace_id}/memory", headers=master_key_header).json()
    assert (memory["proposals_held"], memory["proposals_waiting"]) == (1, 0)

    approved = memory_client.post(f"{_queue_url(workspace_id)}/{held['id']}/approve", headers=master_key_header)
    assert approved.status_code == 204, approved.text
    assert _queue(memory_client, master_key_header, workspace_id) == []
    memory = memory_client.get(f"{API_ROOT}/workspaces/{workspace_id}/memory", headers=master_key_header).json()
    assert (memory["proposals_held"], memory["proposals_waiting"]) == (0, 1)

    # The review policy is still on, and the approved proposal is not held by it again.
    assert _consolidate(memory_client) == 1
    [entry] = _remembered(memory_client, headers, DEPLOY)
    assert (entry["text"], entry["trust"]) == (DEPLOY, "reviewed")
    assert _proposals(db_session_factory) == []


def test_the_queue_leaves_out_member_proposals_and_answers_for_its_own_workspace_only(
    memory_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _switches(memory_client, db_session_factory, master_key_header, write_policy="review")
    _propose(memory_client, headers, DEPLOY)
    assert _consolidate(memory_client) == 1
    # A member's proposal held before member proposals stopped being held, say, is never shown either.
    _propose(memory_client, headers, "I prefer short answers", kind="fact", scope="member")
    with db_session_factory() as session:
        session.execute(
            update(MemoryProposal).where(MemoryProposal.scope == "member").values(state="review", review_reason="model")
        )
        session.commit()

    [held] = _queue(memory_client, master_key_header, workspace_id)
    assert held["text"] == DEPLOY
    with db_session_factory() as session:
        member_proposal = session.scalars(select(MemoryProposal).where(MemoryProposal.scope == "member")).one().id
    for action in ("approve", "discard"):
        response = memory_client.post(
            f"{_queue_url(workspace_id)}/{member_proposal}/{action}", headers=master_key_header
        )
        assert response.status_code == 404, response.text

    other = memory_client.post(f"{API_ROOT}/workspaces", json={"name": "Elsewhere"}, headers=master_key_header)
    assert other.status_code in (200, 201), other.text
    elsewhere = uuid.UUID(other.json()["id"])
    assert _queue(memory_client, master_key_header, elsewhere) == []
    for action in ("approve", "discard"):
        response = memory_client.post(f"{_queue_url(elsewhere)}/{held['id']}/{action}", headers=master_key_header)
        assert response.status_code == 404, response.text
    assert [reason for _, _, reason in _proposals(db_session_factory)].count("policy") == 1


def test_discarding_deletes_the_proposal(
    memory_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _switches(memory_client, db_session_factory, master_key_header, write_policy="review")
    _propose(memory_client, headers, DEPLOY)
    _consolidate(memory_client)
    [held] = _queue(memory_client, master_key_header, workspace_id)

    discarded = memory_client.post(f"{_queue_url(workspace_id)}/{held['id']}/discard", headers=master_key_header)

    assert discarded.status_code == 204, discarded.text
    assert _proposals(db_session_factory) == []
    assert _consolidate(memory_client) == 0
    assert _remembered(memory_client, headers, DEPLOY) == []
    again = memory_client.post(f"{_queue_url(workspace_id)}/{held['id']}/discard", headers=master_key_header)
    assert again.status_code == 404


def test_an_approved_proposal_a_guardrail_held_is_kept_without_being_checked_again(
    memory_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _mandate(memory_client, master_key_header, profile="prompt-injection", mode="block", applies_to_all_workspaces=True)
    guardrails = _Guardrails(monkeypatch)
    _propose(memory_client, headers, INJECTION, kind="fact")
    assert _consolidate(memory_client) == 1

    [held] = _queue(memory_client, master_key_header, workspace_id)
    assert (held["text"], held["reason"]) == (INJECTION, "guardrail")
    approved = memory_client.post(f"{_queue_url(workspace_id)}/{held['id']}/approve", headers=master_key_header)
    assert approved.status_code == 204, approved.text

    assert _consolidate(memory_client) == 1
    assert len(guardrails.calls) == 1
    assert [entry["text"] for entry in _remembered(memory_client, headers, INJECTION)] == [INJECTION]


def test_a_member_proposal_a_guardrail_refuses_is_discarded_never_held(
    memory_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _mandate(memory_client, master_key_header, profile="prompt-injection", mode="block", applies_to_all_workspaces=True)
    _Guardrails(monkeypatch)
    _propose(memory_client, headers, INJECTION, kind="fact", scope="member")

    assert _consolidate(memory_client) == 1

    assert _proposals(db_session_factory) == []
    assert _queue(memory_client, master_key_header, workspace_id) == []
    assert _remembered(memory_client, headers, INJECTION) == []


async def _decide_conflict(self: MemoryModels, proposal: str, existing: Sequence[str]) -> tuple[Decision, ModelUse]:
    """The consolidation model: finds the proposal contradicts the nearest entry."""
    return Decision("conflict", 0), ModelUse("/v1/memory/consolidation", "openai:decide", "openai", True, 40, 8, 3)


@pytest.fixture
def deciding_client(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    """A deployment that names a consolidation model, which finds every proposal contradicts its nearest entry."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(MemoryModels, "embed", _embed)
    monkeypatch.setattr(MemoryModels, "decide", _decide_conflict)
    config = test_config.model_copy(
        update={
            "memory_enabled": True,
            "memory_embedding_model": "openai:embed",
            "memory_consolidation_model": "openai:decide",
        }
    )
    yield from build_test_client(config)


def test_approving_a_conflict_replaces_the_entry_it_contradicts(
    deciding_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    headers = _key(deciding_client, master_key_header, db_session_factory)
    _propose(deciding_client, headers, DEPLOY)
    assert _consolidate(deciding_client) == 1
    [kept] = _remembered(deciding_client, headers, DEPLOY)
    _propose(deciding_client, headers, NEVER_FROM_MAIN)
    assert _consolidate(deciding_client) == 1

    [held] = _queue(deciding_client, master_key_header, workspace_id)
    assert (held["text"], held["reason"]) == (NEVER_FROM_MAIN, "conflict")
    assert held["entry"] == {"id": kept["id"], "text": DEPLOY, "version": 1, "trust": "agent"}

    approved = deciding_client.post(f"{_queue_url(workspace_id)}/{held['id']}/approve", headers=master_key_header)
    assert approved.status_code == 204, approved.text
    # The model would still call it a conflict; the admin's approval settles it without asking again.
    assert _consolidate(deciding_client) == 1

    [entry] = _remembered(deciding_client, headers, DEPLOY)
    assert (entry["text"], entry["version"], entry["trust"]) == (NEVER_FROM_MAIN, 2, "reviewed")
    with db_session_factory() as session:
        replaced = session.get(MemoryEntry, uuid.UUID(kept["id"]))
        assert replaced is not None and replaced.status == "superseded"
    assert _proposals(db_session_factory) == []


def test_a_members_own_conflict_is_settled_by_their_newer_word(
    deciding_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    headers = _key(deciding_client, master_key_header, db_session_factory)
    _propose(deciding_client, headers, DEPLOY, scope="member")
    _consolidate(deciding_client)
    _propose(deciding_client, headers, NEVER_FROM_MAIN, scope="member")

    assert _consolidate(deciding_client) == 1

    [entry] = _remembered(deciding_client, headers, DEPLOY)
    assert (entry["text"], entry["version"], entry["scope"]) == (NEVER_FROM_MAIN, 2, "member")
    assert _proposals(db_session_factory) == []


def _member_session(db_session_factory: Callable[[], Session], workspace_id: uuid.UUID) -> str:
    """A dashboard session for a plain member of the workspace and its organization, who manages neither."""
    token = f"otari-sess-{uuid.uuid4().hex[:8]}"
    with db_session_factory() as session:
        workspace = session.get(Workspace, workspace_id)
        assert workspace is not None
        person = User(email=f"{uuid.uuid4().hex[:8]}@example.test", active_organization_id=workspace.organization_id)
        session.add(person)
        session.flush()
        session.add(
            OrganizationMember(
                organization_id=workspace.organization_id, user_id=person.id, role="member", status="active"
            )
        )
        session.add(WorkspaceMember(workspace_id=workspace_id, user_id=person.id, role="member"))
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=person.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
    return token


def test_only_the_workspaces_managers_see_or_settle_what_waits(
    memory_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    workspace_id = _workspace_id(db_session_factory)
    headers = _key(memory_client, master_key_header, db_session_factory)
    _switches(memory_client, db_session_factory, master_key_header, write_policy="review")
    _propose(memory_client, headers, DEPLOY)
    _consolidate(memory_client)
    [held] = _queue(memory_client, master_key_header, workspace_id)

    # An agent's key, as for the switches.
    assert memory_client.get(_queue_url(workspace_id), headers=headers).status_code in (401, 403)
    for action in ("approve", "discard"):
        response = memory_client.post(f"{_queue_url(workspace_id)}/{held['id']}/{action}", headers=headers)
        assert response.status_code in (401, 403)

    # A member signed in to the dashboard who manages neither the workspace nor its organization.
    memory_client.cookies.set(SESSION_COOKIE_NAME, _member_session(db_session_factory, workspace_id))
    try:
        assert memory_client.get(_queue_url(workspace_id)).status_code == 403
        for action in ("approve", "discard"):
            response = memory_client.post(f"{_queue_url(workspace_id)}/{held['id']}/{action}")
            assert response.status_code == 403, response.text
    finally:
        memory_client.cookies.clear()
    assert _proposals(db_session_factory) == [("workspace", "review", "policy")]
