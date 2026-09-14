"""The Playground serves one signed-in person, and only what is theirs.

Two things are load-bearing here and everything in this file is about one of
them.

**A completion runs as the caller, not as a key and not as the deployment.**
``/api/v1/chat/completions`` refuses a cookie on purpose, because a keyless
request resolves to the *default* workspace and honoring one there would let any
member of any organization spend the default organization's credential
(otari-ai#1880). The Playground's endpoint resolves the caller's own attribution
user and proves their membership of the workspace it will bill before the
pipeline sees anything, so the tests below assert on the rows that came out: the
usage row's ``user_id`` and its null ``api_key_id``, the workspace it was
stamped with, and the refusals for a workspace that is not the caller's.

**A stored row is readable only by its owner.** The hosted original leaked
prompts and model outputs across organizations because a list query trusted a
client-supplied scope (otari-ai#978). So every read and delete here is also
asserted from a second identity, and the expected answer is the 404 a
nonexistent row gets rather than a 403, which would confirm the row exists.

The world is two organizations because one cannot express either property: a
single-tenant fixture makes "the caller's workspace" and "the default workspace"
the same row, which is exactly the conflation both bugs were.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT
from gateway.core.usage_source import PLAYGROUND_USAGE_ENDPOINT, SERVED_HERE_SLUG
from gateway.models.entities import DashboardSession, UsageLog, WorkspaceWebSearchConfig
from gateway.models.entities import User as BillingUser
from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

from .conftest import MODEL_NAME

_PREFIX = f"{API_ROOT}/playground"


@dataclass
class _World:
    """Two organizations, their workspaces, and one session cookie per identity."""

    alpha: uuid.UUID
    beta: uuid.UUID
    workspaces: dict[str, uuid.UUID] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    users: dict[str, uuid.UUID] = field(default_factory=dict)


def _identity(
    session: Session,
    *,
    email: str,
    organization_id: uuid.UUID,
    role: str = "member",
    workspace_ids: tuple[uuid.UUID, ...] = (),
) -> tuple[uuid.UUID, str]:
    """Create an identity with a live dashboard session, and return its cookie."""
    user = User(email=email, full_name=email.split("@")[0].title(), active_organization_id=organization_id)
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
    return user.id, token


@pytest.fixture
def world(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> _World:
    """Two tenants and the identities that act in them."""
    # One master-key call provisions the tenancy root, so the organizations built
    # below sit beside a real default rather than replacing it.
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK

    session = db_session_factory()
    try:
        alpha = Organization(name="Alpha", slug="alpha")
        beta = Organization(name="Beta", slug="beta")
        session.add_all([alpha, beta])
        session.commit()
        session.refresh(alpha)
        session.refresh(beta)

        # Created first, so with no workspace named "Default" this is what
        # ``organization_default_workspace_id`` resolves for alpha, which the
        # omitted-workspace cases lean on.
        alpha_one = Workspace(name="Alpha one", organization_id=alpha.id)
        session.add(alpha_one)
        session.commit()
        alpha_two = Workspace(name="Alpha two", organization_id=alpha.id)
        beta_one = Workspace(name="Beta one", organization_id=beta.id)
        session.add_all([alpha_two, beta_one])
        session.commit()
        for workspace in (alpha_one, alpha_two, beta_one):
            session.refresh(workspace)

        built = _World(alpha=alpha.id, beta=beta.id)
        built.workspaces = {
            "alpha_one": alpha_one.id,
            "alpha_two": alpha_two.id,
            "beta_one": beta_one.id,
        }
        people = {
            # Belongs to one of alpha's two workspaces, which is the case this
            # surface exists for.
            "member": _identity(
                session,
                email="member@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            # A second member of the same workspace, so "may see the workspace"
            # and "owns the row" can be told apart.
            "colleague": _identity(
                session,
                email="colleague@alpha.test",
                organization_id=alpha.id,
                workspace_ids=(alpha_one.id,),
            ),
            # In the organization, in none of its workspaces.
            "newcomer": _identity(session, email="new@alpha.test", organization_id=alpha.id),
            "outsider": _identity(
                session,
                email="owner@beta.test",
                organization_id=beta.id,
                role="owner",
                workspace_ids=(beta_one.id,),
            ),
        }
        built.users = {name: user_id for name, (user_id, _) in people.items()}
        built.sessions = {name: token for name, (_, token) in people.items()}
        return built
    finally:
        session.close()


def _request(
    client: TestClient,
    world: _World,
    who: str,
    method: str,
    path: str,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    client.cookies.set(SESSION_COOKIE_NAME, world.sessions[who])
    try:
        response = client.request(method, path, json=json, params=params)
        is_json = response.headers.get("content-type", "").startswith("application/json")
        body = response.json() if is_json and response.content else None
        return response.status_code, body
    finally:
        client.cookies.clear()


def _ws(world: _World, name: str = "alpha_one") -> dict[str, Any]:
    """The query string that scopes a read to one of the fixture's workspaces."""
    return {"workspace_id": str(world.workspaces[name])}


def _grant(client: TestClient, world: _World, who: str, **flags: bool) -> None:
    code, _ = _request(client, world, who, "PUT", f"{_PREFIX}/consent", json=flags)
    assert code == status.HTTP_200_OK


def _conversation_body(world: _World, *, workspace: str = "alpha_one") -> dict[str, Any]:
    return {
        "workspace_id": str(world.workspaces[workspace]),
        "model": MODEL_NAME,
        "title": "How does OAuth work",
        "messages": [
            {"role": "user", "content": "How does OAuth 2.0 work?"},
            {"role": "assistant", "content": "It delegates authorization.", "reasoning": "thinking"},
        ],
    }


def _comparison_body(world: _World, *, workspace: str = "alpha_one") -> dict[str, Any]:
    return {
        "workspace_id": str(world.workspaces[workspace]),
        "user_question": "Which is better?",
        "model_a": "openai:gpt-4o",
        "model_b": MODEL_NAME,
        "model_a_answer": "The first answer.",
        "model_b_answer": "The second answer.",
        "preference": "model_a",
    }


# =============================================================================
# Consent: the server enforces it, and it is per identity
# =============================================================================


def test_consent_defaults_to_nothing_agreed_and_stores_no_row(client: TestClient, world: _World) -> None:
    """A caller who has never answered reads both flags false, and nothing is written."""
    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/consent")

    assert code == status.HTTP_200_OK
    assert body == {"store_conversations": False, "store_comparisons": False}


def test_saving_without_consent_is_refused(client: TestClient, world: _World) -> None:
    """The gate is the server's, not the page's: a client that skips the prompt is refused."""
    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world))
    assert code == status.HTTP_403_FORBIDDEN

    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/comparisons", json=_comparison_body(world))
    assert code == status.HTTP_403_FORBIDDEN


def test_granting_one_flag_leaves_the_other_alone(client: TestClient, world: _World) -> None:
    """A partial update is partial: the page grants one flag at a time, just in time."""
    _grant(client, world, "member", store_conversations=True)

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/consent")
    assert code == status.HTTP_200_OK
    assert body == {"store_conversations": True, "store_comparisons": False}

    # And the save that flag covers now works, while the other stays refused.
    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world))
    assert code == status.HTTP_201_CREATED
    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/comparisons", json=_comparison_body(world))
    assert code == status.HTTP_403_FORBIDDEN


def test_consent_is_not_shared_between_identities(client: TestClient, world: _World) -> None:
    """One person's grant is not another's, even inside one workspace."""
    _grant(client, world, "member", store_conversations=True, store_comparisons=True)

    code, body = _request(client, world, "colleague", "GET", f"{_PREFIX}/consent")
    assert code == status.HTTP_200_OK
    assert body == {"store_conversations": False, "store_comparisons": False}


def test_withdrawing_consent_keeps_what_was_already_saved(client: TestClient, world: _World) -> None:
    """Withdrawal blocks new saves and deletes nothing: deleting is the owner's to do."""
    _grant(client, world, "member", store_conversations=True)
    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world))
    assert code == status.HTTP_201_CREATED

    _grant(client, world, "member", store_conversations=False)

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/conversations", params=_ws(world))
    assert code == status.HTTP_200_OK
    assert len(body["data"]) == 1

    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world))
    assert code == status.HTTP_403_FORBIDDEN


# =============================================================================
# Saved conversations
# =============================================================================


def test_a_saved_transcript_reads_back_in_the_order_it_was_saved(client: TestClient, world: _World) -> None:
    """``position`` is assigned server-side, so the transcript cannot reorder itself."""
    _grant(client, world, "member", store_conversations=True)
    body = _conversation_body(world)
    body["messages"] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
    ]
    code, created = _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=body)
    assert code == status.HTTP_201_CREATED
    assert created["message_count"] == 4

    code, messages = _request(client, world, "member", "GET", f"{_PREFIX}/conversations/{created['id']}/messages")
    assert code == status.HTTP_200_OK
    assert [m["content"] for m in messages["data"]] == ["first", "second", "third", "fourth"]


def test_the_list_carries_a_turn_count_and_not_the_transcript(client: TestClient, world: _World) -> None:
    """The history list renders a count; sending the bodies would move a transcript per row."""
    _grant(client, world, "member", store_conversations=True)
    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world))
    assert code == status.HTTP_201_CREATED

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/conversations", params=_ws(world))
    assert code == status.HTTP_200_OK
    [row] = body["data"]
    assert row["message_count"] == 2
    assert row["title"] == "How does OAuth work"
    assert "messages" not in row


def test_the_turn_count_is_this_conversation_s_own(client: TestClient, world: _World) -> None:
    """The count is a correlated per-conversation read, not a table-wide aggregate.

    Written with a second identity's transcripts in the table because that is
    the shape a grouped-and-joined count gets wrong if its predicate slips: the
    aggregate would be computed over everybody's turns before the join narrowed
    it, and a mistake there reads as a plausible number rather than as an error.
    """
    _grant(client, world, "member", store_conversations=True)
    _grant(client, world, "colleague", store_conversations=True)

    long_body = _conversation_body(world)
    long_body["messages"] = [{"role": "user", "content": f"turn {i}"} for i in range(7)]
    assert _request(client, world, "colleague", "POST", f"{_PREFIX}/conversations", json=long_body)[0] == (
        status.HTTP_201_CREATED
    )
    assert _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world))[0] == (
        status.HTTP_201_CREATED
    )

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/conversations", params=_ws(world))
    assert code == status.HTTP_200_OK
    assert [row["message_count"] for row in body["data"]] == [2]


def test_a_saved_turn_carries_no_usage_figures(client: TestClient, world: _World) -> None:
    """Tokens, cost and timing describe the request, not the conversation.

    So the save accepts none and the read returns none, and there is no column
    holding them: a resumed transcript reporting an old request's latency as
    this session's would be lying, and the billing record for that request is
    its ``usage_logs`` row.
    """
    _grant(client, world, "member", store_conversations=True)
    code, created = _request(
        client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world)
    )
    assert code == status.HTTP_201_CREATED

    code, messages = _request(client, world, "member", "GET", f"{_PREFIX}/conversations/{created['id']}/messages")
    assert code == status.HTTP_200_OK
    assert set(messages["data"][0]) == {"role", "content", "reasoning"}


def test_reasoning_survives_a_save(client: TestClient, world: _World) -> None:
    """A resumed transcript that lost its thinking block reads as a different answer."""
    _grant(client, world, "member", store_conversations=True)
    code, created = _request(
        client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world)
    )
    assert code == status.HTTP_201_CREATED

    code, messages = _request(client, world, "member", "GET", f"{_PREFIX}/conversations/{created['id']}/messages")
    assert code == status.HTTP_200_OK
    assert messages["data"][1]["reasoning"] == "thinking"


def test_another_identity_cannot_read_or_delete_a_transcript(client: TestClient, world: _World) -> None:
    """404 rather than 403: a 403 would confirm the row exists (otari-ai#978)."""
    _grant(client, world, "member", store_conversations=True)
    code, created = _request(
        client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world)
    )
    assert code == status.HTTP_201_CREATED

    for who in ("colleague", "outsider"):
        code, _ = _request(client, world, who, "GET", f"{_PREFIX}/conversations/{created['id']}/messages")
        assert code == status.HTTP_404_NOT_FOUND
        code, _ = _request(client, world, who, "DELETE", f"{_PREFIX}/conversations/{created['id']}")
        assert code == status.HTTP_404_NOT_FOUND

    # And it is still there for its owner, so neither call half-deleted it.
    code, messages = _request(client, world, "member", "GET", f"{_PREFIX}/conversations/{created['id']}/messages")
    assert code == status.HTTP_200_OK
    assert len(messages["data"]) == 2


def test_a_colleague_in_the_same_workspace_sees_none_of_it(client: TestClient, world: _World) -> None:
    """Workspace membership is not readership: the Playground is per person."""
    _grant(client, world, "member", store_conversations=True)
    _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world))

    code, body = _request(client, world, "colleague", "GET", f"{_PREFIX}/conversations", params=_ws(world))
    assert code == status.HTTP_200_OK
    assert body["data"] == []


def test_deleting_a_transcript_takes_its_turns(client: TestClient, world: _World) -> None:
    """The turns ride the foreign key's CASCADE, so a half-deleted transcript is unreachable."""
    _grant(client, world, "member", store_conversations=True)
    code, created = _request(
        client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world)
    )
    assert code == status.HTTP_201_CREATED

    code, _ = _request(client, world, "member", "DELETE", f"{_PREFIX}/conversations/{created['id']}")
    assert code == status.HTTP_204_NO_CONTENT

    code, _ = _request(client, world, "member", "GET", f"{_PREFIX}/conversations/{created['id']}/messages")
    assert code == status.HTTP_404_NOT_FOUND
    code, _ = _request(client, world, "member", "DELETE", f"{_PREFIX}/conversations/{created['id']}")
    assert code == status.HTTP_404_NOT_FOUND


def test_the_save_body_cannot_move_the_row_to_another_workspace(client: TestClient, world: _World) -> None:
    """``workspace_id`` in the body is resolved through the membership check, not trusted."""
    _grant(client, world, "member", store_conversations=True)

    code, _ = _request(
        client,
        world,
        "member",
        "POST",
        f"{_PREFIX}/conversations",
        json=_conversation_body(world, workspace="beta_one"),
    )
    assert code == status.HTTP_404_NOT_FOUND

    # Alpha two is in the caller's own organization and not one of their
    # workspaces, which must answer the same way.
    code, _ = _request(
        client,
        world,
        "member",
        "POST",
        f"{_PREFIX}/conversations",
        json=_conversation_body(world, workspace="alpha_two"),
    )
    assert code == status.HTTP_404_NOT_FOUND


def test_an_oversized_transcript_is_refused_by_the_schema(client: TestClient, world: _World) -> None:
    """A ceiling is a 422 naming the field, not a database error at write time."""
    _grant(client, world, "member", store_conversations=True)
    body = _conversation_body(world)
    body["messages"] = [{"role": "user", "content": "x"} for _ in range(401)]

    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/conversations", json=body)
    assert code == status.HTTP_422_UNPROCESSABLE_CONTENT


# =============================================================================
# Saved comparisons
# =============================================================================


def test_a_comparison_lists_without_its_answer_bodies(client: TestClient, world: _World) -> None:
    """The list shows a dozen rows and renders no answer, so it does not send them."""
    _grant(client, world, "member", store_comparisons=True)
    code, created = _request(client, world, "member", "POST", f"{_PREFIX}/comparisons", json=_comparison_body(world))
    assert code == status.HTTP_201_CREATED
    assert "model_a_answer" not in created

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/comparisons", params=_ws(world))
    assert code == status.HTTP_200_OK
    [row] = body["data"]
    assert row["model_a"] == "openai:gpt-4o"
    assert row["preference"] == "model_a"
    assert "model_a_answer" not in row
    assert "model_b_answer" not in row


def test_another_identity_cannot_list_or_delete_a_comparison(client: TestClient, world: _World) -> None:
    """The bug otari-ai#978 was: a comparisons list trusting a client-supplied scope."""
    _grant(client, world, "member", store_comparisons=True)
    code, created = _request(client, world, "member", "POST", f"{_PREFIX}/comparisons", json=_comparison_body(world))
    assert code == status.HTTP_201_CREATED

    # A colleague shares the workspace, so their list is a real read of it and
    # comes back empty; an outsider cannot reach the workspace at all.
    code, body = _request(client, world, "colleague", "GET", f"{_PREFIX}/comparisons", params=_ws(world))
    assert code == status.HTTP_200_OK
    assert body["data"] == []
    code, _ = _request(client, world, "outsider", "GET", f"{_PREFIX}/comparisons", params=_ws(world))
    assert code == status.HTTP_404_NOT_FOUND

    for who in ("colleague", "outsider"):
        code, _ = _request(client, world, who, "DELETE", f"{_PREFIX}/comparisons/{created['id']}")
        assert code == status.HTTP_404_NOT_FOUND


def test_an_unrated_preference_value_is_refused(client: TestClient, world: _World) -> None:
    """The preference vocabulary is closed: the table is a dataset, not free text."""
    _grant(client, world, "member", store_comparisons=True)
    body = _comparison_body(world)
    body["preference"] = "whichever"

    code, _ = _request(client, world, "member", "POST", f"{_PREFIX}/comparisons", json=body)
    assert code == status.HTTP_422_UNPROCESSABLE_CONTENT


# =============================================================================
# Pinned models
# =============================================================================


def test_pins_round_trip_in_the_order_they_were_sent(client: TestClient, world: _World) -> None:
    """Order is part of the value: a newly pinned model leads the Favorites group."""
    keys = ["openai:gpt-4o", MODEL_NAME, "anthropic:claude-sonnet-4"]
    code, body = _request(
        client,
        world,
        "member",
        "PUT",
        f"{_PREFIX}/favorite-models",
        json={"model_keys": keys},
        params=_ws(world),
    )
    assert code == status.HTTP_200_OK
    assert body["model_keys"] == keys

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/favorite-models", params=_ws(world))
    assert code == status.HTTP_200_OK
    assert body["model_keys"] == keys


def test_a_reorder_that_adds_nothing_still_changes_the_order(client: TestClient, world: _World) -> None:
    """Which is why ``position`` is stored rather than the list being read by insertion clock."""
    params = _ws(world)
    _request(
        client, world, "member", "PUT", f"{_PREFIX}/favorite-models", json={"model_keys": ["a", "b"]}, params=params
    )
    code, body = _request(
        client, world, "member", "PUT", f"{_PREFIX}/favorite-models", json={"model_keys": ["b", "a"]}, params=params
    )
    assert code == status.HTTP_200_OK
    assert body["model_keys"] == ["b", "a"]

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/favorite-models", params=params)
    assert body["model_keys"] == ["b", "a"]


def test_duplicates_collapse_to_first_occurrence(client: TestClient, world: _World) -> None:
    """Without this the unique constraint refuses a list a client could plausibly send."""
    code, body = _request(
        client,
        world,
        "member",
        "PUT",
        f"{_PREFIX}/favorite-models",
        json={"model_keys": ["a", "b", "a"]},
        params=_ws(world),
    )
    assert code == status.HTTP_200_OK
    assert body["model_keys"] == ["a", "b"]


def test_pins_are_per_workspace_and_per_identity(client: TestClient, world: _World) -> None:
    """A pin belongs to one person looking at one workspace's catalog."""
    _request(
        client,
        world,
        "member",
        "PUT",
        f"{_PREFIX}/favorite-models",
        json={"model_keys": ["a"]},
        params=_ws(world),
    )

    code, body = _request(client, world, "colleague", "GET", f"{_PREFIX}/favorite-models", params=_ws(world))
    assert code == status.HTTP_200_OK
    assert body["model_keys"] == []


def test_too_many_pins_are_refused(client: TestClient, world: _World) -> None:
    code, _ = _request(
        client,
        world,
        "member",
        "PUT",
        f"{_PREFIX}/favorite-models",
        json={"model_keys": [f"m{i}" for i in range(51)]},
        params=_ws(world),
    )
    assert code == status.HTTP_422_UNPROCESSABLE_CONTENT


# =============================================================================
# Which workspace a caller reaches
# =============================================================================


def test_omitting_the_workspace_resolves_the_caller_s_organization_default(client: TestClient, world: _World) -> None:
    """The default is a convenience, and it is still put through the membership check."""
    _grant(client, world, "member", store_conversations=True)
    code, created = _request(
        client, world, "member", "POST", f"{_PREFIX}/conversations", json=_conversation_body(world)
    )
    assert code == status.HTTP_201_CREATED

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/conversations")
    assert code == status.HTTP_200_OK
    assert [row["id"] for row in body["data"]] == [created["id"]]


def test_a_member_of_no_workspace_is_told_so_rather_than_pointed_at_a_parameter(
    client: TestClient, world: _World
) -> None:
    """They named nothing, so 404 on an id they never sent would be no help."""
    code, body = _request(client, world, "newcomer", "GET", f"{_PREFIX}/conversations")
    assert code == status.HTTP_409_CONFLICT
    assert "not a member of a workspace" in body["detail"]


def test_naming_another_organization_s_workspace_answers_404(client: TestClient, world: _World) -> None:
    """Indistinguishable from a workspace that was never created, for every route."""
    beta = _ws(world, "beta_one")
    for path in ("/conversations", "/comparisons", "/favorite-models", "/tools"):
        code, _ = _request(client, world, "member", "GET", f"{_PREFIX}{path}", params=beta)
        assert code == status.HTTP_404_NOT_FOUND, path


def test_an_unknown_workspace_answers_404(client: TestClient, world: _World) -> None:
    code, _ = _request(
        client, world, "member", "GET", f"{_PREFIX}/conversations", params={"workspace_id": str(uuid.uuid4())}
    )
    assert code == status.HTTP_404_NOT_FOUND


# =============================================================================
# Which tools the menu may offer
# =============================================================================


def test_an_unconfigured_tool_reports_unconfigured_rather_than_disabled(client: TestClient, world: _World) -> None:
    """Three states, because a menu that offers what the request path refuses is otari-ai#1419."""
    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/tools", params=_ws(world))

    assert code == status.HTTP_200_OK
    assert body["web_search"]["configured"] is False
    assert body["web_search"]["enabled"] is False
    assert body["web_search"]["reason"] is not None
    assert body["mcp_servers"] == []


def test_a_workspace_can_only_narrow_the_deployment_s_answer(
    client: TestClient,
    world: _World,
    db_session_factory: Callable[[], Session],
) -> None:
    """A workspace row saying ``enabled`` cannot turn on a tool the deployment lacks."""
    session = db_session_factory()
    try:
        session.add(WorkspaceWebSearchConfig(workspace_id=world.workspaces["alpha_one"], enabled=True))
        session.commit()
    finally:
        session.close()

    code, body = _request(client, world, "member", "GET", f"{_PREFIX}/tools", params=_ws(world))
    assert code == status.HTTP_200_OK
    assert body["web_search"] == {
        "configured": False,
        "enabled": False,
        "reason": "No backend is configured on this deployment.",
    }


# =============================================================================
# The completion: billed to the caller, in their workspace, with no key
# =============================================================================


def _mock_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-playground",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


async def _mock_acompletion(**_kwargs: Any) -> ChatCompletion:
    return _mock_completion()


def _chat(
    client: TestClient,
    world: _World,
    who: str,
    *,
    workspace: str | None = "alpha_one",
) -> tuple[int, Any]:
    params = _ws(world, workspace) if workspace else None
    with patch("gateway.api.routes.chat.acompletion") as mock:
        mock.side_effect = _mock_acompletion
        return _request(
            client,
            world,
            who,
            "POST",
            f"{_PREFIX}/chat/completions",
            json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
            params=params,
        )


@pytest.fixture
def _priced(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=master_key_header,
    )
    assert response.status_code == status.HTTP_200_OK, response.text


def test_a_completion_is_billed_to_the_caller_with_no_api_key(
    client: TestClient,
    world: _World,
    db_session: Session,
    _priced: None,
) -> None:
    """The whole point of the endpoint, as the usage row records it.

    ``user_id`` is the caller's own attribution row and ``api_key_id`` is null,
    because there was no key: the session is what authorized this, and the row
    says so rather than borrowing somebody's credential to look ordinary. The
    endpoint label is the Playground's own, which is what keeps in-product
    traffic countable apart from a customer's.
    """
    code, body = _chat(client, world, "member")
    assert code == status.HTTP_200_OK
    assert body["choices"][0]["message"]["content"] == "hi"

    row = db_session.query(UsageLog).one()
    assert row.endpoint == PLAYGROUND_USAGE_ENDPOINT
    assert row.user_id == str(world.users["member"])
    assert row.api_key_id is None
    assert row.workspace_id == world.workspaces["alpha_one"]
    # Still served-here traffic, so the operator's imported-usage mutations
    # cannot touch it: the surface it came from is a different question from
    # who served it (``core/usage_source``).
    assert row.source == SERVED_HERE_SLUG


def test_a_completion_cannot_be_billed_to_a_workspace_that_is_not_the_caller_s(
    client: TestClient,
    world: _World,
    db_session: Session,
    _priced: None,
) -> None:
    """otari-ai#1880 in one assertion: naming a workspace is not reaching it.

    Both refusals happen before the pipeline is entered, so nothing is reserved
    and no row is written.
    """
    code, _ = _chat(client, world, "member", workspace="beta_one")
    assert code == status.HTTP_404_NOT_FOUND
    code, _ = _chat(client, world, "member", workspace="alpha_two")
    assert code == status.HTTP_404_NOT_FOUND

    assert db_session.query(UsageLog).count() == 0


def test_a_completion_streams_server_sent_events(
    client: TestClient,
    world: _World,
    _priced: None,
) -> None:
    """The page renders tokens as they arrive, so the endpoint has to stream them.

    Asserted on the wire rather than on a helper: the frames are what the
    browser's reader parses, so the contract is the ``text/event-stream`` media
    type, ``data:``-prefixed JSON chunks, and the ``[DONE]`` sentinel that ends
    them.
    """

    async def _stream(**_kwargs: Any) -> Any:
        async def chunks() -> Any:
            from any_llm.types.completion import ChatCompletionChunk, ChoiceDelta, ChunkChoice

            for piece in ("Hel", "lo"):
                yield ChatCompletionChunk(
                    id="chunk",
                    object="chat.completion.chunk",
                    created=0,
                    model=MODEL_NAME,
                    choices=[ChunkChoice(index=0, delta=ChoiceDelta(content=piece))],
                )

        return chunks()

    client.cookies.set(SESSION_COOKIE_NAME, world.sessions["member"])
    try:
        with patch("gateway.api.routes.chat.acompletion") as mock:
            mock.side_effect = _stream
            with client.stream(
                "POST",
                f"{_PREFIX}/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                params=_ws(world),
            ) as response:
                assert response.status_code == status.HTTP_200_OK
                assert response.headers["content-type"].startswith("text/event-stream")
                payload = "".join(response.iter_text())
    finally:
        client.cookies.clear()

    assert "data: " in payload
    assert "Hel" in payload
    assert payload.rstrip().endswith("data: [DONE]")


def test_the_public_completion_route_still_refuses_a_cookie(client: TestClient, world: _World) -> None:
    """The control group. A change that made the Playground work by loosening
    ``/chat/completions`` would have reopened otari-ai#1880, and this is what
    would have caught it."""
    code, _ = _request(
        client,
        world,
        "member",
        "POST",
        f"{API_ROOT}/chat/completions",
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert code == status.HTTP_401_UNAUTHORIZED


def test_a_playground_request_does_not_close_the_activation_guide(
    client: TestClient,
    world: _World,
    _priced: None,
) -> None:
    """The guide marks somebody integrating Otari, which a click in our own page is not.

    Its whole message is "your first request arrived", offered once and retired
    on the first successful row. A Playground message is the product being
    demonstrated rather than integrated, so closing on one would congratulate
    somebody for work they have not done and then never offer the guide again.
    """
    workspace_id = str(world.workspaces["alpha_one"])
    path = f"{API_ROOT}/workspaces/{workspace_id}/activation"

    code, before = _request(client, world, "member", "GET", path)
    assert code == status.HTTP_200_OK
    assert before["status"] == "waiting"

    code, _ = _chat(client, world, "member")
    assert code == status.HTTP_200_OK

    code, after = _request(client, world, "member", "GET", path)
    assert code == status.HTTP_200_OK
    assert after["status"] == "waiting"
    assert after["activation_attempt"] is None
    # Nor the "your last attempt" line, which would otherwise report the
    # product talking to itself as the caller's most recent try.
    assert after["latest_attempt"] is None


def test_the_first_completion_provisions_the_caller_s_spend_row_once(
    client: TestClient,
    world: _World,
    db_session: Session,
    _priced: None,
) -> None:
    """The row spend binds to is minted on first use, and committed before dispatch.

    Both halves matter and neither is obvious from the handler. The Playground
    is the first thing a new member touches, so they usually have no attribution
    row yet, and the budget gate updates that row by primary key rather than
    creating one. And the usage row is written on a *different* session, so if
    the insert were still uncommitted when the completion returned, the usage
    row's foreign key would have nothing to point at. The staged row rides
    ``release_session``'s commit before the provider call, which is what makes
    both true.
    """
    assert db_session.query(BillingUser).filter(BillingUser.user_id == str(world.users["member"])).count() == 0

    assert _chat(client, world, "member")[0] == status.HTTP_200_OK
    assert _chat(client, world, "member")[0] == status.HTTP_200_OK

    # One row, not two: it is keyed on the identity, so the second request
    # finds the first one rather than minting a second spend identity.
    owners = db_session.query(BillingUser).filter(BillingUser.user_id == str(world.users["member"])).all()
    assert len(owners) == 1
    assert owners[0].alias == "Member"


def test_a_completion_honors_the_caller_s_own_model_allow_list(
    client: TestClient,
    world: _World,
    db_session: Session,
    _priced: None,
) -> None:
    """A session holds no key to narrow with, so the user default is the effective list.

    The gate that binds here is the pipeline's own, reached through
    ``SessionPrincipal.allowed_models``: without it a session-authorized request
    would be unrestricted, which is the master-key semantics and exactly wrong
    for a member. Refused before dispatch, so nothing is reserved and no row is
    written.
    """
    # Provision the row by making one allowed request, then restrict it.
    assert _chat(client, world, "member")[0] == status.HTTP_200_OK
    owner = db_session.query(BillingUser).filter(BillingUser.user_id == str(world.users["member"])).one()
    owner.allowed_models = ["gemini:some-other-model"]
    db_session.commit()

    code, body = _chat(client, world, "member")

    assert code == status.HTTP_403_FORBIDDEN
    assert MODEL_NAME in body["detail"]
