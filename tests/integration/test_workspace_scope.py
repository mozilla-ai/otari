"""The request plane is scoped to a workspace.

A key belongs to exactly one workspace, and that is where its requests are
recorded. The workspace is read off the key rather than off a request header,
because a caller controls its headers and not which key it holds.
"""

import uuid
from collections.abc import Callable
from typing import Any

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import APIKey, UsageLog


def _default_workspace(client: TestClient, headers: dict[str, str]) -> str:
    context = client.get("/v1/organizations/me", headers=headers).json()
    return str(context["workspace_memberships"][0]["workspace_id"])


def _make_workspace(client: TestClient, headers: dict[str, str], name: str) -> str:
    created = client.post("/v1/workspaces", json={"name": name}, headers=headers)
    assert created.status_code == status.HTTP_201_CREATED, created.text
    return str(created.json()["id"])


def _create_key(client: TestClient, headers: dict[str, str], **body: Any) -> dict[str, Any]:
    response = client.post("/v1/keys", json={"key_name": "k", **body}, headers=headers)
    assert response.status_code == status.HTTP_200_OK, response.text
    payload: dict[str, Any] = response.json()
    return payload


def test_a_key_created_without_a_workspace_lands_in_the_default_one(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    default = _default_workspace(client, master_key_header)

    created = _create_key(client, master_key_header)

    listed = client.get(f"/v1/keys/{created['id']}", headers=master_key_header).json()
    assert listed["workspace_id"] == default


def test_a_key_can_be_created_in_a_named_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    platform = _make_workspace(client, master_key_header, "Platform team")

    created = _create_key(client, master_key_header, workspace_id=platform)

    listed = client.get(f"/v1/keys/{created['id']}", headers=master_key_header).json()
    assert listed["workspace_id"] == platform


def test_the_key_list_filters_by_workspace_and_is_deployment_wide_without_one(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    default = _default_workspace(client, master_key_header)
    platform = _make_workspace(client, master_key_header, "Platform team")
    _create_key(client, master_key_header, key_name="in-default")
    _create_key(client, master_key_header, workspace_id=platform, key_name="in-platform")

    scoped = client.get(f"/v1/keys?workspace_id={platform}", headers=master_key_header).json()
    assert [k["key_name"] for k in scoped] == ["in-platform"]

    in_default = client.get(f"/v1/keys?workspace_id={default}", headers=master_key_header).json()
    assert "in-platform" not in [k["key_name"] for k in in_default]

    # Unset means every key on the deployment, which is what keeps the
    # pre-workspace view working.
    everything = client.get("/v1/keys", headers=master_key_header).json()
    names = [k["key_name"] for k in everything]
    assert "in-default" in names
    assert "in-platform" in names


def test_usage_is_recorded_in_the_workspace_of_the_key_that_authenticated_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The workspace comes off the key, not off anything the caller can send."""
    platform = _make_workspace(client, master_key_header, "Platform team")
    created = _create_key(client, master_key_header, workspace_id=platform, exclude_from_budget=True)

    recorded = client.post(
        "/v1/usage/external-events",
        json={
            "source": "claude_code",
            "events": [
                {
                    "source_event_id": "e1",
                    "model": "claude-3-5-sonnet",
                    "provider": "anthropic",
                    "timestamp": "2026-08-18T00:00:00Z",
                    "input_tokens": 10,
                    "output_tokens": 5,
                }
            ],
        },
        headers={"Otari-Key": created["key"]},
    )
    assert recorded.status_code in (200, 201), recorded.text

    session = db_session_factory()
    try:
        rows = session.query(UsageLog).filter(UsageLog.source == "claude_code").all()
        assert rows, "the event should have been recorded"
        assert {str(row.workspace_id) for row in rows} == {platform}
    finally:
        session.close()


def test_the_usage_list_filters_by_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    default = _default_workspace(client, master_key_header)
    platform = _make_workspace(client, master_key_header, "Platform team")

    session = db_session_factory()
    try:
        key = session.query(APIKey).first()
        assert key is not None
        for workspace_id, model in ((default, "here"), (platform, "there")):
            session.add(
                UsageLog(
                    id=str(uuid.uuid4()),
                    workspace_id=uuid.UUID(workspace_id),
                    api_key_id=key.id,
                    user_id=key.user_id,
                    model=model,
                    provider="p",
                    endpoint="/v1/chat/completions",
                    source="gateway",
                    status="success",
                )
            )
        session.commit()
    finally:
        session.close()

    scoped = client.get(f"/v1/usage?workspace_id={platform}", headers=master_key_header).json()
    assert [row["model"] for row in scoped] == ["there"]

    everything = client.get("/v1/usage", headers=master_key_header).json()
    assert sorted(row["model"] for row in everything) == ["here", "there"]


def test_a_workspace_holding_request_plane_rows_cannot_be_deleted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The four foreign keys are ON DELETE RESTRICT, so the database refuses.

    Without a guard that refusal escaped as a 500: a workspace is a billing
    scope, and deleting one must not take the record of what was spent in it,
    so this is a real conflict with a real reason rather than a server error.
    """
    workspace = client.post(
        "/v1/workspaces",
        json={"name": "Holds a key"},
        headers=master_key_header,
    ).json()
    client.post(
        "/v1/keys",
        json={"user_id": "scoped-owner", "workspace_id": workspace["id"]},
        headers=master_key_header,
    )

    response = client.delete(f"/v1/workspaces/{workspace['id']}", headers=master_key_header)

    assert response.status_code == 409, response.text
    assert "API keys" in response.json()["detail"]


def test_a_key_cannot_be_created_in_a_workspace_that_does_not_exist(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The id comes from the caller, so an unknown one is a bad request.

    Left to the foreign key it answered 500 "Database error" for a value the
    caller supplied and could fix.
    """
    response = client.post(
        "/v1/keys",
        json={
            "user_id": "no-such-workspace",
            "workspace_id": "00000000-0000-0000-0000-000000000000",
        },
        headers=master_key_header,
    )

    assert response.status_code == 404, response.text



def test_deleting_a_workspace_takes_its_ceilings_with_it(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """``scoped_budgets.scope_id`` is not a foreign key, so nothing cascades it.

    A ceiling left behind is the state ``_require_scope_exists`` refuses to
    create, just reached from the other end: it lists, it never binds, and
    nothing surfaces that it stopped mattering.
    """
    workspace = _make_workspace(client, master_key_header, "Departing")
    created = client.post(
        "/v1/scoped-budgets",
        json={"scope_type": "workspace", "scope_id": workspace, "max_budget": 5.0},
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_200_OK, created.text
    budget_id = created.json()["id"]

    deleted = client.delete(f"/v1/workspaces/{workspace}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_200_OK, deleted.text

    orphan = client.get(f"/v1/scoped-budgets/{budget_id}", headers=master_key_header)
    assert orphan.status_code == status.HTTP_404_NOT_FOUND, orphan.text


def test_a_members_workspace_ceiling_goes_with_the_workspace(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The membership rows ride the database cascade, so their ceilings must too."""
    workspace = _make_workspace(client, master_key_header, "Departing with members")
    # Creating a workspace joins its creator to it, so the roster already has the
    # membership row this ceiling is hung on.
    members = client.get(f"/v1/workspaces/{workspace}/members", headers=master_key_header)
    assert members.status_code == status.HTTP_200_OK, members.text
    membership_id = members.json()["data"][0]["id"]
    created = client.post(
        "/v1/scoped-budgets",
        json={"scope_type": "workspace_member", "scope_id": membership_id, "max_budget": 5.0},
        headers=master_key_header,
    )
    assert created.status_code == status.HTTP_200_OK, created.text
    budget_id = created.json()["id"]

    deleted = client.delete(f"/v1/workspaces/{workspace}", headers=master_key_header)
    assert deleted.status_code == status.HTTP_200_OK, deleted.text

    orphan = client.get(f"/v1/scoped-budgets/{budget_id}", headers=master_key_header)
    assert orphan.status_code == status.HTTP_404_NOT_FOUND, orphan.text
