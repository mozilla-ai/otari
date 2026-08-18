"""Tenancy identities own request-plane rows, so a member can hold an API key.

Keys, budgets, and usage hang off the gateway's string-keyed ``users`` table;
members are UUID-keyed ``user`` rows. Nothing joins them, so before this a member
could be added to the roster and then not be given a key. These cover the bridge:
that adding a member mints the owner row, that the id the roster reports is one
``POST /v1/keys`` actually accepts, and that re-adding someone reuses their row
rather than minting a second or resurrecting a clean one.
"""

import uuid
from collections.abc import Callable
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.entities import User as GatewayUser


def _add_member(client: TestClient, headers: dict[str, str], email: str) -> dict[str, Any]:
    response = client.post(
        "/v1/organizations/me/members",
        json={"email": email, "role": "member"},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _roster_row(client: TestClient, headers: dict[str, str], email: str) -> dict[str, Any]:
    roster = client.get("/v1/organizations/me/members", headers=headers).json()
    return next(row for row in roster["data"] if row["email"] == email)


def _gateway_user(session_factory: Callable[[], Session], user_id: str) -> GatewayUser | None:
    session = session_factory()
    try:
        return session.get(GatewayUser, user_id)
    finally:
        session.close()


def test_adding_a_member_mints_the_row_a_key_can_be_billed_to(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    created = _add_member(client, master_key_header, "ada@example.com")

    attribution_user_id = created["attribution_user_id"]
    assert attribution_user_id == str(uuid.UUID(created["user_id"]))

    row = _gateway_user(db_session_factory, attribution_user_id)
    assert row is not None
    assert row.alias == "ada@example.com"
    assert row.deleted_at is None


def test_the_reported_id_is_one_the_keys_endpoint_accepts(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The demo path end to end: add a member, then give her a key."""
    created = _add_member(client, master_key_header, "ada@example.com")

    response = client.post(
        "/v1/keys",
        json={"key_name": "ada-laptop", "user_id": created["attribution_user_id"]},
        headers=master_key_header,
    )

    assert response.status_code == 200, response.text
    assert response.json()["user_id"] == created["attribution_user_id"]


def test_the_roster_carries_the_same_id(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    created = _add_member(client, master_key_header, "ada@example.com")

    assert _roster_row(client, master_key_header, "ada@example.com")["attribution_user_id"] == (
        created["attribution_user_id"]
    )


def test_first_boot_gives_the_operator_one_too(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    roster = client.get("/v1/organizations/me/members", headers=master_key_header).json()
    operator_row = next(row for row in roster["data"] if row["full_name"] == "Operator")

    # The operator identity has no email, so the roster is the only place its id
    # is reported; the point of the assertion is that it is reported at all.
    assert operator_row["attribution_user_id"] == operator_row["user_id"]

    row = _gateway_user(db_session_factory, operator_row["attribution_user_id"])
    assert row is not None
    assert row.alias == "Operator"


def test_re_adding_a_removed_member_reuses_the_same_row(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    created = _add_member(client, master_key_header, "ada@example.com")
    attribution_user_id = created["attribution_user_id"]

    member_id = _roster_row(client, master_key_header, "ada@example.com")["organization_member_id"]
    removed = client.delete(f"/v1/organizations/me/members/{member_id}", headers=master_key_header)
    assert removed.status_code == 200, removed.text

    re_added = _add_member(client, master_key_header, "ada@example.com")

    assert re_added["attribution_user_id"] == attribution_user_id
    session = db_session_factory()
    try:
        rows = session.query(GatewayUser).filter(GatewayUser.alias == "ada@example.com").all()
        assert len(rows) == 1
    finally:
        session.close()


def test_re_adding_revives_a_soft_deleted_row_without_clearing_its_spend(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Reviving restores the owner, not a clean slate.

    Zeroing the counters here would make "remove then re-add" a way to clear
    someone's spend against a budget, so the revive clears ``deleted_at`` and
    leaves ``spend`` alone.
    """
    created = _add_member(client, master_key_header, "ada@example.com")
    attribution_user_id = created["attribution_user_id"]

    session = db_session_factory()
    try:
        row = session.get(GatewayUser, attribution_user_id)
        assert row is not None
        row.spend = 12.5
        session.commit()
    finally:
        session.close()

    deleted = client.delete(f"/v1/users/{attribution_user_id}", headers=master_key_header)
    assert deleted.status_code == 204, deleted.text
    assert _roster_row(client, master_key_header, "ada@example.com")["attribution_user_id"] is None

    member_id = _roster_row(client, master_key_header, "ada@example.com")["organization_member_id"]
    client.delete(f"/v1/organizations/me/members/{member_id}", headers=master_key_header)
    re_added = _add_member(client, master_key_header, "ada@example.com")

    assert re_added["attribution_user_id"] == attribution_user_id
    row = _gateway_user(db_session_factory, attribution_user_id)
    assert row is not None
    assert row.deleted_at is None
    assert row.spend == 12.5

    keyed = client.post(
        "/v1/keys",
        json={"key_name": "ada-again", "user_id": attribution_user_id},
        headers=master_key_header,
    )
    assert keyed.status_code == 200, keyed.text


def test_the_membership_context_carries_the_callers_workspaces(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The shell picks a context and a default workspace from this one call."""
    context = client.get("/v1/organizations/me", headers=master_key_header).json()

    memberships = context["workspace_memberships"]
    assert [m["name"] for m in memberships] == ["Default workspace"]
    assert memberships[0]["role"] == "owner"
    assert memberships[0]["workspace_id"]


def test_a_new_workspace_joins_the_callers_context(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    created = client.post(
        "/v1/workspaces",
        json={"name": "Platform team"},
        headers=master_key_header,
    )
    assert created.status_code == 201, created.text

    context = client.get("/v1/organizations/me", headers=master_key_header).json()

    assert sorted(m["name"] for m in context["workspace_memberships"]) == [
        "Default workspace",
        "Platform team",
    ]


def test_the_context_lists_only_workspaces_the_caller_joined(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """Not a directory: a workspace the caller is not a member of stays out.

    Listing the organization's workspaces is a separate, authorized read; this
    field exists to seed a switcher, so it carries only what the caller belongs
    to.
    """
    from gateway.models.tenancy import Workspace

    context = client.get("/v1/organizations/me", headers=master_key_header).json()
    organization_id = uuid.UUID(context["organization"]["id"])

    session = db_session_factory()
    try:
        session.add(Workspace(name="Someone else's", organization_id=organization_id))
        session.commit()
    finally:
        session.close()

    after = client.get("/v1/organizations/me", headers=master_key_header).json()
    assert [m["name"] for m in after["workspace_memberships"]] == ["Default workspace"]
