"""Organization-member invitations, end to end: issue, list, accept, revoke.

The API test client can only ever act as the one operator identity a
standalone deployment has (owner and superuser), so what a non-manager may not
do is covered at the service layer instead (test_invitation_authorization.py),
the same split test_tenancy_authorization.py's own docstring explains.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.models.tenancy import Invitation


def _invite(
    client: TestClient,
    headers: dict[str, str],
    *,
    email: str,
    role: str = "member",
    workspace_assignments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"email": email, "role": role}
    if workspace_assignments is not None:
        body["workspace_assignments"] = workspace_assignments
    response = client.post("/v1/organizations/me/member-invitations", json=body, headers=headers)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def _token_from(accept_link: str) -> str:
    return accept_link.split("token=")[1]


def _roster_row(client: TestClient, headers: dict[str, str], email: str) -> dict[str, Any]:
    members = client.get("/v1/organizations/me/members", headers=headers).json()["data"]
    return next(row for row in members if row["email"] == email)


def test_invite_lands_invited_with_no_mail_transport_configured(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Without SMTP configured, the invitation is still created and usable, just not emailed."""
    result = _invite(client, master_key_header, email="ada@example.com")

    assert result["status"] == "invited"
    assert result["mail_sent"] is False
    assert "token=" in result["accept_link"]

    row = _roster_row(client, master_key_header, "ada@example.com")
    assert row["status"] == "invited"
    assert row["invitation_id"] == result["invitation_id"]


def test_validate_shows_the_organization_and_role_without_authenticating(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    result = _invite(client, master_key_header, email="bob@example.com", role="admin")
    token = _token_from(result["accept_link"])

    # Deliberately no auth header: the token is the whole credential here.
    preview = client.get(f"/v1/invitations/validate/{token}")

    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["email"] == "bob@example.com"
    assert body["role"] == "admin"
    assert body["organization_name"]


def test_accept_activates_the_membership_and_applies_parked_workspace_assignments(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    workspace_id = client.get("/v1/workspaces", headers=master_key_header).json()["data"][0]["id"]

    result = _invite(
        client,
        master_key_header,
        email="carol@example.com",
        workspace_assignments=[{"workspace_id": workspace_id, "role": "viewer"}],
    )
    token = _token_from(result["accept_link"])

    accept = client.post("/v1/invitations/accept", json={"token": token})
    assert accept.status_code == 200, accept.text
    assert accept.json()["role"] == "member"

    row = _roster_row(client, master_key_header, "carol@example.com")
    assert row["status"] == "active"
    assert row["invitation_id"] is None  # nothing left to act on once accepted

    workspace_members = client.get(f"/v1/workspaces/{workspace_id}/members", headers=master_key_header).json()
    carol = next(m for m in workspace_members["data"] if m["user_id"] == row["user_id"])
    assert carol["role"] == "viewer"
    assert carol["status"] == "active"


def test_accepting_twice_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    result = _invite(client, master_key_header, email="dave@example.com")
    token = _token_from(result["accept_link"])

    first = client.post("/v1/invitations/accept", json={"token": token})
    second = client.post("/v1/invitations/accept", json={"token": token})

    assert first.status_code == 200
    assert second.status_code == 400


def test_accepting_an_unknown_token_is_not_found(client: TestClient) -> None:
    response = client.post("/v1/invitations/accept", json={"token": "not-a-real-token"})
    assert response.status_code == 404


def test_expired_invitation_cannot_be_accepted(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    result = _invite(client, master_key_header, email="erin@example.com")
    token = _token_from(result["accept_link"])

    invitation = db_session.get(Invitation, uuid.UUID(result["invitation_id"]))
    assert invitation is not None
    invitation.expires_at = datetime.now(UTC) - timedelta(hours=1)
    db_session.add(invitation)
    db_session.commit()

    response = client.post("/v1/invitations/accept", json={"token": token})
    assert response.status_code == 400

    db_session.refresh(invitation)
    assert invitation.status == "expired"


def test_revoke_suspends_the_membership_and_the_token_stops_working(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    result = _invite(client, master_key_header, email="frank@example.com")
    token = _token_from(result["accept_link"])

    revoke = client.delete(
        f"/v1/organizations/me/member-invitations/{result['invitation_id']}",
        headers=master_key_header,
    )
    assert revoke.status_code == 200, revoke.text

    members = client.get("/v1/organizations/me/members", headers=master_key_header).json()["data"]
    assert not any(row["email"] == "frank@example.com" for row in members)  # suspended, off the roster

    accept = client.post("/v1/invitations/accept", json={"token": token})
    assert accept.status_code == 400


def test_removing_an_invited_member_directly_also_cancels_the_invitation(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The generic remove path must not leave a still-usable accept link behind.

    `DELETE /me/members/{id}` (not the dedicated revoke endpoint) also
    suspends an `invited` membership, and without also cancelling its
    invitation, accepting it afterwards would silently undo the removal.
    """
    result = _invite(client, master_key_header, email="ivan@example.com")
    token = _token_from(result["accept_link"])

    remove = client.delete(
        f"/v1/organizations/me/members/{result['organization_member_id']}",
        headers=master_key_header,
    )
    assert remove.status_code == 200, remove.text

    accept = client.post("/v1/invitations/accept", json={"token": token})
    assert accept.status_code == 400


def test_patching_an_invited_member_to_suspended_also_cancels_the_invitation(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The same gap, reached through the generic PATCH rather than DELETE."""
    result = _invite(client, master_key_header, email="judy@example.com")
    token = _token_from(result["accept_link"])

    patch = client.patch(
        f"/v1/organizations/me/members/{result['organization_member_id']}",
        json={"status": "suspended"},
        headers=master_key_header,
    )
    assert patch.status_code == 200, patch.text

    accept = client.post("/v1/invitations/accept", json={"token": token})
    assert accept.status_code == 400


def test_reinviting_a_revoked_address_revives_the_membership(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    first = _invite(client, master_key_header, email="grace@example.com", role="viewer")
    client.delete(
        f"/v1/organizations/me/member-invitations/{first['invitation_id']}",
        headers=master_key_header,
    )

    second = _invite(client, master_key_header, email="grace@example.com", role="admin")
    assert second["organization_member_id"] == first["organization_member_id"]
    assert second["role"] == "admin"

    accept = client.post("/v1/invitations/accept", json={"token": _token_from(second["accept_link"])})
    assert accept.status_code == 200


def test_inviting_an_address_with_a_live_membership_conflicts(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _invite(client, master_key_header, email="hank@example.com")

    conflict = _invite_raw(client, master_key_header, email="hank@example.com")
    assert conflict.status_code == 409


def _invite_raw(client: TestClient, headers: dict[str, str], *, email: str) -> Any:
    return client.post(
        "/v1/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers=headers,
    )


def test_revoking_an_unknown_invitation_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.delete(
        f"/v1/organizations/me/member-invitations/{uuid.uuid4()}",
        headers=master_key_header,
    )
    assert response.status_code == 404
