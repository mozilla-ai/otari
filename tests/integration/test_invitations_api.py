"""Organization-member invitations, end to end: issue, list, accept, revoke.

The API test client can only ever act as the one operator identity a
standalone deployment has (owner and superuser), so what a non-manager may not
do is covered at the service layer instead, alongside the rest of tenancy's
authorization matrix (test_tenancy_authorization.py), whose own docstring
explains the same split.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
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


def test_invite_emails_the_accept_link_when_a_transport_is_configured(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other half of the optional-mail design: configured, the message goes out.

    Uses the console transport rather than a patched smtplib, so the whole path
    an operator's SMTP deployment takes runs for real (readiness, rendering,
    the off-loaded send) and only the socket is replaced. The gateway logger
    does not propagate, hence the explicit handler.
    """
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        result = _invite(client, master_key_header, email="mailed@example.com")
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert result["mail_sent"] is True
    # Absolute, because it has to mean something outside a browser.
    assert result["accept_link"].startswith("https://otari.example.com/#/accept-invitation?token=")
    assert "You're invited to join" in caplog.text
    assert result["accept_link"] in caplog.text
    # The recipient is redacted in the log line even on the success path.
    assert "mailed@example.com" not in caplog.text


def test_invite_is_not_emailed_without_a_public_base_url(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative accept link is useless in an inbox, so a transport alone is not enough."""
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", None)

    result = _invite(client, master_key_header, email="relative@example.com")

    assert result["mail_sent"] is False
    assert result["accept_link"].startswith("/#/accept-invitation?token=")


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
    preview = client.post("/v1/invitations/validate", json={"token": token})

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


def test_accepting_mints_the_attribution_user_so_the_member_can_own_a_key(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """An accepted invitee must be offerable as a key owner, the same as a member added directly.

    ``create_active_organization_member_for_user`` calls
    ``get_or_create_attribution_user`` before it commits; ``accept_invitation``
    did not, which would have left an accepted invitee's roster row with no
    ``attribution_user_id`` and no key of their own possible.
    """
    result = _invite(client, master_key_header, email="nadia@example.com")
    accept = client.post(
        "/v1/invitations/accept", json={"token": _token_from(result["accept_link"])}
    )
    assert accept.status_code == 200, accept.text

    row = _roster_row(client, master_key_header, "nadia@example.com")
    assert row["attribution_user_id"] is not None

    key = client.post(
        "/v1/keys",
        json={"key_name": "nadia's key", "user_id": row["attribution_user_id"]},
        headers=master_key_header,
    )
    assert key.status_code == 200, key.text
    assert key.json()["user_id"] == row["attribution_user_id"]


def test_a_workspace_deleted_after_invite_but_before_accept_still_lets_the_invitee_in(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Acceptance can arrive days later; the parked workspace ids are re-checked, not trusted.

    Refusing the whole accept over one vanished assignment would be worse than
    the bug it would be guarding against: this is a public, unauthenticated
    endpoint the recipient has no way to retry differently, so the invitation
    would be stuck `pending` forever, and the 404's body would name a
    workspace id to a caller who has only ever held a token. Dropping the
    vanished assignment and applying the rest lets the invitee become an
    active member missing just that one grant, which an operator can restore
    from the workspace roster once they notice.
    """
    created = client.post(
        "/v1/workspaces", json={"name": "Temporary"}, headers=master_key_header
    ).json()
    kept = client.post(
        "/v1/workspaces", json={"name": "Kept"}, headers=master_key_header
    ).json()

    result = _invite(
        client,
        master_key_header,
        email="karen@example.com",
        workspace_assignments=[
            {"workspace_id": created["id"], "role": "viewer"},
            {"workspace_id": kept["id"], "role": "viewer"},
        ],
    )
    token = _token_from(result["accept_link"])

    delete = client.delete(f"/v1/workspaces/{created['id']}", headers=master_key_header)
    assert delete.status_code == 200, delete.text

    accept = client.post("/v1/invitations/accept", json={"token": token})
    assert accept.status_code == 200, accept.text

    row = _roster_row(client, master_key_header, "karen@example.com")
    assert row["status"] == "active"

    members = client.get(f"/v1/workspaces/{kept['id']}/members", headers=master_key_header).json()
    assert any(member["user_id"] == row["user_id"] for member in members["data"])


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


def test_patching_an_invited_member_straight_to_active_also_cancels_the_invitation(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A different escape from ``invited`` than suspension, with the same stale-token risk.

    ``PATCH`` can activate an invited member directly, bypassing
    ``accept_invitation`` entirely. If the pending invitation survived that,
    its token would still resolve; removing the member later (any path) would
    then leave a token that could silently reactivate them.
    """
    result = _invite(client, master_key_header, email="leo@example.com")
    token = _token_from(result["accept_link"])

    patch = client.patch(
        f"/v1/organizations/me/members/{result['organization_member_id']}",
        json={"status": "active"},
        headers=master_key_header,
    )
    assert patch.status_code == 200, patch.text

    accept = client.post("/v1/invitations/accept", json={"token": token})
    assert accept.status_code == 400, accept.text


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
    """The message has to say "pending invitation", not "active member": it isn't one yet."""
    _invite(client, master_key_header, email="hank@example.com")

    conflict = _invite_raw(client, master_key_header, email="hank@example.com")
    assert conflict.status_code == 409
    assert "pending invitation" in conflict.text
    assert "active member" not in conflict.text


def test_inviting_an_address_who_is_already_an_active_member_conflicts(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    add = client.post(
        "/v1/organizations/me/members",
        json={"email": "iris@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert add.status_code == 201, add.text

    conflict = _invite_raw(client, master_key_header, email="iris@example.com")
    assert conflict.status_code == 409
    assert "active member" in conflict.text


def test_reinviting_after_the_previous_invitation_expired_supersedes_it(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Expiry is lazy: a link nobody ever opened must not dead-end every future invite.

    `_resolve_pending_invitation` only flips a `pending` row to `expired` when
    someone presents its token, so an unopened link's own `expires_at` can be
    long past while its stored status still reads `pending`. Without a
    time-based re-check on the invite path (rather than trusting the stored
    status), re-inviting the same address would answer 409 forever, with only
    a revoke-then-invite as the way through.
    """
    first = _invite(client, master_key_header, email="jill@example.com", role="viewer")
    first_token = _token_from(first["accept_link"])

    invitation = db_session.get(Invitation, uuid.UUID(first["invitation_id"]))
    assert invitation is not None
    invitation.expires_at = datetime.now(UTC) - timedelta(hours=1)
    db_session.add(invitation)
    db_session.commit()

    second = _invite(client, master_key_header, email="jill@example.com", role="admin")
    assert second["organization_member_id"] == first["organization_member_id"]
    assert second["role"] == "admin"

    db_session.refresh(invitation)
    assert invitation.status == "expired"

    # The superseded link is dead, not merely redundant.
    stale_accept = client.post("/v1/invitations/accept", json={"token": first_token})
    assert stale_accept.status_code == 400, stale_accept.text

    accept = client.post("/v1/invitations/accept", json={"token": _token_from(second["accept_link"])})
    assert accept.status_code == 200, accept.text


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
