"""Organization-member invitations, end to end: issue, list, accept, revoke.

The API test client can only ever act as the one operator identity a
standalone deployment has (owner and superuser), so what a non-manager may not
do is covered at the service layer instead, alongside the rest of tenancy's
authorization matrix (test_tenancy_authorization.py), whose own docstring
explains the same split. The invitee-side inbox splits the same way: its rules
are in test_invitee_membership_inbox.py, because every one of them needs a
second identity, and what is here is the route wiring.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.tenancy import Invitation, Organization, OrganizationMember, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME


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
    response = client.post(f"{API_ROOT}/organizations/me/member-invitations", json=body, headers=headers)
    assert response.status_code == 201, response.text
    result: dict[str, Any] = response.json()
    return result


def _token_from(accept_link: str) -> str:
    return accept_link.split("token=")[1]


PASSWORD = "correct-horse-battery"  # pragma: allowlist secret


def _switch(client: TestClient, headers: dict[str, str], organization_id: str) -> None:
    switched = client.post(
        f"{API_ROOT}/organizations/me/switch", json={"organization_id": organization_id}, headers=headers
    )
    assert switched.status_code == 200, switched.text


def _roster_row(client: TestClient, headers: dict[str, str], email: str) -> dict[str, Any]:
    members = client.get(f"{API_ROOT}/organizations/me/members", headers=headers).json()["data"]
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
    # Delivered, so the link went to the mailbox alone and not back to the inviter.
    assert result["accept_link"] is None
    assert "You're invited to join" in caplog.text
    # Absolute, because it has to mean something outside a browser.
    assert "https://otari.example.com/#/accept-invitation?token=" in caplog.text
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
    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token})

    assert preview.status_code == 200, preview.text
    body = preview.json()
    assert body["email"] == "bob@example.com"
    assert body["role"] == "admin"
    assert body["organization_name"]


def test_accept_activates_the_membership_and_applies_parked_workspace_assignments(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"][0]["id"]

    result = _invite(
        client,
        master_key_header,
        email="carol@example.com",
        workspace_assignments=[{"workspace_id": workspace_id, "role": "viewer"}],
    )
    token = _token_from(result["accept_link"])

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert accept.status_code == 200, accept.text
    assert accept.json()["role"] == "member"

    row = _roster_row(client, master_key_header, "carol@example.com")
    assert row["status"] == "active"
    assert row["invitation_id"] is None  # nothing left to act on once accepted

    workspace_members = client.get(f"{API_ROOT}/workspaces/{workspace_id}/members", headers=master_key_header).json()
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
    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(result["accept_link"])})
    assert accept.status_code == 200, accept.text

    row = _roster_row(client, master_key_header, "nadia@example.com")
    assert row["attribution_user_id"] is not None

    key = client.post(
        f"{API_ROOT}/keys",
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
    created = client.post(f"{API_ROOT}/workspaces", json={"name": "Temporary"}, headers=master_key_header).json()
    kept = client.post(f"{API_ROOT}/workspaces", json={"name": "Kept"}, headers=master_key_header).json()

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

    delete = client.delete(f"{API_ROOT}/workspaces/{created['id']}", headers=master_key_header)
    assert delete.status_code == 200, delete.text

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert accept.status_code == 200, accept.text

    row = _roster_row(client, master_key_header, "karen@example.com")
    assert row["status"] == "active"

    members = client.get(f"{API_ROOT}/workspaces/{kept['id']}/members", headers=master_key_header).json()
    assert any(member["user_id"] == row["user_id"] for member in members["data"])


def test_accepting_twice_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    result = _invite(client, master_key_header, email="dave@example.com")
    token = _token_from(result["accept_link"])

    first = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    second = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})

    assert first.status_code == 200
    assert second.status_code == 400


def test_accepting_an_unknown_token_is_not_found(client: TestClient) -> None:
    response = client.post(f"{API_ROOT}/invitations/accept", json={"token": "not-a-real-token"})
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

    response = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert response.status_code == 400

    db_session.refresh(invitation)
    assert invitation.status == "expired"


def test_a_shared_link_lets_the_invitee_set_a_password_and_sign_in_without_mail(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """With no mail transport, the link an operator hands over is the whole way in."""
    result = _invite(client, master_key_header, email="grace@example.com")
    assert result["mail_sent"] is False
    token = _token_from(result["accept_link"])

    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token})
    assert preview.json()["needs_password"] is True

    accept = client.post(
        f"{API_ROOT}/invitations/accept",
        json={"token": token, "password": PASSWORD, "full_name": "Grace Hopper"},
    )
    assert accept.status_code == 200, accept.text
    assert accept.json()["password_set"] is True

    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "grace@example.com", "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text
    row = _roster_row(client, master_key_header, "grace@example.com")
    assert row["status"] == "active"
    assert row["full_name"] == "Grace Hopper"


def test_a_first_password_from_a_link_requires_the_published_terms(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The link is held to the rule signup and a reset claim follow, so it is not a way around the terms."""
    monkeypatch.setattr(test_config, "terms_url", "https://example.com/terms")
    token = _token_from(_invite(client, master_key_header, email="tess@example.com")["accept_link"])

    refused = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})
    assert refused.status_code == 400, refused.text
    assert "terms" in refused.json()["detail"]
    # Nothing was accepted, so the same link still works once the terms are.
    accepted = client.post(
        f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD, "terms_accepted": True}
    )
    assert accepted.status_code == 200, accepted.text

    invitee = db_session.query(User).filter(col(User.email) == "tess@example.com").one()
    assert invitee.terms_accepted_at is not None


def test_accepting_without_a_password_still_works_and_sets_none(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    result = _invite(client, master_key_header, email="hedy@example.com")

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(result["accept_link"])})

    assert accept.status_code == 200, accept.text
    assert accept.json()["password_set"] is False
    assert accept.json()["verification_pending"] is False
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "hedy@example.com", "password": PASSWORD})
    assert signed_in.status_code == 401


def test_accepting_after_an_unverified_signup_says_the_address_still_needs_verifying(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting does not verify (whoever signed up may not hold the mailbox), so the page has to say so."""
    result = _invite(client, master_key_header, email="hedy@example.com")
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")
    signed_up = client.post(f"{API_ROOT}/auth/signup", json={"email": "hedy@example.com", "password": PASSWORD})
    assert signed_up.status_code == 200, signed_up.text

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(result["accept_link"])})

    assert accept.status_code == 200, accept.text
    assert accept.json()["verification_pending"] is True
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "hedy@example.com", "password": PASSWORD})
    assert signed_in.status_code == 403


def test_a_forwarded_link_cannot_set_a_password_on_an_address_that_can_already_sign_in(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """A verified address is one someone already signs in to, whether by password or by provider."""
    result = _invite(client, master_key_header, email="ida@example.com")
    token = _token_from(result["accept_link"])
    invitee = db_session.get(User, uuid.UUID(_roster_row(client, master_key_header, "ida@example.com")["user_id"]))
    assert invitee is not None
    invitee.email_verified_at = datetime.now(UTC)
    db_session.add(invitee)
    db_session.commit()

    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token})
    assert preview.json()["needs_password"] is False

    refused = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})
    assert refused.status_code == 400, refused.text
    db_session.refresh(invitee)
    assert invitee.hashed_password is None
    # Refused as a whole: the invitation is still there to accept without a password.
    assert _roster_row(client, master_key_header, "ida@example.com")["status"] == "invited"
    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert accept.status_code == 200, accept.text


def test_a_link_claim_is_recorded_as_vouched_rather_than_proven(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """The inviter holds the link too, so a password chosen through it is the organization's word."""
    result = _invite(client, master_key_header, email="joan@example.com")
    accept = client.post(
        f"{API_ROOT}/invitations/accept",
        json={"token": _token_from(result["accept_link"]), "password": PASSWORD},
    )
    assert accept.status_code == 200, accept.text

    invitee = db_session.get(User, uuid.UUID(_roster_row(client, master_key_header, "joan@example.com")["user_id"]))
    assert invitee is not None
    db_session.refresh(invitee)
    assert invitee.email_verified_at is not None
    assert invitee.email_vouched_at is not None


def test_a_link_from_another_organization_cannot_claim_an_address_already_on_a_roster(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Any session can create an organization and invite any address, so its link must not claim one.

    Kay is added to the deployment's organization and has never signed in. A
    second organization, created by the same caller as anyone could, invites her
    and holds the accept link it gets back: that link must not choose her password.
    """
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "kay@example.com", "role": "owner"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Mallory"}, headers=master_key_header)
    assert created.status_code == 201, created.text
    _switch(client, master_key_header, created.json()["id"])
    token = _token_from(_invite(client, master_key_header, email="kay@example.com")["accept_link"])

    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token})
    assert preview.json()["needs_password"] is False
    refused = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})
    assert refused.status_code == 400, refused.text

    invitee = db_session.get(User, uuid.UUID(added.json()["user_id"]))
    assert invitee is not None
    db_session.refresh(invitee)
    assert invitee.hashed_password is None
    assert invitee.email_verified_at is None
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "kay@example.com", "password": PASSWORD})
    assert signed_in.status_code == 401
    # Joining without a password still works; it claims nothing.
    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert accept.status_code == 200, accept.text


def test_an_address_another_organization_vouched_for_cannot_be_added_until_it_is_proven(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Pre-claiming: vouch for an address nobody holds yet, then wait for a real organization to add it.

    Mallory invites a fresh address and chooses its password through her own
    link. The deployment's organization adding or inviting that address later
    would hand Mallory a seat in it, so both are refused until the address is
    proven.
    """
    home = client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).json()["organization"]["id"]
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Mallory"}, headers=master_key_header)
    assert created.status_code == 201, created.text
    _switch(client, master_key_header, created.json()["id"])
    token = _token_from(_invite(client, master_key_header, email="lin@example.com")["accept_link"])
    claimed = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})
    assert claimed.status_code == 200, claimed.text
    _switch(client, master_key_header, home)

    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "lin@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 409, added.text
    invited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": "lin@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert invited.status_code == 409, invited.text

    # Once the address is proven (a reset by email, a provider sign-in), it is an ordinary identity.
    identity = db_session.query(User).filter(col(User.email) == "lin@example.com").one()
    identity.email_vouched_at = None
    db_session.add(identity)
    db_session.commit()
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "lin@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text


def test_a_password_that_breaks_the_policy_is_refused_before_anything_is_accepted(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    result = _invite(client, master_key_header, email="joan@example.com")

    refused = client.post(
        f"{API_ROOT}/invitations/accept",
        json={"token": _token_from(result["accept_link"]), "password": "short"},
    )

    assert refused.status_code == 400, refused.text
    assert "at least" in refused.json()["detail"]
    assert _roster_row(client, master_key_header, "joan@example.com")["status"] == "invited"


def test_revoke_suspends_the_membership_and_the_token_stops_working(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    result = _invite(client, master_key_header, email="frank@example.com")
    token = _token_from(result["accept_link"])

    revoke = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{result['invitation_id']}",
        headers=master_key_header,
    )
    assert revoke.status_code == 200, revoke.text

    members = client.get(f"{API_ROOT}/organizations/me/members", headers=master_key_header).json()["data"]
    assert not any(row["email"] == "frank@example.com" for row in members)  # suspended, off the roster

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
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
        f"{API_ROOT}/organizations/me/members/{result['organization_member_id']}",
        headers=master_key_header,
    )
    assert remove.status_code == 200, remove.text

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert accept.status_code == 400


def test_patching_an_invited_member_to_suspended_also_cancels_the_invitation(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """The same gap, reached through the generic PATCH rather than DELETE."""
    result = _invite(client, master_key_header, email="judy@example.com")
    token = _token_from(result["accept_link"])

    patch = client.patch(
        f"{API_ROOT}/organizations/me/members/{result['organization_member_id']}",
        json={"status": "suspended"},
        headers=master_key_header,
    )
    assert patch.status_code == 200, patch.text

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
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
        f"{API_ROOT}/organizations/me/members/{result['organization_member_id']}",
        json={"status": "active"},
        headers=master_key_header,
    )
    assert patch.status_code == 200, patch.text

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert accept.status_code == 400, accept.text


def test_reinviting_a_revoked_address_revives_the_membership(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    first = _invite(client, master_key_header, email="grace@example.com", role="viewer")
    client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{first['invitation_id']}",
        headers=master_key_header,
    )

    second = _invite(client, master_key_header, email="grace@example.com", role="admin")
    assert second["organization_member_id"] == first["organization_member_id"]
    assert second["role"] == "admin"

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(second["accept_link"])})
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
        f"{API_ROOT}/organizations/me/members",
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
    stale_accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": first_token})
    assert stale_accept.status_code == 400, stale_accept.text

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(second["accept_link"])})
    assert accept.status_code == 200, accept.text


def _invite_raw(client: TestClient, headers: dict[str, str], *, email: str) -> Any:
    return client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers=headers,
    )


def test_revoking_an_unknown_invitation_is_not_found(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{uuid.uuid4()}",
        headers=master_key_header,
    )
    assert response.status_code == 404


def _operator_user_id(client: TestClient, headers: dict[str, str], db_session: Session) -> uuid.UUID:
    """The identity behind the master key, read back through its own membership.

    No route reports the caller's ``user_id`` directly; the membership context
    reports the row that joins them to their organization, and that row is
    where the id lives.
    """
    context = client.get(f"{API_ROOT}/organizations/me", headers=headers)
    assert context.status_code == 200, context.text
    membership = db_session.get(OrganizationMember, uuid.UUID(context.json()["organization_member_id"]))
    assert membership is not None
    return membership.user_id


def _invitation_waiting_on(
    db_session: Session,
    user_id: uuid.UUID,
    *,
    role: str = "member",
    expires_in: timedelta = timedelta(days=7),
) -> tuple[Organization, OrganizationMember, Invitation]:
    """Put a pending invitation in front of ``user_id``, in a second organization.

    Built directly rather than through ``POST /me/member-invitations``, because
    that route invites *somebody else*: the caller already holds an active
    membership in the organization it would write to, and
    ``uq_organization_member_organization_user`` allows them only the one. A
    second organization is the only place this identity can hold an ``invited``
    membership at all, which is also the real shape of the case the inbox
    exists for.
    """
    organization = Organization(name="Second", slug=f"second-{uuid.uuid4().hex[:8]}")
    db_session.add(organization)
    db_session.flush()
    membership = OrganizationMember(
        organization_id=organization.id,
        user_id=user_id,
        role=role,
        status="invited",
    )
    db_session.add(membership)
    db_session.flush()
    invitation = Invitation(
        organization_id=organization.id,
        organization_member_id=membership.id,
        email="operator@example.com",
        token_hash=uuid.uuid4().hex,
        workspace_assignments=[],
        expires_at=datetime.now(UTC) + expires_in,
    )
    db_session.add(invitation)
    db_session.commit()
    return organization, membership, invitation


def test_the_inbox_lists_an_invitation_waiting_on_the_caller(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    organization, membership, invitation = _invitation_waiting_on(
        db_session,
        _operator_user_id(client, master_key_header, db_session),
        role="admin",
    )

    response = client.get(f"{API_ROOT}/organizations/me/pending-memberships", headers=master_key_header)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    (waiting,) = body["data"]
    assert waiting["organization_member_id"] == str(membership.id)
    assert waiting["invitation_id"] == str(invitation.id)
    assert waiting["organization_id"] == str(organization.id)
    assert waiting["organization_name"] == "Second"
    assert waiting["role"] == "admin"


def test_the_inbox_is_the_callers_own_and_not_the_roster_it_administers(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """An invitation the caller *sent* is not one waiting on them."""
    _invite(client, master_key_header, email="someone-else@example.com")

    response = client.get(f"{API_ROOT}/organizations/me/pending-memberships", headers=master_key_header)
    assert response.status_code == 200, response.text
    assert response.json() == {"data": [], "count": 0}


def test_accepting_from_the_inbox_activates_the_membership(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    _, membership, invitation = _invitation_waiting_on(
        db_session,
        _operator_user_id(client, master_key_header, db_session),
    )

    response = client.post(
        f"{API_ROOT}/organizations/me/pending-memberships/{membership.id}/accept",
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "organization_name": "Second",
        "role": "member",
        "password_set": False,
        "verification_pending": False,
        "can_sign_in": True,
        "ways_in": [],
    }

    db_session.refresh(membership)
    db_session.refresh(invitation)
    assert membership.status == "active"
    assert invitation.status == "accepted"
    assert (
        client.get(f"{API_ROOT}/organizations/me/pending-memberships", headers=master_key_header).json()["count"] == 0
    )


def test_declining_from_the_inbox_cancels_and_suspends_the_pair(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    _, membership, invitation = _invitation_waiting_on(
        db_session,
        _operator_user_id(client, master_key_header, db_session),
    )

    response = client.post(
        f"{API_ROOT}/organizations/me/pending-memberships/{membership.id}/decline",
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Invitation declined"}

    db_session.refresh(membership)
    db_session.refresh(invitation)
    assert membership.status == "suspended"
    assert invitation.status == "cancelled"
    assert (
        client.get(f"{API_ROOT}/organizations/me/pending-memberships", headers=master_key_header).json()["count"] == 0
    )


def test_a_lapsed_invitation_is_absent_from_the_inbox_over_http(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Expiry is lazy, so a row that never had its token presented still reads `pending`."""
    _, membership, _ = _invitation_waiting_on(
        db_session,
        _operator_user_id(client, master_key_header, db_session),
        expires_in=timedelta(hours=-1),
    )

    listed = client.get(f"{API_ROOT}/organizations/me/pending-memberships", headers=master_key_header)
    assert listed.json() == {"data": [], "count": 0}

    refused = client.post(
        f"{API_ROOT}/organizations/me/pending-memberships/{membership.id}/accept",
        headers=master_key_header,
    )
    assert refused.status_code == 400, refused.text


@pytest.mark.parametrize("action", ["accept", "decline"])
def test_an_unknown_pending_membership_is_a_404(
    client: TestClient,
    master_key_header: dict[str, str],
    action: str,
) -> None:
    response = client.post(
        f"{API_ROOT}/organizations/me/pending-memberships/{uuid.uuid4()}/{action}",
        headers=master_key_header,
    )
    assert response.status_code == 404, response.text


def _bulk_invite(client: TestClient, headers: dict[str, str], emails: list[str], **extra: Any) -> dict[str, Any]:
    response = client.post(
        f"{API_ROOT}/organizations/me/member-invitations/bulk",
        json={"emails": emails, **extra},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def test_bulk_invite_reports_each_address_and_one_refusal_does_not_stop_the_rest(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _invite(client, master_key_header, email="pending@example.com")
    workspace_id = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"][0]["id"]

    result = _bulk_invite(
        client,
        master_key_header,
        # A refusal between two good addresses, a duplicate and a malformed one.
        ["one@example.com", "pending@example.com", "not-an-address", "two@example.com", "ONE@example.com"],
        role="admin",
        workspace_assignments=[{"workspace_id": workspace_id, "role": "member"}],
    )

    assert [row["email"] for row in result["invited"]] == ["one@example.com", "two@example.com"]
    assert all(row["role"] == "admin" and row["mail_sent"] is False for row in result["invited"])
    # Every submitted address is accounted for, the repeat included.
    assert [row["email"] for row in result["failed"]] == ["pending@example.com", "not-an-address", "ONE@example.com"]
    assert "more than once" in result["failed"][2]["detail"]
    assert all(row["detail"] for row in result["failed"])

    # Committed, on the roster, and each link works on its own.
    assert _roster_row(client, master_key_header, "two@example.com")["status"] == "invited"
    accepted = client.post(
        f"{API_ROOT}/invitations/accept",
        json={"token": _token_from(result["invited"][0]["accept_link"])},
    )
    assert accepted.status_code == 200, accepted.text
    placements = _roster_row(client, master_key_header, "one@example.com")
    assert placements["status"] == "active"


def test_bulk_invite_emails_every_address_when_a_transport_is_configured(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")

    result = _bulk_invite(client, master_key_header, [f"mailed{n}@example.com" for n in range(7)])

    assert len(result["invited"]) == 7
    assert result["failed"] == []
    assert all(row["mail_sent"] is True for row in result["invited"])
    assert all(row["accept_link"] is None for row in result["invited"])


def test_bulk_invite_refuses_an_empty_list(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(
        f"{API_ROOT}/organizations/me/member-invitations/bulk",
        json={"emails": []},
        headers=master_key_header,
    )
    assert response.status_code == 422


# ---------- the lifecycle of a link once its membership moves on ----------


def _add_directly(client: TestClient, headers: dict[str, str], *, email: str, role: str = "member") -> str:
    added = client.post(f"{API_ROOT}/organizations/me/members", json={"email": email, "role": role}, headers=headers)
    assert added.status_code == 201, added.text
    return str(added.json()["organization_member_id"])


def _members(client: TestClient, headers: dict[str, str]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = client.get(f"{API_ROOT}/organizations/me/members", headers=headers).json()["data"]
    return members


def _admin_session(client: TestClient, headers: dict[str, str], db_session: Session) -> str:
    """An admin of the operator's organization, signed in, and its session cookie."""
    result = _invite(client, headers, email="admin@example.com", role="admin")
    accepted = client.post(
        f"{API_ROOT}/invitations/accept", json={"token": _token_from(result["accept_link"]), "password": PASSWORD}
    )
    assert accepted.status_code == 200, accepted.text
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "admin@example.com", "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text
    cookie = signed_in.cookies[SESSION_COOKIE_NAME]
    client.cookies.clear()
    return cookie


def test_a_stale_link_cannot_readmit_a_member_who_was_removed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Invite, add directly, remove: the first link must not bring the member back."""
    result = _invite(client, master_key_header, email="sam@example.com")
    member_id = _add_directly(client, master_key_header, email="sam@example.com")
    removed = client.delete(f"{API_ROOT}/organizations/me/members/{member_id}", headers=master_key_header)
    assert removed.status_code in (200, 204), removed.text

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(result["accept_link"])})

    assert accept.status_code == 400, accept.text
    assert all(row["email"] != "sam@example.com" for row in _members(client, master_key_header))


def test_a_pending_link_on_a_membership_that_left_invited_is_refused(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """A pending row on a membership that is no longer invited must not be accepted."""
    result = _invite(client, master_key_header, email="sam@example.com")
    membership = db_session.execute(
        select(OrganizationMember).where(col(OrganizationMember.id) == uuid.UUID(result["organization_member_id"]))
    ).scalar_one()
    membership.status = "suspended"
    db_session.commit()

    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(result["accept_link"])})

    assert accept.status_code == 400, accept.text
    db_session.refresh(membership)
    assert membership.status == "suspended"


def test_revoking_a_stale_invitation_leaves_an_active_member_active(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    result = _invite(client, master_key_header, email="sam@example.com")
    membership = db_session.execute(
        select(OrganizationMember).where(col(OrganizationMember.id) == uuid.UUID(result["organization_member_id"]))
    ).scalar_one()
    membership.status = "active"
    db_session.commit()

    revoked = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{result['invitation_id']}", headers=master_key_header
    )

    assert revoked.status_code == 200, revoked.text
    assert _roster_row(client, master_key_header, "sam@example.com")["status"] == "active"
    invitation = db_session.get(Invitation, uuid.UUID(result["invitation_id"]))
    assert invitation is not None
    db_session.refresh(invitation)
    assert invitation.status == "cancelled"


def test_a_revoked_invitation_elsewhere_does_not_stop_a_link_setting_a_first_password(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """Only a live membership in another organization vouches for the address."""
    first = _invite(client, master_key_header, email="ada@example.com")
    revoked = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{first['invitation_id']}", headers=master_key_header
    )
    assert revoked.status_code == 200, revoked.text
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Second"}, headers=master_key_header)
    assert created.status_code == 201, created.text
    _switch(client, master_key_header, created.json()["id"])
    token = _token_from(_invite(client, master_key_header, email="ada@example.com")["accept_link"])

    assert client.post(f"{API_ROOT}/invitations/validate", json={"token": token}).json()["needs_password"] is True
    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})
    assert accept.status_code == 200, accept.text


def test_a_refused_first_password_names_the_way_in_this_deployment_offers(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _add_directly(client, master_key_header, email="kay@example.com")
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Mallory"}, headers=master_key_header)
    _switch(client, master_key_header, created.json()["id"])
    token = _token_from(_invite(client, master_key_header, email="kay@example.com")["accept_link"])

    refused = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})

    assert refused.status_code == 400, refused.text
    # No mail and no provider here, so neither is offered as a way in.
    detail = refused.json()["detail"]
    assert "Forgot your password" not in detail
    assert "administers this gateway" in detail


def test_a_refused_first_password_offers_the_provider_and_the_reset_this_deployment_has(
    client: TestClient, master_key_header: dict[str, str], test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_directly(client, master_key_header, email="kay@example.com")
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Mallory"}, headers=master_key_header)
    _switch(client, master_key_header, created.json()["id"])
    # Invited while no mail goes out, so the link comes back to share; the ways in are switched on afterwards.
    token = _token_from(_invite(client, master_key_header, email="kay@example.com")["accept_link"])
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://gw.example.com")
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")

    refused = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == (
        "This link cannot set a password for this address. Accept without one, then sign in with Google, "
        'or choose a password with "Forgot your password?" on the sign-in screen.'
    )


def test_an_admin_can_reinvite_an_address_whose_lapsed_invitation_offered_owner(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    result = _invite(client, master_key_header, email="olga@example.com", role="owner")
    client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{result['invitation_id']}", headers=master_key_header
    )
    client.cookies.set(SESSION_COOKIE_NAME, _admin_session(client, master_key_header, db_session))

    reinvited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations", json={"email": "olga@example.com", "role": "member"}
    )
    owner_again = client.post(
        f"{API_ROOT}/organizations/me/member-invitations", json={"email": "olga@example.com", "role": "owner"}
    )

    assert reinvited.status_code == 201, reinvited.text
    assert owner_again.status_code == 400, owner_again.text
    assert owner_again.json()["detail"] == "Only organization owners can grant the owner role"


def test_an_expired_link_says_expired_every_time(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    result = _invite(client, master_key_header, email="eli@example.com")
    invitation = db_session.get(Invitation, uuid.UUID(result["invitation_id"]))
    assert invitation is not None
    invitation.expires_at = datetime.now(UTC) - timedelta(hours=1)
    db_session.commit()
    token = _token_from(result["accept_link"])

    first = client.post(f"{API_ROOT}/invitations/validate", json={"token": token})
    second = client.post(f"{API_ROOT}/invitations/validate", json={"token": token})

    assert first.status_code == second.status_code == 400
    assert first.json()["detail"] == second.json()["detail"]
    assert "expired" in second.json()["detail"]
    assert "Ask whoever invited you" in second.json()["detail"]


def test_an_unknown_link_answers_in_a_sentence_with_a_next_step(client: TestClient) -> None:
    response = client.post(f"{API_ROOT}/invitations/validate", json={"token": "not-a-real-token"})

    assert response.status_code == 404
    assert response.json()["detail"].startswith("This invitation link is not valid")
    assert "Ask whoever invited you" in response.json()["detail"]


def test_a_deactivated_account_cannot_be_invited_or_added(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    _add_directly(client, master_key_header, email="dee@example.com")
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Second"}, headers=master_key_header)
    _switch(client, master_key_header, created.json()["id"])
    identity = db_session.execute(select(User).where(col(User.email) == "dee@example.com")).scalar_one()
    identity.is_active = False
    db_session.commit()

    invited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": "dee@example.com", "role": "member"},
        headers=master_key_header,
    )
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "dee@example.com", "role": "member"},
        headers=master_key_header,
    )

    assert invited.status_code == added.status_code == 409
    # Refused without the reason: the admin asking may be another tenant's.
    assert invited.json()["detail"] == "dee@example.com cannot be added to or invited into this organization"
    assert "deactivated" not in added.json()["detail"]


@pytest.mark.parametrize("name", ["Acme\r\nBcc: x@example.com", "Acme\x07", "Acme\u2028Bcc: x@example.com"])
def test_an_organization_name_with_a_control_character_is_refused(
    client: TestClient, master_key_header: dict[str, str], name: str
) -> None:
    created = client.post(f"{API_ROOT}/organizations", json={"name": name}, headers=master_key_header)
    renamed = client.patch(f"{API_ROOT}/organizations/me", json={"name": name}, headers=master_key_header)

    assert created.status_code == 400, created.text
    assert renamed.status_code == 400, renamed.text


# ---------- invitation mail limits ----------


def _mail_on(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")


def test_an_organization_over_its_invitation_mail_limit_gets_the_links_instead(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh tenant on an open-signup deployment once mailed 300 invitations in two seconds."""
    _mail_on(test_config, monkeypatch)
    monkeypatch.setattr(test_config, "invitation_mail_limit_per_organization", 2)

    result = _bulk_invite(client, master_key_header, ["a@example.com", "b@example.com", "c@example.com"])

    sent = [row for row in result["invited"] if row["mail_sent"]]
    held = [row for row in result["invited"] if not row["mail_sent"]]
    assert len(sent) == 2
    assert len(held) == 1
    # Still invited, with the link to share by hand, and said why.
    assert held[0]["mail_limited"] is True
    assert held[0]["accept_link"].startswith("https://otari.example.com/#/accept-invitation?token=")
    assert all(row["mail_limited"] is False for row in sent)
    single = _invite(client, master_key_header, email="d@example.com")
    assert single["mail_sent"] is False
    assert single["mail_limited"] is True


def test_revoking_and_reinviting_cannot_flood_one_address(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mail_on(test_config, monkeypatch)
    monkeypatch.setattr(test_config, "invitation_mail_limit_per_address", 2)

    outcomes = []
    for _ in range(3):
        result = _invite(client, master_key_header, email="victim@example.com")
        outcomes.append((result["mail_sent"], result["mail_limited"]))
        revoked = client.delete(
            f"{API_ROOT}/organizations/me/member-invitations/{result['invitation_id']}", headers=master_key_header
        )
        assert revoked.status_code == 200, revoked.text

    assert outcomes == [(True, False), (True, False), (False, True)]


# ---------- reissuing a link ----------


def _reissue(client: TestClient, headers: dict[str, str], invitation_id: str) -> Any:
    return client.post(f"{API_ROOT}/organizations/me/member-invitations/{invitation_id}/reissue", headers=headers)


def test_inviting_an_address_already_invited_says_how_to_send_a_new_link(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    _invite(client, master_key_header, email="lost@example.com")

    again = _invite_raw(client, master_key_header, email="lost@example.com")

    assert again.status_code == 409, again.text
    assert "Reissue it" in again.json()["detail"]


def test_reissuing_gives_a_new_link_and_kills_the_old_one(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    """On a deployment with no mail, the link is gone once the invite dialog closes; reissuing gets a new one."""
    first = _invite(client, master_key_header, email="lost@example.com", role="admin")

    reissued = _reissue(client, master_key_header, first["invitation_id"])

    assert reissued.status_code == 200, reissued.text
    body = reissued.json()
    assert body["invitation_id"] == first["invitation_id"]
    assert body["role"] == "admin"
    assert body["mail_sent"] is False
    assert body["accept_link"] != first["accept_link"]
    old = client.post(f"{API_ROOT}/invitations/validate", json={"token": _token_from(first["accept_link"])})
    assert old.status_code == 404, old.text
    accepted = client.post(
        f"{API_ROOT}/invitations/accept", json={"token": _token_from(body["accept_link"]), "password": PASSWORD}
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["role"] == "admin"


def test_reissuing_emails_the_new_link_where_mail_is_configured(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _invite(client, master_key_header, email="lost@example.com")
    _mail_on(test_config, monkeypatch)

    reissued = _reissue(client, master_key_header, first["invitation_id"])

    assert reissued.status_code == 200, reissued.text
    assert reissued.json()["mail_sent"] is True
    assert reissued.json()["accept_link"] is None


def test_an_invitation_that_expired_after_it_was_opened_can_still_be_revoked_or_reissued(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    """Opening an expired link records the expiry, and the roster then lost the invitation to act on."""
    first = _invite(client, master_key_header, email="late@example.com")
    invitation = db_session.get(Invitation, uuid.UUID(first["invitation_id"]))
    assert invitation is not None
    invitation.expires_at = datetime.now(UTC) - timedelta(hours=1)
    db_session.commit()
    opened = client.post(f"{API_ROOT}/invitations/validate", json={"token": _token_from(first["accept_link"])})
    assert opened.status_code == 400, opened.text

    row = _roster_row(client, master_key_header, "late@example.com")
    assert row["status"] == "invited"
    assert row["invitation_id"] == first["invitation_id"]
    assert datetime.fromisoformat(row["invitation_expires_at"]) < datetime.now(UTC)

    reissued = _reissue(client, master_key_header, first["invitation_id"])
    assert reissued.status_code == 200, reissued.text
    row = _roster_row(client, master_key_header, "late@example.com")
    assert datetime.fromisoformat(row["invitation_expires_at"]) > datetime.now(UTC)

    invitation.expires_at = datetime.now(UTC) - timedelta(hours=1)
    invitation.status = "expired"
    db_session.commit()
    revoked = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{first['invitation_id']}", headers=master_key_header
    )
    assert revoked.status_code == 200, revoked.text
    assert all(row["email"] != "late@example.com" for row in _members(client, master_key_header))


def test_revoking_or_reissuing_a_used_invitation_is_refused_in_the_inviters_words(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    result = _invite(client, master_key_header, email="done@example.com")
    accepted = client.post(f"{API_ROOT}/invitations/accept", json={"token": _token_from(result["accept_link"])})
    assert accepted.status_code == 200, accepted.text

    revoked = client.delete(
        f"{API_ROOT}/organizations/me/member-invitations/{result['invitation_id']}", headers=master_key_header
    )
    reissued = _reissue(client, master_key_header, result["invitation_id"])

    assert revoked.status_code == reissued.status_code == 400
    assert revoked.json()["detail"] == "This invitation was already accepted, so it cannot be revoked or reissued."
    assert "Ask whoever invited you" not in reissued.json()["detail"]


def test_an_admin_cannot_reissue_an_owner_invitation(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    result = _invite(client, master_key_header, email="olga@example.com", role="owner")
    client.cookies.set(SESSION_COOKIE_NAME, _admin_session(client, master_key_header, db_session))

    reissued = client.post(f"{API_ROOT}/organizations/me/member-invitations/{result['invitation_id']}/reissue")

    assert reissued.status_code == 400, reissued.text


# ---------- the invitee's way in ----------


def test_a_deactivated_invitee_cannot_use_a_link_issued_before_the_deactivation(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    result = _invite(client, master_key_header, email="dee@example.com")
    token = _token_from(result["accept_link"])
    identity = db_session.execute(select(User).where(col(User.email) == "dee@example.com")).scalar_one()
    identity.is_active = False
    db_session.commit()

    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token})
    accept = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})

    assert preview.status_code == accept.status_code == 400
    # Refused as a link that is no longer valid, without the reason.
    assert "deactivated" not in preview.json()["detail"]
    assert _roster_row(client, master_key_header, "dee@example.com")["status"] == "invited"


def test_two_organizations_inviting_one_new_address_tell_it_how_to_get_in(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither link may set a password, so the page has to say which way in exists instead."""
    _invite(client, master_key_header, email="new@example.com")
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Second"}, headers=master_key_header)
    _switch(client, master_key_header, created.json()["id"])
    token = _token_from(_invite(client, master_key_header, email="new@example.com")["accept_link"])

    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token}).json()
    assert (preview["needs_password"], preview["can_sign_in"], preview["ways_in"]) == (False, False, [])

    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")
    _mail_on(test_config, monkeypatch)
    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token}).json()
    assert preview["ways_in"] == ["provider", "password_reset"]

    accepted = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["can_sign_in"] is False
    assert accepted.json()["ways_in"] == ["provider", "password_reset"]


def test_an_invitee_who_can_already_sign_in_is_told_so(
    client: TestClient, master_key_header: dict[str, str], db_session: Session
) -> None:
    first = _invite(client, master_key_header, email="kim@example.com")
    claimed = client.post(
        f"{API_ROOT}/invitations/accept", json={"token": _token_from(first["accept_link"]), "password": PASSWORD}
    )
    assert claimed.json()["can_sign_in"] is True
    # Proven since, as a reset by email would, so another organization may invite it.
    identity = db_session.execute(select(User).where(col(User.email) == "kim@example.com")).scalar_one()
    identity.email_vouched_at = None
    db_session.commit()
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Second"}, headers=master_key_header)
    _switch(client, master_key_header, created.json()["id"])
    token = _token_from(_invite(client, master_key_header, email="kim@example.com")["accept_link"])

    preview = client.post(f"{API_ROOT}/invitations/validate", json={"token": token}).json()

    assert (preview["needs_password"], preview["can_sign_in"], preview["ways_in"]) == (False, True, [])


def test_the_refusal_for_an_address_vouched_elsewhere_does_not_mention_the_other_organization(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    home = client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).json()["organization"]["id"]
    created = client.post(f"{API_ROOT}/organizations", json={"name": "Mallory"}, headers=master_key_header)
    _switch(client, master_key_header, created.json()["id"])
    token = _token_from(_invite(client, master_key_header, email="lin@example.com")["accept_link"])
    claimed = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})
    assert claimed.status_code == 200, claimed.text
    _switch(client, master_key_header, home)

    refused = _invite_raw(client, master_key_header, email="lin@example.com")

    assert refused.status_code == 409, refused.text
    assert "organization's invitation" not in refused.json()["detail"]
    assert "another organization" not in refused.json()["detail"]


def test_an_address_ending_in_a_dot_is_refused_rather_than_silently_not_mailed(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    refused = _invite_raw(client, master_key_header, email="x@example.test.")

    assert refused.status_code == 400, refused.text
