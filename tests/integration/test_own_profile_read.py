"""The signed-in identity reading itself: ``GET /api/v1/auth/profile``.

The membership context (``GET /api/v1/organizations/me``) publishes the same
``caller``, but it 404s for an identity with no active membership. These pin
that the profile read answers for that identity too, and that it matches the
context's ``caller`` wherever both answer.
"""

from typing import Any

from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT

PASSWORD = "correct-horse-battery"  # pragma: allowlist secret


def _member_with_a_password(client: TestClient, headers: dict[str, str], *, email: str) -> dict[str, Any]:
    """Invite an address and accept for it with a first password, as a shared link does."""
    invited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers=headers,
    )
    assert invited.status_code == 201, invited.text
    result: dict[str, Any] = invited.json()
    token = result["accept_link"].split("token=")[1]
    accepted = client.post(
        f"{API_ROOT}/invitations/accept",
        json={"token": token, "password": PASSWORD, "full_name": "Grace Hopper"},
    )
    assert accepted.status_code == 200, accepted.text
    return result


def test_an_identity_with_no_organization_can_still_read_who_it_is(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    invited = _member_with_a_password(client, master_key_header, email="grace@example.com")
    removed = client.delete(
        f"{API_ROOT}/organizations/me/members/{invited['organization_member_id']}",
        headers=master_key_header,
    )
    assert removed.status_code == 200, removed.text

    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "grace@example.com", "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text

    assert client.get(f"{API_ROOT}/organizations/me").status_code == 404
    response = client.get(f"{API_ROOT}/auth/profile")
    assert response.status_code == 200, response.text
    profile = response.json()
    assert profile["email"] == "grace@example.com"
    assert profile["full_name"] == "Grace Hopper"
    assert profile["has_password"] is True
    assert profile["claims_deployment"] is False


def test_the_profile_is_the_caller_the_membership_context_carries(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    _member_with_a_password(client, master_key_header, email="grace@example.com")
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": "grace@example.com", "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text

    context = client.get(f"{API_ROOT}/organizations/me")
    assert context.status_code == 200, context.text
    profile = client.get(f"{API_ROOT}/auth/profile")
    assert profile.status_code == 200, profile.text
    assert profile.json() == context.json()["caller"]


def test_the_profile_needs_a_credential(client: TestClient) -> None:
    assert client.get(f"{API_ROOT}/auth/profile").status_code == 401
