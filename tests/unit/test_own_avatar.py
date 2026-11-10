"""The signed-in identity picking its avatar: ``PUT /api/v1/auth/profile/avatar``.

The pick used to live in one browser's storage, so a colleague listing the
organization saw the face the id hashes to, never the one picked. These pin
that the pick is stored on the identity and published wherever the dashboard
draws that person: the caller on the membership context, and the member row
everybody else in the organization reads.

Unit rather than integration, matching ``test_own_profile.py``.
"""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app
from gateway.models.tenancy import AVATAR_FACE_COUNT

MASTER_KEY = "sk-test-master"
OPERATOR = {"Otari-Key": MASTER_KEY}
PASSWORD = "correct-horse-battery"  # pragma: allowlist secret


def _client(tmp_path: Path) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'avatar-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
    )
    return TestClient(create_app(config))


def _sign_in_as_operator(client: TestClient) -> None:
    response = client.post(f"{API_ROOT}/auth/session", json={"master_key": MASTER_KEY})
    assert response.status_code == 200, response.text


def _caller(client: TestClient) -> dict[str, Any]:
    response = client.get(f"{API_ROOT}/organizations/me")
    assert response.status_code == 200, response.text
    caller: dict[str, Any] = response.json()["caller"]
    return caller


def _members_by_email(client: TestClient) -> dict[str | None, dict[str, Any]]:
    response = client.get(f"{API_ROOT}/organizations/me/members", headers=OPERATOR)
    assert response.status_code == 200, response.text
    return {member["email"]: member for member in response.json()["data"]}


def _sign_in_as_new_member(client: TestClient, *, email: str) -> None:
    """Invite an address, accept with a first password, and take a session as that member."""
    invited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": email, "role": "member"},
        headers=OPERATOR,
    )
    assert invited.status_code == 201, invited.text
    token = invited.json()["accept_link"].split("token=")[1]
    accepted = client.post(f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD})
    assert accepted.status_code == 200, accepted.text
    signed_in = client.post(f"{API_ROOT}/auth/session", json={"email": email, "password": PASSWORD})
    assert signed_in.status_code == 200, signed_in.text


def test_nobody_has_a_pick_until_they_make_one(tmp_path: Path) -> None:
    """Null, so the dashboard draws the face the id gives, as it did before picks were stored."""
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        assert _caller(client)["avatar_face"] is None


def test_a_pick_is_what_the_membership_context_reports(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        response = client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 1234})

        assert response.status_code == 200, response.text
        assert response.json()["avatar_face"] == 1234
        assert _caller(client)["avatar_face"] == 1234
        assert client.get(f"{API_ROOT}/auth/profile").json()["avatar_face"] == 1234


def test_the_response_is_the_caller_the_context_carries(tmp_path: Path) -> None:
    """Same shape as the name change, so the client seats it where it read the old one."""
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        before = _caller(client)

        response = client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 7})

        assert response.json() == {**before, "avatar_face": 7}


def test_a_colleague_listing_the_organization_sees_the_pick(tmp_path: Path) -> None:
    """The bug this column exists for: another member's list showed the hashed face."""
    with _client(tmp_path) as client:
        _sign_in_as_new_member(client, email="grace@example.com")
        picked = client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 4242})
        assert picked.status_code == 200, picked.text

        assert _members_by_email(client)["grace@example.com"]["avatar_face"] == 4242


def test_a_pick_is_the_callers_own_and_nobody_elses(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in_as_new_member(client, email="grace@example.com")
        client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 4242})

        members = _members_by_email(client)
        others = [member for email, member in members.items() if email != "grace@example.com"]
        assert others
        assert all(member["avatar_face"] is None for member in others)


def test_null_goes_back_to_the_face_the_id_gives(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 1234})

        response = client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": None})

        assert response.status_code == 200, response.text
        assert _caller(client)["avatar_face"] is None


def test_both_ends_of_the_set_are_accepted(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        for face in (0, AVATAR_FACE_COUNT - 1):
            response = client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": face})
            assert response.status_code == 200, response.text
            assert _caller(client)["avatar_face"] == face


def test_a_face_outside_the_set_is_refused_and_the_pick_kept(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 1234})

        for face in (-1, AVATAR_FACE_COUNT, 2**63, 1.5, "12", True):
            response = client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": face})
            assert response.status_code == 422, (face, response.text)

        assert _caller(client)["avatar_face"] == 1234


def test_an_omitted_face_is_refused(tmp_path: Path) -> None:
    """A face or null; an empty body is neither."""
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        assert client.put(f"{API_ROOT}/auth/profile/avatar", json={}).status_code == 422


def test_picking_leaves_the_name_alone(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in_as_operator(client)
        client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "Ada Lovelace"})

        client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 1234})

        assert _caller(client)["full_name"] == "Ada Lovelace"


def test_it_needs_a_credential(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.put(f"{API_ROOT}/auth/profile/avatar", json={"face": 1234}).status_code == 401
