"""The signed-in identity naming itself: ``PATCH /api/v1/auth/profile``.

``full_name`` has been published on the membership context since
mozilla-ai/otari#832, so the shell can draw a person rather than a role, but
nothing authenticated could write it: an identity added to a roster by address
was drawn by its address permanently. These pin the write and the two answers
that are easy to get wrong, clearing a name and refusing to touch an address.

Unit rather than integration, matching ``test_password_sign_in.py``: route,
service and identity behavior that runs unchanged on the SQLite file each test
stands up.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app
from gateway.models.tenancy import MAX_FULL_NAME_LENGTH
from gateway.services.tenancy.provisioning_service import OPERATOR_FULL_NAME

MASTER_KEY = "sk-test-master"


def _config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'profile-test.db'}",
        master_key=MASTER_KEY,
        require_pricing=False,
    )


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(_config(tmp_path)))


def _sign_in(client: TestClient) -> None:
    """Mint the dashboard session the account page calls this endpoint from."""
    response = client.post(f"{API_ROOT}/auth/session", json={"master_key": MASTER_KEY})
    assert response.status_code == 200, response.text


def _caller(client: TestClient) -> dict[str, object]:
    """The identity as the membership context publishes it, which is what the shell draws."""
    response = client.get(f"{API_ROOT}/organizations/me")
    assert response.status_code == 200, response.text
    caller: dict[str, object] = response.json()["caller"]
    return caller


def test_a_name_set_here_is_what_the_membership_context_reports(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in(client)
        response = client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "Ada Lovelace"})

        assert response.status_code == 200, response.text
        assert response.json()["full_name"] == "Ada Lovelace"
        assert _caller(client)["full_name"] == "Ada Lovelace"


def test_the_response_is_the_caller_the_context_carries(tmp_path: Path) -> None:
    """Same shape, so a client can seat the answer where it read the old value."""
    with _client(tmp_path) as client:
        _sign_in(client)
        before = _caller(client)

        response = client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "Ada Lovelace"})

        assert response.json() == {**before, "full_name": "Ada Lovelace"}


def test_surrounding_and_inner_whitespace_is_collapsed(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in(client)
        response = client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "  Ada\t Lovelace \n"})

        assert response.json()["full_name"] == "Ada Lovelace"


def test_null_clears_the_name(tmp_path: Path) -> None:
    """Back to the state a roster entry added by address starts in, which readers handle."""
    with _client(tmp_path) as client:
        _sign_in(client)
        client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "Ada Lovelace"})

        response = client.patch(f"{API_ROOT}/auth/profile", json={"full_name": None})

        assert response.status_code == 200, response.text
        assert response.json()["full_name"] is None
        assert _caller(client)["full_name"] is None


def test_a_whitespace_only_name_clears_it_rather_than_storing_blanks(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        _sign_in(client)
        client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "Ada Lovelace"})

        response = client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "   "})

        assert response.json()["full_name"] is None


def test_a_name_past_the_column_width_is_refused(tmp_path: Path) -> None:
    """Refused with a 422 the caller can read, rather than truncated or 500 by the driver."""
    with _client(tmp_path) as client:
        _sign_in(client)
        response = client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "a" * (MAX_FULL_NAME_LENGTH + 1)})

        assert response.status_code == 422
        assert _caller(client)["full_name"] == OPERATOR_FULL_NAME


def test_an_omitted_name_is_refused(tmp_path: Path) -> None:
    """One field, two meanings: a name or null. An empty body is neither."""
    with _client(tmp_path) as client:
        _sign_in(client)
        assert client.patch(f"{API_ROOT}/auth/profile", json={}).status_code == 422


def test_the_sign_in_address_is_untouched(tmp_path: Path) -> None:
    """Changing an address is a credential change with a flow behind it; this is not it."""
    with _client(tmp_path) as client:
        _sign_in(client)
        client.put(f"{API_ROOT}/auth/password", json={"email": "ada@example.com", "new_password": "a-real-password"})

        client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "Ada Lovelace"})

        assert _caller(client)["email"] == "ada@example.com"


def test_it_needs_a_credential(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.patch(f"{API_ROOT}/auth/profile", json={"full_name": "Ada Lovelace"})

        assert response.status_code == 401


def test_the_master_key_in_a_header_names_the_operator(tmp_path: Path) -> None:
    """The header path resolves the bootstrap operator, the identity that key stands for."""
    with _client(tmp_path) as client:
        response = client.patch(
            f"{API_ROOT}/auth/profile",
            json={"full_name": "Ada Lovelace"},
            headers={"Otari-Key": MASTER_KEY},
        )

        assert response.status_code == 200, response.text
        assert response.json()["full_name"] == "Ada Lovelace"
