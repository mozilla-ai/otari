"""``max_api_key_lifetime_days`` caps the lifetime a minted API key is given.

The cap binds on both key surfaces, create and update: an ask past the cap is clamped
to the key's creation plus the cap, a mint naming no expiry gets one at the cap, and a
clamped write answers with ``Otari-Key-Lifetime-Capped`` so the caller learns its ask
moved. A write that would clamp on a key the cap has already outlived is refused with
409 rather than storing an expiry that ends the key. Unset, the default, nothing moves,
which the last tests hold.

The pure arithmetic has its own unit tests (``tests/unit/test_key_lifetime_cap.py``);
these exercise the routes, the header, and the member surface's ``changes`` path.
"""

from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, KEY_LIFETIME_CAPPED_HEADER, GatewayConfig
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

from .conftest import build_test_client

_CAP_DAYS = 30
_TOLERANCE = timedelta(minutes=5)


@pytest.fixture
def capped_config(test_config: GatewayConfig) -> GatewayConfig:
    return test_config.model_copy(update={"max_api_key_lifetime_days": _CAP_DAYS})


@pytest.fixture
def capped_client(capped_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(capped_config)


def _parse(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _near(value: datetime | None, expected: datetime) -> bool:
    return value is not None and abs(value - expected) <= _TOLERANCE


def _backdate(db_session_factory: Callable[[], Session], key_id: str, days: int) -> None:
    """Move a key's creation back in time, as a cap lowered after its mint would leave it."""
    session = db_session_factory()
    try:
        row = session.get(APIKey, key_id)
        assert row is not None
        row.created_at = datetime.now(UTC) - timedelta(days=days)
        session.commit()
    finally:
        session.close()


# =============================================================================
# Operator surface: POST /keys and PATCH /keys/{id}
# =============================================================================


def test_create_clamps_an_expiry_past_the_cap(capped_client: TestClient, master_key_header: dict[str, str]) -> None:
    asked = datetime.now(UTC) + timedelta(days=365)
    response = capped_client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "far", "expires_at": asked.isoformat()},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert _near(_parse(body["expires_at"]), datetime.now(UTC) + timedelta(days=_CAP_DAYS))
    header = response.headers.get(KEY_LIFETIME_CAPPED_HEADER)
    assert header is not None
    assert _parse(header) == _parse(body["expires_at"])


def test_create_without_an_expiry_gets_one_at_the_cap(
    capped_client: TestClient, master_key_header: dict[str, str]
) -> None:
    response = capped_client.post(f"{API_ROOT}/keys", json={"key_name": "silent"}, headers=master_key_header)
    assert response.status_code == 200, response.text
    body = response.json()

    assert _near(_parse(body["expires_at"]), datetime.now(UTC) + timedelta(days=_CAP_DAYS))
    assert response.headers.get(KEY_LIFETIME_CAPPED_HEADER) is not None


def test_create_within_the_cap_is_answered_as_asked(
    capped_client: TestClient, master_key_header: dict[str, str]
) -> None:
    asked = datetime.now(UTC) + timedelta(days=10)
    response = capped_client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "near", "expires_at": asked.isoformat()},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert _parse(body["expires_at"]) == asked
    assert response.headers.get(KEY_LIFETIME_CAPPED_HEADER) is None


def test_update_clamps_an_expiry_past_the_key_creation(
    capped_client: TestClient, master_key_header: dict[str, str]
) -> None:
    created = capped_client.post(f"{API_ROOT}/keys", json={"key_name": "to-patch"}, headers=master_key_header).json()
    created_at = _parse(created["created_at"])
    assert created_at is not None

    response = capped_client.patch(
        f"{API_ROOT}/keys/{created['id']}",
        json={"expires_at": (datetime.now(UTC) + timedelta(days=365)).isoformat()},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # Capped from the key's creation, not from this request.
    assert _near(_parse(body["expires_at"]), created_at + timedelta(days=_CAP_DAYS))
    assert response.headers.get(KEY_LIFETIME_CAPPED_HEADER) is not None


def test_update_without_an_expiry_change_leaves_the_expiry_alone(
    capped_client: TestClient, master_key_header: dict[str, str]
) -> None:
    asked = datetime.now(UTC) + timedelta(days=5)
    created = capped_client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "keep-expiry", "expires_at": asked.isoformat()},
        headers=master_key_header,
    ).json()

    response = capped_client.patch(
        f"{API_ROOT}/keys/{created['id']}",
        json={"key_name": "renamed"},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert _parse(body["expires_at"]) == asked
    assert response.headers.get(KEY_LIFETIME_CAPPED_HEADER) is None


def test_update_past_an_outlived_cap_is_refused(
    capped_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """A renewal on a key older than the cap is refused rather than stored as an expiry already past."""
    created = capped_client.post(f"{API_ROOT}/keys", json={"key_name": "outlived"}, headers=master_key_header).json()
    minted_expiry = _parse(created["expires_at"])
    assert minted_expiry is not None
    _backdate(db_session_factory, created["id"], days=100)

    renewed = capped_client.patch(
        f"{API_ROOT}/keys/{created['id']}",
        json={"expires_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
        headers=master_key_header,
    )
    assert renewed.status_code == 409, renewed.text
    assert "max_api_key_lifetime_days" in renewed.json()["detail"]
    assert KEY_LIFETIME_CAPPED_HEADER not in renewed.headers

    # Clearing the expiry is also a write of the ceiling, so it is refused the same way.
    cleared = capped_client.patch(
        f"{API_ROOT}/keys/{created['id']}",
        json={"expires_at": None},
        headers=master_key_header,
    )
    assert cleared.status_code == 409, cleared.text

    stored = capped_client.get(f"{API_ROOT}/keys/{created['id']}", headers=master_key_header).json()
    assert _parse(stored["expires_at"]) == minted_expiry


# =============================================================================
# Member surface: POST and PATCH /organizations/me/keys
# =============================================================================


@pytest.fixture
def member_session(db_session_factory: Callable[[], Session]) -> tuple[str, str]:
    """An organization owner's dashboard session token, in a tenant of their own."""
    session = db_session_factory()
    try:
        organization = Organization(name="Capped", slug="capped")
        session.add(organization)
        session.commit()
        session.refresh(organization)
        session.add(Workspace(name="Default", organization_id=organization.id))
        session.commit()

        user = User(email="owner@capped.test", full_name="Owner", active_organization_id=organization.id)
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(OrganizationMember(organization_id=organization.id, user_id=user.id, role="owner", status="active"))
        token = "otari-sess-capped-owner"
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
        return token, str(user.id)
    finally:
        session.close()


def _as_member(client: TestClient, token: str) -> None:
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE_NAME, token)


def test_a_member_create_is_clamped_and_told(capped_client: TestClient, member_session: tuple[str, str]) -> None:
    token, _ = member_session
    _as_member(capped_client, token)
    try:
        response = capped_client.post(
            f"{API_ROOT}/organizations/me/keys",
            json={"key_name": "member-far", "expires_at": (datetime.now(UTC) + timedelta(days=365)).isoformat()},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert _near(_parse(body["expires_at"]), datetime.now(UTC) + timedelta(days=_CAP_DAYS))
        assert response.headers.get(KEY_LIFETIME_CAPPED_HEADER) is not None
    finally:
        capped_client.cookies.clear()


def test_a_member_update_is_clamped_and_told(capped_client: TestClient, member_session: tuple[str, str]) -> None:
    token, _ = member_session
    _as_member(capped_client, token)
    try:
        created = capped_client.post(
            f"{API_ROOT}/organizations/me/keys",
            json={"key_name": "member-far", "expires_at": (datetime.now(UTC) + timedelta(days=365)).isoformat()},
        ).json()
        key_id: str = created["id"]
        created_at = _parse(created["created_at"])
        assert created_at is not None

        response = capped_client.patch(
            f"{API_ROOT}/organizations/me/keys/{key_id}",
            json={"expires_at": (datetime.now(UTC) + timedelta(days=365)).isoformat()},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert _near(_parse(body["expires_at"]), created_at + timedelta(days=_CAP_DAYS))
        assert response.headers.get(KEY_LIFETIME_CAPPED_HEADER) is not None
    finally:
        capped_client.cookies.clear()


def test_a_member_update_past_an_outlived_cap_is_refused(
    capped_client: TestClient,
    member_session: tuple[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    """The member surface's ``changes`` path refuses the same write the operator surface does."""
    token, _ = member_session
    _as_member(capped_client, token)
    try:
        created = capped_client.post(
            f"{API_ROOT}/organizations/me/keys",
            json={"key_name": "member-outlived"},
        ).json()
        _backdate(db_session_factory, created["id"], days=100)

        response = capped_client.patch(
            f"{API_ROOT}/organizations/me/keys/{created['id']}",
            json={"expires_at": (datetime.now(UTC) + timedelta(days=1)).isoformat()},
        )
        assert response.status_code == 409, response.text
        assert KEY_LIFETIME_CAPPED_HEADER not in response.headers
    finally:
        capped_client.cookies.clear()


# =============================================================================
# Unset, the default, is no cap
# =============================================================================


def test_no_cap_keeps_a_mint_as_asked(client: TestClient, master_key_header: dict[str, str]) -> None:
    asked = datetime.now(UTC) + timedelta(days=3650)
    response = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": "unbounded-expiry", "expires_at": asked.isoformat()},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert _parse(body["expires_at"]) == asked
    assert response.headers.get(KEY_LIFETIME_CAPPED_HEADER) is None

    silent = client.post(f"{API_ROOT}/keys", json={"key_name": "unbounded"}, headers=master_key_header)
    assert silent.status_code == 200, silent.text
    assert silent.json()["expires_at"] is None
    assert silent.headers.get(KEY_LIFETIME_CAPPED_HEADER) is None


def test_the_cap_appears_on_the_settings_view(capped_client: TestClient, master_key_header: dict[str, str]) -> None:
    """The deployment's own Settings view carries the value, so an operator can read it back."""
    response = capped_client.get(f"{API_ROOT}/settings", headers=master_key_header)
    assert response.status_code == 200, response.text
    entries: list[dict[str, Any]] = response.json()["config"]
    cap_entries = [entry for entry in entries if entry["key"] == "max_api_key_lifetime_days"]
    assert len(cap_entries) == 1
    assert cap_entries[0]["value"] == _CAP_DAYS
