"""The operator's email blocklist (`/api/v1/admin/blocked-emails`) and every flow it gates, end to end."""

import logging
import re
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

PASSWORD = "a-real-password"  # pragma: allowlist secret
_TOKEN_IN_LINK = re.compile(r"token=([\w-]+)")
_BLOCKED = f"{API_ROOT}/admin/blocked-emails"


@pytest.fixture
def mail(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")


@pytest.fixture
def mail_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the console transport's messages, which carry the links a test follows."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.fixture
def organization_id(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> uuid.UUID:
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == 200
    session = db_session_factory()
    try:
        return session.execute(select(Organization).where(col(Organization.slug) == "default")).scalar_one().id
    finally:
        session.close()


def _block(client: TestClient, headers: dict[str, str], value: str, **extra: Any) -> dict[str, Any]:
    response = client.post(_BLOCKED, json={"value": value, **extra}, headers=headers)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _session_for(
    session_factory: Callable[[], Session], *, organization_id: uuid.UUID, email: str, is_superuser: bool = False
) -> tuple[uuid.UUID, str]:
    """An identity holding a live dashboard session, and the cookie that names it."""
    session = session_factory()
    try:
        user = User(email=email, active_organization_id=organization_id, is_superuser=is_superuser)
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(
            OrganizationMember(organization_id=organization_id, user_id=user.id, role="member", status="active")
        )
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
    finally:
        session.close()


def _session_count(session_factory: Callable[[], Session], user_id: uuid.UUID) -> int:
    session = session_factory()
    try:
        return len(
            session.execute(select(DashboardSession).where(col(DashboardSession.user_id) == user_id)).scalars().all()
        )
    finally:
        session.close()


def _signed_up_and_verified(
    client: TestClient, master_key_header: dict[str, str], mail_log: pytest.LogCaptureFixture, *, email: str
) -> None:
    """An account added by an admin, claimed through signup and verified, so it can sign in by password."""
    added = client.post(
        f"{API_ROOT}/organizations/me/members", json={"email": email, "role": "member"}, headers=master_key_header
    )
    assert added.status_code == 201, added.text
    mail_log.clear()
    assert client.post(f"{API_ROOT}/auth/signup", json={"email": email, "password": PASSWORD}).status_code == 200
    match = _TOKEN_IN_LINK.search(mail_log.text)
    assert match, mail_log.text
    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": match.group(1), "password": PASSWORD})
    assert verified.status_code == 200, verified.text


# =============================================================================
# Managing the list
# =============================================================================


def test_an_operator_blocks_lists_and_unblocks_an_entry(client: TestClient, master_key_header: dict[str, str]) -> None:
    created = _block(client, master_key_header, "@Spam.Example", reason="throwaway signups")
    assert created["pattern"] == "spam.example"
    assert created["kind"] == "domain"
    assert created["reason"] == "throwaway signups"
    assert created["blocked_account_count"] == 0
    address = _block(client, master_key_header, "Person@Other.Example")
    assert address["kind"] == "address"
    assert address["pattern"] == "person@other.example"

    listed = client.get(_BLOCKED, headers=master_key_header).json()
    assert listed["count"] == 2
    assert {row["pattern"] for row in listed["data"]} == {"spam.example", "person@other.example"}

    removed = client.delete(f"{_BLOCKED}/{created['id']}", headers=master_key_header)
    assert removed.status_code == 204, removed.text
    assert client.get(_BLOCKED, headers=master_key_header).json()["count"] == 1
    assert client.delete(f"{_BLOCKED}/{created['id']}", headers=master_key_header).status_code == 404


def test_a_duplicate_or_a_malformed_entry_is_refused(client: TestClient, master_key_header: dict[str, str]) -> None:
    _block(client, master_key_header, "spam.example")
    duplicate = client.post(_BLOCKED, json={"value": "*.SPAM.example"}, headers=master_key_header)
    assert duplicate.status_code == 409, duplicate.text
    malformed = client.post(_BLOCKED, json={"value": "not a domain"}, headers=master_key_header)
    assert malformed.status_code == 400, malformed.text


def test_an_operator_cannot_block_their_own_address(
    client: TestClient, db_session_factory: Callable[[], Session], organization_id: uuid.UUID
) -> None:
    _, token = _session_for(
        db_session_factory, organization_id=organization_id, email="operator@ops.example", is_superuser=True
    )
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        own_address = client.post(_BLOCKED, json={"value": "operator@ops.example"})
        own_domain = client.post(_BLOCKED, json={"value": "ops.example"})
    finally:
        client.cookies.clear()
    assert own_address.status_code == 400, own_address.text
    assert own_domain.status_code == 400, own_domain.text


def test_a_non_operator_is_told_the_list_does_not_exist(
    client: TestClient, db_session_factory: Callable[[], Session], organization_id: uuid.UUID
) -> None:
    _, token = _session_for(db_session_factory, organization_id=organization_id, email="ada@example.com")
    client.cookies.set(SESSION_COOKIE_NAME, token)
    try:
        listed = client.get(_BLOCKED)
        created = client.post(_BLOCKED, json={"value": "spam.example"})
        removed = client.delete(f"{_BLOCKED}/{uuid.uuid4()}")
    finally:
        client.cookies.clear()
    assert listed.status_code == 404, listed.text
    assert created.status_code == 404, created.text
    assert removed.status_code == 404, removed.text


# =============================================================================
# What an entry gates
# =============================================================================


def test_blocking_a_domain_ends_the_sessions_of_the_accounts_under_it_and_no_others(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    organization_id: uuid.UUID,
) -> None:
    apex_id, apex_token = _session_for(db_session_factory, organization_id=organization_id, email="a@spam.example")
    sub_id, _ = _session_for(db_session_factory, organization_id=organization_id, email="b@mail.spam.example")
    other_id, _ = _session_for(db_session_factory, organization_id=organization_id, email="c@notspam.example")

    created = _block(client, master_key_header, "spam.example")

    assert created["blocked_account_count"] == 2
    assert _session_count(db_session_factory, apex_id) == 0
    assert _session_count(db_session_factory, sub_id) == 0
    assert _session_count(db_session_factory, other_id) == 1
    client.cookies.set(SESSION_COOKIE_NAME, apex_token)
    try:
        assert client.get(f"{API_ROOT}/organizations/me").status_code == 401
    finally:
        client.cookies.clear()


def test_a_blocked_account_cannot_sign_in_until_it_is_unblocked(
    client: TestClient, master_key_header: dict[str, str], mail: None, mail_log: pytest.LogCaptureFixture
) -> None:
    _signed_up_and_verified(client, master_key_header, mail_log, email="ada@example.com")
    entry = _block(client, master_key_header, "ada@example.com")

    refused = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert refused.status_code == 403, refused.text
    assert "blocked" in refused.json()["detail"]

    assert client.delete(f"{_BLOCKED}/{entry['id']}", headers=master_key_header).status_code == 204
    allowed = client.post(f"{API_ROOT}/auth/session", json={"email": "ada@example.com", "password": PASSWORD})
    assert allowed.status_code == 200, allowed.text


def test_signup_resend_and_reset_send_nothing_to_a_blocked_address(
    client: TestClient, master_key_header: dict[str, str], mail: None, mail_log: pytest.LogCaptureFixture
) -> None:
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "ada@spam.example", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    _block(client, master_key_header, "spam.example")
    mail_log.clear()

    signup = client.post(f"{API_ROOT}/auth/signup", json={"email": "ada@spam.example", "password": PASSWORD})
    resend = client.post(f"{API_ROOT}/auth/resend-verification", json={"email": "ada@spam.example"})
    reset = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "ada@spam.example"})

    # The same answers an unknown address gets, so the response does not reveal the block.
    assert signup.status_code == 200, signup.text
    assert resend.status_code == 200, resend.text
    assert reset.status_code in (200, 202), reset.text
    assert not _TOKEN_IN_LINK.search(mail_log.text), mail_log.text


def test_a_blocked_address_cannot_be_added_or_invited(
    client: TestClient, master_key_header: dict[str, str], mail: None
) -> None:
    _block(client, master_key_header, "spam.example")

    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "ada@spam.example", "role": "member"},
        headers=master_key_header,
    )
    invited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": "ada@mail.spam.example", "role": "member"},
        headers=master_key_header,
    )
    bulk = client.post(
        f"{API_ROOT}/organizations/me/member-invitations/bulk",
        json={"emails": ["ok@example.com", "bob@spam.example"], "role": "member"},
        headers=master_key_header,
    )

    assert added.status_code == 400, added.text
    assert invited.status_code == 400, invited.text
    assert bulk.status_code == 200, bulk.text
    assert [failure["email"] for failure in bulk.json()["failed"]] == ["bob@spam.example"]


def test_an_invitation_sent_before_the_block_cannot_be_accepted(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    invited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": "ada@spam.example", "role": "member"},
        headers=master_key_header,
    )
    assert invited.status_code == 201, invited.text
    token = invited.json()["accept_link"].split("token=")[1]
    _block(client, master_key_header, "spam.example")

    accepted = client.post(f"{API_ROOT}/invitations/accept", json={"token": token})

    assert accepted.status_code == 400, accepted.text


def test_a_verification_link_mailed_before_the_block_verifies_nothing(
    client: TestClient, master_key_header: dict[str, str], mail: None, mail_log: pytest.LogCaptureFixture
) -> None:
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "ada@spam.example", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    mail_log.clear()
    assert (
        client.post(f"{API_ROOT}/auth/signup", json={"email": "ada@spam.example", "password": PASSWORD}).status_code
        == 200
    )
    match = _TOKEN_IN_LINK.search(mail_log.text)
    assert match, mail_log.text
    _block(client, master_key_header, "spam.example")

    verified = client.post(f"{API_ROOT}/auth/verify-email", json={"token": match.group(1), "password": PASSWORD})

    assert verified.status_code == 400, verified.text


def test_an_operator_cannot_send_a_blocked_account_a_password_link(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    added = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": "ada@spam.example", "role": "member"},
        headers=master_key_header,
    )
    assert added.status_code == 201, added.text
    _block(client, master_key_header, "spam.example")

    link = client.post(f"{API_ROOT}/admin/users/{added.json()['user_id']}/password-link", headers=master_key_header)

    assert link.status_code == 409, link.text
    assert "blocked" in link.json()["detail"]
