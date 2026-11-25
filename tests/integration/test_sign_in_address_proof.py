"""What a sign-in through a provider proves about the address an account holds.

``user.email_proven_at`` is what domain auto-join and the add-or-invite refusals read to
decide that the holder of an address controls it. A returning sign-in may stamp it only
where the provider actually asserted that address and this deployment trusts the provider
to assert one. An organization's own connection is supplied by a tenant, so it never does.
"""

import uuid
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlmodel import col, select

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.identity import EmailAssertion
from gateway.models.tenancy import OrganizationMember, User
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client
from .test_auth_sso_routes import (
    ISSUER,
    ORIGIN,
    _add_claim,
    _add_connection,
    _add_organization,
    _authorize,
    stub_exchange,
)

OTHER_DOMAIN_ADDRESS = "member@acme.example"
SAME_DOMAIN_ADDRESS = "other@connected.example"
CHOSEN_PASSWORD = "a-chosen-password"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decrypting a connection's stored client secret needs a key configured."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture
def client(test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    """A client whose base URL is ``ORIGIN``, matching ``public_base_url``."""
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    yield from build_test_client(test_config, base_url=ORIGIN)


def _sign_in(client: TestClient, slug: str, code: str) -> Any:
    bound = _authorize(client, slug)
    response = client.post(f"{API_ROOT}/auth/sso/callback", json={"code": code, "state": bound["state"]})
    assert response.status_code == 200, response.text
    return response.json()


def _account(db_session: Session, account_id: uuid.UUID) -> User:
    """Read the account back, past anything this session already holds from before the request wrote."""
    db_session.expire_all()
    return db_session.execute(select(User).where(col(User.id) == account_id)).scalars().one()


def test_a_returning_sign_in_does_not_prove_an_address_the_provider_never_asserted(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An account records an address through its session, then signs in again.

    The connection's issuer asserts its own address and says nothing about the one the
    account now holds, so the second sign-in leaves ``email_proven_at`` null and the
    organization that proved that address does not admit the account.
    """
    connected = _add_organization(db_session, slug="connected")
    _add_connection(db_session, connected.id, issuer=ISSUER)
    _add_claim(db_session, connected.id, domain="connected.example")
    elsewhere = _add_organization(db_session, slug="elsewhere")
    _add_claim(db_session, elsewhere.id, domain="acme.example")

    stub_exchange(monkeypatch, subject="a-subject", email="someone@connected.example", full_name="Someone")

    first = _sign_in(client, "connected", "code-one")
    account_id = uuid.UUID(first["user_id"])

    recorded = client.put(
        f"{API_ROOT}/auth/password",
        json={"email": OTHER_DOMAIN_ADDRESS, "new_password": CHOSEN_PASSWORD},
    )
    assert recorded.status_code == 200, recorded.text
    assert _account(db_session, account_id).email_proven_at is None

    _sign_in(client, "connected", "code-two")

    account = _account(db_session, account_id)
    assert account.email == OTHER_DOMAIN_ADDRESS
    assert account.email_proven_at is None
    memberships = (
        db_session.execute(select(OrganizationMember).where(col(OrganizationMember.user_id) == account_id))
        .scalars()
        .all()
    )
    assert [m.organization_id for m in memberships] == [connected.id]


def test_a_connection_a_tenant_supplies_proves_nothing_even_about_the_address_it_asserts(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The issuer asserts the account's own address, verified, and the sign-in still proves nothing.

    Its verification and the address both line up, so the only thing left refusing is that a tenant
    supplies the issuer and can make it say whatever it likes.
    """
    connected = _add_organization(db_session, slug="connected")
    _add_connection(db_session, connected.id, issuer=ISSUER)
    _add_claim(db_session, connected.id, domain="connected.example")

    stub_exchange(monkeypatch, subject="a-subject", email="someone@connected.example", full_name="Someone")

    first = _sign_in(client, "connected", "code-one")
    account_id = uuid.UUID(first["user_id"])

    recorded = client.put(
        f"{API_ROOT}/auth/password",
        json={"email": SAME_DOMAIN_ADDRESS, "new_password": CHOSEN_PASSWORD},
    )
    assert recorded.status_code == 200, recorded.text

    stub_exchange(
        monkeypatch,
        subject="a-subject",
        email=SAME_DOMAIN_ADDRESS,
        full_name="Someone",
        email_verified=EmailAssertion.VERIFIED,
    )
    _sign_in(client, "connected", "code-two")

    account = _account(db_session, account_id)
    assert account.email == SAME_DOMAIN_ADDRESS
    assert account.email_proven_at is None
