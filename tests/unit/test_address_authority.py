"""What may record a sign-in address, and what that recording proves.

``user.email_proven_at`` is the column auto-join and the add-or-invite refusals
read to decide that the holder of an address controls it. Only an authority that
really proves the holder may set it.

Deployment authority does: the master key in a header, and the first-boot
operator claiming an unclaimed deployment. A dashboard session does not, because
a session says who is calling and nothing about which mailbox answers. An account
with no address yet is not evidence of deployment authority either, which is the
case these tests pin: the deployment operator is not the only identity that can
hold a NULL address.

Unit rather than integration: route and service behavior that runs unchanged on
the SQLite file each test stands up.
"""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.main import create_app
from gateway.models.tenancy import DashboardSession, User
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

MASTER_KEY = "sk-test-master"  # pragma: allowlist secret
OPERATOR_EMAIL = "operator@example.com"
OPERATOR_PASSWORD = "operator-password"  # pragma: allowlist secret
CLAIMED_ADDRESS = "someone@another-company.test"
NEW_PASSWORD = "a-chosen-password"  # pragma: allowlist secret
SESSION_TOKEN = "test-session-token"  # pragma: allowlist secret
CONNECTION_PROVIDER = "oidc:https://idp.another-company.test"
PROVED_ADDRESS = "member@another-company.test"


def _db_path(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'address-authority.db'}"


def _client(tmp_path: Path) -> TestClient:
    config = GatewayConfig(
        database_url=_db_path(tmp_path),
        master_key=MASTER_KEY,
        require_pricing=False,
    )
    return TestClient(create_app(config))


def _claim_the_deployment(client: TestClient) -> None:
    """Claim the deployment with the master key, so later tests are not first boot."""
    response = client.put(
        f"{API_ROOT}/auth/password",
        json={"email": OPERATOR_EMAIL, "new_password": OPERATOR_PASSWORD},
        headers={"Otari-Key": MASTER_KEY},
    )
    assert response.status_code == 200, response.text


def _account_without_an_address(tmp_path: Path) -> uuid.UUID:
    """Create an active account holding no address and no password, and a session for it.

    This is the state an account provisioned by an organization's own connection
    starts in. It is built directly rather than through a sign-in, because what is
    under test is what the password route does with such an account, not how one
    comes to exist.
    """
    engine = create_engine(_db_path(tmp_path))
    with sessionmaker(bind=engine)() as session:
        account = User(
            email=None,
            full_name="No Address Yet",
            hashed_password=None,
            is_active=True,
            oauth_provider=CONNECTION_PROVIDER,
        )
        session.add(account)
        session.flush()
        session.add(
            DashboardSession(
                token_hash=hash_session_token(SESSION_TOKEN),
                user_id=account.id,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )
        session.commit()
        return account.id


def _account(tmp_path: Path, account_id: uuid.UUID) -> User:
    engine = create_engine(_db_path(tmp_path))
    with sessionmaker(bind=engine)() as session:
        account = session.execute(select(User).where(col(User.id) == account_id)).scalars().one()
        session.expunge(account)
        return account


def test_a_session_cookie_claiming_an_address_does_not_prove_it(tmp_path: Path) -> None:
    """A caller holding only a session records an address nobody has proved.

    The address becomes usable, so ``email_verified_at`` is stamped, but
    ``email_proven_at`` stays NULL and auto-join and the add-or-invite refusals
    therefore refuse it.
    """
    with _client(tmp_path) as client:
        _claim_the_deployment(client)
        account_id = _account_without_an_address(tmp_path)
        client.cookies.set(SESSION_COOKIE_NAME, SESSION_TOKEN)

        response = client.put(
            f"{API_ROOT}/auth/password",
            json={"email": CLAIMED_ADDRESS, "new_password": NEW_PASSWORD},
        )

        assert response.status_code == 200, response.text
        account = _account(tmp_path, account_id)
        assert account.email == CLAIMED_ADDRESS
        assert account.email_verified_at is not None
        assert account.email_proven_at is None


def _proved_account_without_a_password(tmp_path: Path) -> uuid.UUID:
    """An account whose mailbox proved its address and that holds no password.

    The ordinary state of somebody who signs in through a provider or a passkey.
    """
    engine = create_engine(_db_path(tmp_path))
    now = datetime.now(UTC)
    with sessionmaker(bind=engine)() as session:
        account = User(
            email=PROVED_ADDRESS,
            full_name="Proved Member",
            hashed_password=None,
            is_active=True,
            oauth_provider=CONNECTION_PROVIDER,
            email_verified_at=now,
            email_proven_at=now,
        )
        session.add(account)
        session.flush()
        session.add(
            DashboardSession(
                token_hash=hash_session_token(SESSION_TOKEN),
                user_id=account.id,
                expires_at=now + timedelta(hours=1),
            )
        )
        session.commit()
        return account.id


def test_a_first_password_leaves_an_established_proof_alone(tmp_path: Path) -> None:
    """Setting a first password establishes nothing about the address, so it retires nothing.

    An authority that does not prove the holder retires an earlier proof where it
    supplies the address. Supplying none is not that case, and treating it as one
    would drop the proof of everyone who signs in with a provider and later adds
    a password.
    """
    with _client(tmp_path) as client:
        _claim_the_deployment(client)
        account_id = _proved_account_without_a_password(tmp_path)
        client.cookies.set(SESSION_COOKIE_NAME, SESSION_TOKEN)

        response = client.put(f"{API_ROOT}/auth/password", json={"new_password": NEW_PASSWORD})

        assert response.status_code == 200, response.text
        account = _account(tmp_path, account_id)
        assert account.email == PROVED_ADDRESS
        assert account.email_proven_at is not None


def test_the_first_claim_of_a_deployment_proves_the_address_without_the_master_key(tmp_path: Path) -> None:
    """The operator claiming an unclaimed deployment holds deployment authority through a session.

    Pins the first-claim half by itself: no master key is presented, so only that half
    can answer. This is the one path where a cookie-authenticated caller proves an address.
    """
    with _client(tmp_path) as client:
        response = client.post(f"{API_ROOT}/auth/session", json={"master_key": MASTER_KEY})
        assert response.status_code == 200, response.text

        response = client.put(
            f"{API_ROOT}/auth/password",
            json={"email": OPERATOR_EMAIL, "new_password": OPERATOR_PASSWORD},
        )

        assert response.status_code == 200, response.text
        engine = create_engine(_db_path(tmp_path))
        with sessionmaker(bind=engine)() as session:
            operator = session.execute(select(User).where(col(User.email) == OPERATOR_EMAIL)).scalars().one()
            assert operator.email_proven_at is not None


def test_the_master_key_claiming_an_address_proves_it(tmp_path: Path) -> None:
    """Deployment authority still proves the holder, which first boot depends on."""
    with _client(tmp_path) as client:
        _claim_the_deployment(client)

        engine = create_engine(_db_path(tmp_path))
        with sessionmaker(bind=engine)() as session:
            operator = session.execute(select(User).where(col(User.email) == OPERATOR_EMAIL)).scalars().one()

            assert operator.email_verified_at is not None
            assert operator.email_proven_at is not None
