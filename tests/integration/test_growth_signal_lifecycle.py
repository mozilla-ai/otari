"""What actually crosses ``GrowthSignalPort`` on a running app.

The port shipped fully wired and entirely uncalled (otari#796), which is a
failure nothing reports: the core adapter is a Null Object, so a build that
binds a real CRM gets a resolved port that is simply never reached and no test,
log line or type error says so. These tests close that by binding a recording
adapter through the container, the way an overlay binds its own, and asserting
on what the seam carried rather than on what the route answered.

The negatives carry as much as the positives. Signup's enumeration-safety is a
property of the *response*, and a signal fired on an address nobody has touched
would leak through the seam what the body refuses to say, so the untouched-address
case asserts silence. The second key asserts silence too: ``record_activation``
puts first-occurrence detection on the caller, and a milestone that fires on
every key is not a milestone.
"""

import logging
import re
import uuid
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.identity import EmailAssertion, ProviderIdentity
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.ports.growth_signal_port import GrowthActivationEvent, GrowthSignalPort
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token

PASSWORD = "a-real-password"  # pragma: allowlist secret
_KEYS = f"{API_ROOT}/organizations/me/keys"


class RecordingGrowthAdapter:
    """A stand-in for an overlay-bound adapter, recording what it was told.

    Records rather than asserts, so a test can name the one call it expects and
    still fail on an extra one it did not.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def record_signup(
        self,
        *,
        background_tasks: Any,
        user_id: uuid.UUID,
        email: str,
        full_name: str | None,
        created_at: datetime,
    ) -> None:
        self.calls.append(
            ("signup", {"user_id": user_id, "email": email, "full_name": full_name, "created_at": created_at})
        )

    async def record_activation(
        self,
        *,
        background_tasks: Any,
        event: GrowthActivationEvent,
        user_id: uuid.UUID,
        email: str,
    ) -> None:
        self.calls.append(("activation", {"event": event, "user_id": user_id, "email": email}))

    async def record_onboarding_completed(self, **kwargs: Any) -> None:
        self.calls.append(("onboarding", kwargs))

    async def record_profile_updated(self, **kwargs: Any) -> None:
        self.calls.append(("profile_updated", kwargs))

    async def record_account_deleted(self, **kwargs: Any) -> None:
        self.calls.append(("account_deleted", kwargs))

    def of(self, name: str) -> list[dict[str, Any]]:
        return [payload for recorded, payload in self.calls if recorded == name]


@pytest.fixture
def growth(client: TestClient) -> RecordingGrowthAdapter:
    """Rebind the port on the booted app, as ``OTARI_BOOTSTRAP`` would.

    Through the container and not ``dependency_overrides`` so the resolution
    path under test is the real one: ``get_growth_signal_port`` asking the
    container for whatever this build bound. The ``client`` fixture gives each
    test a container of its own, so nothing has to be unbound afterwards.
    """
    recorder = RecordingGrowthAdapter()
    container: Any = client.app.state.container  # type: ignore[attr-defined]
    container.bind(GrowthSignalPort, lambda session: recorder)
    return recorder


@pytest.fixture
def mail(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Signup refuses before writing anything unless the link can be mailed."""
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")


def _add_member(client: TestClient, master_key_header: dict[str, str], *, email: str) -> None:
    response = client.post(
        f"{API_ROOT}/organizations/me/members",
        json={"email": email, "role": "member"},
        headers=master_key_header,
    )
    assert response.status_code == status.HTTP_201_CREATED, response.text


def test_claiming_a_roster_identity_notifies_the_signup(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
    mail: None,
) -> None:
    """The whole record a vendor needs to create a contact crosses the seam."""
    _add_member(client, master_key_header, email="ada@example.com")

    response = client.post(
        f"{API_ROOT}/auth/signup",
        json={"email": "ada@example.com", "password": PASSWORD, "full_name": "Ada Lovelace"},
    )
    assert response.status_code == status.HTTP_200_OK, response.text

    signups = growth.of("signup")
    assert len(signups) == 1, growth.calls
    assert signups[0]["email"] == "ada@example.com"
    assert signups[0]["full_name"] == "Ada Lovelace"
    assert isinstance(signups[0]["user_id"], uuid.UUID)
    assert isinstance(signups[0]["created_at"], datetime)
    # ``SIGNED_UP`` overlaps ``record_signup`` and the port settles the overlap
    # in that method's favor: a signup is one notification, not two.
    assert growth.of("activation") == []


def test_signup_on_an_untouched_address_notifies_nothing(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    mail: None,
) -> None:
    """Enumeration-safety has to hold through the seam, not just in the body.

    The response is the same 200 either way (``test_signup_api.py``), so a
    signal here would tell an operator's vendor precisely what the route
    declines to tell the caller: that the address was on the roster.
    """
    response = client.post(
        f"{API_ROOT}/auth/signup",
        json={"email": "nobody@example.com", "password": PASSWORD},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    assert growth.calls == []


def test_a_second_signup_on_a_claimed_address_notifies_nothing(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
    mail: None,
) -> None:
    """A replayed signup is one of the enumeration-safe returns, so it is silent."""
    _add_member(client, master_key_header, email="grace@example.com")
    for _ in range(2):
        response = client.post(
            f"{API_ROOT}/auth/signup",
            json={"email": "grace@example.com", "password": PASSWORD},
        )
        assert response.status_code == status.HTTP_200_OK, response.text

    assert len(growth.of("signup")) == 1, growth.calls


def _sign_in_with_google(client: TestClient, monkeypatch: pytest.MonkeyPatch, *, email: str) -> None:
    async def _exchange(_config: GatewayConfig, provider: str, **_: Any) -> ProviderIdentity:
        return ProviderIdentity(
            provider=provider,
            subject=f"{email}-subject",
            email=email,
            full_name="Ada Lovelace",
            email_verified=EmailAssertion.VERIFIED,
        )

    monkeypatch.setattr("gateway.api.routes.auth_oauth.exchange_code", _exchange)
    response = client.post(f"{API_ROOT}/auth/oauth/google/callback", json={"code": "c", "state": "s"})
    assert response.status_code == status.HTTP_200_OK, response.text
    client.cookies.clear()


@pytest.fixture
def oauth(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(test_config, "public_base_url", "http://testserver")
    monkeypatch.setattr(test_config, "oauth_google_client_id", "google-id")
    monkeypatch.setattr(test_config, "oauth_google_client_secret", "google-secret")


def test_a_provider_registration_notifies_the_signup_and_a_later_sign_in_does_not(
    client: TestClient,
    test_config: GatewayConfig,
    growth: RecordingGrowthAdapter,
    oauth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "open_signup", True)

    _sign_in_with_google(client, monkeypatch, email="new@example.com")
    _sign_in_with_google(client, monkeypatch, email="new@example.com")

    signups = growth.of("signup")
    assert [signup["email"] for signup in signups] == ["new@example.com"], growth.calls
    assert signups[0]["full_name"] == "Ada Lovelace"


def test_a_provider_sign_in_that_claims_an_added_address_notifies_the_signup(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
    oauth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _add_member(client, master_key_header, email="ada@example.com")

    _sign_in_with_google(client, monkeypatch, email="ada@example.com")

    assert [signup["email"] for signup in growth.of("signup")] == ["ada@example.com"], growth.calls
    assert growth.of("signup")[0]["full_name"] == "Ada Lovelace"


def test_a_first_password_from_an_invitation_link_notifies_the_signup(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
) -> None:
    invited = client.post(
        f"{API_ROOT}/organizations/me/member-invitations",
        json={"email": "ida@example.com", "role": "member"},
        headers=master_key_header,
    )
    assert invited.status_code == status.HTTP_201_CREATED, invited.text
    token = invited.json()["accept_link"].split("token=")[1]

    accepted = client.post(
        f"{API_ROOT}/invitations/accept", json={"token": token, "password": PASSWORD, "full_name": "Ida Wells"}
    )

    assert accepted.status_code == status.HTTP_200_OK, accepted.text
    assert [signup["email"] for signup in growth.of("signup")] == ["ida@example.com"], growth.calls
    # The name the accept wrote, not the empty one the invitee was loaded with.
    assert growth.of("signup")[0]["full_name"] == "Ida Wells"


def _complete_a_reset(client: TestClient, caplog: pytest.LogCaptureFixture, *, email: str) -> None:
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    caplog.clear()
    try:
        requested = client.post(f"{API_ROOT}/auth/password/reset", json={"email": email})
    finally:
        gateway_logger.removeHandler(caplog.handler)
    assert requested.status_code == status.HTTP_200_OK, requested.text
    match = re.search(r"token=([\w-]+)", caplog.text)
    assert match, caplog.text
    confirmed = client.post(
        f"{API_ROOT}/auth/password/reset/confirm", json={"token": match.group(1), "new_password": PASSWORD}
    )
    assert confirmed.status_code == status.HTTP_204_NO_CONTENT, confirmed.text


def test_a_reset_that_claims_an_added_address_notifies_the_signup_and_a_later_reset_does_not(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
    mail: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _add_member(client, master_key_header, email="ada@example.com")

    _complete_a_reset(client, caplog, email="ada@example.com")
    _complete_a_reset(client, caplog, email="ada@example.com")

    assert [signup["email"] for signup in growth.of("signup")] == ["ada@example.com"], growth.calls


@pytest.fixture
def member(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> Generator[tuple[uuid.UUID, str]]:
    """A member of the bootstrapped organization, and their session cookie."""
    # One master-key call provisions the tenancy root, so the identity below
    # joins a real organization and workspace rather than inventing them.
    assert client.get(f"{API_ROOT}/organizations/me", headers=master_key_header).status_code == status.HTTP_200_OK

    session = db_session_factory()
    try:
        organization = session.execute(select(Organization)).scalars().first()
        assert organization is not None
        workspace = (
            session.execute(select(Workspace).where(col(Workspace.organization_id) == organization.id))
            .scalars()
            .first()
        )
        assert workspace is not None

        user = User(
            email="member@example.com",
            full_name="Member Example",
            active_organization_id=organization.id,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        session.add(
            OrganizationMember(
                organization_id=organization.id,
                user_id=user.id,
                role="member",
                status="active",
            )
        )
        session.add(
            WorkspaceMember(
                workspace_id=workspace.id,
                user_id=user.id,
                role="member",
                status="active",
            )
        )
        token = "otari-sess-member"
        session.add(
            DashboardSession(
                token_hash=hash_session_token(token),
                user_id=user.id,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(hours=12),
            )
        )
        session.commit()
        yield user.id, token
    finally:
        session.close()


def test_a_members_first_key_notifies_the_activation_and_the_second_does_not(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    member: tuple[uuid.UUID, str],
) -> None:
    """The milestone is derived from the owner's own keys, so it fires once."""
    user_id, cookie = member
    client.cookies.set(SESSION_COOKIE_NAME, cookie)

    first = client.post(_KEYS, json={"key_name": "first"})
    assert first.status_code == status.HTTP_200_OK, first.text

    activations = growth.of("activation")
    assert len(activations) == 1, growth.calls
    assert activations[0]["event"] is GrowthActivationEvent.API_KEY_CREATED
    assert activations[0]["user_id"] == user_id
    assert activations[0]["email"] == "member@example.com"

    second = client.post(_KEYS, json={"key_name": "second"})
    assert second.status_code == status.HTTP_200_OK, second.text
    assert len(growth.of("activation")) == 1, growth.calls


def test_an_operator_with_no_address_notifies_nothing(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
) -> None:
    """A header master key names nobody, and the bootstrap operator holds no address.

    ``record_activation`` takes an ``email`` because that is what a vendor keys
    a person on, and the operator identity deliberately has none
    (``services/tenancy/provisioning_service``), so this fires nothing rather
    than inventing a stand-in.
    """
    response = client.post(_KEYS, json={"key_name": "operator"}, headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    assert growth.calls == []
