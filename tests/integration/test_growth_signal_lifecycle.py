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
every key is not a milestone. ``record_action`` is the opposite contract: one
call per thing created, and none for an edit or a refused write.
"""

import logging
import re
import uuid
from collections.abc import Callable, Generator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from any_llm.types.model import Model
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlmodel import col

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.models.identity import EmailAssertion, ProviderIdentity
from gateway.models.tenancy import DashboardSession, Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.ports.growth_signal_port import GrowthAction, GrowthActivationEvent, GrowthSignalPort
from gateway.services import mcp_stateless
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.mcp_stateless import ServerCheck
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.password_service import hash_password
from gateway.services.secret_box import generate_secret_key
from gateway.services.tenancy.organization_service import OrganizationService

from .webauthn_helpers import SoftwareAuthenticator, challenge_of

PASSWORD = "a-real-password"  # pragma: allowlist secret
_KEYS = f"{API_ROOT}/organizations/me/keys"
_GUARDRAILS = f"{API_ROOT}/organizations/me/guardrails"
_INVITATIONS = f"{API_ROOT}/organizations/me/member-invitations"
_PROVIDER_KEYS = f"{API_ROOT}/organizations/me/provider-keys"
_WEB_SEARCH_KEYS = f"{API_ROOT}/organizations/me/web-search-keys"
_ROUTING_POLICIES = f"{API_ROOT}/organizations/me/routing-policies"
_BUDGETS = f"{API_ROOT}/organizations/me/budgets"
_MEMBER_EMAIL = "member@example.com"


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

    async def record_sign_in(
        self,
        *,
        background_tasks: Any,
        user_id: uuid.UUID,
        email: str,
        signed_in_at: datetime,
    ) -> None:
        self.calls.append(("sign_in", {"user_id": user_id, "email": email, "signed_in_at": signed_in_at}))

    async def record_activation(
        self,
        *,
        background_tasks: Any,
        event: GrowthActivationEvent,
        user_id: uuid.UUID,
        email: str,
    ) -> None:
        self.calls.append(("activation", {"event": event, "user_id": user_id, "email": email}))

    async def record_action(
        self,
        *,
        background_tasks: Any,
        action: GrowthAction,
        user_id: uuid.UUID,
        email: str | None,
        organization_id: uuid.UUID,
        details: Mapping[str, str] | None = None,
    ) -> None:
        self.calls.append(
            (
                "action",
                {
                    "action": action,
                    "user_id": user_id,
                    "email": email,
                    "organization_id": organization_id,
                    "details": details,
                },
            )
        )

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
    monkeypatch.setattr(test_config, "public_base_url", "https://testserver")
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
    than inventing a stand-in. Nor is the key an action: the header master key
    is the deployment's credential, not a person's.
    """
    response = client.post(_KEYS, json={"key_name": "operator"}, headers=master_key_header)
    assert response.status_code == status.HTTP_200_OK, response.text
    assert growth.calls == []


# -- sign-in ----------------------------------------------------------------


def _give_password(db_session_factory: Callable[[], Session], user_id: uuid.UUID) -> None:
    """A verified address and a password, which is what a password sign-in asks of an identity."""
    session = db_session_factory()
    try:
        user = session.get(User, user_id)
        assert user is not None
        user.hashed_password = hash_password(PASSWORD)
        user.email_verified_at = datetime.now(UTC)
        session.commit()
    finally:
        session.close()


def test_a_password_sign_in_notifies_the_sign_in(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    member: tuple[uuid.UUID, str],
    db_session_factory: Callable[[], Session],
) -> None:
    user_id, _cookie = member
    _give_password(db_session_factory, user_id)
    before = datetime.now(UTC)

    response = client.post(f"{API_ROOT}/auth/session", json={"email": _MEMBER_EMAIL, "password": PASSWORD})

    assert response.status_code == status.HTTP_200_OK, response.text
    assert [name for name, _ in growth.calls] == ["sign_in"], growth.calls
    [sign_in] = growth.of("sign_in")
    assert (sign_in["user_id"], sign_in["email"]) == (user_id, _MEMBER_EMAIL)
    assert sign_in["signed_in_at"] >= before


def test_a_refused_password_sign_in_notifies_nothing(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    member: tuple[uuid.UUID, str],
    db_session_factory: Callable[[], Session],
) -> None:
    _give_password(db_session_factory, member[0])

    response = client.post(f"{API_ROOT}/auth/session", json={"email": _MEMBER_EMAIL, "password": "not-the-password"})

    assert response.status_code == status.HTTP_401_UNAUTHORIZED, response.text
    assert growth.calls == []


def test_a_master_key_sign_in_notifies_nothing_even_for_an_operator_with_an_address(
    client: TestClient,
    test_config: GatewayConfig,
    growth: RecordingGrowthAdapter,
    db_session_factory: Callable[[], Session],
) -> None:
    """The master key is the deployment's credential, not a person's, so nobody signed in."""
    first = client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key})
    assert first.status_code == status.HTTP_200_OK, first.text
    session = db_session_factory()
    try:
        operator = session.get(User, uuid.UUID(first.json()["user_id"]))
        assert operator is not None
        operator.email = "operator@example.com"
        session.commit()
    finally:
        session.close()
    client.cookies.clear()

    again = client.post(f"{API_ROOT}/auth/session", json={"master_key": test_config.master_key})

    assert again.status_code == status.HTTP_200_OK, again.text
    assert growth.calls == []


def test_every_provider_sign_in_notifies_a_sign_in_beside_the_one_signup(
    client: TestClient,
    test_config: GatewayConfig,
    growth: RecordingGrowthAdapter,
    oauth: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registering sign-in is both: the account was made, and it was used."""
    monkeypatch.setattr(test_config, "open_signup", True)

    _sign_in_with_google(client, monkeypatch, email="new@example.com")
    _sign_in_with_google(client, monkeypatch, email="new@example.com")

    assert [name for name, _ in growth.calls] == ["signup", "sign_in", "sign_in"], growth.calls
    assert {sign_in["email"] for sign_in in growth.of("sign_in")} == {"new@example.com"}


def test_a_passkey_sign_in_notifies_the_sign_in(
    client: TestClient,
    test_config: GatewayConfig,
    growth: RecordingGrowthAdapter,
    member: tuple[uuid.UUID, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "public_base_url", "http://testserver")
    authenticator = SoftwareAuthenticator(rp_id="testserver", origin="http://testserver")
    user_id, cookie = member
    client.cookies.set(SESSION_COOKIE_NAME, cookie)
    options = client.post(f"{API_ROOT}/auth/webauthn/register/options")
    assert options.status_code == status.HTTP_200_OK, options.text
    registered = client.post(
        f"{API_ROOT}/auth/webauthn/register",
        json={"credential": authenticator.register(challenge_of(options.json()))},
    )
    assert registered.status_code == status.HTTP_201_CREATED, registered.text
    client.cookies.clear()

    options = client.post(f"{API_ROOT}/auth/webauthn/authenticate/options")
    assert options.status_code == status.HTTP_200_OK, options.text
    signed_in = client.post(
        f"{API_ROOT}/auth/webauthn/authenticate",
        json={"credential": authenticator.authenticate(challenge_of(options.json()))},
    )

    assert signed_in.status_code == status.HTTP_200_OK, signed_in.text
    assert [(sign_in["user_id"], sign_in["email"]) for sign_in in growth.of("sign_in")] == [(user_id, _MEMBER_EMAIL)]


# -- the first guardrail ----------------------------------------------------


def _make_admin(db_session_factory: Callable[[], Session], user_id: uuid.UUID) -> None:
    """Guardrails are mandated by an organization's owners and admins."""
    session = db_session_factory()
    try:
        membership = session.execute(
            select(OrganizationMember).where(col(OrganizationMember.user_id) == user_id)
        ).scalar_one()
        membership.role = "admin"
        session.commit()
    finally:
        session.close()


def test_an_organizations_first_guardrail_notifies_the_activation_and_the_second_does_not(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    member: tuple[uuid.UUID, str],
    db_session_factory: Callable[[], Session],
) -> None:
    user_id, cookie = member
    _make_admin(db_session_factory, user_id)
    client.cookies.set(SESSION_COOKIE_NAME, cookie)

    first = client.post(
        _GUARDRAILS, json={"profile": "prompt-injection", "mode": "block", "applies_to_all_workspaces": True}
    )
    assert first.status_code == status.HTTP_201_CREATED, first.text
    second = client.post(
        _GUARDRAILS, json={"profile": "toxicity", "mode": "monitor", "applies_to_all_workspaces": True}
    )
    assert second.status_code == status.HTTP_201_CREATED, second.text

    assert growth.of("activation") == [
        {"event": GrowthActivationEvent.FIRST_GUARDRAIL_CONFIGURED, "user_id": user_id, "email": _MEMBER_EMAIL}
    ], growth.calls


def test_a_guardrail_the_caller_may_not_mandate_notifies_nothing(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    member: tuple[uuid.UUID, str],
) -> None:
    """A plain member is refused before anything is read or written, so nothing was configured."""
    client.cookies.set(SESSION_COOKIE_NAME, member[1])

    response = client.post(
        _GUARDRAILS, json={"profile": "prompt-injection", "mode": "block", "applies_to_all_workspaces": True}
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN, response.text
    assert growth.calls == []


def test_a_guardrail_mandated_with_the_master_key_notifies_nothing(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
) -> None:
    response = client.post(
        _GUARDRAILS,
        json={"profile": "prompt-injection", "mode": "block", "applies_to_all_workspaces": True},
        headers=master_key_header,
    )

    assert response.status_code == status.HTTP_201_CREATED, response.text
    assert growth.calls == []


# Every action.
#
# One call per thing created, after its write commits, for the admin who made
# it, in their organization. An edit, a refused write and the deployment-wide
# twin routes report nothing.


@pytest.fixture
def admin(client: TestClient, member: tuple[uuid.UUID, str], db_session_factory: Callable[[], Session]) -> uuid.UUID:
    """The member, promoted to admin and signed in, since these routes are owners and admins only."""
    user_id, cookie = member
    _make_admin(db_session_factory, user_id)
    client.cookies.set(SESSION_COOKIE_NAME, cookie)
    return user_id


@pytest.fixture
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Credentials are encrypted at rest, so a write that stores one needs a key to encrypt with."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _actions(
    client: TestClient, growth: RecordingGrowthAdapter, user_id: uuid.UUID
) -> list[tuple[GrowthAction, Mapping[str, str] | None]]:
    """Each reported action with its details, after checking every one names the admin and their organization."""
    organization = client.get(f"{API_ROOT}/organizations/me")
    assert organization.status_code == status.HTTP_200_OK, organization.text
    organization_id = uuid.UUID(organization.json()["organization"]["id"])
    actions = growth.of("action")
    for action in actions:
        assert (action["user_id"], action["email"], action["organization_id"]) == (
            user_id,
            _MEMBER_EMAIL,
            organization_id,
        ), action
    return [(action["action"], action["details"]) for action in actions]


def _workspace_of(client: TestClient) -> str:
    listed = client.get(f"{API_ROOT}/workspaces")
    assert listed.status_code == status.HTTP_200_OK, listed.text
    workspace_id: str = listed.json()["data"][0]["id"]
    return workspace_id


def test_every_key_is_an_action(client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID) -> None:
    for name in ("first", "second"):
        response = client.post(_KEYS, json={"key_name": name})
        assert response.status_code == status.HTTP_200_OK, response.text

    assert _actions(client, growth, admin) == [(GrowthAction.API_KEY_CREATED, None)] * 2


def test_every_guardrail_is_an_action_and_a_refused_one_is_not(
    client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID
) -> None:
    body = {"profile": "prompt-injection", "mode": "block", "applies_to_all_workspaces": True}
    assert client.post(_GUARDRAILS, json=body).status_code == status.HTTP_201_CREATED
    assert client.post(_GUARDRAILS, json={**body, "profile": "toxicity"}).status_code == status.HTTP_201_CREATED
    # The same profile again is refused.
    assert client.post(_GUARDRAILS, json=body).status_code == status.HTTP_409_CONFLICT

    assert _actions(client, growth, admin) == [(GrowthAction.GUARDRAIL_CREATED, None)] * 2


def test_every_invitation_is_an_action_carrying_its_role(
    client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID
) -> None:
    single = client.post(_INVITATIONS, json={"email": "ada@example.com", "role": "admin"})
    assert single.status_code == status.HTTP_201_CREATED, single.text
    bulk = client.post(
        f"{_INVITATIONS}/bulk",
        # The member is refused, as already in the organization, so two of three are invited.
        json={"emails": ["grace@example.com", _MEMBER_EMAIL, "ida@example.com"], "role": "member"},
    )
    assert bulk.status_code == status.HTTP_200_OK, bulk.text
    assert len(bulk.json()["invited"]) == 2, bulk.text
    # Pending already, so refused.
    again = client.post(_INVITATIONS, json={"email": "ada@example.com", "role": "admin"})
    assert again.status_code == status.HTTP_409_CONFLICT, again.text

    assert _actions(client, growth, admin) == [
        (GrowthAction.MEMBER_INVITED, {"role": "admin"}),
        (GrowthAction.MEMBER_INVITED, {"role": "member"}),
        (GrowthAction.MEMBER_INVITED, {"role": "member"}),
    ]


def test_a_bulk_invitation_that_invites_nobody_is_no_action(
    client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID
) -> None:
    bulk = client.post(f"{_INVITATIONS}/bulk", json={"emails": [_MEMBER_EMAIL], "role": "member"})
    assert bulk.status_code == status.HTTP_200_OK, bulk.text
    assert bulk.json()["invited"] == [], bulk.text

    assert growth.calls == []


def test_every_provider_key_is_an_action_carrying_its_provider(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    admin: uuid.UUID,
    secret_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def listed(impl_name: str, **_: object) -> ProviderDiscovery:
        # Answered here rather than dialed: a new key lists what its credential reaches.
        model = Model(id="a-model", object="model", created=0, owned_by=impl_name)
        return ProviderDiscovery(provider=impl_name, models=[model], error=None)

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", listed)

    for provider, name in (("openai", "primary"), ("anthropic", "primary")):
        response = client.post(_PROVIDER_KEYS, json={"provider": provider, "name": name, "api_key": "sk-test-1"})
        assert response.status_code == status.HTTP_201_CREATED, response.text
    created = response.json()
    duplicate = client.post(_PROVIDER_KEYS, json={"provider": "anthropic", "name": "primary", "api_key": "sk-test-2"})
    assert duplicate.status_code == status.HTTP_409_CONFLICT, duplicate.text
    renamed = client.patch(f"{_PROVIDER_KEYS}/{created['id']}", json={"name": "renamed"})
    assert renamed.status_code == status.HTTP_200_OK, renamed.text

    assert _actions(client, growth, admin) == [
        (GrowthAction.PROVIDER_KEY_ADDED, {"provider": "openai"}),
        (GrowthAction.PROVIDER_KEY_ADDED, {"provider": "anthropic"}),
    ]


def test_every_search_or_fetch_key_is_an_action_carrying_its_provider_and_kind(
    client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID, secret_key: None
) -> None:
    search = client.post(_WEB_SEARCH_KEYS, json={"provider": "tavily", "name": "ours", "api_key": "tvly-1"})
    assert search.status_code == status.HTTP_201_CREATED, search.text
    fetch = client.post(
        _WEB_SEARCH_KEYS, json={"provider": "exa", "name": "pages", "api_key": "exa-1", "kind": "fetch"}
    )
    assert fetch.status_code == status.HTTP_201_CREATED, fetch.text
    unknown = client.post(_WEB_SEARCH_KEYS, json={"provider": "google", "name": "g", "api_key": "k"})
    assert unknown.status_code == status.HTTP_400_BAD_REQUEST, unknown.text
    renamed = client.patch(f"{_WEB_SEARCH_KEYS}/{search.json()['id']}", json={"name": "renamed"})
    assert renamed.status_code == status.HTTP_200_OK, renamed.text

    assert _actions(client, growth, admin) == [
        (GrowthAction.WEB_SEARCH_KEY_ADDED, {"provider": "tavily", "kind": "web_search"}),
        (GrowthAction.WEB_SEARCH_KEY_ADDED, {"provider": "exa", "kind": "web_fetch"}),
    ]


def test_every_mcp_server_is_an_action(
    client: TestClient,
    growth: RecordingGrowthAdapter,
    admin: uuid.UUID,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def reachable(_server: object) -> ServerCheck:
        # The connection check would otherwise dial the address stored below.
        return ServerCheck(tool_names=[])

    monkeypatch.setattr(mcp_stateless, "check_stored_server", reachable)
    servers = f"{API_ROOT}/workspaces/{_workspace_of(client)}/mcp-servers"

    first = client.post(servers, json={"name": "github", "url": "https://93.184.216.34/mcp"})
    assert first.status_code == status.HTTP_201_CREATED, first.text
    second = client.post(servers, json={"name": "linear", "url": "https://93.184.216.35/mcp"})
    assert second.status_code == status.HTTP_201_CREATED, second.text
    duplicate = client.post(servers, json={"name": "github", "url": "https://93.184.216.36/mcp"})
    assert duplicate.status_code == status.HTTP_409_CONFLICT, duplicate.text
    edited = client.patch(f"{servers}/{first.json()['id']}", json={"enabled": False})
    assert edited.status_code == status.HTTP_200_OK, edited.text

    assert _actions(client, growth, admin) == [(GrowthAction.MCP_SERVER_CREATED, None)] * 2


def test_every_new_routing_policy_is_an_action_and_the_first_a_milestone(
    client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID
) -> None:
    workspace_id = _workspace_of(client)

    def write(name: str, target: str, **extra: str) -> None:
        body = {"name": name, "spec": {"select": [{"default": target}]}, "workspace_id": workspace_id, **extra}
        response = client.post(_ROUTING_POLICIES, json=body)
        assert response.status_code == status.HTTP_200_OK, response.text

    write("fast", "openai:gpt-4o-mini")
    write("fast", "openai:gpt-4o")
    write("quick", "openai:gpt-4o", rename_from="fast")
    write("careful", "anthropic:claude-3-5-haiku-latest")
    # Renaming a policy that is not stored is refused, and creates nothing.
    missing = client.post(
        _ROUTING_POLICIES,
        json={
            "name": "other",
            "rename_from": "nothing-by-this-name",
            "spec": {"select": [{"default": "openai:gpt-4o"}]},
            "workspace_id": workspace_id,
        },
    )
    assert missing.status_code == status.HTTP_404_NOT_FOUND, missing.text

    assert _actions(client, growth, admin) == [(GrowthAction.ROUTING_POLICY_CREATED, None)] * 2
    assert growth.of("activation") == [
        {"event": GrowthActivationEvent.FIRST_ROUTE_CONFIGURED, "user_id": admin, "email": _MEMBER_EMAIL}
    ], growth.calls


def test_a_deployment_wide_routing_policy_reports_nothing(
    client: TestClient, master_key_header: dict[str, str], growth: RecordingGrowthAdapter
) -> None:
    response = client.post(
        f"{API_ROOT}/routing/policies",
        json={"name": "fast", "spec": {"select": [{"default": "openai:gpt-4o-mini"}]}},
        headers=master_key_header,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    assert growth.calls == []


def test_every_budget_is_an_action_and_the_first_a_milestone(
    client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID
) -> None:
    body = {"max_budget": 250.0, "reset_alignment": "calendar_month"}
    created = client.post(_BUDGETS, json={**body, "name": "Engineering monthly"})
    assert created.status_code == status.HTTP_201_CREATED, created.text
    second = client.post(_BUDGETS, json={**body, "name": "Research monthly"})
    assert second.status_code == status.HTTP_201_CREATED, second.text
    raised = client.patch(f"{_BUDGETS}/{created.json()['budget_id']}", json={"max_budget": 500.0})
    assert raised.status_code == status.HTTP_200_OK, raised.text
    malformed = client.post(_BUDGETS, json={**body, "name": "Broken", "max_budget": "a lot"})
    assert malformed.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, malformed.text

    assert _actions(client, growth, admin) == [(GrowthAction.BUDGET_CREATED, None)] * 2
    assert growth.of("activation") == [
        {"event": GrowthActivationEvent.BUDGET_RULE_SET, "user_id": admin, "email": _MEMBER_EMAIL}
    ], growth.calls


def test_a_deployment_wide_budget_reports_nothing(
    client: TestClient, master_key_header: dict[str, str], growth: RecordingGrowthAdapter
) -> None:
    response = client.post(f"{API_ROOT}/budgets", json={"max_budget": 100.0}, headers=master_key_header)

    assert response.status_code == status.HTTP_200_OK, response.text
    assert growth.calls == []


def test_the_setup_guides_first_key_is_an_action_and_its_rotation_is_not(
    client: TestClient, growth: RecordingGrowthAdapter, admin: uuid.UUID
) -> None:
    issue = f"{API_ROOT}/workspaces/{_workspace_of(client)}/activation/key"

    first = client.post(issue)
    assert first.status_code == status.HTTP_201_CREATED, first.text
    rotated = client.post(issue)
    assert rotated.status_code == status.HTTP_201_CREATED, rotated.text
    assert rotated.json()["key_id"] == first.json()["key_id"]

    assert _actions(client, growth, admin) == [(GrowthAction.API_KEY_CREATED, None)]


def test_the_header_master_key_reports_no_action_on_any_tenant_route(
    client: TestClient,
    master_key_header: dict[str, str],
    growth: RecordingGrowthAdapter,
    secret_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment's credential, not a person's, even on the routes a person uses."""

    async def listed(impl_name: str, **_: object) -> ProviderDiscovery:
        model = Model(id="a-model", object="model", created=0, owned_by=impl_name)
        return ProviderDiscovery(provider=impl_name, models=[model], error=None)

    async def reachable(_server: object) -> ServerCheck:
        return ServerCheck(tool_names=[])

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", listed)
    monkeypatch.setattr(mcp_stateless, "check_stored_server", reachable)
    listed_workspaces = client.get(f"{API_ROOT}/workspaces", headers=master_key_header)
    assert listed_workspaces.status_code == status.HTTP_200_OK, listed_workspaces.text
    workspace_id = listed_workspaces.json()["data"][0]["id"]

    writes = [
        (_INVITATIONS, {"email": "ada@example.com", "role": "member"}),
        (f"{_INVITATIONS}/bulk", {"emails": ["grace@example.com"], "role": "member"}),
        (_PROVIDER_KEYS, {"provider": "openai", "name": "primary", "api_key": "sk-test-1"}),
        (_WEB_SEARCH_KEYS, {"provider": "tavily", "name": "ours", "api_key": "tvly-1"}),
        (f"{API_ROOT}/workspaces/{workspace_id}/mcp-servers", {"name": "github", "url": "https://93.184.216.34/mcp"}),
        (
            _ROUTING_POLICIES,
            {"name": "fast", "spec": {"select": [{"default": "openai:gpt-4o-mini"}]}, "workspace_id": workspace_id},
        ),
        (_BUDGETS, {"name": "Engineering monthly", "max_budget": 250.0, "reset_alignment": "calendar_month"}),
        (f"{API_ROOT}/workspaces/{workspace_id}/activation/key", None),
    ]
    for path, body in writes:
        response = client.post(path, json=body, headers=master_key_header)
        assert response.status_code < status.HTTP_300_MULTIPLE_CHOICES, (path, response.text)

    assert growth.calls == []


def test_a_report_comes_after_what_the_write_does_next(
    client: TestClient,
    admin: uuid.UUID,
    secret_key: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter that raises cannot cost a write its models, its health check or its emails.

    The port promises never to raise, so this is defense in depth: each report
    is the last thing its write does.
    """
    done: list[str] = []

    class RaisingGrowthAdapter(RecordingGrowthAdapter):
        async def record_action(self, **kwargs: Any) -> None:
            done.append(f"reported {kwargs['action']}")
            raise RuntimeError("the vendor is down")

    container: Any = client.app.state.container  # type: ignore[attr-defined]
    container.bind(GrowthSignalPort, lambda session: RaisingGrowthAdapter())

    async def listed(impl_name: str, **_: object) -> ProviderDiscovery:
        done.append("models listed")
        model = Model(id="a-model", object="model", created=0, owned_by=impl_name)
        return ProviderDiscovery(provider=impl_name, models=[model], error=None)

    async def checked(_server: object) -> ServerCheck:
        done.append("server checked")
        return ServerCheck(tool_names=[])

    mail_invitation = OrganizationService._mail_invitation

    async def mailed(self: OrganizationService, **kwargs: Any) -> Any:
        done.append("invitation mailed")
        return await mail_invitation(self, **kwargs)

    monkeypatch.setattr("gateway.services.providers._org_provider_model_service.test_provider_credentials", listed)
    monkeypatch.setattr(mcp_stateless, "check_stored_server", checked)
    monkeypatch.setattr(OrganizationService, "_mail_invitation", mailed)

    writes = [
        (_PROVIDER_KEYS, {"provider": "openai", "name": "primary", "api_key": "sk-test-1"}),
        (
            f"{API_ROOT}/workspaces/{_workspace_of(client)}/mcp-servers",
            {"name": "github", "url": "https://93.184.216.34/mcp"},
        ),
        (_INVITATIONS, {"email": "ada@example.com", "role": "member"}),
        (f"{_INVITATIONS}/bulk", {"emails": ["grace@example.com", "ida@example.com"], "role": "member"}),
    ]
    for path, body in writes:
        with pytest.raises(RuntimeError, match="the vendor is down"):
            client.post(path, json=body)

    assert done == [
        "models listed",
        "reported provider_key_added",
        "server checked",
        "reported mcp_server_created",
        "invitation mailed",
        "reported member_invited",
        "invitation mailed",
        "invitation mailed",
        "reported member_invited",
    ]
