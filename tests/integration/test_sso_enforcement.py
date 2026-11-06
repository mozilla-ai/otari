"""An organization's SSO enforcement: its state, the break-glass preconditions, and who it blocks.

``sso_blocks_membership`` is pure and needs no database; its cases are kept here rather than in
``tests/unit`` so the whole layer this task adds reads as one file.
"""

import functools
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.identity_provider_adapter import DeploymentIdentityProviderAdapter
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OIDCConnectionEnforcedError,
    OIDCConnectionNotFoundError,
    OIDCConnectionUnprovenError,
    PasswordLinkUnavailableError,
    SsoBreakGlassRequiredError,
)
from gateway.exceptions.organizations_exceptions import InvitationPasswordNotAcceptedError, NotAuthorizedError
from gateway.models.identity import EmailAssertion, OAuthIdentityLink, OIDCConnection, oidc_identity_provider
from gateway.models.tenancy import DashboardSession, InviteOrganizationMemberRequest, Organization, User
from gateway.repositories.identity import IdentityRepositories
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.repositories.tenancy.organization_domain_repository import OrganizationDomainRepository
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.identity import IdentityService, OIDCConnectionService
from gateway.services.mail import Mailer
from gateway.services.organizations import OrganizationClaimedDomains
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tenancy import (
    DomainAutoJoinListener,
    OrganizationService,
    VerifiedDomainLookup,
    forget_unproven_access,
    hash_token,
    sso_blocks_membership,
)

from .tenancy_helpers import (
    NO_BLOCKED_ADDRESSES,
    NO_MAIL_LIMIT,
    NO_TERMS,
    create_member,
    create_organization,
    membership_writes,
)

pytestmark = pytest.mark.asyncio

_ISSUER = "https://sso.acme.example"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storing a connection's client secret needs a key configured."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _connection_service(db: AsyncSession) -> OIDCConnectionService:
    uow = UnitOfWork(db)
    return OIDCConnectionService(
        uow,
        IdentityRepositories.on(uow),
        organizations=OrganizationService(db, membership_listener=None),
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(uow)),
        allow_private_hosts=False,
    )


async def _add_connection(
    db: AsyncSession, organization_id: uuid.UUID, *, first_sign_in_at: datetime | None
) -> OIDCConnection:
    connection = OIDCConnection(
        organization_id=organization_id,
        issuer=_ISSUER,
        resolved_issuer=_ISSUER,
        client_id="a-client-id",
        encrypted_client_secret=encrypt_secret("a-client-secret"),
        client_secret_last4="cret",
        display_name="Acme SSO",
        discovery_document={"issuer": _ISSUER},
        discovered_at=datetime.now(UTC),
        first_sign_in_at=first_sign_in_at,
        enabled=True,
    )
    db.add(connection)
    await db.flush()
    await db.refresh(connection)
    return connection


async def _member_with_email(
    db: AsyncSession, organization: Organization, *, role: str, email: str, full_name: str
) -> User:
    user = await UserRepository(db).create_local_identity(
        email=email, full_name=full_name, active_organization_id=organization.id
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role=role
    )
    return user


async def _reloaded(db: AsyncSession, organization: Organization) -> Organization:
    """Reload every column of ``organization`` now, inside this ``await``.

    NOTE: A write under test may have rolled the session back, which expires every attribute on
    ``organization``. A plain attribute access after that reloads lazily outside any ``await``,
    which asyncio SQLAlchemy cannot do.
    """
    await db.refresh(organization)
    return organization


async def _link_member(db: AsyncSession, member: User, connection: OIDCConnection) -> None:
    """Give ``member`` a sign-in link to ``connection``, as a completed SSO sign-in would."""
    db.add(
        OAuthIdentityLink(
            provider=oidc_identity_provider(connection.resolved_issuer),
            subject=f"subject-{member.id}",
            user_id=member.id,
            email_at_link=member.email,
            email_assertion_at_link=EmailAssertion.VERIFIED,
        )
    )
    await db.flush()


async def _dashboard_session(db: AsyncSession, user: User) -> str:
    """Give ``user`` a live dashboard session, and return its cookie token."""
    token = f"otari-sess-{user.id}"
    db.add(
        DashboardSession(
            token_hash=hash_session_token(token),
            user_id=user.id,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
    )
    await db.flush()
    return token


def _identity_service(db: AsyncSession) -> IdentityService:
    """Build the identity service as a request builds it.

    Every sign-in this file exercises resolves through an account already created, so the
    identity-provider policy it is built with is never consulted.
    """
    uow = UnitOfWork(db)
    organizations = OrganizationService(db, membership_listener=None, uow=uow)
    policy = DeploymentIdentityProviderAdapter(
        UserRepository(db), organizations, functools.partial(forget_unproven_access, db), open_signup=False
    )
    return IdentityService(
        uow,
        IdentityRepositories.on(uow),
        identity_provider=policy,
        domain_claims=VerifiedDomainLookup(OrganizationDomainRepository(uow)),
        organizations=organizations,
        sign_in_listener=DomainAutoJoinListener(OrganizationDomainRepository(uow), OrganizationMemberRepository(uow)),
        session_ttl_hours=1,
        terms_url=None,
        mailer=Mailer(GatewayConfig(mail_transport="none")),
        open_signup=False,
        password_reset_expiry_hours=2,
    )


async def _live_session_count(db: AsyncSession, user_id: uuid.UUID) -> int:
    result = await db.execute(
        select(func.count()).select_from(DashboardSession).where(DashboardSession.user_id == user_id)
    )
    return result.scalar_one()


# =============================================================================
# The predicate
# =============================================================================


async def test_enforcement_blocks_a_member_with_no_link_for_this_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset()) is True


async def test_enforcement_does_not_block_the_break_glass_owner(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    organization.sso_required = True
    organization.sso_break_glass_user_id = owner.id
    async_db.add(organization)
    await async_db.flush()

    assert sso_blocks_membership(organization=organization, user=owner, linked_providers=frozenset()) is False


async def test_enforcement_does_not_block_the_master_key_operator(async_db: AsyncSession) -> None:
    """The deployment's operator reaches an enforcing organization; an ordinary member with no link does not."""
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    operator = await create_member(async_db, organization, role="member", full_name="The Operator")
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert sso_blocks_membership(organization=organization, user=operator, linked_providers=frozenset()) is False
    assert sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset()) is True


async def test_enforcement_does_not_block_a_member_linked_to_this_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert (
        sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset({"oidc:acme"}))
        is False
    )


async def test_enforcement_blocks_nobody_while_turned_off(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert sso_blocks_membership(organization=organization, user=member, linked_providers=frozenset()) is False


# =============================================================================
# Setting enforcement
# =============================================================================


async def test_enforcement_cannot_be_set_before_one_sign_in_has_succeeded(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=None)
    await async_db.commit()

    with pytest.raises(OIDCConnectionUnprovenError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
        )
    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_enforcement_is_permitted_once_the_connection_has_signed_somebody_in(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert (await _reloaded(async_db, organization)).sso_required is True


async def test_enforcement_cannot_be_set_without_an_owner_exception_or_an_explicit_waiver(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    with pytest.raises(SsoBreakGlassRequiredError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=None, break_glass_waived=False
        )
    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_enforcement_is_permitted_with_an_explicit_waiver(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert (await _reloaded(async_db, organization)).sso_required is True


async def test_enforcement_is_permitted_with_a_break_glass_owner_exception(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )

    stored = await _reloaded(async_db, organization)
    assert stored.sso_required is True
    assert stored.sso_break_glass_user_id == owner.id


async def test_enforcement_refuses_a_break_glass_account_that_is_not_an_owner(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    with pytest.raises(SsoBreakGlassRequiredError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=member.id, break_glass_waived=False
        )
    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_enforcement_cannot_be_set_for_an_organization_with_no_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )

    with pytest.raises(OIDCConnectionNotFoundError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
        )


async def test_disabling_enforcement_needs_no_precondition(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=None)

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=False, break_glass_user_id=None, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_disabling_enforcement_succeeds_once_its_connection_is_gone(async_db: AsyncSession) -> None:
    """An organization must always have a way to turn enforcement off, even with no connection left."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()
    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )
    uow = UnitOfWork(async_db)
    async with uow:
        await IdentityRepositories.on(uow).connections.delete(connection)

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=False, break_glass_user_id=None, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is False


@pytest.mark.parametrize("role", ["member", "viewer"])
async def test_enforcement_cannot_be_set_by_a_caller_who_cannot_manage_the_organization(
    async_db: AsyncSession, role: str
) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    caller = await _member_with_email(async_db, organization, role=role, email="mo@acme.example", full_name="Mo Member")

    with pytest.raises(NotAuthorizedError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=caller, required=True, break_glass_user_id=None, break_glass_waived=True
        )
    assert (await _reloaded(async_db, organization)).sso_required is False


# =============================================================================
# Deleting the connection
# =============================================================================


async def test_deleting_an_enforced_connection_is_refused(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()
    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )

    with pytest.raises(OIDCConnectionEnforcedError):
        await _connection_service(async_db).delete_for_user(user=owner)

    await async_db.refresh(owner)
    stored = await _connection_service(async_db).get_for_user(user=owner)
    assert stored is not None
    assert stored.id == connection.id
    assert (await _reloaded(async_db, organization)).sso_required is True


async def test_disabling_enforcement_recovers_the_lockout_a_deleted_connection_would_leave(
    client: TestClient, async_db: AsyncSession
) -> None:
    """Enforcement blocks every member once the connection that admitted them is deleted.

    Turning it off must still succeed and still reach them, with no connection required to do it.
    """
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    other = await create_organization(async_db, slug="other")
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=member.id, role="member"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()
    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )
    uow = UnitOfWork(async_db)
    async with uow:
        await IdentityRepositories.on(uow).connections.delete(connection)
    member.active_organization_id = other.id
    async_db.add(member)
    token = await _dashboard_session(async_db, member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)
    blocked = client.post(f"{API_ROOT}/organizations/me/switch", json={"organization_id": str(organization.id)})
    assert blocked.status_code == 404, blocked.text

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=False, break_glass_user_id=None, break_glass_waived=False
    )

    recovered = client.post(f"{API_ROOT}/organizations/me/switch", json={"organization_id": str(organization.id)})
    assert recovered.status_code == 200, recovered.text
    assert recovered.json()["organization"]["id"] == str(organization.id)


# =============================================================================
# Signing in
# =============================================================================


async def test_a_blocked_member_signs_in_and_sees_only_their_other_organization(async_db: AsyncSession) -> None:
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    other = await create_organization(async_db, slug="other")
    member = await _member_with_email(
        async_db, enforcing, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=member.id, role="member"
    )
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(member)

    assert signed_in.active_organization_id == other.id


async def test_repairing_the_pointer_moves_a_refused_member_off_that_organization(
    async_db: AsyncSession,
) -> None:
    """The context read repairs the pointer, so it must not leave somebody pointed at an organization refusing them.

    The pointer starts on the enforcing organization with a live membership, which is what an
    unenforced repair returns unchanged.
    """
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    other = await create_organization(async_db, slug="other")
    member = await _member_with_email(
        async_db, enforcing, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=member.id, role="member"
    )
    await async_db.commit()
    await _reloaded(async_db, enforcing)
    await _reloaded(async_db, other)

    context = await OrganizationService(async_db, membership_listener=None).get_active_membership_context_for_user(
        member, sso_enforcement=_connection_service(async_db)
    )

    assert context.organization.id == other.id
    repaired = await async_db.get(User, member.id)
    assert repaired is not None
    assert repaired.active_organization_id == other.id


async def test_a_password_sign_in_reaches_no_enforcing_organization(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(member)

    assert signed_in.active_organization_id is None
    assert signed_in.session_token


async def test_a_passkey_sign_in_reaches_no_enforcing_organization(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="passkey@acme.example", full_name="Pat Passkey"
    )
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(member)

    assert signed_in.active_organization_id is None
    assert signed_in.session_token


async def test_a_member_linked_through_the_connection_reaches_the_enforcing_organization(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    linked_member = await _member_with_email(
        async_db, organization, role="member", email="li@acme.example", full_name="Li Linked"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await _link_member(async_db, linked_member, connection)
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(linked_member)

    assert signed_in.active_organization_id == organization.id


async def test_the_deployment_operator_reaches_an_enforcing_organization(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    operator = await create_member(async_db, organization, role="member", full_name="The Operator")
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(operator)

    assert signed_in.active_organization_id == organization.id


async def test_one_tenant_enforcement_does_not_lock_a_person_out_of_another(async_db: AsyncSession) -> None:
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    await _member_with_email(async_db, enforcing, role="member", email="mo@acme.example", full_name="Mo Member")
    elsewhere = await create_organization(async_db, slug="other")
    bystander = await _member_with_email(
        async_db, elsewhere, role="member", email="by@other.example", full_name="By Stander"
    )
    bystander.hashed_password = "not-a-real-hash"
    bystander.email_verified_at = datetime.now(UTC)
    async_db.add(bystander)
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(bystander)
    assert signed_in.active_organization_id == elsewhere.id

    link = await _identity_service(async_db).issue_password_link(bystander.id)
    assert link.link is not None


# =============================================================================
# Minting a credential outside the connection
# =============================================================================


async def test_accept_invitation_with_a_password_is_refused_for_an_enforcing_organization(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    service = OrganizationService(async_db, **membership_writes(async_db))
    invitation = await service.invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email="newcomer@acme.example"),
        config=GatewayConfig(),
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )
    assert invitation.accept_link is not None
    token = invitation.accept_link.split("token=")[1]
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.commit()

    with pytest.raises(InvitationPasswordNotAcceptedError):
        await service.accept_invitation(
            token, claims=NO_TERMS, blocked_addresses=NO_BLOCKED_ADDRESSES, password="correct-horse-battery-staple"
        )


async def test_a_password_link_is_refused_for_a_member_of_an_enforcing_organization(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    member.hashed_password = "not-a-real-hash"
    member.email_verified_at = datetime.now(UTC)
    async_db.add(member)
    await async_db.commit()

    with pytest.raises(PasswordLinkUnavailableError):
        await _identity_service(async_db).issue_password_link(member.id)


async def test_setting_a_dashboard_password_is_refused_for_a_member_of_an_enforcing_organization(
    client: TestClient, async_db: AsyncSession
) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    token = await _dashboard_session(async_db, member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.put(f"{API_ROOT}/auth/password", json={"new_password": "correct-horse-battery-staple"})

    assert response.status_code == 409, response.text
    await async_db.refresh(member)
    assert member.hashed_password is None


async def test_requesting_a_password_reset_cannot_be_told_apart_from_an_unknown_address(
    client: TestClient,
    async_db: AsyncSession,
    test_config: GatewayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "mail_transport", "console")
    monkeypatch.setattr(test_config, "public_base_url", "https://otari.example.com")
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    member.hashed_password = "not-a-real-hash"
    member.email_verified_at = datetime.now(UTC)
    async_db.add(member)
    await async_db.commit()

    blocked = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "mo@acme.example"})
    unknown = client.post(f"{API_ROOT}/auth/password/reset", json={"email": "nobody@acme.example"})

    assert blocked.status_code == unknown.status_code == 200, (blocked.text, unknown.text)
    assert blocked.json() == unknown.json()
    await async_db.refresh(member)
    assert member.password_reset_token_hash is None


async def test_resetting_a_password_is_refused_for_an_account_an_enforcing_organization_blocks(
    client: TestClient, async_db: AsyncSession
) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    member.hashed_password = "not-a-real-hash"
    member.email_verified_at = datetime.now(UTC)
    token = "a-reset-token"
    member.password_reset_token_hash = hash_token(token)
    member.password_reset_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
    async_db.add(member)
    await async_db.commit()

    response = client.post(
        f"{API_ROOT}/auth/password/reset/confirm", json={"token": token, "new_password": "a-new-password"}
    )

    assert response.status_code == 400, response.text
    await async_db.refresh(member)
    assert member.hashed_password == "not-a-real-hash"


# =============================================================================
# Listing memberships under enforcement
# =============================================================================


async def test_memberships_listing_hides_an_organization_that_blocks_the_caller(
    client: TestClient, async_db: AsyncSession
) -> None:
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    other = await create_organization(async_db, slug="other")
    member = await _member_with_email(
        async_db, enforcing, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=member.id, role="member"
    )
    member.active_organization_id = other.id
    async_db.add(member)
    token = await _dashboard_session(async_db, member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.get(f"{API_ROOT}/organizations/me/memberships")

    assert response.status_code == 200, response.text
    slugs = {row["organization"]["slug"] for row in response.json()["data"]}
    assert slugs == {"other"}


async def test_memberships_listing_includes_an_organization_the_caller_is_linked_into(
    client: TestClient, async_db: AsyncSession
) -> None:
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    linked_member = await _member_with_email(
        async_db, enforcing, role="member", email="li@acme.example", full_name="Li Linked"
    )
    connection = await _add_connection(async_db, enforcing.id, first_sign_in_at=datetime.now(UTC))
    await _link_member(async_db, linked_member, connection)
    token = await _dashboard_session(async_db, linked_member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.get(f"{API_ROOT}/organizations/me/memberships")

    assert response.status_code == 200, response.text
    slugs = {row["organization"]["slug"] for row in response.json()["data"]}
    assert slugs == {"acme"}


# =============================================================================
# Switching enforcement on: live sessions
# =============================================================================


async def test_switching_enforcement_on_revokes_the_affected_live_sessions(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()
    signed_in = await _identity_service(async_db).sign_in(member)
    assert signed_in.session_token

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert await _live_session_count(async_db, member.id) == 0


async def test_switching_enforcement_on_leaves_an_unaffected_session_alone(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    elsewhere = await create_organization(async_db, slug="other")
    bystander = await _member_with_email(
        async_db, elsewhere, role="member", email="by@other.example", full_name="By Stander"
    )
    await async_db.commit()
    signed_in = await _identity_service(async_db).sign_in(bystander)
    assert signed_in.session_token

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert await _live_session_count(async_db, bystander.id) == 1


async def test_switching_enforcement_on_leaves_a_linked_members_session_alive(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    linked_member = await _member_with_email(
        async_db, organization, role="member", email="li@acme.example", full_name="Li Linked"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await _link_member(async_db, linked_member, connection)
    await async_db.commit()
    signed_in = await _identity_service(async_db).sign_in(linked_member)
    assert signed_in.session_token

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert await _live_session_count(async_db, linked_member.id) == 1


async def test_saving_enforcement_again_revokes_a_member_whose_link_is_now_gone(async_db: AsyncSession) -> None:
    """Enforcement was already on when the member's link disappeared; saving it again must still catch them."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await _link_member(async_db, member, connection)
    await async_db.commit()
    signed_in = await _identity_service(async_db).sign_in(member)
    assert signed_in.session_token

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )
    assert await _live_session_count(async_db, member.id) == 1

    await async_db.execute(
        delete(OAuthIdentityLink).where(
            OAuthIdentityLink.user_id == member.id,
            OAuthIdentityLink.provider == oidc_identity_provider(connection.resolved_issuer),
        )
    )
    await async_db.commit()

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )

    assert await _live_session_count(async_db, member.id) == 0


# =============================================================================
# Switching into an enforcing organization
# =============================================================================


async def test_switching_into_an_enforcing_organization_is_refused_for_a_blocked_member(
    client: TestClient, async_db: AsyncSession
) -> None:
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    other = await create_organization(async_db, slug="other")
    member = await _member_with_email(
        async_db, enforcing, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=member.id, role="member"
    )
    member.active_organization_id = other.id
    async_db.add(member)
    token = await _dashboard_session(async_db, member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.post(f"{API_ROOT}/organizations/me/switch", json={"organization_id": str(enforcing.id)})

    assert response.status_code == 404, response.text
    await async_db.refresh(member)
    assert member.active_organization_id == other.id


async def test_switching_into_an_enforcing_organization_succeeds_for_a_linked_member(
    client: TestClient, async_db: AsyncSession
) -> None:
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    other = await create_organization(async_db, slug="other")
    linked_member = await _member_with_email(
        async_db, enforcing, role="member", email="li@acme.example", full_name="Li Linked"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=linked_member.id, role="member"
    )
    linked_member.active_organization_id = other.id
    async_db.add(linked_member)
    connection = await _add_connection(async_db, enforcing.id, first_sign_in_at=datetime.now(UTC))
    await _link_member(async_db, linked_member, connection)
    token = await _dashboard_session(async_db, linked_member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.post(f"{API_ROOT}/organizations/me/switch", json={"organization_id": str(enforcing.id)})

    assert response.status_code == 200, response.text
    assert response.json()["organization"]["id"] == str(enforcing.id)


async def test_switching_into_a_non_enforcing_organization_is_unaffected(
    client: TestClient, async_db: AsyncSession
) -> None:
    organization = await create_organization(async_db, slug="acme")
    other = await create_organization(async_db, slug="other")
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=other.id, user_id=member.id, role="member"
    )
    member.active_organization_id = other.id
    async_db.add(member)
    token = await _dashboard_session(async_db, member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)

    response = client.post(f"{API_ROOT}/organizations/me/switch", json={"organization_id": str(organization.id)})

    assert response.status_code == 200, response.text
    assert response.json()["organization"]["id"] == str(organization.id)
