"""An organization's SSO enforcement: its state, the break-glass preconditions, and who it blocks.

``sso_blocks_membership`` is pure and needs no database; its cases are kept here rather than in
``tests/unit`` so the whole layer this task adds reads as one file.
"""

import asyncio
import functools
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import col

from gateway.adapters.identity_provider_adapter import DeploymentIdentityProviderAdapter
from gateway.core.config import API_ROOT, OAUTH_PROVIDERS, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OIDCConnectionDisabledError,
    OIDCConnectionEnforcedError,
    OIDCConnectionNotFoundError,
    OIDCConnectionUnprovenError,
    PasswordLinkUnavailableError,
    SsoBreakGlassRequiredError,
    SsoBreakGlassRouteRequiredError,
    SsoSignInRequiredError,
)
from gateway.exceptions.organizations_exceptions import InvitationPasswordNotAcceptedError, NotAuthorizedError
from gateway.models.identity import EmailAssertion, OAuthIdentityLink, OIDCConnection, oidc_identity_provider
from gateway.models.tenancy import (
    ActiveOrganizationMemberUpdateRequest,
    DashboardSession,
    InviteOrganizationMemberRequest,
    Organization,
    OrganizationMember,
    User,
    WebAuthnCredential,
)
from gateway.repositories.identity import IdentityRepositories, OAuthPendingStateRepository
from gateway.repositories.tenancy import OrganizationMemberRepository, UserRepository
from gateway.repositories.tenancy.organization_domain_repository import OrganizationDomainRepository
from gateway.schemas.identity import OIDCConnectionRequest
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME, hash_session_token
from gateway.services.identity import IdentityService, OIDCConnectionService
from gateway.services.identity._discovery import DiscoveredProvider
from gateway.services.mail import Mailer
from gateway.services.organizations import OrganizationClaimedDomains
from gateway.services.password_service import hash_password
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
    membership_listener_on,
    membership_writes,
)

pytestmark = pytest.mark.asyncio

_ISSUER = "https://sso.acme.example"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Storing a connection's client secret needs a key configured."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _connection_service(
    db: AsyncSession, *, offered_providers: frozenset[str] = frozenset(OAUTH_PROVIDERS)
) -> OIDCConnectionService:
    uow = UnitOfWork(db)
    return OIDCConnectionService(
        uow,
        IdentityRepositories.on(uow),
        organizations=OrganizationService(db, membership_listener=None),
        domains=OrganizationClaimedDomains(OrganizationDomainRepository(uow)),
        sign_in_states=OAuthPendingStateRepository(db),
        allow_private_hosts=False,
        passkey_rp_id="otari.test",
        offered_providers=offered_providers,
    )


async def _add_connection(
    db: AsyncSession,
    organization_id: uuid.UUID,
    *,
    first_sign_in_at: datetime | None,
    issuer: str = _ISSUER,
    enabled: bool = True,
) -> OIDCConnection:
    connection = OIDCConnection(
        organization_id=organization_id,
        issuer=issuer,
        resolved_issuer=issuer,
        client_id="a-client-id",
        encrypted_client_secret=encrypt_secret("a-client-secret"),
        client_secret_last4="cret",
        display_name="Acme SSO",
        discovery_document={"issuer": issuer},
        discovered_at=datetime.now(UTC),
        first_sign_in_at=first_sign_in_at,
        enabled=enabled,
    )
    db.add(connection)
    await db.flush()
    await db.refresh(connection)
    return connection


async def _member_with_email(
    db: AsyncSession, organization: Organization, *, role: str, email: str, full_name: str, password: bool = False
) -> User:
    """Create a member of ``organization``, optionally holding a password as a route in of their own."""
    user = await UserRepository(db).create_local_identity(
        email=email, full_name=full_name, active_organization_id=organization.id
    )
    if password:
        user.hashed_password = hash_password("correct-horse-battery-staple")
        db.add(user)
        await db.flush()
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
            asserted_email=member.email,
            email_assertion=EmailAssertion.VERIFIED,
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
    organizations = OrganizationService(db, membership_listener=membership_listener_on(uow), uow=uow)
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


async def _membership_of(db: AsyncSession, organization: Organization, user: User) -> OrganizationMember:
    """The active membership the enforcement rule is asked about."""
    membership = await OrganizationMemberRepository(db).get_active_by_organization_and_user(organization.id, user.id)
    assert membership is not None
    return membership


async def test_enforcement_blocks_a_member_with_no_link_for_this_connection(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert (
        sso_blocks_membership(
            organization=organization,
            membership=await _membership_of(async_db, organization, member),
            is_linked=False,
            is_deployment_operator=False,
        )
        is True
    )


async def test_enforcement_does_not_block_the_break_glass_owner(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    organization.sso_required = True
    organization.sso_break_glass_user_id = owner.id
    async_db.add(organization)
    await async_db.flush()

    assert (
        sso_blocks_membership(
            organization=organization,
            membership=await _membership_of(async_db, organization, owner),
            is_linked=False,
            is_deployment_operator=False,
        )
        is False
    )


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

    # The operator fact comes from the deployment, not from the shape of the account: an operator
    # who has set a password holds an address like anybody else.
    assert (
        sso_blocks_membership(
            organization=organization,
            membership=await _membership_of(async_db, organization, operator),
            is_linked=False,
            is_deployment_operator=True,
        )
        is False
    )
    assert (
        sso_blocks_membership(
            organization=organization,
            membership=await _membership_of(async_db, organization, member),
            is_linked=False,
            is_deployment_operator=False,
        )
        is True
    )


async def test_enforcement_blocks_a_demoted_break_glass_owner(async_db: AsyncSession) -> None:
    """The stored id outlives the role, so the exception has to follow the role rather than the id."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    organization.sso_required = True
    organization.sso_break_glass_user_id = owner.id
    async_db.add(organization)
    await async_db.flush()
    membership = await _membership_of(async_db, organization, owner)
    membership.role = "member"
    async_db.add(membership)
    await async_db.flush()

    assert (
        sso_blocks_membership(
            organization=organization,
            membership=membership,
            is_linked=False,
            is_deployment_operator=False,
        )
        is True
    )


async def test_enforcement_blocks_an_address_less_account_that_holds_no_authority(
    async_db: AsyncSession,
) -> None:
    """An account with no address is not an operator: only the deployment says who is."""
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await async_db.flush()
    nameless = await create_member(async_db, organization, role="member", full_name="No Address")

    assert (
        sso_blocks_membership(
            organization=organization,
            membership=await _membership_of(async_db, organization, nameless),
            is_linked=False,
            is_deployment_operator=False,
        )
        is True
    )


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
        sso_blocks_membership(
            organization=organization,
            membership=await _membership_of(async_db, organization, member),
            is_linked=True,
            is_deployment_operator=False,
        )
        is False
    )


async def test_enforcement_blocks_nobody_while_turned_off(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )

    assert (
        sso_blocks_membership(
            organization=organization,
            membership=await _membership_of(async_db, organization, member),
            is_linked=False,
            is_deployment_operator=False,
        )
        is False
    )


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


async def test_enforcement_cannot_be_required_through_a_disabled_connection(async_db: AsyncSession) -> None:
    """Requiring a connection nobody can sign in through refuses every member holding no exemption."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    connection.enabled = False
    async_db.add(connection)
    await async_db.commit()

    with pytest.raises(OIDCConnectionDisabledError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
        )

    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_enforcement_can_be_turned_off_while_the_named_owner_is_no_longer_one(
    async_db: AsyncSession,
) -> None:
    """Turning it off cannot be refused over an account the exception no longer grants anything to."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    demoted = await _member_with_email(
        async_db, organization, role="owner", email="was@acme.example", full_name="Was Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    organization.sso_required = True
    organization.sso_break_glass_user_id = demoted.id
    async_db.add(organization)
    membership = await _membership_of(async_db, organization, demoted)
    membership.role = "member"
    async_db.add(membership)
    await async_db.commit()

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=False, break_glass_user_id=demoted.id, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_demoting_the_break_glass_owner_ends_the_session_the_exception_kept_alive(
    async_db: AsyncSession,
) -> None:
    """No later request re-asks the rule, so the session has to go when the exception stops applying.

    The exception holds only while the named account is an active owner. Demoted, they are refused
    like any unlinked member, and the session minted while they were exempt would otherwise keep
    acting in the organization until it expired.
    """
    organization = await create_organization(async_db, slug=f"acme-{uuid.uuid4().hex[:8]}")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    exempt = await _member_with_email(
        async_db,
        organization,
        role="owner",
        email="glass@acme.example",
        full_name="Glass Owner",
        password=True,
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    organization.sso_required = True
    organization.sso_break_glass_user_id = exempt.id
    async_db.add(organization)
    await _dashboard_session(async_db, exempt)
    await async_db.commit()
    assert await _live_session_count(async_db, exempt.id) == 1
    membership = await _membership_of(async_db, organization, exempt)

    service = OrganizationService(async_db, **membership_writes(async_db))
    await service.update_active_organization_member_for_user(
        user=owner,
        organization_member_id=membership.id,
        request=ActiveOrganizationMemberUpdateRequest(role="member"),
        sso_enforcement=_connection_service(async_db),
    )

    assert await _live_session_count(async_db, exempt.id) == 0


async def test_a_demotion_whose_revocation_fails_does_not_stand(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The change and the revocation it calls for are one step, so a failed revocation takes it back.

    Committed separately, the demotion would persist while the session it no longer admits stayed
    usable, which is the direction this must not fail in.
    """
    organization = await create_organization(async_db, slug=f"acme-{uuid.uuid4().hex[:8]}")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    exempt = await _member_with_email(
        async_db,
        organization,
        role="owner",
        email="glass@acme.example",
        full_name="Glass Owner",
        password=True,
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    organization.sso_required = True
    organization.sso_break_glass_user_id = exempt.id
    async_db.add(organization)
    await _dashboard_session(async_db, exempt)
    await async_db.commit()
    membership = await _membership_of(async_db, organization, exempt)
    # Held before the call: a rollback expires every attribute on the instances, and reading one
    # afterwards reloads it lazily outside an await.
    organization_id, exempt_id = organization.id, exempt.id
    enforcement = _connection_service(async_db)

    async def _fail(*, organization: Organization) -> None:
        raise RuntimeError("the sessions could not be revoked")

    monkeypatch.setattr(enforcement, "revoke_refused_sessions", _fail)
    service = OrganizationService(async_db, **membership_writes(async_db))

    with pytest.raises(RuntimeError, match="could not be revoked"):
        await service.update_active_organization_member_for_user(
            user=owner,
            organization_member_id=membership.id,
            request=ActiveOrganizationMemberUpdateRequest(role="member"),
            sso_enforcement=enforcement,
        )

    kept = await OrganizationMemberRepository(async_db).get_active_by_organization_and_user(organization_id, exempt_id)
    assert kept is not None
    assert kept.role == "owner"
    assert await _live_session_count(async_db, exempt_id) == 1


async def test_enforcement_can_be_turned_off_while_the_named_owner_is_no_longer_a_member(
    async_db: AsyncSession,
) -> None:
    """The dashboard resends the stored account, so a removed one must not block the way out.

    Removal suspends the membership rather than deleting it, which is the status used here: a
    status nothing writes would prove nothing about the path a person actually takes.
    """
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    removed = await _member_with_email(
        async_db, organization, role="owner", email="gone@acme.example", full_name="Gone Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    organization.sso_required = True
    organization.sso_break_glass_user_id = removed.id
    async_db.add(organization)
    membership = await _membership_of(async_db, organization, removed)
    membership.status = "suspended"
    async_db.add(membership)
    await async_db.commit()

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=False, break_glass_user_id=removed.id, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_turning_enforcement_off_still_refuses_an_account_this_deployment_does_not_hold(
    async_db: AsyncSession,
) -> None:
    """The column is a foreign key, so an id naming nobody is refused rather than left to the database."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    with pytest.raises(SsoBreakGlassRequiredError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=False, break_glass_user_id=uuid.uuid4(), break_glass_waived=False
        )


async def test_enforcing_and_disabling_at_once_cannot_leave_an_organization_nobody_can_enter(
    async_db: AsyncSession,
) -> None:
    """Turning enforcement on and pausing the connection are read-then-write over each other.

    Neither has a unique index to lose to, so only the row lock on the organization keeps what each
    one read still true when it writes. Without it both commit, and the organization then requires
    a connection it has paused, which admits nobody. Neither path reads the provider, so there is
    no discovery to stand in for.
    """
    organization = await create_organization(async_db, slug=f"acme-{uuid.uuid4().hex[:8]}")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()
    owner_id, organization_id = owner.id, organization.id
    engine = create_async_engine(async_db.get_bind().engine.url)

    async def enforce(session: AsyncSession) -> object:
        actor = await UserRepository(session).get(owner_id)
        assert actor is not None
        await _connection_service(session).set_sso_enforcement_for_user(
            user=actor, required=True, break_glass_user_id=None, break_glass_waived=True
        )
        return None

    async def disable(session: AsyncSession) -> object:
        actor = await UserRepository(session).get(owner_id)
        assert actor is not None
        return await _connection_service(session).set_enabled_for_user(user=actor, enabled=False)

    async def run(attempt: Callable[[AsyncSession], Awaitable[object]]) -> object:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            try:
                return await attempt(session)
            except Exception as error:  # noqa: BLE001 - the outcome is the assertion
                return error

    try:
        outcomes = list(await asyncio.gather(run(enforce), run(disable)))
    finally:
        await engine.dispose()

    # One of the two has to win, or the test would pass on a pair that both failed for an unrelated
    # reason. Either way round the loser reads the winner's column and refuses on it.
    refused = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(refused) == 1
    assert isinstance(refused[0], OIDCConnectionDisabledError)

    stored = await _reloaded(async_db, organization)
    enabled = (
        await async_db.execute(
            select(OIDCConnection.enabled).where(col(OIDCConnection.organization_id) == organization_id)
        )
    ).scalar_one()
    assert not (stored.sso_required and not enabled)


async def test_enforcing_and_deleting_at_once_cannot_require_a_connection_that_is_gone(
    async_db: AsyncSession,
) -> None:
    """The third path over the same pair: deleting the connection reads the switch, enabling reads it.

    Deleting a child row takes no lock on the organization, so only the settings lock keeps what
    each one read true when it writes. Without it both commit and the organization requires a
    connection nobody can reach.
    """
    organization = await create_organization(async_db, slug=f"acme-{uuid.uuid4().hex[:8]}")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()
    owner_id, organization_id = owner.id, organization.id
    engine = create_async_engine(async_db.get_bind().engine.url)

    async def enforce(session: AsyncSession) -> object:
        actor = await UserRepository(session).get(owner_id)
        assert actor is not None
        await _connection_service(session).set_sso_enforcement_for_user(
            user=actor, required=True, break_glass_user_id=None, break_glass_waived=True
        )
        return None

    async def remove(session: AsyncSession) -> object:
        actor = await UserRepository(session).get(owner_id)
        assert actor is not None
        await _connection_service(session).delete_for_user(user=actor)
        return None

    async def run(attempt: Callable[[AsyncSession], Awaitable[object]]) -> object:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            try:
                return await attempt(session)
            except Exception as error:  # noqa: BLE001 - the outcome is the assertion
                return error

    try:
        outcomes = list(await asyncio.gather(run(enforce), run(remove)))
    finally:
        await engine.dispose()

    refused = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(refused) == 1

    stored = await _reloaded(async_db, organization)
    remaining = (
        await async_db.execute(
            select(func.count())
            .select_from(OIDCConnection)
            .where(col(OIDCConnection.organization_id) == organization_id)
        )
    ).scalar_one()
    assert not (stored.sso_required and remaining == 0)


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


async def test_the_preflight_offers_an_owner_who_keeps_a_route_of_their_own(async_db: AsyncSession) -> None:
    """The picker's whole purpose: an owner who could still sign in while the provider is down."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    report = await _connection_service(async_db).preflight_sso_enforcement_for_user(user=owner)

    assert [candidate.email for candidate in report.break_glass_candidates] == ["owner@acme.example"]


async def test_the_preflight_offers_no_member_who_is_not_an_owner(async_db: AsyncSession) -> None:
    """Only an owner may be named, because only an owner is accepted when the switch is set."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
    )
    await _member_with_email(
        async_db, organization, role="member", password=True, email="mo@acme.example", full_name="Mo Member"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    report = await _connection_service(async_db).preflight_sso_enforcement_for_user(user=owner)

    offered = [candidate.email for candidate in report.break_glass_candidates]
    assert "mo@acme.example" not in offered
    assert offered == ["owner@acme.example"]


async def test_the_preflight_offers_no_owner_whose_only_way_in_is_the_connection(
    async_db: AsyncSession,
) -> None:
    """Naming them would not let them sign in while the connection is unavailable."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
    )
    linked = await _member_with_email(
        async_db, organization, role="owner", email="li@acme.example", full_name="Li Linked"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await _link_member(async_db, linked, connection)
    await async_db.commit()

    report = await _connection_service(async_db).preflight_sso_enforcement_for_user(user=owner)

    assert [candidate.email for candidate in report.break_glass_candidates] == ["owner@acme.example"]


async def test_the_preflight_offers_no_deployment_operator(async_db: AsyncSession) -> None:
    """Enforcement never refuses one, so naming it as the exception would achieve nothing."""
    organization = await create_organization(async_db, slug="acme")
    operator = await _member_with_email(
        async_db, organization, role="owner", password=True, email="ops@acme.example", full_name="The Operator"
    )
    operator.is_superuser = True
    async_db.add(operator)
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    report = await _connection_service(async_db).preflight_sso_enforcement_for_user(user=operator)

    assert report.break_glass_candidates == []


async def test_the_preflight_offers_an_owner_of_an_organization_the_deployment_is_not_operated_by(
    async_db: AsyncSession,
) -> None:
    """The hosted shape: a customer's own owner is offered, while the deployment's operator is not.

    The exclusion is scoped to a deployment operator rather than to whoever owns the organization,
    so an organization on a deployment somebody else runs keeps its own candidates.
    """
    organization = await create_organization(async_db, slug="acme")
    operator = await _member_with_email(
        async_db, organization, role="owner", password=True, email="ops@otari.example", full_name="The Operator"
    )
    operator.is_superuser = True
    async_db.add(operator)
    customer = await _member_with_email(
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    report = await _connection_service(async_db).preflight_sso_enforcement_for_user(user=customer)

    assert [candidate.email for candidate in report.break_glass_candidates] == ["owner@acme.example"]


async def test_enforcement_is_permitted_with_a_break_glass_owner_exception(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
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
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
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
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
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
        async_db, organization, role="owner", password=True, email="owner@acme.example", full_name="Ada Owner"
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

    signed_in = await _identity_service(async_db).sign_in(member, asserted_email=None)

    assert signed_in.active_organization_id == other.id


def _stub_discovery(monkeypatch: pytest.MonkeyPatch, *, resolved_issuer: str) -> None:
    """Stand in for a successful discovery of ``resolved_issuer``, the one call that leaves the process."""

    async def _discover(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
        return DiscoveredProvider(
            resolved_issuer=resolved_issuer,
            document={"issuer": resolved_issuer},
            discovered_at=datetime.now(UTC),
        )

    monkeypatch.setattr("gateway.services.identity._connection_service.discover_connection_metadata", _discover)


async def test_repointing_an_enforcing_connection_revokes_a_member_it_now_blocks(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link names the issuer it was made at, so storing a different one leaves a linked member with none."""
    new_issuer = "https://new-sso.acme.example"
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    connection = await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await _link_member(async_db, member, connection)
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    await _dashboard_session(async_db, member)
    await async_db.commit()
    assert await _live_session_count(async_db, member.id) == 1

    _stub_discovery(monkeypatch, resolved_issuer=new_issuer)
    await _connection_service(async_db).save_for_user(
        user=owner,
        request=OIDCConnectionRequest(
            issuer=new_issuer,
            client_id="a-client-id",
            client_secret="s3cr3t-client-value",
            display_name="Acme SSO",
            default_role="member",
        ),
    )

    assert await _live_session_count(async_db, member.id) == 0


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


async def test_a_password_sign_in_is_refused_where_every_organization_requires_the_connection(
    async_db: AsyncSession,
) -> None:
    """Refused rather than signed in to nowhere: a session that reaches no organization grants nothing."""
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await async_db.commit()

    with pytest.raises(SsoSignInRequiredError) as refusal:
        await _identity_service(async_db).sign_in(member, asserted_email=None)

    reason = str(refusal.value)
    assert "its own provider" in reason
    assert "acme" not in reason


async def test_a_refused_sign_in_leaves_no_session_behind(async_db: AsyncSession) -> None:
    """Asserted against the table, because a refusal that still seated a session would grant the access it refused."""
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    await async_db.commit()

    member_id = member.id

    with pytest.raises(SsoSignInRequiredError):
        await _identity_service(async_db).sign_in(member, asserted_email=None)

    seated = (
        await async_db.execute(
            select(func.count()).select_from(DashboardSession).where(col(DashboardSession.user_id) == member_id)
        )
    ).scalar_one()
    assert seated == 0


async def test_turning_enforcement_on_ends_the_session_of_a_member_it_now_refuses(
    client: TestClient, async_db: AsyncSession
) -> None:
    """What protects a session seated before enforcement, driven over HTTP.

    NOTE: There is no second enforcement check on the routes themselves. A session that survived
    this revocation would still resolve the organization it was seated in, so the revocation is the
    control rather than a convenience, and that is why it is asserted end to end here.
    """
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    member = await _member_with_email(
        async_db, organization, role="member", email="mo@acme.example", full_name="Mo Member", password=True
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(member, asserted_email=None)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, signed_in.session_token)
    assert client.get(f"{API_ROOT}/organizations/me/members").status_code == 200

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=None, break_glass_waived=True
    )
    await async_db.commit()

    assert client.get(f"{API_ROOT}/organizations/me/members").status_code == 401


async def test_a_passkey_sign_in_is_refused_where_every_organization_requires_the_connection(
    async_db: AsyncSession,
) -> None:
    """A passkey proves the same account, so enforcement refuses it for the same reason a password is refused."""
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    member = await _member_with_email(
        async_db, organization, role="member", email="passkey@acme.example", full_name="Pat Passkey"
    )
    await async_db.commit()

    with pytest.raises(SsoSignInRequiredError):
        await _identity_service(async_db).sign_in(member, asserted_email=None)


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

    signed_in = await _identity_service(async_db).sign_in(linked_member, asserted_email=None)

    assert signed_in.active_organization_id == organization.id


async def test_the_deployment_operator_reaches_an_enforcing_organization(async_db: AsyncSession) -> None:
    """Held by the deployment's own authority, not by the account lacking an address.

    An operator who has claimed the deployment holds an address and a password like anybody else, so
    the exemption has to follow the authority rather than the shape of the row.
    """
    organization = await create_organization(async_db, slug="acme")
    organization.sso_required = True
    organization.sso_break_glass_waived = True
    async_db.add(organization)
    operator = await _member_with_email(
        async_db, organization, role="member", email="operator@acme.example", full_name="The Operator"
    )
    operator.is_superuser = True
    async_db.add(operator)
    await async_db.commit()

    signed_in = await _identity_service(async_db).sign_in(operator, asserted_email=None)

    assert signed_in.active_organization_id == organization.id


async def test_enforcement_is_refused_when_the_named_owner_could_not_sign_in_without_it(
    async_db: AsyncSession,
) -> None:
    """An exemption for an account that cannot authenticate without the connection is no exemption."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    with pytest.raises(SsoBreakGlassRouteRequiredError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
        )

    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_a_link_at_a_provider_nobody_configured_is_not_a_route_in(async_db: AsyncSession) -> None:
    """A link needs its provider to still be offered: one nobody configured signs nobody in."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    owner.hashed_password = None
    async_db.add(owner)
    async_db.add(
        OAuthIdentityLink(
            provider="google",
            subject="google-subject",
            user_id=owner.id,
            asserted_email=owner.email,
            email_assertion=EmailAssertion.VERIFIED,
        )
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    service = _connection_service(async_db, offered_providers=frozenset())
    with pytest.raises(SsoBreakGlassRouteRequiredError):
        await service.set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
        )

    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_a_link_at_a_connection_nobody_can_use_is_not_a_route_in(async_db: AsyncSession) -> None:
    """A link naming an organization's own connection is a way in only while a connection serves it.

    The organization administers that provider rather than the deployment, so the deployment's own
    list of offered providers says nothing about it. What says something is whether any enabled
    connection still resolves to the issuer the link names.
    """
    other_issuer = "https://other-sso.example.com"
    organization = await create_organization(async_db, slug=f"acme-{uuid.uuid4().hex[:8]}")
    elsewhere = await create_organization(async_db, slug=f"other-{uuid.uuid4().hex[:8]}")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    owner.hashed_password = None
    async_db.add(owner)
    async_db.add(
        OAuthIdentityLink(
            provider=oidc_identity_provider(other_issuer),
            subject="other-subject",
            user_id=owner.id,
            asserted_email=owner.email,
            email_assertion=EmailAssertion.VERIFIED,
        )
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await _add_connection(async_db, elsewhere.id, first_sign_in_at=None, issuer=other_issuer, enabled=False)
    await async_db.commit()

    with pytest.raises(SsoBreakGlassRouteRequiredError):
        await _connection_service(async_db).set_sso_enforcement_for_user(
            user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
        )

    assert (await _reloaded(async_db, organization)).sso_required is False


async def test_a_link_at_a_connection_somebody_can_use_is_a_route_in(async_db: AsyncSession) -> None:
    """The same link, while an enabled connection resolves to that issuer, is the way in."""
    other_issuer = "https://other-sso.example.com"
    organization = await create_organization(async_db, slug=f"acme-{uuid.uuid4().hex[:8]}")
    elsewhere = await create_organization(async_db, slug=f"other-{uuid.uuid4().hex[:8]}")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    owner.hashed_password = None
    async_db.add(owner)
    async_db.add(
        OAuthIdentityLink(
            provider=oidc_identity_provider(other_issuer),
            subject="other-subject",
            user_id=owner.id,
            asserted_email=owner.email,
            email_assertion=EmailAssertion.VERIFIED,
        )
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await _add_connection(async_db, elsewhere.id, first_sign_in_at=None, issuer=other_issuer, enabled=True)
    await async_db.commit()

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is True


async def test_a_link_at_a_configured_provider_is_a_route_in(async_db: AsyncSession) -> None:
    """The same link, on a deployment that offers that provider, is the way in the exception needs."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    owner.hashed_password = None
    async_db.add(owner)
    async_db.add(
        OAuthIdentityLink(
            provider="google",
            subject="google-subject",
            user_id=owner.id,
            asserted_email=owner.email,
            email_assertion=EmailAssertion.VERIFIED,
        )
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is True


async def test_enforcement_is_permitted_where_the_named_owner_keeps_a_passkey(async_db: AsyncSession) -> None:
    """A passkey needs no provider, so it is a route in while the connection is unavailable."""
    organization = await create_organization(async_db, slug="acme")
    owner = await _member_with_email(
        async_db, organization, role="owner", email="owner@acme.example", full_name="Ada Owner"
    )
    async_db.add(
        WebAuthnCredential(
            user_id=owner.id,
            name="Ada's key",
            rp_id="otari.test",
            credential_id="a-credential",
            public_key="a-public-key",
            sign_count=0,
        )
    )
    await _add_connection(async_db, organization.id, first_sign_in_at=datetime.now(UTC))
    await async_db.commit()

    await _connection_service(async_db).set_sso_enforcement_for_user(
        user=owner, required=True, break_glass_user_id=owner.id, break_glass_waived=False
    )

    assert (await _reloaded(async_db, organization)).sso_required is True


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

    signed_in = await _identity_service(async_db).sign_in(bystander, asserted_email=None)
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


async def test_setting_a_dashboard_password_is_allowed_where_another_organization_admits_them(
    client: TestClient, async_db: AsyncSession
) -> None:
    """Enforcement attaches to a membership, so one organization's setting does not take the password
    that signs the person into another."""
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

    response = client.put(f"{API_ROOT}/auth/password", json={"new_password": "correct-horse-battery-staple"})

    assert response.status_code == 200, response.text
    await async_db.refresh(member)
    assert member.hashed_password is not None


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


async def test_memberships_listing_counts_and_pages_what_is_left_after_a_refusal(
    client: TestClient, async_db: AsyncSession
) -> None:
    """The refusal is applied before the page is cut, so the count is the caller's real choice of destinations."""
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    first = await create_organization(async_db, slug="first")
    second = await create_organization(async_db, slug="second")
    member = await _member_with_email(
        async_db, enforcing, role="member", email="mo@acme.example", full_name="Mo Member"
    )
    for organization in (first, second):
        await OrganizationMemberRepository(async_db).create_membership(
            organization_id=organization.id, user_id=member.id, role="member"
        )
    member.active_organization_id = first.id
    async_db.add(member)
    token = await _dashboard_session(async_db, member)
    await async_db.commit()
    client.cookies.set(SESSION_COOKIE_NAME, token)

    seen: set[str] = set()
    for skip in (0, 1):
        response = client.get(f"{API_ROOT}/organizations/me/memberships?skip={skip}&limit=1")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["count"] == 2
        assert len(body["data"]) == 1
        seen.add(body["data"][0]["organization"]["slug"])

    assert seen == {"first", "second"}


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
    signed_in = await _identity_service(async_db).sign_in(member, asserted_email=None)
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
    signed_in = await _identity_service(async_db).sign_in(bystander, asserted_email=None)
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
    signed_in = await _identity_service(async_db).sign_in(linked_member, asserted_email=None)
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
    signed_in = await _identity_service(async_db).sign_in(member, asserted_email=None)
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


async def test_deleting_an_organization_leaves_nobody_acting_in_one_that_refuses_them(
    client: TestClient, async_db: AsyncSession
) -> None:
    """The deletion clears the pointer, so the enforcing organization is not somewhere it can land."""
    enforcing = await create_organization(async_db, slug="acme")
    enforcing.sso_required = True
    enforcing.sso_break_glass_waived = True
    async_db.add(enforcing)
    doomed = await create_organization(async_db, slug="doomed")
    elsewhere = await create_organization(async_db, slug="elsewhere")
    owner = await _member_with_email(async_db, doomed, role="owner", email="owner@acme.example", full_name="Ada Owner")
    member = await _member_with_email(async_db, doomed, role="member", email="mo@acme.example", full_name="Mo Member")
    # The owner keeps somewhere to land; the member's only other organization is the enforcing one.
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=elsewhere.id, user_id=owner.id, role="member"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=enforcing.id, user_id=member.id, role="member"
    )
    member_token = await _dashboard_session(async_db, member)
    owner_token = await _dashboard_session(async_db, owner)
    await async_db.commit()

    client.cookies.set(SESSION_COOKIE_NAME, owner_token)
    deleted = client.delete(f"{API_ROOT}/organizations/me")
    assert deleted.status_code == 200, deleted.text

    await async_db.refresh(member)
    assert member.active_organization_id is None
    client.cookies.set(SESSION_COOKIE_NAME, member_token)
    assert client.get(f"{API_ROOT}/organizations/me").status_code == 404


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
