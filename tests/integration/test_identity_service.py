"""Sign-in through the identity service, and the core identity provider policy."""

import functools
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from gateway.adapters.identity_provider_adapter import DeploymentIdentityProviderAdapter
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OAuthEmailNotVerifiedError,
    OAuthIdentityLinkConflictError,
    OAuthIdentityUnknownError,
    TermsNotAcceptedError,
)
from gateway.models.base import Base
from gateway.models.identity import (
    EmailAssertion,
    OAuthIdentityLink,
    OIDCConnection,
    ProviderIdentity,
    oidc_identity_provider,
)
from gateway.models.tenancy import Organization, OrganizationDomain, OrganizationDomainRole, User, Workspace
from gateway.ports.identity_provider_port import IdentityProviderPort, IdentityResolution
from gateway.repositories.identity import IdentityRepositories, OAuthIdentityLinkRepository, OIDCConnectionRepository
from gateway.repositories.tenancy import OrganizationDomainRepository, OrganizationMemberRepository, UserRepository
from gateway.services.identity import IdentityService
from gateway.services.mail import Mailer
from gateway.services.tenancy import (
    DomainAutoJoinListener,
    OrganizationService,
    VerifiedDomainLookup,
    forget_unproven_access,
)

from .tenancy_helpers import create_member, create_organization, membership_writes

pytestmark = pytest.mark.asyncio

TENANT_PROVIDER = "acme-oidc"
SSO_ISSUER = "https://login.acme.example"


def _service_with(
    db: AsyncSession, uow: UnitOfWork, policy: IdentityProviderPort, *, terms_url: str | None = None
) -> IdentityService:
    """Build the service as its request builder does, on ``policy``."""
    return IdentityService(
        uow,
        IdentityRepositories.on(uow),
        identity_provider=policy,
        domain_claims=VerifiedDomainLookup(OrganizationDomainRepository(uow)),
        organizations=OrganizationService(db, membership_listener=None, uow=uow),
        sign_in_listener=DomainAutoJoinListener(OrganizationDomainRepository(uow), OrganizationMemberRepository(uow)),
        session_ttl_hours=1,
        terms_url=terms_url,
        mailer=Mailer(GatewayConfig(mail_transport="none")),
        open_signup=False,
        password_reset_expiry_hours=2,
    )


def _service(db: AsyncSession, *, open_signup: bool) -> IdentityService:
    writes = membership_writes(db)
    adapter = DeploymentIdentityProviderAdapter(
        UserRepository(db),
        OrganizationService(db, **writes),
        functools.partial(forget_unproven_access, db),
        open_signup=open_signup,
    )
    return _service_with(db, writes["uow"], adapter)


def _verified(provider: str, email: str) -> ProviderIdentity:
    return ProviderIdentity(
        provider=provider,
        subject="provider-subject",
        email=email,
        full_name=None,
        email_verified=EmailAssertion.VERIFIED,
    )


async def _count(db: AsyncSession, model: type[SQLModel] | type[Base]) -> int:
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


class _RefusesEverything:
    """An identity provider policy that fails the test if an SSO sign-in ever reaches it."""

    async def resolve(self, external: ProviderIdentity) -> IdentityResolution:
        raise AssertionError("An SSO identity reached the identity provider port")


def _sso_service(db: AsyncSession, *, terms_url: str | None = None) -> IdentityService:
    """Build the service for an SSO sign-in, on a policy an SSO identity must never reach."""
    return _service_with(db, UnitOfWork(db), _RefusesEverything(), terms_url=terms_url)


def _sso_identity(
    *,
    issuer: str = SSO_ISSUER,
    subject: str = "sso-subject",
    email: str | None,
    full_name: str | None = "Ada Lovelace",
    # Defaults to what a connection reports in practice. ``trust_email_verified`` is off for every
    # connection, so apron-auth sends no claim at all and the assertion is unasserted.
    email_verified: EmailAssertion = EmailAssertion.UNASSERTED,
) -> ProviderIdentity:
    return ProviderIdentity(
        provider=oidc_identity_provider(issuer),
        subject=subject,
        email=email,
        full_name=full_name,
        email_verified=email_verified,
    )


async def _add_connection(
    db: AsyncSession,
    organization_id: uuid.UUID,
    *,
    issuer: str = SSO_ISSUER,
    default_role: OrganizationDomainRole = "member",
) -> OIDCConnection:
    uow = UnitOfWork(db)
    async with uow:
        return await OIDCConnectionRepository(uow).add(
            organization_id=organization_id,
            issuer=issuer,
            resolved_issuer=issuer,
            client_id="a-client-id",
            encrypted_client_secret="encrypted-secret",
            client_secret_last4="cret",
            display_name="Example IdP",
            discovery_document={"issuer": issuer},
            discovered_at=datetime.now(UTC),
            default_role=default_role,
        )


async def _add_claim(db: AsyncSession, organization_id: uuid.UUID, *, domain: str) -> OrganizationDomain:
    row = await OrganizationDomainRepository(db).create_domain(
        organization_id=organization_id,
        domain=domain,
        default_role="member",
        enabled=True,
        verification_token="tok" + uuid.uuid4().hex[:16],
    )
    row.verified_at = datetime.now(UTC)
    db.add(row)
    await db.commit()
    return row


async def test_a_first_sso_sign_in_creates_an_account_with_no_email(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")

    signed_in = await _sso_service(async_db).sign_in_with_oauth(_sso_identity(email="ada@acme.example"))

    assert signed_in.new_account is True
    assert signed_in.account.email is None
    assert signed_in.account.full_name == "Ada Lovelace"
    assert signed_in.active_organization_id == organization.id


async def test_the_claimed_address_is_recorded_on_the_link_only(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")

    signed_in = await _sso_service(async_db).sign_in_with_oauth(
        _sso_identity(subject="sso-subject", email="ada@acme.example")
    )

    link = await async_db.get(OAuthIdentityLink, (oidc_identity_provider(SSO_ISSUER), "sso-subject"))
    assert link is not None
    assert link.email_at_link == "ada@acme.example"
    assert signed_in.account.email is None


async def test_an_sso_account_gets_no_personal_organization(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")
    organizations_before = await _count(async_db, Organization)
    workspaces_before = await _count(async_db, Workspace)

    signed_in = await _sso_service(async_db).sign_in_with_oauth(_sso_identity(email="ada@acme.example"))

    assert await _count(async_db, Organization) == organizations_before
    assert await _count(async_db, Workspace) == workspaces_before
    membership = await OrganizationMemberRepository(async_db).get_active_by_organization_and_user(
        organization.id, signed_in.account.id
    )
    assert membership is not None
    assert membership.role == "member"


async def test_an_sso_sign_in_succeeds_with_open_signup_off(async_db: AsyncSession) -> None:
    """A connection is its own authority over its own directory, so ``open_signup=False`` does not block it."""
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")

    signed_in = await _sso_service(async_db).sign_in_with_oauth(_sso_identity(email="ada@acme.example"))

    assert signed_in.account.active_organization_id == organization.id


async def test_an_sso_sign_in_succeeds_with_no_email_verified_claim(async_db: AsyncSession) -> None:
    """A connection reports no assertion at all, and the DNS proof is what admits the identity.

    ``trust_email_verified`` is off for every connection, so requiring an assertion here would
    refuse every sign-in a real provider produces.
    """
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")

    signed_in = await _sso_service(async_db).sign_in_with_oauth(
        _sso_identity(email="ada@acme.example", email_verified=EmailAssertion.UNASSERTED)
    )

    assert signed_in.new_account is True


async def test_an_sso_sign_in_is_refused_when_the_provider_denies_the_address(async_db: AsyncSession) -> None:
    """An issuer that says outright the address is unverified is turned away, unlike one that is silent."""
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")
    accounts_before = await _count(async_db, User)

    with pytest.raises(OAuthEmailNotVerifiedError):
        await _sso_service(async_db).sign_in_with_oauth(
            _sso_identity(email="ada@acme.example", email_verified=EmailAssertion.UNVERIFIED)
        )

    assert await _count(async_db, User) == accounts_before


async def test_an_authenticated_person_matching_no_organization_is_refused(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    # No domain claim: nothing proves the identity's email domain, so the connection admits nobody.
    accounts_before = await _count(async_db, User)

    with pytest.raises(OAuthIdentityUnknownError):
        await _sso_service(async_db).sign_in_with_oauth(_sso_identity(email="ada@acme.example"))

    assert await _count(async_db, User) == accounts_before


async def test_one_issuer_shared_by_two_organizations_is_one_account_with_two_memberships(
    async_db: AsyncSession,
) -> None:
    """Two organizations sharing one issuer each admit only their own proven domain.

    The second membership comes from an ordinary admin grant, not from SSO. The one account SSO
    provisioned is not confined to one organization, and tolerates holding both.
    """
    acme = await create_organization(async_db, slug="acme")
    globex = await create_organization(async_db, slug="globex")
    await _add_connection(async_db, acme.id)
    await _add_connection(async_db, globex.id)
    await _add_claim(async_db, acme.id, domain="acme.example")

    signed_in = await _sso_service(async_db).sign_in_with_oauth(_sso_identity(email="ada@acme.example"))
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=globex.id, user_id=signed_in.account.id, role="member"
    )
    await async_db.commit()

    members = OrganizationMemberRepository(async_db)
    assert await members.get_active_by_organization_and_user(acme.id, signed_in.account.id) is not None
    assert await members.get_active_by_organization_and_user(globex.id, signed_in.account.id) is not None


async def test_a_first_sso_sign_in_takes_the_terms_gate(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")
    service = _sso_service(async_db, terms_url="https://example.com/terms")
    identity = _sso_identity(email="ada@acme.example")

    with pytest.raises(TermsNotAcceptedError):
        await service.sign_in_with_oauth(identity)
    assert await _count(async_db, User) == 0

    signed_in = await service.sign_in_with_oauth(identity, terms_accepted=True)

    assert signed_in.account.terms_accepted_at is not None


async def test_an_sso_identity_never_reaches_the_identity_provider_port(async_db: AsyncSession) -> None:
    """``_sso_service`` binds a policy that raises on any call, so a successful sign-in proves it unused."""
    organization = await create_organization(async_db, slug="acme")
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")

    signed_in = await _sso_service(async_db).sign_in_with_oauth(_sso_identity(email="ada@acme.example"))

    assert signed_in.account.email is None


async def test_an_sso_account_is_told_apart_from_the_master_key_operator(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    operator = await UserRepository(async_db).create_local_identity(
        full_name="Operator", active_organization_id=organization.id
    )
    await async_db.commit()
    await _add_connection(async_db, organization.id)
    await _add_claim(async_db, organization.id, domain="acme.example")

    signed_in = await _sso_service(async_db).sign_in_with_oauth(_sso_identity(email="ada@acme.example"))

    assert operator.email is None
    assert operator.oauth_provider is None
    assert signed_in.account.email is None
    assert signed_in.account.oauth_provider is not None


async def test_a_tenant_provider_cannot_claim_an_account_by_its_verified_address(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    account = await UserRepository(async_db).create_local_identity(
        email="ada@example.com", full_name="Ada", active_organization_id=organization.id
    )
    await async_db.commit()

    with pytest.raises(OAuthIdentityUnknownError):
        await _service(async_db, open_signup=False).sign_in_with_oauth(_verified(TENANT_PROVIDER, "ada@example.com"))

    await async_db.refresh(account)
    assert account.email_verified_at is None
    assert account.oauth_provider is None
    assert await _count(async_db, OAuthIdentityLink) == 0


async def test_a_tenant_provider_registers_no_account_under_open_signup(async_db: AsyncSession) -> None:
    accounts_before = await _count(async_db, User)

    with pytest.raises(OAuthIdentityUnknownError):
        await _service(async_db, open_signup=True).sign_in_with_oauth(_verified(TENANT_PROVIDER, "nova@example.com"))

    assert await _count(async_db, User) == accounts_before
    assert await _count(async_db, OAuthIdentityLink) == 0


async def test_open_signup_links_the_account_it_registers(async_db: AsyncSession) -> None:
    signed_in = await _service(async_db, open_signup=True).sign_in_with_oauth(_verified("google", "nova@example.com"))

    link = await async_db.get(OAuthIdentityLink, ("google", "provider-subject"))
    assert link is not None
    assert link.user_id == signed_in.account.id
    assert signed_in.session_token


async def test_starting_a_session_records_the_sign_in_on_the_account(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    account = await UserRepository(async_db).create_local_identity(
        email="ada@example.com", full_name="Ada", active_organization_id=organization.id
    )
    await async_db.commit()
    assert account.last_sign_in_at is None

    await _service(async_db, open_signup=False).start_session(account)

    await async_db.refresh(account)
    assert account.last_sign_in_at is not None


class _ReturnsAccount:
    """An identity provider policy that signs every identity in as one account."""

    def __init__(self, account: User, *, new_account: bool = False) -> None:
        self._account = account
        self._new_account = new_account

    async def resolve(self, external: ProviderIdentity) -> IdentityResolution:
        return IdentityResolution(identity=self._account, new_account=self._new_account)


async def test_a_deactivated_account_from_the_policy_is_refused_and_not_linked(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    account = await UserRepository(async_db).create_local_identity(
        email="ada@example.com", full_name="Ada", active_organization_id=organization.id, is_active=False
    )
    await async_db.commit()
    service = _service_with(async_db, UnitOfWork(async_db), _ReturnsAccount(account))

    with pytest.raises(OAuthIdentityUnknownError):
        await service.sign_in_with_oauth(_verified("google", "ada@example.com"))

    assert await _count(async_db, OAuthIdentityLink) == 0


class _LinksElsewhereFirst:
    """A policy during which a concurrent sign-in links the same identity to another account."""

    def __init__(self, account: User, elsewhere: User, links: OAuthIdentityLinkRepository) -> None:
        self._account = account
        self._elsewhere = elsewhere
        self._links = links

    async def resolve(self, external: ProviderIdentity) -> IdentityResolution:
        await self._links.add(external, user_id=self._elsewhere.id)
        return IdentityResolution(identity=self._account, new_account=False)


async def test_an_identity_linked_elsewhere_during_the_sign_in_is_refused_as_a_conflict(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="acme")
    users = UserRepository(async_db)
    account = await users.create_local_identity(
        email="ada@example.com", full_name="Ada", active_organization_id=organization.id
    )
    elsewhere = await users.create_local_identity(
        email="grace@example.com", full_name="Grace", active_organization_id=organization.id
    )
    await async_db.commit()
    uow = UnitOfWork(async_db)
    links = OAuthIdentityLinkRepository(uow)
    service = _service_with(async_db, uow, _LinksElsewhereFirst(account, elsewhere, links))

    with pytest.raises(OAuthIdentityLinkConflictError):
        await service.sign_in_with_oauth(_verified("google", "ada@example.com"))

    assert await _count(async_db, OAuthIdentityLink) == 0


async def test_any_policy_that_reports_a_new_account_is_held_to_the_published_terms(async_db: AsyncSession) -> None:
    """The rule is the service's, so an adapter an overlay binds cannot leave it out."""
    organization = await create_organization(async_db, slug="acme")
    account = await UserRepository(async_db).create_local_identity(
        email="ada@example.com", full_name="Ada", active_organization_id=organization.id
    )
    await async_db.commit()
    policy = _ReturnsAccount(account, new_account=True)
    service = _service_with(async_db, UnitOfWork(async_db), policy, terms_url="https://example.com/terms")

    with pytest.raises(TermsNotAcceptedError):
        await service.sign_in_with_oauth(_verified("google", "ada@example.com"))
    assert await _count(async_db, OAuthIdentityLink) == 0
    # The refusal rolled back, which expired the account the policy holds.
    await async_db.refresh(account)
    assert account.terms_accepted_at is None

    signed_in = await service.sign_in_with_oauth(_verified("google", "ada@example.com"), terms_accepted=True)

    await async_db.refresh(signed_in.account)
    assert signed_in.account.terms_accepted_at is not None


async def _squatted(db: AsyncSession, email: str) -> User:
    """An address a stranger signed up with under open signup and never verified."""
    registration = await OrganizationService(db, **membership_writes(db)).provision_signup_tenancy(
        email=email, full_name="Mallory Squat"
    )
    account = registration.identity
    account.hashed_password = "squatter-hash"  # pragma: allowlist secret
    db.add(account)
    await db.commit()
    return account


def _named(provider: str, email: str, full_name: str | None) -> ProviderIdentity:
    return ProviderIdentity(
        provider=provider,
        subject="provider-subject",
        email=email,
        full_name=full_name,
        email_verified=EmailAssertion.VERIFIED,
    )


async def _organization_name(db: AsyncSession, organization_id: object) -> str:
    db.expire_all()
    organization = await db.get(Organization, organization_id)
    assert organization is not None
    return organization.name


async def test_taking_back_a_squatted_address_drops_the_strangers_name_and_renames_their_organization(
    async_db: AsyncSession,
) -> None:
    squatted = await _squatted(async_db, "ada@example.com")
    organization_id = squatted.default_organization_id
    assert await _organization_name(async_db, organization_id) == "Mallory Squat's organization"

    signed_in = await _service(async_db, open_signup=True).sign_in_with_oauth(
        _named("google", "ada@example.com", "Ada Lovelace")
    )

    assert signed_in.active_organization_id == organization_id
    assert signed_in.account.full_name == "Ada Lovelace"
    assert signed_in.account.hashed_password is None
    assert await _organization_name(async_db, organization_id) == "Ada Lovelace's organization"


async def test_taking_back_a_squatted_address_names_the_organization_from_the_address_without_a_provider_name(
    async_db: AsyncSession,
) -> None:
    squatted = await _squatted(async_db, "ada@example.com")

    signed_in = await _service(async_db, open_signup=True).sign_in_with_oauth(_named("google", "ada@example.com", None))

    assert signed_in.account.full_name is None
    assert await _organization_name(async_db, squatted.default_organization_id) == "ada's organization"


async def test_taking_back_a_squatted_address_keeps_the_name_of_an_organization_someone_else_belongs_to(
    async_db: AsyncSession,
) -> None:
    squatted = await _squatted(async_db, "ada@example.com")
    organization = await async_db.get(Organization, squatted.default_organization_id)
    assert organization is not None
    await create_member(async_db, organization, role="member", full_name="Grace")
    await async_db.commit()

    signed_in = await _service(async_db, open_signup=True).sign_in_with_oauth(
        _named("google", "ada@example.com", "Ada Lovelace")
    )

    assert signed_in.account.full_name == "Ada Lovelace"
    assert await _organization_name(async_db, organization.id) == "Mallory Squat's organization"


async def _former_member(db: AsyncSession, email: str, *, vouched: bool) -> tuple[User, Organization]:
    """An account an organization added and then removed, whose address was proven, or only vouched for."""
    organization = await create_organization(db, slug="acme")
    account = await UserRepository(db).create_local_identity(
        email=email, full_name="Ada", active_organization_id=organization.id
    )
    account.email_verified_at = datetime.now(UTC)
    account.email_vouched_at = account.email_verified_at if vouched else None
    account.hashed_password = "member-hash"  # pragma: allowlist secret
    db.add(account)
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=account.id, role="member", status="suspended"
    )
    await db.commit()
    return account, organization


async def test_proving_a_vouched_address_nobody_still_wants_founds_its_own_organization(
    async_db: AsyncSession,
) -> None:
    """A provider proves an address an organization only vouched for, which is when open signup claims it."""
    account, former = await _former_member(async_db, "ada@example.com", vouched=True)

    signed_in = await _service(async_db, open_signup=True).sign_in_with_oauth(
        _named("google", "ada@example.com", "Ada Lovelace")
    )

    assert signed_in.active_organization_id is not None
    assert signed_in.active_organization_id != former.id
    assert signed_in.account.default_organization_id == signed_in.active_organization_id
    owner = await OrganizationMemberRepository(async_db).get_active_by_organization_and_user(
        signed_in.active_organization_id, account.id
    )
    assert owner is not None
    assert owner.role == "owner"
    assert await _organization_name(async_db, signed_in.active_organization_id) == "Ada Lovelace's organization"


async def test_a_proven_address_that_lost_its_last_membership_signs_in_to_no_organization(
    async_db: AsyncSession,
) -> None:
    """An address is claimed once, so a later sign-in leaves founding an organization to the person."""
    _, former = await _former_member(async_db, "ada@example.com", vouched=False)
    organizations_before = await _count(async_db, Organization)

    signed_in = await _service(async_db, open_signup=True).sign_in_with_oauth(
        _named("google", "ada@example.com", "Ada Lovelace")
    )

    assert signed_in.active_organization_id is None
    assert await _count(async_db, Organization) == organizations_before
    assert signed_in.account.full_name == "Ada"
    assert signed_in.account.active_organization_id == former.id
