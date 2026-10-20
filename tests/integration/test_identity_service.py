"""Sign-in through the identity service, and the core identity provider policy."""

import functools
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from gateway.adapters.identity_provider_adapter import DeploymentIdentityProviderAdapter
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OAuthIdentityLinkConflictError,
    OAuthIdentityUnknownError,
    TermsNotAcceptedError,
)
from gateway.models.base import Base
from gateway.models.identity import EmailAssertion, OAuthIdentityLink, ProviderIdentity
from gateway.models.tenancy import Organization, User
from gateway.ports.identity_provider_port import IdentityProviderPort, IdentityResolution
from gateway.repositories.identity import IdentityRepositories, OAuthIdentityLinkRepository
from gateway.repositories.tenancy import OrganizationDomainRepository, OrganizationMemberRepository, UserRepository
from gateway.services.identity import IdentityService
from gateway.services.mail import Mailer
from gateway.services.tenancy import DomainAutoJoinListener, OrganizationService, forget_unproven_access
from gateway.services.tenancy.workspace_listener import NullWorkspaceListener

from .tenancy_helpers import create_member, create_organization, membership_writes

pytestmark = pytest.mark.asyncio

TENANT_PROVIDER = "acme-oidc"


def _service_with(
    db: AsyncSession, uow: UnitOfWork, policy: IdentityProviderPort, *, terms_url: str | None = None
) -> IdentityService:
    """Build the service as its request builder does, on ``policy``."""
    return IdentityService(
        uow,
        IdentityRepositories.on(uow),
        identity_provider=policy,
        organizations=OrganizationService(db, membership_listener=None, workspace_listener=NullWorkspaceListener()),
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
