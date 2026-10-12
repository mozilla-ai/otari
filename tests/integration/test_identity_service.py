"""OAuth sign-in through the identity service and the core identity provider policy."""

import functools

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.identity_provider_adapter import DeploymentIdentityProviderAdapter
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import OAuthIdentityLinkConflictError, OAuthIdentityUnknownError
from gateway.models.identity import EmailAssertion, OAuthIdentityLink, ProviderIdentity
from gateway.models.tenancy import User
from gateway.ports.identity_provider_port import IdentityResolution
from gateway.repositories.identity import OAuthIdentityLinkRepository
from gateway.repositories.tenancy import UserRepository
from gateway.services.identity import IdentityService
from gateway.services.tenancy import OrganizationService, forget_unproven_access

from .tenancy_helpers import create_organization, membership_writes

pytestmark = pytest.mark.asyncio

TENANT_PROVIDER = "acme-oidc"


def _service(db: AsyncSession, *, open_signup: bool) -> IdentityService:
    writes = membership_writes(db)
    adapter = DeploymentIdentityProviderAdapter(
        UserRepository(db),
        OrganizationService(db, **writes),
        functools.partial(forget_unproven_access, db),
        open_signup=open_signup,
    )
    return IdentityService(
        writes["uow"],
        identity_provider=adapter,
        links=OAuthIdentityLinkRepository(writes["uow"]),
        users=UserRepository(writes["uow"]),
    )


def _verified(provider: str, email: str) -> ProviderIdentity:
    return ProviderIdentity(
        provider=provider,
        subject="provider-subject",
        email=email,
        full_name=None,
        email_verified=EmailAssertion.VERIFIED,
    )


async def _count(db: AsyncSession, model: type[User] | type[OAuthIdentityLink]) -> int:
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
    resolution = await _service(async_db, open_signup=True).sign_in_with_oauth(_verified("google", "nova@example.com"))

    link = await async_db.get(OAuthIdentityLink, ("google", "provider-subject"))
    assert link is not None
    assert link.user_id == resolution.identity.id


class _ReturnsAccount:
    """An identity provider policy that signs every identity in as one account."""

    def __init__(self, account: User) -> None:
        self._account = account

    async def resolve(self, external: ProviderIdentity) -> IdentityResolution:
        return IdentityResolution(identity=self._account, new_account=False)


async def test_a_deactivated_account_from_the_policy_is_refused_and_not_linked(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    account = await UserRepository(async_db).create_local_identity(
        email="ada@example.com", full_name="Ada", active_organization_id=organization.id, is_active=False
    )
    await async_db.commit()
    uow = UnitOfWork(async_db)
    service = IdentityService(
        uow,
        identity_provider=_ReturnsAccount(account),
        links=OAuthIdentityLinkRepository(uow),
        users=UserRepository(uow),
    )

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
    service = IdentityService(
        uow,
        identity_provider=_LinksElsewhereFirst(account, elsewhere, links),
        links=links,
        users=UserRepository(uow),
    )

    with pytest.raises(OAuthIdentityLinkConflictError):
        await service.sign_in_with_oauth(_verified("google", "ada@example.com"))

    assert await _count(async_db, OAuthIdentityLink) == 0
