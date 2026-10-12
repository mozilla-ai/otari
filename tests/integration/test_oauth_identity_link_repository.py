"""Provider identity links, read and written in a Unit of Work."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import OAuthIdentityLinkConflictError
from gateway.models.identity import EmailAssertion, ProviderIdentity
from gateway.repositories.identity import OAuthIdentityLinkRepository

from .tenancy_helpers import create_member, create_organization

pytestmark = pytest.mark.asyncio


def _identity(subject: str, *, email: str | None = None) -> ProviderIdentity:
    return ProviderIdentity(
        provider="google", subject=subject, email=email, full_name=None, email_verified=EmailAssertion.UNASSERTED
    )


async def test_a_link_is_found_by_its_provider_identity_and_by_its_account(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    member = await create_member(async_db, organization, role="member", full_name="Ada")
    uow = UnitOfWork(async_db)
    links = OAuthIdentityLinkRepository(uow)

    async with uow:
        await links.add(_identity("ada-subject", email="ada@example.com"), user_id=member.id)

    async with uow:
        by_identity = await links.get_by_identity(provider="google", subject="ada-subject")
        by_account = await links.get_for_user(provider="google", user_id=member.id)
        elsewhere = await links.get_for_user(provider="github", user_id=member.id)

    assert by_identity is not None
    assert by_identity.user_id == member.id
    assert by_identity.email_at_link == "ada@example.com"
    assert by_identity.email_assertion_at_link is EmailAssertion.UNASSERTED
    assert by_identity.linked_at.tzinfo is not None
    assert by_account is by_identity
    assert elsewhere is None


async def test_an_account_holds_one_link_per_provider(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    member = await create_member(async_db, organization, role="member", full_name="Ada")
    uow = UnitOfWork(async_db)
    links = OAuthIdentityLinkRepository(uow)
    async with uow:
        await links.add(_identity("ada-subject"), user_id=member.id)

    with pytest.raises(OAuthIdentityLinkConflictError):
        async with uow:
            await links.add(_identity("another-subject"), user_id=member.id)


async def test_an_identity_links_to_one_account(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    first = await create_member(async_db, organization, role="member", full_name="Ada")
    second = await create_member(async_db, organization, role="member", full_name="Grace")
    uow = UnitOfWork(async_db)
    links = OAuthIdentityLinkRepository(uow)
    async with uow:
        await links.add(_identity("a-subject"), user_id=first.id)

    with pytest.raises(OAuthIdentityLinkConflictError):
        async with uow:
            await links.add(_identity("a-subject"), user_id=second.id)
