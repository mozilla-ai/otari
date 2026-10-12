"""Provider identity links, read and written in a Unit of Work."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import OAuthIdentityLinkConflictError
from gateway.repositories.identity import OAuthIdentityLinkRepository

from .tenancy_helpers import create_member, create_organization

pytestmark = pytest.mark.asyncio


async def test_a_link_is_found_by_its_provider_identity_and_by_its_account(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    member = await create_member(async_db, organization, role="member", full_name="Ada")
    uow = UnitOfWork(async_db)
    links = OAuthIdentityLinkRepository(uow)

    async with uow:
        await links.add(
            provider="google", subject="ada-subject", user_id=member.id, email="ada@example.com", email_verified=True
        )

    async with uow:
        by_identity = await links.get_by_identity("google", "ada-subject")
        by_account = await links.get_for_user(member.id, "google")
        elsewhere = await links.get_for_user(member.id, "github")

    assert by_identity is not None
    assert by_identity.user_id == member.id
    assert by_identity.email_at_link == "ada@example.com"
    assert by_identity.email_verified_at_link is True
    assert by_identity.linked_at.tzinfo is not None
    assert by_account is by_identity
    assert elsewhere is None


async def test_an_account_holds_one_link_per_provider(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    member = await create_member(async_db, organization, role="member", full_name="Ada")
    uow = UnitOfWork(async_db)
    links = OAuthIdentityLinkRepository(uow)
    async with uow:
        await links.add(provider="google", subject="ada-subject", user_id=member.id, email=None, email_verified=True)

    with pytest.raises(OAuthIdentityLinkConflictError):
        async with uow:
            await links.add(
                provider="google", subject="another-subject", user_id=member.id, email=None, email_verified=True
            )


async def test_an_identity_links_to_one_account(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    first = await create_member(async_db, organization, role="member", full_name="Ada")
    second = await create_member(async_db, organization, role="member", full_name="Grace")
    uow = UnitOfWork(async_db)
    links = OAuthIdentityLinkRepository(uow)
    async with uow:
        await links.add(provider="google", subject="a-subject", user_id=first.id, email=None, email_verified=True)

    with pytest.raises(OAuthIdentityLinkConflictError):
        async with uow:
            await links.add(provider="google", subject="a-subject", user_id=second.id, email=None, email_verified=True)
