"""Data access for organization memberships, for behavior the roster and search tests do not cover."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import Organization, User
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(
    db: AsyncSession, organization: Organization, *, full_name: str, email: str | None = None, status: str = "active"
) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name, active_organization_id=organization.id, email=email
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id, user_id=user.id, role="member", status=status
    )
    return user


async def test_active_member_addresses_keeps_only_active_members_here(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="active-member-addresses")
    other = await _organization(async_db, slug="active-member-addresses-other")
    repository = OrganizationMemberRepository(async_db)
    await _member(async_db, organization, full_name="Active Here", email="Grace@X.test")
    await _member(async_db, organization, full_name="Suspended Here", email="ada@x.test", status="suspended")
    await _member(async_db, other, full_name="Active Elsewhere", email="mia@x.test")

    found = await repository.active_member_addresses(
        organization.id, ["GRACE@x.test", "ada@x.test", "mia@x.test", "nobody@x.test"]
    )

    assert found == {"grace@x.test"}


async def test_active_member_addresses_with_nothing_to_look_up_matches_nothing(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="active-member-addresses-empty")
    repository = OrganizationMemberRepository(async_db)

    assert await repository.active_member_addresses(organization.id, []) == set()


async def test_active_member_addresses_finds_a_stored_address_with_stray_whitespace(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="active-member-addresses-whitespace")
    repository = OrganizationMemberRepository(async_db)
    await _member(async_db, organization, full_name="Legacy", email=" Grace@X.test ")

    found = await repository.active_member_addresses(organization.id, ["grace@x.test"])

    assert found == {"grace@x.test"}
