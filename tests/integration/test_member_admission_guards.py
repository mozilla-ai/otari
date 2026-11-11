"""Who an organization may put on its roster without the address's owner agreeing.

Two rules, at the service layer, where the caller can be an ordinary owner rather
than the deployment operator the API tests act as:

- under ``open_signup``, adding a member straight to ``active`` is the operator's
  alone, and everyone else invites;
- an address at a domain another organization has proven is refused, whether
  added or invited.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.exceptions.organizations_exceptions import (
    DirectMemberAddUnavailableError,
    OrganizationMemberDomainClaimedError,
)
from gateway.models.tenancy import (
    DOMAIN_PROOF_TTL,
    ActiveOrganizationMemberCreateRequest,
    BulkInviteOrganizationMembersRequest,
    InviteOrganizationMemberRequest,
    Organization,
)
from gateway.repositories.tenancy import OrganizationDomainRepository, UserRepository
from gateway.services.tenancy import OrganizationService

from .tenancy_helpers import (
    NO_BLOCKED_ADDRESSES,
    NO_MAIL_LIMIT,
    create_member,
    create_organization,
    membership_writes,
)

pytestmark = pytest.mark.asyncio

_TEST_CONFIG = GatewayConfig()
DOMAIN = "acme.example"
ADDRESS = f"ada@{DOMAIN}"


async def _prove(db: AsyncSession, organization: Organization, *, verified_at: datetime | None = None) -> None:
    domains = OrganizationDomainRepository(db)
    row = await domains.create_domain(
        organization_id=organization.id,
        domain=DOMAIN,
        default_role="member",
        enabled=True,
        verification_token="token",  # pragma: allowlist secret
    )
    await domains.mark_verified(row, verified_at=verified_at or datetime.now(UTC))


async def test_an_owner_cannot_add_a_member_directly_under_open_signup(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")

    with pytest.raises(DirectMemberAddUnavailableError):
        await OrganizationService(async_db, **membership_writes(async_db)).create_active_organization_member_for_user(
            user=owner,
            request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
            blocked_addresses=NO_BLOCKED_ADDRESSES,
            open_signup=True,
        )

    assert await UserRepository(async_db).get_by_email("ada@example.com") is None


async def test_an_owner_can_still_invite_under_open_signup(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")

    invited = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=_TEST_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert invited.status == "invited"


async def test_an_address_at_a_domain_another_organization_proved_cannot_be_added(async_db: AsyncSession) -> None:
    proving = await create_organization(async_db, slug="acme")
    await _prove(async_db, proving)
    other = await create_organization(async_db, slug="globex")
    owner = await create_member(async_db, other, role="owner", full_name="Owner")

    with pytest.raises(OrganizationMemberDomainClaimedError):
        await OrganizationService(async_db, **membership_writes(async_db)).create_active_organization_member_for_user(
            user=owner,
            request=ActiveOrganizationMemberCreateRequest(email=ADDRESS),
            blocked_addresses=NO_BLOCKED_ADDRESSES,
            open_signup=False,
        )

    assert await UserRepository(async_db).get_by_email(ADDRESS) is None


async def test_an_address_at_a_domain_another_organization_proved_cannot_be_invited(async_db: AsyncSession) -> None:
    proving = await create_organization(async_db, slug="acme")
    await _prove(async_db, proving)
    other = await create_organization(async_db, slug="globex")
    owner = await create_member(async_db, other, role="owner", full_name="Owner")

    with pytest.raises(OrganizationMemberDomainClaimedError):
        await OrganizationService(async_db, **membership_writes(async_db)).invite_active_organization_member_for_user(
            user=owner,
            request=InviteOrganizationMemberRequest(email=ADDRESS),
            config=_TEST_CONFIG,
            mail_limit=NO_MAIL_LIMIT,
            blocked_addresses=NO_BLOCKED_ADDRESSES,
        )

    assert await UserRepository(async_db).get_by_email(ADDRESS) is None


async def test_a_bulk_invite_reports_an_address_at_a_proven_domain_as_failed(async_db: AsyncSession) -> None:
    proving = await create_organization(async_db, slug="acme")
    await _prove(async_db, proving)
    other = await create_organization(async_db, slug="globex")
    owner = await create_member(async_db, other, role="owner", full_name="Owner")

    result = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).invite_active_organization_members_for_user(
        user=owner,
        request=BulkInviteOrganizationMembersRequest(emails=[ADDRESS, "grace@example.com"]),
        config=_TEST_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert [failure.email for failure in result.failed] == [ADDRESS]
    assert [invited.email for invited in result.invited] == ["grace@example.com"]


async def test_the_organization_that_proved_the_domain_can_invite_its_addresses(async_db: AsyncSession) -> None:
    proving = await create_organization(async_db, slug="acme")
    await _prove(async_db, proving)
    owner = await create_member(async_db, proving, role="owner", full_name="Owner")

    invited = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email=ADDRESS),
        config=_TEST_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert invited.status == "invited"


async def test_an_expired_proof_no_longer_protects_the_domain(async_db: AsyncSession) -> None:
    proving = await create_organization(async_db, slug="acme")
    await _prove(async_db, proving, verified_at=datetime.now(UTC) - DOMAIN_PROOF_TTL * 2)
    other = await create_organization(async_db, slug="globex")
    owner = await create_member(async_db, other, role="owner", full_name="Owner")

    invited = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email=ADDRESS),
        config=_TEST_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert invited.status == "invited"
