"""Who an organization may put on its roster without the address's owner agreeing.

At the service layer, where the caller can be an ordinary owner rather than the
deployment operator the API tests act as. Three rules:

- under ``open_signup``, adding a member straight to ``active`` is the operator's
  alone, and everyone else invites;
- under ``open_signup``, an accept link that was not emailed goes back to the
  operator alone;
- an address at a domain another organization has proven is refused, whether
  added, invited or reinvited.

Under ``open_signup``, a membership nobody accepted cannot be made active by an
update either, which would be a direct add by another route.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.exceptions.organizations_exceptions import (
    DirectMemberAddUnavailableError,
    MembershipActivationNeedsAcceptanceError,
    OrganizationMemberDomainClaimedError,
)
from gateway.models.tenancy import (
    DOMAIN_PROOF_TTL,
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberUpdateRequest,
    BulkInviteOrganizationMembersRequest,
    InviteOrganizationMemberRequest,
    Organization,
    User,
)
from gateway.repositories.tenancy import OrganizationDomainRepository, OrganizationMemberRepository, UserRepository
from gateway.services.tenancy import OrganizationService

from .tenancy_helpers import (
    NO_BLOCKED_ADDRESSES,
    NO_MAIL_LIMIT,
    NO_SSO_ENFORCEMENT,
    NO_TERMS,
    create_member,
    create_organization,
    membership_writes,
)

pytestmark = pytest.mark.asyncio

# Closed signup and no mail, whatever the environment says, so an unsent link is always returned.
_TEST_CONFIG = GatewayConfig().model_copy(update={"open_signup": False, "public_base_url": None})
_OPEN_SIGNUP_CONFIG = _TEST_CONFIG.model_copy(update={"open_signup": True})
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
        config=_OPEN_SIGNUP_CONFIG,
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


async def test_an_invitation_cannot_be_reissued_once_another_organization_proves_its_domain(
    async_db: AsyncSession,
) -> None:
    other = await create_organization(async_db, slug="globex")
    owner = await create_member(async_db, other, role="owner", full_name="Owner")
    service = OrganizationService(async_db, **membership_writes(async_db))
    invited = await service.invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email=ADDRESS),
        config=_TEST_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )
    await _prove(async_db, await create_organization(async_db, slug="acme"))

    with pytest.raises(OrganizationMemberDomainClaimedError):
        await service.reissue_organization_member_invitation_for_user(
            user=owner,
            invitation_id=invited.invitation_id,
            config=_TEST_CONFIG,
            mail_limit=NO_MAIL_LIMIT,
            blocked_addresses=NO_BLOCKED_ADDRESSES,
        )


async def test_an_owner_cannot_activate_an_invited_member_under_open_signup(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationService(async_db, **membership_writes(async_db))
    invited = await service.invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=_OPEN_SIGNUP_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    with pytest.raises(MembershipActivationNeedsAcceptanceError):
        await service.update_active_organization_member_for_user(
            user=owner,
            organization_member_id=invited.organization_member_id,
            request=ActiveOrganizationMemberUpdateRequest(status="active"),
            sso_enforcement=NO_SSO_ENFORCEMENT,
            open_signup=True,
        )

    membership = await OrganizationMemberRepository(async_db).get(invited.organization_member_id)
    assert membership is not None
    assert membership.status == "invited"


async def test_an_owner_cannot_activate_a_revoked_invitee_under_open_signup(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationService(async_db, **membership_writes(async_db))
    invited = await service.invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=_OPEN_SIGNUP_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )
    await service.remove_active_organization_member_for_user(
        user=owner, organization_member_id=invited.organization_member_id
    )

    with pytest.raises(MembershipActivationNeedsAcceptanceError):
        await service.update_active_organization_member_for_user(
            user=owner,
            organization_member_id=invited.organization_member_id,
            request=ActiveOrganizationMemberUpdateRequest(status="active"),
            sso_enforcement=NO_SSO_ENFORCEMENT,
            open_signup=True,
        )


async def test_an_owner_can_reactivate_a_member_who_had_joined_under_open_signup(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    member = await create_member(async_db, organization, role="member", full_name="Ada")
    membership = await OrganizationMemberRepository(async_db).get_by_organization_and_user(organization.id, member.id)
    assert membership is not None
    service = OrganizationService(async_db, **membership_writes(async_db))
    await service.remove_active_organization_member_for_user(user=owner, organization_member_id=membership.id)

    updated = await service.update_active_organization_member_for_user(
        user=owner,
        organization_member_id=membership.id,
        request=ActiveOrganizationMemberUpdateRequest(status="active"),
        sso_enforcement=NO_SSO_ENFORCEMENT,
        open_signup=True,
    )

    assert updated.status == "active"


async def test_an_owner_can_reactivate_a_member_who_accepted_an_invitation_under_open_signup(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="acme")
    owner = await create_member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationService(async_db, **membership_writes(async_db))
    # Invited where signup is closed, the one place an owner gets the unsent link to accept with.
    invited = await service.invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=_TEST_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )
    assert invited.accept_link is not None
    await service.accept_invitation(
        invited.accept_link.split("token=", 1)[1], claims=NO_TERMS, blocked_addresses=NO_BLOCKED_ADDRESSES
    )
    await service.remove_active_organization_member_for_user(
        user=owner, organization_member_id=invited.organization_member_id
    )

    updated = await service.update_active_organization_member_for_user(
        user=owner,
        organization_member_id=invited.organization_member_id,
        request=ActiveOrganizationMemberUpdateRequest(status="active"),
        sso_enforcement=NO_SSO_ENFORCEMENT,
        open_signup=True,
    )

    assert updated.status == "active"


async def _operator(db: AsyncSession, organization: Organization) -> User:
    operator = await create_member(db, organization, role="owner", full_name="Operator")
    operator.is_superuser = True
    await db.flush()
    return operator


async def test_a_deployment_operator_can_add_a_member_directly_under_open_signup(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    operator = await _operator(async_db, organization)

    added = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).create_active_organization_member_for_user(
        user=operator,
        request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
        blocked_addresses=NO_BLOCKED_ADDRESSES,
        open_signup=True,
    )

    assert added.status == "active"


async def test_a_deployment_operator_can_invite_an_address_at_a_domain_another_organization_proved(
    async_db: AsyncSession,
) -> None:
    await _prove(async_db, await create_organization(async_db, slug="acme"))
    operator = await _operator(async_db, await create_organization(async_db, slug="globex"))

    invited = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).invite_active_organization_member_for_user(
        user=operator,
        request=InviteOrganizationMemberRequest(email=ADDRESS),
        config=_OPEN_SIGNUP_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert invited.status == "invited"


async def test_a_deployment_operator_can_add_an_address_at_a_domain_another_organization_proved(
    async_db: AsyncSession,
) -> None:
    await _prove(async_db, await create_organization(async_db, slug="acme"))
    operator = await _operator(async_db, await create_organization(async_db, slug="globex"))

    added = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).create_active_organization_member_for_user(
        user=operator,
        request=ActiveOrganizationMemberCreateRequest(email=ADDRESS),
        blocked_addresses=NO_BLOCKED_ADDRESSES,
        open_signup=True,
    )

    assert added.status == "active"


async def test_a_deployment_operator_can_reissue_once_another_organization_proves_the_domain(
    async_db: AsyncSession,
) -> None:
    operator = await _operator(async_db, await create_organization(async_db, slug="globex"))
    service = OrganizationService(async_db, **membership_writes(async_db))
    invited = await service.invite_active_organization_member_for_user(
        user=operator,
        request=InviteOrganizationMemberRequest(email=ADDRESS),
        config=_OPEN_SIGNUP_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )
    await _prove(async_db, await create_organization(async_db, slug="acme"))

    reissued = await service.reissue_organization_member_invitation_for_user(
        user=operator,
        invitation_id=invited.invitation_id,
        config=_OPEN_SIGNUP_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert reissued.status == "invited"


async def test_a_deployment_operator_can_activate_an_invited_member_under_open_signup(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="acme")
    operator = await _operator(async_db, organization)
    service = OrganizationService(async_db, **membership_writes(async_db))
    invited = await service.invite_active_organization_member_for_user(
        user=operator,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=_OPEN_SIGNUP_CONFIG,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    updated = await service.update_active_organization_member_for_user(
        user=operator,
        organization_member_id=invited.organization_member_id,
        request=ActiveOrganizationMemberUpdateRequest(status="active"),
        sso_enforcement=NO_SSO_ENFORCEMENT,
        open_signup=True,
    )

    assert updated.status == "active"


async def _caller(db: AsyncSession, organization: Organization, *, operator: bool) -> User:
    if operator:
        return await _operator(db, organization)
    return await create_member(db, organization, role="owner", full_name="Owner")


_LINK_CASES = [(_OPEN_SIGNUP_CONFIG, False, False), (_OPEN_SIGNUP_CONFIG, True, True), (_TEST_CONFIG, False, True)]


@pytest.mark.parametrize(("config", "operator", "shared"), _LINK_CASES)
async def test_an_invite_returns_an_unsent_accept_link_only_to_a_caller_who_may_admit_without_acceptance(
    async_db: AsyncSession, config: GatewayConfig, operator: bool, shared: bool
) -> None:
    organization = await create_organization(async_db, slug="acme")
    caller = await _caller(async_db, organization, operator=operator)

    invited = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).invite_active_organization_member_for_user(
        user=caller,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=config,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert (invited.accept_link is not None) is shared


@pytest.mark.parametrize(("config", "operator", "shared"), _LINK_CASES)
async def test_a_bulk_invite_returns_unsent_accept_links_only_to_a_caller_who_may_admit_without_acceptance(
    async_db: AsyncSession, config: GatewayConfig, operator: bool, shared: bool
) -> None:
    organization = await create_organization(async_db, slug="acme")
    caller = await _caller(async_db, organization, operator=operator)

    result = await OrganizationService(
        async_db, **membership_writes(async_db)
    ).invite_active_organization_members_for_user(
        user=caller,
        request=BulkInviteOrganizationMembersRequest(emails=["ada@example.com", "grace@example.com"]),
        config=config,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert len(result.invited) == 2
    assert all((invited.accept_link is not None) is shared for invited in result.invited)


@pytest.mark.parametrize(("config", "operator", "shared"), _LINK_CASES)
async def test_a_reissue_returns_an_unsent_accept_link_only_to_a_caller_who_may_admit_without_acceptance(
    async_db: AsyncSession, config: GatewayConfig, operator: bool, shared: bool
) -> None:
    organization = await create_organization(async_db, slug="acme")
    caller = await _caller(async_db, organization, operator=operator)
    service = OrganizationService(async_db, **membership_writes(async_db))
    invited = await service.invite_active_organization_member_for_user(
        user=caller,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=config,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    reissued = await service.reissue_organization_member_invitation_for_user(
        user=caller,
        invitation_id=invited.invitation_id,
        config=config,
        mail_limit=NO_MAIL_LIMIT,
        blocked_addresses=NO_BLOCKED_ADDRESSES,
    )

    assert (reissued.accept_link is not None) is shared
