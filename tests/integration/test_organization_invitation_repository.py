"""Tests for the invitation repository and the membership origin repository."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import Organization
from gateway.repositories.organizations import NewInvitation
from gateway.repositories.organizations.membership_origin_repository import MembershipOriginRepository
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository
from gateway.repositories.tenancy import OrganizationMemberRepository

from .conftest import _to_async_url
from .tenancy_helpers import create_member, create_organization

pytestmark = pytest.mark.asyncio

_FAR_FUTURE = datetime(2030, 1, 1, tzinfo=UTC)


def _invitation(
    organization: Organization, *, email: str, token_hash: str, expires_at: datetime = _FAR_FUTURE
) -> NewInvitation:
    return NewInvitation(
        email=email,
        expires_at=expires_at,
        invited_by_user_id=None,
        organization_id=organization.id,
        role="member",
        token_hash=token_hash,
        workspace_assignments=[],
    )


async def test_a_delete_waits_for_a_concurrent_one_and_then_finds_nothing_to_remove(postgres_url: str) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def delete(session: AsyncSession, invitation_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
        return await OrganizationInvitationRepository(session).delete_in_organization(invitation_id, organization_id)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="concurrent-delete")
            invitation = await OrganizationInvitationRepository(setup).add(
                _invitation(organization, email="ada@example.com", token_hash="concurrent")
            )
            await setup.commit()
            invitation_id, organization_id = invitation.id, organization.id

        async with sessions() as first, sessions() as second:
            first_result = await delete(first, invitation_id, organization_id)
            waiting = asyncio.create_task(delete(second, invitation_id, organization_id))
            # The first delete has not committed, so the second cannot finish.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(waiting), timeout=0.5)
            await first.commit()
            second_result = await waiting
    finally:
        await engine.dispose()

    assert first_result is True
    assert second_result is False


async def test_a_membership_origin_is_recorded(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="membership-origin")
        inviter = await create_member(async_db, organization, role="owner", full_name="Inviter")
        joiner = await create_member(async_db, organization, role="member", full_name="Joiner")
        membership = await OrganizationMemberRepository(async_db).get_by_organization_and_user(
            organization.id, joiner.id
        )
        assert membership is not None

        repository = MembershipOriginRepository(uow)
        invited_at = datetime(2026, 1, 1, tzinfo=UTC)
        joined_at = datetime(2026, 1, 3, tzinfo=UTC)
        await repository.record(
            organization_member_id=membership.id,
            invited_by_user_id=inviter.id,
            invited_at=invited_at,
            joined_at=joined_at,
        )

        origin = await repository.get(membership.id)
        assert origin is not None
        assert (origin.invited_by_user_id, origin.invited_at, origin.joined_at) == (inviter.id, invited_at, joined_at)


async def test_a_second_invitation_for_one_address_in_one_organization_is_refused(
    async_db: AsyncSession,
) -> None:
    organization = await create_organization(async_db, slug="duplicate-invite")
    uow = UnitOfWork(async_db)

    with pytest.raises(IntegrityError):
        async with uow:
            repository = OrganizationInvitationRepository(uow)
            await repository.add(_invitation(organization, email="ada@example.com", token_hash="t1"))
            await repository.add(_invitation(organization, email="ada@example.com", token_hash="t2"))


async def test_an_invitation_in_another_organization_is_not_found(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="owner-org")
    other = await create_organization(async_db, slug="other-org")
    uow = UnitOfWork(async_db)

    async with uow:
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(_invitation(organization, email="ada@example.com", token_hash="t1"))

        assert await repository.get_in_organization(invitation.id, other.id) is None
        assert await repository.get_in_organization(invitation.id, organization.id) == invitation


async def test_deleting_an_invitation_makes_it_unreachable_in_the_same_session(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="delete-identity-map")
    uow = UnitOfWork(async_db)

    async with uow:
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(_invitation(organization, email="ada@example.com", token_hash="t1"))

        assert await repository.delete_in_organization(invitation.id, organization.id) is True
        assert await repository.get(invitation.id) is None


async def test_deleting_an_invitation_reports_whether_a_row_was_removed(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="delete-invite")
    uow = UnitOfWork(async_db)

    async with uow:
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(_invitation(organization, email="ada@example.com", token_hash="t1"))

        assert await repository.delete_in_organization(invitation.id, organization.id) is True
        assert await repository.delete_in_organization(invitation.id, organization.id) is False


async def test_deleting_in_the_wrong_organization_leaves_the_row(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="delete-scope-org")
    other = await create_organization(async_db, slug="delete-scope-other")
    uow = UnitOfWork(async_db)

    async with uow:
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(_invitation(organization, email="ada@example.com", token_hash="t1"))

        assert await repository.delete_in_organization(invitation.id, other.id) is False
        assert await repository.get_in_organization(invitation.id, organization.id) is not None


async def test_expired_invitations_older_than_a_cutoff_are_deleted(async_db: AsyncSession) -> None:
    cutoff = datetime(2026, 1, 10, tzinfo=UTC)
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="expired-invite")
        repository = OrganizationInvitationRepository(uow)
        expired = await repository.add(
            _invitation(
                organization,
                email="expired@example.com",
                token_hash="expired",
                expires_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        not_expired = await repository.add(
            _invitation(
                organization, email="later@example.com", token_hash="later", expires_at=datetime(2026, 2, 1, tzinfo=UTC)
            )
        )

        assert await repository.delete_expired_before(cutoff) == 1
        assert await repository.get_in_organization(expired.id, organization.id) is None
        assert await repository.get_in_organization(not_expired.id, organization.id) is not None


async def test_invitations_are_listed_and_counted_only_for_their_own_organization(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="list-org")
        other = await create_organization(async_db, slug="other-list-org")
        repository = OrganizationInvitationRepository(uow)
        await repository.add(_invitation(organization, email="one@example.com", token_hash="t1"))
        await repository.add(_invitation(organization, email="two@example.com", token_hash="t2"))
        await repository.add(_invitation(other, email="three@example.com", token_hash="t3"))

        rows, total = await repository.list_for_organization(organization.id, skip=0, limit=10)

        assert total == 2
        assert {row.email for row in rows} == {"one@example.com", "two@example.com"}


async def test_list_for_organization_pages_oldest_first_and_honors_skip_and_limit(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="list-order")
        repository = OrganizationInvitationRepository(uow)
        # The insertion order and the address order both differ from the creation order.
        newest = await repository.add(_invitation(organization, email="b@example.com", token_hash="b"))
        oldest = await repository.add(_invitation(organization, email="c@example.com", token_hash="c"))
        middle = await repository.add(_invitation(organization, email="a@example.com", token_hash="a"))
        newest.created_at = datetime(2026, 1, 3, tzinfo=UTC)
        oldest.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        middle.created_at = datetime(2026, 1, 2, tzinfo=UTC)
        await async_db.flush()

        page, total = await repository.list_for_organization(organization.id, skip=1, limit=1)

        assert total == 3
        assert [row.id for row in page] == [middle.id]


async def test_the_same_address_can_be_invited_by_two_organizations(async_db: AsyncSession) -> None:
    first = await create_organization(async_db, slug="org-one")
    second = await create_organization(async_db, slug="org-two")
    uow = UnitOfWork(async_db)

    async with uow:
        repository = OrganizationInvitationRepository(uow)
        in_first = await repository.add(_invitation(first, email="ada@example.com", token_hash="token-one"))
        in_second = await repository.add(_invitation(second, email="ada@example.com", token_hash="token-two"))

        assert await repository.get_by_organization_and_email(first.id, "ada@example.com") == in_first
        assert await repository.get_by_organization_and_email(second.id, "ada@example.com") == in_second
        assert await repository.get_by_token_hash("token-one") == in_first
        assert await repository.get_by_token_hash("token-two") == in_second


async def test_the_sweep_keeps_a_row_expiring_exactly_at_the_cutoff_and_deletes_one_just_before(
    async_db: AsyncSession,
) -> None:
    cutoff = datetime(2026, 1, 10, tzinfo=UTC)
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="sweep-boundary")
        repository = OrganizationInvitationRepository(uow)
        at_cutoff = await repository.add(
            _invitation(organization, email="at@example.com", token_hash="at", expires_at=cutoff)
        )
        just_before = await repository.add(
            _invitation(
                organization,
                email="before@example.com",
                token_hash="before",
                expires_at=cutoff - timedelta(microseconds=1),
            )
        )

        assert await repository.delete_expired_before(cutoff) == 1
        assert await repository.get_in_organization(at_cutoff.id, organization.id) is not None
        assert await repository.get_in_organization(just_before.id, organization.id) is None


async def test_an_address_is_stored_and_found_in_its_normalized_form(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="normalized-address")
    uow = UnitOfWork(async_db)

    async with uow:
        repository = OrganizationInvitationRepository(uow)
        added = await repository.add(_invitation(organization, email="  Ada@Example.COM ", token_hash="t1"))
        found = await repository.get_by_organization_and_email(organization.id, "ADA@example.com")

        assert added.email == "ada@example.com"
        assert found is not None
        assert found.id == added.id


async def test_a_second_invitation_that_differs_only_in_case_is_refused(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="case-duplicate")
    uow = UnitOfWork(async_db)

    with pytest.raises(IntegrityError):
        async with uow:
            repository = OrganizationInvitationRepository(uow)
            await repository.add(_invitation(organization, email="ada@example.com", token_hash="t1"))
            await repository.add(_invitation(organization, email="ADA@EXAMPLE.COM", token_hash="t2"))


async def test_text_that_is_not_an_address_cannot_be_invited_and_matches_nothing(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="not-an-address")
    uow = UnitOfWork(async_db)

    async with uow:
        repository = OrganizationInvitationRepository(uow)

        assert await repository.get_by_organization_and_email(organization.id, "not an address") is None
        with pytest.raises(ValueError, match="email address"):
            await repository.add(_invitation(organization, email="not an address", token_hash="t1"))
