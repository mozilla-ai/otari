"""Tests for the invitation repository and the membership origin repository."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import Organization
from gateway.repositories.organizations import DueInvitation, NewInvitation
from gateway.repositories.organizations.membership_origin_repository import MembershipOriginRepository
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository
from gateway.repositories.tenancy import OrganizationMemberRepository

from .conftest import _to_async_url
from .tenancy_helpers import create_member, create_organization

pytestmark = pytest.mark.asyncio

_FAR_FUTURE = datetime(2030, 1, 1, tzinfo=UTC)
_NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)
_HOLD = timedelta(minutes=2)
_HELD_UNTIL = _NOW + _HOLD
_MAX_ATTEMPTS = 3


def _invitation(
    organization: Organization,
    *,
    email: str,
    token_hash: str,
    expires_at: datetime = _FAR_FUTURE,
    next_send_at: datetime | None = None,
) -> NewInvitation:
    return NewInvitation(
        email=email,
        expires_at=expires_at,
        invited_by_user_id=None,
        next_send_at=next_send_at,
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


async def test_a_claim_returns_the_values_of_the_invitation_and_counts_the_attempt(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="claim-values")
        inviter = await create_member(async_db, organization, role="owner", full_name="Inviter")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            NewInvitation(
                email="ada@example.com",
                expires_at=_FAR_FUTURE,
                invited_by_user_id=inviter.id,
                next_send_at=_NOW,
                organization_id=organization.id,
                role="admin",
                token_hash="seed",
                workspace_assignments=[{"workspace_id": "w1", "role": "member"}],
            )
        )

        first = await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="first"
        )
        second = await repository.claim_due_send(
            now=_HELD_UNTIL, held_until=_FAR_FUTURE, max_attempts=_MAX_ATTEMPTS, token_hash="second"
        )

        assert first == DueInvitation(
            attempt=1,
            declined_at=None,
            email="ada@example.com",
            expires_at=_FAR_FUTURE,
            id=invitation.id,
            invited_by_user_id=inviter.id,
            organization_id=organization.id,
            role="admin",
            workspace_assignments=[{"workspace_id": "w1", "role": "member"}],
        )
        assert second is not None
        assert second.attempt == 2


async def test_a_due_invitation_is_claimed_once(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="claim-once")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )

        first = await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="first"
        )
        second = await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="second"
        )

        assert first is not None
        assert first.id == invitation.id
        assert second is None


async def test_of_two_claims_made_before_either_commits_only_one_gets_the_invitation(postgres_url: str) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def claim(session: AsyncSession, token_hash: str) -> DueInvitation | None:
        return await OrganizationInvitationRepository(session).claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash=token_hash
        )

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="concurrent-claim")
            invitation = await OrganizationInvitationRepository(setup).add(
                _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
            )
            await setup.commit()

        async with sessions() as first, sessions() as second:
            first_claim = await claim(first, "first")
            waiting = asyncio.create_task(claim(second, "second"))
            # The first claim has not committed, so the second cannot finish.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(waiting), timeout=0.5)
            await first.commit()
            second_claim = await waiting
            await second.commit()

        async with sessions() as check:
            stored = await OrganizationInvitationRepository(check).get(invitation.id)
    finally:
        await engine.dispose()

    assert first_claim is not None
    assert first_claim.id == invitation.id
    assert second_claim is None
    assert stored is not None
    assert (stored.token_hash, stored.delivery_attempts, stored.next_send_at) == ("first", 1, _HELD_UNTIL)


async def test_the_claim_of_the_last_attempt_leaves_no_send_due(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)
    later = _HELD_UNTIL + _HOLD

    async with uow:
        organization = await create_organization(async_db, slug="claim-last-attempt")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )

        first = await repository.claim_due_send(now=_NOW, held_until=_HELD_UNTIL, max_attempts=2, token_hash="first")
        second = await repository.claim_due_send(now=_HELD_UNTIL, held_until=later, max_attempts=2, token_hash="second")
        third = await repository.claim_due_send(
            now=_FAR_FUTURE, held_until=_FAR_FUTURE + _HOLD, max_attempts=2, token_hash="third"
        )

        assert first is not None
        assert second is not None
        assert second.attempt == 2
        assert third is None
        stored = await repository.get(invitation.id)
        assert stored is not None
        assert (stored.token_hash, stored.delivery_attempts, stored.next_send_at) == ("second", 2, None)


async def test_an_invitation_is_due_by_its_due_time_alone(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="claim-due-time-alone")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(now=_NOW, held_until=_HELD_UNTIL, max_attempts=1, token_hash="first")
        await repository.retry_send(invitation.id, token_hash="first", next_send_at=_HELD_UNTIL)

        again = await repository.claim_due_send(
            now=_HELD_UNTIL, held_until=_HELD_UNTIL + _HOLD, max_attempts=1, token_hash="second"
        )

        assert again is not None
        assert again.attempt == 2
        stored = await repository.get(invitation.id)
        assert stored is not None
        assert (stored.token_hash, stored.next_send_at) == ("second", None)


async def test_a_claim_that_would_hold_the_invitation_for_no_time_is_refused(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="claim-no-hold")
        repository = OrganizationInvitationRepository(uow)
        await repository.add(_invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW))

        with pytest.raises(ValueError, match="held"):
            await repository.claim_due_send(now=_NOW, held_until=_NOW, max_attempts=_MAX_ATTEMPTS, token_hash="claim")

        assert await repository.get_by_token_hash("seed") is not None


async def test_an_invitation_with_no_send_due_is_not_claimed(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="claim-not-due")
        repository = OrganizationInvitationRepository(uow)
        await repository.add(_invitation(organization, email="none@example.com", token_hash="none"))
        await repository.add(
            _invitation(
                organization,
                email="later@example.com",
                token_hash="later",
                next_send_at=_NOW + timedelta(microseconds=1),
            )
        )

        assert (
            await repository.claim_due_send(
                now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="claim"
            )
            is None
        )


async def test_the_invitation_that_has_been_due_longest_is_claimed_first(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="claim-order")
        repository = OrganizationInvitationRepository(uow)
        newer = await repository.add(
            _invitation(organization, email="newer@example.com", token_hash="newer", next_send_at=_NOW)
        )
        older = await repository.add(
            _invitation(
                organization,
                email="older@example.com",
                token_hash="older",
                next_send_at=_NOW - timedelta(minutes=1),
            )
        )

        first = await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="first"
        )
        second = await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="second"
        )

        assert first is not None
        assert second is not None
        assert (first.id, second.id) == (older.id, newer.id)


async def test_a_claim_replaces_the_token_hash_of_the_invitation(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="claim-hash")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="old", next_send_at=_NOW)
        )

        await repository.claim_due_send(now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="new")

        assert await repository.get_by_token_hash("old") is None
        found = await repository.get_by_token_hash("new")
        assert found is not None
        assert (found.id, found.token_hash, found.next_send_at, found.delivery_attempts) == (
            invitation.id,
            "new",
            _HELD_UNTIL,
            1,
        )


async def test_finishing_a_send_leaves_no_send_due(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="finish-send")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="claim"
        )

        assert await repository.finish_send(invitation.id, token_hash="claim") is True
        assert invitation.next_send_at is None
        assert (
            await repository.claim_due_send(
                now=_FAR_FUTURE, held_until=_FAR_FUTURE + _HOLD, max_attempts=_MAX_ATTEMPTS, token_hash="next"
            )
            is None
        )


async def test_retrying_a_send_makes_it_due_at_the_later_time(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)
    retry_at = _NOW + timedelta(minutes=5)

    async with uow:
        organization = await create_organization(async_db, slug="retry-send")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="claim"
        )

        assert await repository.retry_send(invitation.id, token_hash="claim", next_send_at=retry_at) is True
        assert invitation.next_send_at == retry_at
        assert (
            await repository.claim_due_send(
                now=retry_at - timedelta(microseconds=1),
                held_until=_FAR_FUTURE,
                max_attempts=_MAX_ATTEMPTS,
                token_hash="early",
            )
            is None
        )
        again = await repository.claim_due_send(
            now=retry_at, held_until=_FAR_FUTURE, max_attempts=_MAX_ATTEMPTS, token_hash="again"
        )
        assert again is not None
        assert again.attempt == 2


async def test_a_result_from_an_earlier_claim_changes_nothing(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="stale-result")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="earlier"
        )
        await repository.claim_due_send(
            now=_HELD_UNTIL, held_until=_FAR_FUTURE, max_attempts=_MAX_ATTEMPTS, token_hash="current"
        )

        assert await repository.finish_send(invitation.id, token_hash="earlier") is False
        assert await repository.retry_send(invitation.id, token_hash="earlier", next_send_at=_NOW) is False
        found = await repository.get_by_token_hash("current")
        assert found is not None
        assert found.next_send_at == _FAR_FUTURE


async def test_a_result_for_a_deleted_invitation_changes_nothing(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    async with uow:
        organization = await create_organization(async_db, slug="deleted-result")
        repository = OrganizationInvitationRepository(uow)
        invitation = await repository.add(
            _invitation(organization, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="claim"
        )
        await repository.delete_in_organization(invitation.id, organization.id)

        assert await repository.finish_send(invitation.id, token_hash="claim") is False
        assert await repository.retry_send(invitation.id, token_hash="claim", next_send_at=_NOW) is False
