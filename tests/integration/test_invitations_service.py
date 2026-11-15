"""The invitation service against real PostgreSQL transactions, for the races SQLite cannot exercise."""

import asyncio
import time
import uuid
from collections.abc import Collection, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.adapters.rate_limit_store_adapter import InMemoryRateLimitStore
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import (
    AddressAlreadyInvitedError,
    InvitationLimitCeiling,
    InvitationsNeedSsoError,
    MembershipUpdateError,
)
from gateway.models.organizations import OrganizationInvitation
from gateway.models.tenancy import Organization, User
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    WorkspaceRepository,
)
from gateway.services.mail import Mailer
from gateway.services.organizations import OrganizationClaimedDomains
from gateway.services.organizations._invitation_limits import Admission, InvitationLimits
from gateway.services.organizations._invitations import (
    ActingMember,
    BulkInvitations,
    InvitationOffer,
    InvitationPolicy,
    Invitations,
    InvitationState,
    IssuedInvitation,
    OfferedWorkspace,
)

from .conftest import _to_async_url
from .tenancy_helpers import create_member, create_organization

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 7, 1, 12, tzinfo=UTC)
_POLICY = InvitationPolicy(
    expires_after=timedelta(days=7), per_address_per_day=10, per_day=10, resend_limit=3, resend_wait=timedelta(hours=1)
)


def _config() -> GatewayConfig:
    """A deployment with no mail transport, so an invite returns a hand-carried link."""
    return GatewayConfig(mail_transport="none")


def _invitations(
    session: AsyncSession, *, store: InMemoryRateLimitStore, policy: InvitationPolicy = _POLICY
) -> Invitations:
    uow = UnitOfWork(session)
    domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
    return Invitations(
        uow,
        OrganizationRepositories.on(uow),
        clock=lambda: _NOW,
        domains=domains,
        mailer=Mailer(_config()),
        policy=policy,
        store=store,
    )


@pytest.fixture(autouse=True)
def _no_mail_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OTARI_MAIL_FROM_EMAIL", "OTARI_MAIL_TRANSPORT", "OTARI_PUBLIC_BASE_URL", "OTARI_SMTP_HOST"):
        monkeypatch.delenv(name, raising=False)


async def test_two_bulk_invites_that_share_an_address(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()
    shared_address = "shared@example.com"

    # Whichever call's batched insert reaches the shared address first holds its row open here.
    # The other call's own insert then blocks on Postgres's unique index until this one commits.
    # The database decides which call loses the address, not the order this test starts them in.
    real_add_many = OrganizationInvitationRepository.add_many
    inserted = asyncio.Event()
    released = asyncio.Event()
    first_to_arrive: list[None] = []

    async def add_many_that_pauses_the_first_arrival(
        self: OrganizationInvitationRepository, news: Sequence[NewInvitation]
    ) -> list[OrganizationInvitation]:
        rows = await real_add_many(self, news)
        if any(new.email == shared_address for new in news) and not first_to_arrive:
            first_to_arrive.append(None)
            inserted.set()
            await released.wait()
        return rows

    monkeypatch.setattr(OrganizationInvitationRepository, "add_many", add_many_that_pauses_the_first_arrival)

    async def invite_many(session: AsyncSession, by: ActingMember, emails: list[str]) -> BulkInvitations:
        service = _invitations(session, store=store)
        return await service.invite_many(by, emails, InvitationOffer(role="member", workspaces=()))

    async def release_once_the_first_insert_is_staged() -> None:
        await inserted.wait()
        await asyncio.sleep(0.2)
        released.set()

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="shared-invite-race")
            first_owner = await create_member(setup, organization, role="owner", full_name="First")
            second_owner = await create_member(setup, organization, role="owner", full_name="Second")
            await setup.commit()

        async with sessions() as first, sessions() as second:
            by_first = ActingMember(organization_id=organization.id, user_id=first_owner.id)
            by_second = ActingMember(organization_id=organization.id, user_id=second_owner.id)

            first_task = asyncio.create_task(invite_many(first, by_first, ["first-only@example.com", shared_address]))
            second_task = asyncio.create_task(
                invite_many(second, by_second, ["second-only@example.com", shared_address])
            )
            releaser = asyncio.create_task(release_once_the_first_insert_is_staged())
            first_result, second_result = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=10.0)
            await releaser
            await first.commit()
            await second.commit()
    finally:
        await engine.dispose()

    results = [first_result, second_result]
    winners = [result for result in results if not result.refused]
    losers = [result for result in results if result.refused]
    assert len(winners) == 1
    assert len(losers) == 1
    [winner] = winners
    [loser] = losers

    assert len(winner.issued) == 2
    assert len(loser.issued) == 1
    [refused] = loser.refused
    assert refused.email == shared_address
    assert isinstance(refused.refusal, AddressAlreadyInvitedError)

    # The loser's admission for the shared address was given back, so a third attempt still has room.
    probe_limits = InvitationLimits(store, per_day=_POLICY.per_day, per_address_per_day=_POLICY.per_address_per_day)
    probe = await probe_limits.admit(account_id=uuid.uuid4(), email=shared_address, organization_id=organization.id)
    assert isinstance(probe, Admission)


async def test_two_bulk_invites_sharing_two_addresses_in_opposite_order_serialize_on_the_settings_lock(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()
    address_x = "ax-shared@example.com"
    address_y = "by-shared@example.com"

    # Each call takes the organization's settings lock before it ever reaches add_many, so the
    # two calls cannot overlap here regardless of the order they named their addresses in:
    # whichever call arrives first finishes its whole write and commits before the other call's
    # own insert is even attempted. Address order has nothing left to protect.
    real_add_many = OrganizationInvitationRepository.add_many
    inserted = asyncio.Event()
    released = asyncio.Event()
    first_to_arrive: list[None] = []

    async def add_many_that_pauses_the_first_arrival(
        self: OrganizationInvitationRepository, news: Sequence[NewInvitation]
    ) -> list[OrganizationInvitation]:
        rows = await real_add_many(self, news)
        if not first_to_arrive:
            first_to_arrive.append(None)
            inserted.set()
            await released.wait()
        return rows

    monkeypatch.setattr(OrganizationInvitationRepository, "add_many", add_many_that_pauses_the_first_arrival)

    async def invite_many(session: AsyncSession, by: ActingMember, emails: list[str]) -> BulkInvitations:
        service = _invitations(session, store=store)
        return await service.invite_many(by, emails, InvitationOffer(role="member", workspaces=()))

    async def release_once_the_first_insert_is_staged() -> None:
        await inserted.wait()
        await asyncio.sleep(0.2)
        released.set()

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="two-address-deadlock")
            first_owner = await create_member(setup, organization, role="owner", full_name="First")
            second_owner = await create_member(setup, organization, role="owner", full_name="Second")
            await setup.commit()

        async with sessions() as first, sessions() as second:
            by_first = ActingMember(organization_id=organization.id, user_id=first_owner.id)
            by_second = ActingMember(organization_id=organization.id, user_id=second_owner.id)

            first_task = asyncio.create_task(invite_many(first, by_first, [address_x, address_y]))
            second_task = asyncio.create_task(invite_many(second, by_second, [address_y, address_x]))
            releaser = asyncio.create_task(release_once_the_first_insert_is_staged())
            first_result, second_result = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=10.0)
            await releaser
            await first.commit()
            await second.commit()
    finally:
        await engine.dispose()

    results = [first_result, second_result]
    winners = [result for result in results if not result.refused]
    losers = [result for result in results if result.refused]
    assert len(winners) == 1
    assert len(losers) == 1
    [winner] = winners
    [loser] = losers

    assert {issued.invitation.email for issued in winner.issued} == {address_x, address_y}
    assert loser.issued == []
    assert {refused.email for refused in loser.refused} == {address_x, address_y}
    for refused in loser.refused:
        assert isinstance(refused.refusal, AddressAlreadyInvitedError)


async def test_two_bulk_invites_replacing_the_same_expired_invitations_serialize_on_the_settings_lock(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()
    address_x = "ex-shared@example.com"
    address_y = "ey-shared@example.com"

    # The settings lock each call takes before this runs already keeps the two calls from
    # overlapping at all, so whichever call arrives first replaces and inserts both addresses,
    # and commits, before the other call's own call to this is even attempted.
    real_delete = OrganizationInvitationRepository.delete_expired_in_organization
    paused = asyncio.Event()
    released = asyncio.Event()
    first_to_arrive: list[None] = []

    async def delete_that_pauses_the_first_arrival(
        self: OrganizationInvitationRepository,
        invitation_ids: Collection[uuid.UUID],
        organization_id: uuid.UUID,
        *,
        expired_by: datetime,
    ) -> int:
        result = await real_delete(self, invitation_ids, organization_id, expired_by=expired_by)
        if not first_to_arrive:
            first_to_arrive.append(None)
            paused.set()
            await released.wait()
        return result

    monkeypatch.setattr(
        OrganizationInvitationRepository, "delete_expired_in_organization", delete_that_pauses_the_first_arrival
    )

    async def invite_many(session: AsyncSession, by: ActingMember, emails: list[str]) -> BulkInvitations:
        service = _invitations(session, store=store)
        return await service.invite_many(by, emails, InvitationOffer(role="member", workspaces=()))

    async def release_once_the_first_delete_is_staged() -> None:
        await paused.wait()
        await asyncio.sleep(0.2)
        released.set()

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="replace-expired-opposite-order")
            first_owner = await create_member(setup, organization, role="owner", full_name="First")
            second_owner = await create_member(setup, organization, role="owner", full_name="Second")
            repository = OrganizationInvitationRepository(setup)
            await repository.add_many(
                [
                    NewInvitation(
                        created_at=_NOW - timedelta(days=10),
                        email=address,
                        expires_at=_NOW - timedelta(days=1),
                        invited_by_user_id=None,
                        next_send_at=None,
                        organization_id=organization.id,
                        role="member",
                        token_hash=f"expired-{index}",
                        workspace_assignments=[],
                    )
                    for index, address in enumerate((address_x, address_y))
                ]
            )
            await setup.commit()

        async with sessions() as first, sessions() as second:
            by_first = ActingMember(organization_id=organization.id, user_id=first_owner.id)
            by_second = ActingMember(organization_id=organization.id, user_id=second_owner.id)

            first_task = asyncio.create_task(invite_many(first, by_first, [address_x, address_y]))
            second_task = asyncio.create_task(invite_many(second, by_second, [address_y, address_x]))
            releaser = asyncio.create_task(release_once_the_first_delete_is_staged())
            first_result, second_result = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=10.0)
            await releaser
            await first.commit()
            await second.commit()
    finally:
        await engine.dispose()

    results = [first_result, second_result]
    winners = [result for result in results if not result.refused]
    losers = [result for result in results if result.refused]
    assert len(winners) == 1
    assert len(losers) == 1
    [winner] = winners
    [loser] = losers

    assert {issued.invitation.email for issued in winner.issued} == {address_x, address_y}
    assert loser.issued == []
    assert {refused.email for refused in loser.refused} == {address_x, address_y}
    for refused in loser.refused:
        assert isinstance(refused.refusal, AddressAlreadyInvitedError)


async def test_an_owner_demoted_mid_invite_cannot_still_grant_the_owner_role(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # The invite holds here, between its read-only decision and its write, until the demotion has
    # committed, so the write's own re-check sees the caller's current role rather than the one it
    # decided on.
    real_admit = InvitationLimits.admit
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def admit_that_waits_for_the_demotion(
        self: InvitationLimits, *, account_id: uuid.UUID, email: str, organization_id: uuid.UUID
    ) -> Admission | InvitationLimitCeiling:
        arrived.set()
        await released.wait()
        return await real_admit(self, account_id=account_id, email=email, organization_id=organization_id)

    monkeypatch.setattr(InvitationLimits, "admit", admit_that_waits_for_the_demotion)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="owner-demoted-mid-invite")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            await setup.commit()

        async with sessions() as invite_session, sessions() as demote_session:
            by = ActingMember(organization_id=organization.id, user_id=owner.id)

            async def invite() -> IssuedInvitation | MembershipUpdateError:
                try:
                    return await _invitations(invite_session, store=store).invite(
                        by, "new-owner@example.com", InvitationOffer(role="owner", workspaces=())
                    )
                except MembershipUpdateError as refusal:
                    return refusal

            async def demote_once_the_invite_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                members = OrganizationMemberRepository(demote_session)
                member = await members.get_active_by_organization_and_user(organization.id, owner.id)
                assert member is not None
                await members.update_membership(member, {"role": "admin"})
                await demote_session.commit()
                released.set()

            invite_task = asyncio.create_task(invite())
            demote_task = asyncio.create_task(demote_once_the_invite_is_waiting())
            result = await asyncio.wait_for(invite_task, timeout=10.0)
            await demote_task
    finally:
        await engine.dispose()

    assert isinstance(result, MembershipUpdateError)
    async with sessions() as check:
        existing = await OrganizationInvitationRepository(check).get_by_organization_and_emails(
            organization.id, ["new-owner@example.com"]
        )
    assert existing == []


async def test_an_owner_demoted_before_the_settings_lock_still_refuses_the_write(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # A different gap from the owner-demotion race above: the role there is already stale by
    # the time the write transaction opens. Here it is still current when the write opens, and
    # pausing where the settings lock is taken lets a demotion commit before anything holds the
    # organization row, so the role read that follows the lock, once it resumes, must see it.
    real_lock_settings = OrganizationRepository.lock_settings
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def lock_settings_that_waits_for_the_demotion(
        self: OrganizationRepository, organization_id: uuid.UUID
    ) -> None:
        arrived.set()
        await released.wait()
        await real_lock_settings(self, organization_id)

    monkeypatch.setattr(OrganizationRepository, "lock_settings", lock_settings_that_waits_for_the_demotion)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="owner-demoted-before-settings-lock")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            await setup.commit()

        async with sessions() as invite_session, sessions() as demote_session:
            by = ActingMember(organization_id=organization.id, user_id=owner.id)

            async def invite() -> IssuedInvitation | MembershipUpdateError:
                try:
                    return await _invitations(invite_session, store=store).invite(
                        by, "new-owner@example.com", InvitationOffer(role="owner", workspaces=())
                    )
                except MembershipUpdateError as refusal:
                    return refusal

            async def demote_once_the_write_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                members = OrganizationMemberRepository(demote_session)
                member = await members.get_active_by_organization_and_user(organization.id, owner.id)
                assert member is not None
                await members.update_membership(member, {"role": "admin"})
                await demote_session.commit()
                released.set()

            invite_task = asyncio.create_task(invite())
            demote_task = asyncio.create_task(demote_once_the_write_is_waiting())
            result = await asyncio.wait_for(invite_task, timeout=10.0)
            await demote_task
    finally:
        await engine.dispose()

    assert isinstance(result, MembershipUpdateError)
    async with sessions() as check:
        existing = await OrganizationInvitationRepository(check).get_by_organization_and_emails(
            organization.id, ["new-owner@example.com"]
        )
    assert existing == []


async def test_sso_turned_on_mid_invite_still_refuses_the_write(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # Same gap as the owner-demotion race above, with the organization's own SSO requirement
    # turned on in it instead of the caller's role.
    real_admit = InvitationLimits.admit
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def admit_that_waits_for_sso_to_turn_on(
        self: InvitationLimits, *, account_id: uuid.UUID, email: str, organization_id: uuid.UUID
    ) -> Admission | InvitationLimitCeiling:
        arrived.set()
        await released.wait()
        return await real_admit(self, account_id=account_id, email=email, organization_id=organization_id)

    monkeypatch.setattr(InvitationLimits, "admit", admit_that_waits_for_sso_to_turn_on)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="sso-turned-on-mid-invite")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            await setup.commit()

        async with sessions() as invite_session, sessions() as sso_session:
            by = ActingMember(organization_id=organization.id, user_id=owner.id)

            async def invite() -> IssuedInvitation | InvitationsNeedSsoError:
                try:
                    return await _invitations(invite_session, store=store).invite(
                        by, "invitee@example.com", InvitationOffer(role="member", workspaces=())
                    )
                except InvitationsNeedSsoError as refusal:
                    return refusal

            async def enforce_sso_once_the_invite_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                organizations = OrganizationRepository(sso_session)
                row = await organizations.get(organization.id)
                assert row is not None
                await organizations.update_organization(row, {"sso_required": True})
                await sso_session.commit()
                released.set()

            invite_task = asyncio.create_task(invite())
            sso_task = asyncio.create_task(enforce_sso_once_the_invite_is_waiting())
            result = await asyncio.wait_for(invite_task, timeout=10.0)
            await sso_task
    finally:
        await engine.dispose()

    assert isinstance(result, InvitationsNeedSsoError)
    async with sessions() as check:
        existing = await OrganizationInvitationRepository(check).get_by_organization_and_emails(
            organization.id, ["invitee@example.com"]
        )
    assert existing == []


async def test_sso_turned_on_during_the_write_takes_effect_only_after_the_insert_commits(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # The write transaction's own SSO re-check (the second of the two the service makes) must
    # hold the settings lock across that read, so an enable racing it cannot land between the
    # read and the insert it guards. Pause there, after the read has already answered False, and
    # let a second session race its own enable against the still-open write.
    real_is_sso_enforced = OrganizationRepository.is_sso_enforced
    checks_seen: list[None] = []
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def is_sso_enforced_that_pauses_the_write_phase_check(
        self: OrganizationRepository, organization_id: uuid.UUID
    ) -> bool:
        result = await real_is_sso_enforced(self, organization_id)
        checks_seen.append(None)
        if len(checks_seen) == 2:
            arrived.set()
            await released.wait()
        return result

    monkeypatch.setattr(OrganizationRepository, "is_sso_enforced", is_sso_enforced_that_pauses_the_write_phase_check)

    invite_committed_at: list[float] = []
    sso_committed_at: list[float] = []

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="sso-turned-on-during-write")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            await setup.commit()

        async with sessions() as invite_session, sessions() as sso_session:
            by = ActingMember(organization_id=organization.id, user_id=owner.id)

            async def invite() -> IssuedInvitation:
                result = await _invitations(invite_session, store=store).invite(
                    by, "invitee@example.com", InvitationOffer(role="member", workspaces=())
                )
                invite_committed_at.append(time.monotonic())
                return result

            async def enable_sso_once_the_write_is_open() -> None:
                await arrived.wait()
                organizations = OrganizationRepository(sso_session)
                row = await organizations.get(organization.id)
                assert row is not None
                await organizations.update_organization(row, {"sso_required": True})
                await sso_session.commit()
                sso_committed_at.append(time.monotonic())

            invite_task = asyncio.create_task(invite())
            sso_task = asyncio.create_task(enable_sso_once_the_write_is_open())
            await asyncio.wait_for(arrived.wait(), timeout=10.0)
            # Gives the enable above time to reach the database and either block on the write
            # transaction's lock, or, unlocked, run straight through.
            await asyncio.sleep(0.2)
            released.set()
            result = await asyncio.wait_for(invite_task, timeout=10.0)
            await sso_task
    finally:
        await engine.dispose()

    assert result.email == "invitee@example.com"
    assert len(invite_committed_at) == 1
    assert len(sso_committed_at) == 1
    # The enable can only have taken effect once the insert it raced had already committed.
    assert invite_committed_at[0] < sso_committed_at[0]


async def test_invite_and_page_work_against_real_transactions_and_joins(postgres_url: str) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    try:
        async with sessions() as session:
            organization: Organization = await create_organization(session, slug="invitations-smoke")
            owner: User = await create_member(session, organization, role="owner", full_name="Owner")
            await session.commit()

        async with sessions() as session:
            workspace = await WorkspaceRepository(session).create_workspace(
                name="Engineering", organization_id=organization.id, created_by_user_id=owner.id
            )
            await session.commit()

        async with sessions() as session:
            service = _invitations(session, store=store)
            by = ActingMember(organization_id=organization.id, user_id=owner.id)
            issued = await service.invite(
                by,
                "Grace@Example.com",
                InvitationOffer(role="member", workspaces=(OfferedWorkspace(role="admin", workspace_id=workspace.id),)),
            )
            await session.commit()

        assert issued.accept_link is not None
        assert issued.email == "Grace@Example.com"
        assert issued.invitation.email == "grace@example.com"
        assert issued.invitation.workspaces == (OfferedWorkspace(role="admin", workspace_id=workspace.id),)

        async with sessions() as session:
            service = _invitations(session, store=store)
            entries, total = await service.page(by, skip=0, limit=10)
    finally:
        await engine.dispose()

    assert total == 1
    [entry] = entries
    assert entry.state == InvitationState.PENDING
    assert entry.email == "grace@example.com"
