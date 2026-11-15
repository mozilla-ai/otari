"""The invitation service against real PostgreSQL transactions, for the races SQLite cannot exercise."""

import asyncio
import re
import time
import uuid
from collections.abc import Collection, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gateway.adapters.rate_limit_store_adapter import InMemoryRateLimitStore
from gateway.core.config import GatewayConfig
from gateway.core.tokens import hash_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import (
    AddressAlreadyInvitedError,
    InvitationLimitCeiling,
    InvitationNotFoundError,
    InvitationResendLimitError,
    InvitationResendTooSoonError,
    InvitationsNeedSsoError,
    MembershipUpdateError,
)
from gateway.models.organizations import OrganizationInvitation
from gateway.models.tenancy import Organization, User
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories, SendRequest, SendResult
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


def _available_at(error: InvitationResendTooSoonError) -> datetime:
    """Parse the UTC timestamp an ``InvitationResendTooSoonError`` carries in its message."""
    match = re.search(r"at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", str(error))
    assert match is not None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


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


_RESEND_POLICY = InvitationPolicy(
    expires_after=timedelta(days=7),
    per_address_per_day=100,
    per_day=3,
    resend_limit=10,
    resend_wait=timedelta(hours=1),
)
# Past the wait since the invite's own send, which counts as the first, so neither resend is too soon to ask for.
_RESEND_NOW = _NOW + _RESEND_POLICY.resend_wait + timedelta(minutes=1)


def _resend_service(session: AsyncSession, *, store: InMemoryRateLimitStore) -> Invitations:
    """Build a service past its own resend wait, for a resend that races something other than itself."""
    uow = UnitOfWork(session)
    domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
    return Invitations(
        uow,
        OrganizationRepositories.on(uow),
        clock=lambda: _RESEND_NOW,
        domains=domains,
        mailer=Mailer(_config()),
        policy=_RESEND_POLICY,
        store=store,
    )


async def test_a_resend_blocks_on_the_organizations_lock_a_role_change_holds(postgres_url: str) -> None:
    """``organizations.lock`` (``FOR UPDATE``) conflicts with the ``FOR NO KEY UPDATE`` that `lock_settings` takes.

    A role change holds the stronger lock for its whole transaction, so a resend that reaches its
    own settings lock while that transaction is still open must wait for it to end. Resend, not
    invite, is raced here: a resend's write only updates the invitation row it already has, so
    nothing else on its path can touch the organization row, and a wait can only come from
    `lock_settings` itself.
    """
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="resend-waits-on-row-lock")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            by = ActingMember(organization_id=organization.id, user_id=owner.id)
            issued = await _invitations(setup, store=store).invite(
                by, "nova@example.com", InvitationOffer(role="member", workspaces=())
            )
            await setup.commit()

        async with sessions() as locker, sessions() as resend_session:
            await OrganizationRepository(locker).lock(organization.id)

            async def resend() -> IssuedInvitation:
                return await _resend_service(resend_session, store=store).resend(by, issued.invitation.id)

            resending = asyncio.create_task(resend())
            # The role-change lock is still open here, so the resend cannot have reached its write yet.
            done, _ = await asyncio.wait({resending}, timeout=0.5)
            assert not done, "the resend must wait for the lock a role change holds"

            await locker.commit()
            reissued = await asyncio.wait_for(resending, timeout=10.0)
    finally:
        await engine.dispose()

    assert reissued.invitation.email == "nova@example.com"


async def test_an_owner_demoted_mid_resend_cannot_still_mint_an_owner_link(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # The resend holds here, between its read-only decision and its write, until the demotion
    # has committed, so the write's own re-check sees the caller's current role rather than the
    # one it decided on before this wait. Patched only once the invitation to resend already
    # exists, so the invite that creates it is not itself caught by the pause.
    real_admit = InvitationLimits.admit
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def admit_that_waits_for_the_demotion(
        self: InvitationLimits, *, account_id: uuid.UUID, email: str, organization_id: uuid.UUID
    ) -> Admission | InvitationLimitCeiling:
        arrived.set()
        await released.wait()
        return await real_admit(self, account_id=account_id, email=email, organization_id=organization_id)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="owner-demoted-mid-resend")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            by = ActingMember(organization_id=organization.id, user_id=owner.id)
            issued = await _invitations(setup, store=store).invite(
                by, "new-owner@example.com", InvitationOffer(role="owner", workspaces=())
            )
            await setup.commit()

        monkeypatch.setattr(InvitationLimits, "admit", admit_that_waits_for_the_demotion)

        async with sessions() as resend_session, sessions() as demote_session:

            async def resend() -> IssuedInvitation | MembershipUpdateError:
                try:
                    return await _resend_service(resend_session, store=store).resend(by, issued.invitation.id)
                except MembershipUpdateError as refusal:
                    return refusal

            async def demote_once_the_resend_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                members = OrganizationMemberRepository(demote_session)
                member = await members.get_active_by_organization_and_user(organization.id, owner.id)
                assert member is not None
                await members.update_membership(member, {"role": "admin"})
                await demote_session.commit()
                released.set()

            resend_task = asyncio.create_task(resend())
            demote_task = asyncio.create_task(demote_once_the_resend_is_waiting())
            result = await asyncio.wait_for(resend_task, timeout=10.0)
            await demote_task
    finally:
        await engine.dispose()

    assert isinstance(result, MembershipUpdateError)
    assert issued.accept_link is not None
    original_token = issued.accept_link.rsplit("token=", 1)[-1]
    async with sessions() as check:
        current = await OrganizationInvitationRepository(check).get_in_organization(
            issued.invitation.id, organization.id
        )
    assert current is not None
    assert current.send_count == 1
    assert current.token_hash == hash_token(original_token)


async def test_two_resends_at_once(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # The settings lock each resend takes before this already decides which one proceeds: the
    # one that wins it reaches here, pauses, and commits once released, uncontended. The other,
    # blocked on that lock until the commit lands, then updates against a send count that has
    # already moved, so it loses rather than racing to the same answer as the winner.
    real_request_send = OrganizationInvitationRepository.request_send
    arrived = asyncio.Event()
    released = asyncio.Event()
    first_to_arrive: list[None] = []

    async def request_send_that_pauses_the_first_arrival(
        self: OrganizationInvitationRepository,
        invitation_id: uuid.UUID,
        organization_id: uuid.UUID,
        request: SendRequest,
        *,
        expected_send_count: int,
    ) -> SendResult:
        if not first_to_arrive:
            first_to_arrive.append(None)
            arrived.set()
            await released.wait()
        return await real_request_send(
            self, invitation_id, organization_id, request, expected_send_count=expected_send_count
        )

    monkeypatch.setattr(OrganizationInvitationRepository, "request_send", request_send_that_pauses_the_first_arrival)

    def resend_service(session: AsyncSession) -> Invitations:
        uow = UnitOfWork(session)
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
        return Invitations(
            uow,
            OrganizationRepositories.on(uow),
            clock=lambda: _RESEND_NOW,
            domains=domains,
            mailer=Mailer(_config()),
            policy=_RESEND_POLICY,
            store=store,
        )

    async def attempt(
        session: AsyncSession, by: ActingMember, invitation_id: uuid.UUID
    ) -> IssuedInvitation | InvitationResendTooSoonError:
        try:
            issued = await resend_service(session).resend(by, invitation_id)
        except InvitationResendTooSoonError as refusal:
            return refusal
        await session.commit()
        return issued

    async def release_once_the_first_is_waiting() -> None:
        await arrived.wait()
        await asyncio.sleep(0.2)
        released.set()

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="two-resends-at-once")
            first_owner = await create_member(setup, organization, role="owner", full_name="First")
            second_owner = await create_member(setup, organization, role="owner", full_name="Second")
            service = _invitations(setup, store=store)
            by_first = ActingMember(organization_id=organization.id, user_id=first_owner.id)
            issued = await service.invite(by_first, "shared@example.com", InvitationOffer(role="member", workspaces=()))
            await setup.commit()

        async with sessions() as first, sessions() as second:
            by_first = ActingMember(organization_id=organization.id, user_id=first_owner.id)
            by_second = ActingMember(organization_id=organization.id, user_id=second_owner.id)
            first_task = asyncio.create_task(attempt(first, by_first, issued.invitation.id))
            second_task = asyncio.create_task(attempt(second, by_second, issued.invitation.id))
            releaser = asyncio.create_task(release_once_the_first_is_waiting())
            first_result, second_result = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=10.0)
            await releaser
    finally:
        await engine.dispose()

    results = [first_result, second_result]
    winners = [result for result in results if isinstance(result, IssuedInvitation)]
    losers = [result for result in results if isinstance(result, InvitationResendTooSoonError)]
    assert len(winners) == 1
    assert len(losers) == 1
    [winner] = winners
    [loser] = losers

    # The loser's own re-read, not a cached one: the time it is given is later than now, not a
    # moment it has already proved has passed.
    assert _available_at(loser) > _RESEND_NOW

    assert winner.accept_link is not None
    token = winner.accept_link.rsplit("token=", 1)[-1]
    async with sessions() as check:
        row = await OrganizationInvitationRepository(check).get_by_token_hash(hash_token(token))
    assert row is not None

    # The loser's own admission was given back: a third request against the same organization,
    # at the ceiling the invite and the winning resend already spent two of, still fits.
    probe_limits = InvitationLimits(store, per_day=_RESEND_POLICY.per_day, per_address_per_day=100)
    probe = await probe_limits.admit(
        account_id=uuid.uuid4(), email="unrelated@example.com", organization_id=organization.id
    )
    assert isinstance(probe, Admission)


_LIMIT_CROSSING_POLICY = InvitationPolicy(
    expires_after=timedelta(days=7),
    per_address_per_day=100,
    per_day=3,
    resend_limit=1,
    resend_wait=timedelta(hours=1),
)
_LIMIT_CROSSING_NOW = _NOW + _LIMIT_CROSSING_POLICY.resend_wait + timedelta(minutes=1)


async def test_two_resends_at_once_when_the_winner_crosses_the_limit(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    real_request_send = OrganizationInvitationRepository.request_send
    arrived = asyncio.Event()
    released = asyncio.Event()
    first_to_arrive: list[None] = []

    async def request_send_that_pauses_the_first_arrival(
        self: OrganizationInvitationRepository,
        invitation_id: uuid.UUID,
        organization_id: uuid.UUID,
        request: SendRequest,
        *,
        expected_send_count: int,
    ) -> SendResult:
        if not first_to_arrive:
            first_to_arrive.append(None)
            arrived.set()
            await released.wait()
        return await real_request_send(
            self, invitation_id, organization_id, request, expected_send_count=expected_send_count
        )

    monkeypatch.setattr(OrganizationInvitationRepository, "request_send", request_send_that_pauses_the_first_arrival)

    def resend_service(session: AsyncSession) -> Invitations:
        uow = UnitOfWork(session)
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
        return Invitations(
            uow,
            OrganizationRepositories.on(uow),
            clock=lambda: _LIMIT_CROSSING_NOW,
            domains=domains,
            mailer=Mailer(_config()),
            policy=_LIMIT_CROSSING_POLICY,
            store=store,
        )

    async def attempt(
        session: AsyncSession, by: ActingMember, invitation_id: uuid.UUID
    ) -> IssuedInvitation | InvitationResendLimitError:
        try:
            issued = await resend_service(session).resend(by, invitation_id)
        except InvitationResendLimitError as refusal:
            return refusal
        await session.commit()
        return issued

    async def release_once_the_first_is_waiting() -> None:
        await arrived.wait()
        await asyncio.sleep(0.2)
        released.set()

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="two-resends-cross-the-limit")
            first_owner = await create_member(setup, organization, role="owner", full_name="First")
            second_owner = await create_member(setup, organization, role="owner", full_name="Second")
            service = _invitations(setup, store=store)
            by_first = ActingMember(organization_id=organization.id, user_id=first_owner.id)
            issued = await service.invite(by_first, "shared@example.com", InvitationOffer(role="member", workspaces=()))
            await setup.commit()

        async with sessions() as first, sessions() as second:
            by_first = ActingMember(organization_id=organization.id, user_id=first_owner.id)
            by_second = ActingMember(organization_id=organization.id, user_id=second_owner.id)
            first_task = asyncio.create_task(attempt(first, by_first, issued.invitation.id))
            second_task = asyncio.create_task(attempt(second, by_second, issued.invitation.id))
            releaser = asyncio.create_task(release_once_the_first_is_waiting())
            first_result, second_result = await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=10.0)
            await releaser
    finally:
        await engine.dispose()

    results = [first_result, second_result]
    winners = [result for result in results if isinstance(result, IssuedInvitation)]
    losers = [result for result in results if isinstance(result, InvitationResendLimitError)]
    assert len(winners) == 1
    assert len(losers) == 1


async def test_a_resend_racing_a_cancel_is_not_found_once_the_cancel_wins(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # The resend holds here, past every refusal that costs nothing, until the cancel has
    # committed. Its own write then matches no row, and its re-read must see that for itself.
    real_request_send = OrganizationInvitationRepository.request_send
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def request_send_that_waits_for_the_cancel(
        self: OrganizationInvitationRepository,
        invitation_id: uuid.UUID,
        organization_id: uuid.UUID,
        request: SendRequest,
        *,
        expected_send_count: int,
    ) -> SendResult:
        arrived.set()
        await released.wait()
        return await real_request_send(
            self, invitation_id, organization_id, request, expected_send_count=expected_send_count
        )

    monkeypatch.setattr(OrganizationInvitationRepository, "request_send", request_send_that_waits_for_the_cancel)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="resend-races-a-cancel")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            service = _invitations(setup, store=store)
            by = ActingMember(organization_id=organization.id, user_id=owner.id)
            issued = await service.invite(by, "shared@example.com", InvitationOffer(role="member", workspaces=()))
            await setup.commit()

        async with sessions() as resend_session, sessions() as cancel_session:

            async def resend() -> IssuedInvitation | InvitationNotFoundError:
                try:
                    return await _resend_service(resend_session, store=store).resend(by, issued.invitation.id)
                except InvitationNotFoundError as refusal:
                    return refusal

            async def cancel_once_resend_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                await _invitations(cancel_session, store=store).cancel(by, issued.invitation.id)
                await cancel_session.commit()
                released.set()

            resend_task = asyncio.create_task(resend())
            canceler_task = asyncio.create_task(cancel_once_resend_is_waiting())
            resend_result = await asyncio.wait_for(resend_task, timeout=10.0)
            await canceler_task
    finally:
        await engine.dispose()

    assert isinstance(resend_result, InvitationNotFoundError)
    async with sessions() as check:
        assert await OrganizationInvitationRepository(check).get(issued.invitation.id) is None


async def test_sso_turned_on_while_a_resend_holds_its_admission_refuses_the_write(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # The resend's own early SSO read is the cheap refusal, taken before anything holds a lock.
    # Pausing in its admission, after that read has already answered False, lets an enable commit
    # before the resend takes its own settings lock, so the re-check under that lock must see it.
    real_admit = InvitationLimits.admit
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def admit_that_waits_for_sso_to_turn_on(
        self: InvitationLimits, *, account_id: uuid.UUID, email: str, organization_id: uuid.UUID
    ) -> Admission | InvitationLimitCeiling:
        arrived.set()
        await released.wait()
        return await real_admit(self, account_id=account_id, email=email, organization_id=organization_id)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="sso-turned-on-during-resends-admission")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            service = _invitations(setup, store=store)
            by = ActingMember(organization_id=organization.id, user_id=owner.id)
            issued = await service.invite(by, "invitee@example.com", InvitationOffer(role="member", workspaces=()))
            await setup.commit()

        # Patched only now: the setup invite above must run its own admission unpaused.
        monkeypatch.setattr(InvitationLimits, "admit", admit_that_waits_for_sso_to_turn_on)

        async with sessions() as resend_session, sessions() as sso_session:

            async def resend() -> IssuedInvitation | InvitationsNeedSsoError:
                try:
                    return await _resend_service(resend_session, store=store).resend(by, issued.invitation.id)
                except InvitationsNeedSsoError as refusal:
                    return refusal

            async def enforce_sso_once_the_resend_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                organizations = OrganizationRepository(sso_session)
                row = await organizations.get(organization.id)
                assert row is not None
                await organizations.update_organization(row, {"sso_required": True})
                await sso_session.commit()
                released.set()

            resend_task = asyncio.create_task(resend())
            sso_task = asyncio.create_task(enforce_sso_once_the_resend_is_waiting())
            result = await asyncio.wait_for(resend_task, timeout=10.0)
            await sso_task
    finally:
        await engine.dispose()

    assert isinstance(result, InvitationsNeedSsoError)
    async with sessions() as check:
        row = await OrganizationInvitationRepository(check).get_in_organization(issued.invitation.id, organization.id)
    assert row is not None
    assert row.send_count == 1


async def test_a_resend_races_the_delivery_workers_claim(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()

    # The resend holds here until the worker's claim has committed, so its write lands on top
    # of a send already in flight and must end it rather than race it to the same answer.
    real_request_send = OrganizationInvitationRepository.request_send
    arrived = asyncio.Event()
    released = asyncio.Event()

    async def request_send_that_waits_for_the_claim(
        self: OrganizationInvitationRepository,
        invitation_id: uuid.UUID,
        organization_id: uuid.UUID,
        request: SendRequest,
        *,
        expected_send_count: int,
    ) -> SendResult:
        arrived.set()
        await released.wait()
        return await real_request_send(
            self, invitation_id, organization_id, request, expected_send_count=expected_send_count
        )

    monkeypatch.setattr(OrganizationInvitationRepository, "request_send", request_send_that_waits_for_the_claim)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="resend-races-the-worker")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            # A deployment that can deliver links, so the invite leaves a send due for the worker to claim.
            setup_uow = UnitOfWork(setup)
            service = Invitations(
                setup_uow,
                OrganizationRepositories.on(setup_uow),
                clock=lambda: _NOW,
                domains=OrganizationClaimedDomains(OrganizationDomainRepository(setup_uow)),
                mailer=Mailer(
                    GatewayConfig(
                        mail_from_email="otari@example.com",
                        mail_transport="smtp",
                        public_base_url="https://otari.example.com",
                        smtp_host="smtp.example.com",
                    )
                ),
                policy=_POLICY,
                store=store,
            )
            by = ActingMember(organization_id=organization.id, user_id=owner.id)
            issued = await service.invite(by, "shared@example.com", InvitationOffer(role="member", workspaces=()))
            await setup.commit()

        async with sessions() as resend_session, sessions() as worker_session:

            async def resend() -> IssuedInvitation:
                return await _resend_service(resend_session, store=store).resend(by, issued.invitation.id)

            async def claim_once_resend_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                worker_repository = OrganizationInvitationRepository(worker_session)
                claimed = await worker_repository.claim_due_send(
                    now=_NOW, held_until=_NOW + timedelta(minutes=10), max_attempts=3, token_hash="worker-claim"
                )
                assert claimed is not None
                await worker_session.commit()
                released.set()

            resend_task = asyncio.create_task(resend())
            worker_task = asyncio.create_task(claim_once_resend_is_waiting())
            resent = await asyncio.wait_for(resend_task, timeout=10.0)
            await worker_task
    finally:
        await engine.dispose()

    assert resent.accept_link is not None
    token = resent.accept_link.rsplit("token=", 1)[-1]
    async with sessions() as check:
        repository = OrganizationInvitationRepository(check)
        assert await repository.get_by_token_hash("worker-claim") is None
        row = await repository.get_by_token_hash(hash_token(token))
    assert row is not None
    assert row.delivery_attempts == 0


async def test_a_resend_renewing_an_expired_invitation_survives_a_bulk_invite_replacing_it(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = InMemoryRateLimitStore()
    address = "renewed@example.com"

    # The invite holds here, past its own decision to replace the expired row, until the resend has
    # renewed and committed it, so the delete's own expiry re-check sees the renewal rather than the
    # invite's earlier, stale read. Only the invite's own call pauses: the resend admits too, and
    # pausing that one as well would wait on a release that only the resend's own return triggers.
    real_admit = InvitationLimits.admit
    arrived = asyncio.Event()
    released = asyncio.Event()
    first_to_arrive: list[None] = []

    async def admit_that_waits_for_the_resend(
        self: InvitationLimits, *, account_id: uuid.UUID, email: str, organization_id: uuid.UUID
    ) -> Admission | InvitationLimitCeiling:
        if not first_to_arrive:
            first_to_arrive.append(None)
            arrived.set()
            await released.wait()
        return await real_admit(self, account_id=account_id, email=email, organization_id=organization_id)

    monkeypatch.setattr(InvitationLimits, "admit", admit_that_waits_for_the_resend)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="resend-renews-what-invite-replaces")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            [expired] = await OrganizationInvitationRepository(setup).add_many(
                [
                    NewInvitation(
                        created_at=_NOW - timedelta(days=10),
                        email=address,
                        expires_at=_NOW - timedelta(days=1),
                        invited_by_user_id=None,
                        next_send_at=None,
                        organization_id=organization.id,
                        role="member",
                        token_hash="expired",
                        workspace_assignments=[],
                    )
                ]
            )
            invitation_id = expired.id
            await setup.commit()

        async with sessions() as invite_session, sessions() as resend_session:
            by = ActingMember(organization_id=organization.id, user_id=owner.id)

            async def invite() -> IssuedInvitation | AddressAlreadyInvitedError:
                try:
                    return await _invitations(invite_session, store=store).invite(
                        by, address, InvitationOffer(role="member", workspaces=())
                    )
                except AddressAlreadyInvitedError as refusal:
                    return refusal

            async def resend_once_the_invite_is_waiting() -> None:
                await arrived.wait()
                await asyncio.sleep(0.2)
                await _invitations(resend_session, store=store).resend(by, invitation_id)
                await resend_session.commit()
                released.set()

            invite_task = asyncio.create_task(invite())
            resend_task = asyncio.create_task(resend_once_the_invite_is_waiting())
            result = await asyncio.wait_for(invite_task, timeout=10.0)
            await resend_task
    finally:
        await engine.dispose()

    assert isinstance(result, AddressAlreadyInvitedError)
    async with sessions() as check:
        survivor = await OrganizationInvitationRepository(check).get(invitation_id)
    assert survivor is not None
    assert survivor.expires_at > _NOW
    assert survivor.token_hash != "expired"
