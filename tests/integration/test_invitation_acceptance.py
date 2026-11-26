"""Accepting an invitation against real PostgreSQL transactions, for the races SQLite cannot exercise."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.tokens import generate_shared_token, generate_token, hash_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import InvitationNoLongerValidError, MembershipUpdateError
from gateway.models.identity import AddressAuthority
from gateway.models.organizations import OrganizationInvitation
from gateway.models.tenancy import Organization, OrganizationDomain, User, Workspace, WorkspaceMember
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories, SendRequest
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.mail import Mailer
from gateway.services.organizations import (
    DeliveryRule,
    InvitationFactsReader,
    OrganizationClaimedDomains,
)
from gateway.services.organizations._acceptance import AcceptingPerson, InvitationAcceptance
from gateway.services.organizations._invitation_link import InvitationLink
from gateway.services.tenancy import OrganizationService

from .conftest import _to_async_url
from .tenancy_helpers import (
    NO_BLOCKED_ADDRESSES,
    NO_SSO_ENFORCEMENT,
    create_member,
    create_organization,
    create_workspace,
    membership_listener_on,
)

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 7, 1, 12, tzinfo=UTC)
# Bounds a whole race, so an actor whose checkpoint is never reached fails the
# test rather than hanging it. Generous, because a cold connection is slow.
_RACE_DEADLINE = 10.0
_TOKEN = "an-invitation-token"


def _config(*, mail_transport: str = "smtp") -> GatewayConfig:
    """A deployment that can deliver links by default, so the delivery rule is not consulted."""
    return GatewayConfig(
        mail_from_email="otari@example.com",
        mail_transport=mail_transport,
        public_base_url="https://otari.example.com",
        smtp_host="smtp.example.com",
    )


def _acceptance(session: AsyncSession) -> InvitationAcceptance:
    """The acceptance service with the real admission collaborator, as a route would hold it."""
    uow = UnitOfWork(session)
    domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
    return InvitationAcceptance(
        uow,
        OrganizationRepositories.on(uow),
        admission=OrganizationService(session, membership_listener=membership_listener_on(uow), uow=uow),
        clock=lambda: _NOW,
        link=_link(session, uow=uow),
        facts=InvitationFactsReader(
            domains=domains,
            organizations=OrganizationRepository(uow),
            users=UserRepository(uow),
            workspaces=WorkspaceRepository(uow),
        ),
        sso=NO_SSO_ENFORCEMENT,
    )


def _link(session: AsyncSession, *, uow: UnitOfWork | None = None, mail_transport: str = "smtp") -> InvitationLink:
    """The one reader of what a link still opens, on a deployment that can deliver them.

    Built on the caller's Unit of Work where there is one, because its repositories read inside that
    caller's block.
    """
    uow = uow or UnitOfWork(session)
    domains = OrganizationClaimedDomains(OrganizationDomainRepository(uow))
    return InvitationLink(
        uow,
        OrganizationRepositories.on(uow),
        blocked=NO_BLOCKED_ADDRESSES,
        clock=lambda: _NOW,
        delivery=DeliveryRule(domains, NO_BLOCKED_ADDRESSES),
        mailer=Mailer(_config(mail_transport=mail_transport)),
    )


async def _invite(
    session: AsyncSession,
    organization: Organization,
    *,
    email: str,
    token: str = _TOKEN,
    workspace_ids: list[uuid.UUID] | None = None,
) -> None:
    uow = UnitOfWork(session)
    async with uow:
        row = await OrganizationRepositories.on(uow).invitations.add(
            NewInvitation(
                created_at=_NOW,
                email=email,
                expires_at=_NOW + timedelta(days=7),
                invited_by_user_id=None,
                next_send_at=None,
                organization_id=organization.id,
                role="member",
                token_hash=hash_token(token),
                workspace_assignments=[
                    {"workspace_id": str(workspace_id), "role": "member"} for workspace_id in (workspace_ids or [])
                ],
            )
        )
        assert row is not None


async def _invitee(session: AsyncSession, organization: Organization, *, email: str) -> User:
    """An identity holding the invited address and a member of nothing."""
    return await UserRepository(session).create_local_identity(
        full_name="Invitee", active_organization_id=organization.id, email=email
    )


@pytest.fixture(autouse=True)
def _no_mail_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OTARI_MAIL_FROM_EMAIL", "OTARI_MAIL_TRANSPORT", "OTARI_PUBLIC_BASE_URL", "OTARI_SMTP_HOST"):
        monkeypatch.delenv(name, raising=False)


async def test_two_accepts_of_one_link_give_one_membership(postgres_url: str) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    address = "ada@example.com"

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="accept-raced")
            await create_member(setup, organization, role="owner", full_name="Owner")
            invitee = await _invitee(setup, organization, email=address)
            await setup.commit()
            await _invite(setup, organization, email=address)
            await setup.commit()
            by = AcceptingPerson(email=address, user_id=invitee.id)

        async with sessions() as first, sessions() as second:
            outcomes = await asyncio.gather(
                _acceptance(first).accept(_TOKEN, by=by),
                _acceptance(second).accept(_TOKEN, by=by),
                return_exceptions=True,
            )

        async with sessions() as check:
            memberships = (
                await check.execute(
                    select(col(OrganizationMemberRepository(check).model_class.id)).where(
                        col(OrganizationMemberRepository(check).model_class.user_id) == invitee.id
                    )
                )
            ).all()
    finally:
        await engine.dispose()

    refused = [outcome for outcome in outcomes if isinstance(outcome, InvitationNoLongerValidError)]
    accepted = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    assert len(accepted) == 1
    assert len(refused) == 1
    assert len(memberships) == 1


async def test_an_accept_racing_a_cancel_finds_the_link_gone(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    address = "grace@example.com"
    arrived = asyncio.Event()
    released = asyncio.Event()
    real_delete = OrganizationInvitationRepository.delete_by_token_hash

    async def delete_that_waits(
        self: OrganizationInvitationRepository, token_hash: str
    ) -> OrganizationInvitation | None:
        arrived.set()
        await released.wait()
        return await real_delete(self, token_hash)

    monkeypatch.setattr(OrganizationInvitationRepository, "delete_by_token_hash", delete_that_waits)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="accept-vs-cancel")
            await create_member(setup, organization, role="owner", full_name="Owner")
            invitee = await _invitee(setup, organization, email=address)
            await setup.commit()
            await _invite(setup, organization, email=address)
            await setup.commit()
            by = AcceptingPerson(email=address, user_id=invitee.id)

        async with sessions() as accepting, sessions() as cancelling:

            async def accept() -> InvitationNoLongerValidError | object:
                try:
                    return await _acceptance(accepting).accept(_TOKEN, by=by)
                except InvitationNoLongerValidError as refusal:
                    return refusal

            async def cancel_once_the_accept_is_waiting() -> None:
                await arrived.wait()
                uow = UnitOfWork(cancelling)
                async with uow:
                    repositories = OrganizationRepositories.on(uow)
                    rows = await repositories.invitations.get_by_organization_and_emails(organization.id, [address])
                    assert rows
                    assert await repositories.invitations.delete_in_organization(rows[0].id, organization.id)
                released.set()

            async with asyncio.timeout(_RACE_DEADLINE), asyncio.TaskGroup() as actors:
                accept_task = actors.create_task(accept())
                actors.create_task(cancel_once_the_accept_is_waiting())
            outcome = accept_task.result()
    finally:
        await engine.dispose()

    assert isinstance(outcome, InvitationNoLongerValidError)


async def test_an_accept_racing_a_decline_is_refused(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A decline that commits while an acceptance is in flight ends the link, so the acceptance finds none."""
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    address = "pia@example.com"
    arrived = asyncio.Event()
    released = asyncio.Event()
    real_delete = OrganizationInvitationRepository.delete_by_token_hash

    async def delete_that_waits(
        self: OrganizationInvitationRepository, token_hash: str
    ) -> OrganizationInvitation | None:
        arrived.set()
        await released.wait()
        return await real_delete(self, token_hash)

    monkeypatch.setattr(OrganizationInvitationRepository, "delete_by_token_hash", delete_that_waits)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="accept-vs-decline")
            await create_member(setup, organization, role="owner", full_name="Owner")
            invitee = await _invitee(setup, organization, email=address)
            await setup.commit()
            await _invite(setup, organization, email=address)
            await setup.commit()
            by = AcceptingPerson(email=address, user_id=invitee.id)

        async with sessions() as accepting, sessions() as declining:

            async def accept() -> InvitationNoLongerValidError | object:
                try:
                    return await _acceptance(accepting).accept(_TOKEN, by=by)
                except InvitationNoLongerValidError as refusal:
                    return refusal

            async def decline_once_the_accept_is_waiting() -> None:
                await arrived.wait()
                uow = UnitOfWork(declining)
                async with uow:
                    declined = await OrganizationRepositories.on(uow).invitations.decline_by_token_hash(
                        hash_token(_TOKEN), at=_NOW
                    )
                    assert declined is not None
                released.set()

            async with asyncio.timeout(_RACE_DEADLINE), asyncio.TaskGroup() as actors:
                accept_task = actors.create_task(accept())
                actors.create_task(decline_once_the_accept_is_waiting())
            outcome = accept_task.result()

        async with sessions() as check:
            memberships = (
                (
                    await check.execute(
                        select(col(OrganizationMemberRepository(check).model_class.id)).where(
                            col(OrganizationMemberRepository(check).model_class.user_id) == invitee.id
                        )
                    )
                )
                .scalars()
                .all()
            )
            # scalar_one, so a deleted row fails here: the DELETE must not have matched.
            declined_at = (
                await check.execute(
                    select(col(OrganizationInvitation.declined_at)).where(
                        col(OrganizationInvitation.token_hash) == hash_token(_TOKEN)
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert isinstance(outcome, InvitationNoLongerValidError)
    assert memberships == []
    assert declined_at == _NOW


async def test_an_accept_racing_a_workspace_deletion_joins_without_it(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    address = "ida@example.com"
    arrived = asyncio.Event()
    released = asyncio.Event()
    real_delete = OrganizationInvitationRepository.delete_by_token_hash

    async def delete_that_waits(
        self: OrganizationInvitationRepository, token_hash: str
    ) -> OrganizationInvitation | None:
        arrived.set()
        await released.wait()
        return await real_delete(self, token_hash)

    monkeypatch.setattr(OrganizationInvitationRepository, "delete_by_token_hash", delete_that_waits)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="accept-vs-workspace")
            owner = await create_member(setup, organization, role="owner", full_name="Owner")
            workspace = await create_workspace(setup, organization, name="Engineering", owner=owner)
            invitee = await _invitee(setup, organization, email=address)
            await setup.commit()
            await _invite(setup, organization, email=address, workspace_ids=[workspace.id])
            await setup.commit()
            by = AcceptingPerson(email=address, user_id=invitee.id)
            workspace_id = workspace.id

        async with sessions() as accepting, sessions() as deleting:

            async def delete_the_workspace_once_the_accept_is_waiting() -> None:
                await arrived.wait()
                uow = UnitOfWork(deleting)
                async with uow:
                    workspace = await deleting.get(Workspace, workspace_id)
                    assert workspace is not None
                    await WorkspaceRepository(uow).delete_workspace(workspace)
                released.set()

            async with asyncio.timeout(_RACE_DEADLINE), asyncio.TaskGroup() as actors:
                accept_task = actors.create_task(_acceptance(accepting).accept(_TOKEN, by=by))
                actors.create_task(delete_the_workspace_once_the_accept_is_waiting())
            accepted = accept_task.result()

        async with sessions() as check:
            placements = (
                (
                    await check.execute(
                        select(col(WorkspaceMember.workspace_id)).where(col(WorkspaceMember.user_id) == invitee.id)
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()

    assert accepted.organization_id == organization.id
    assert list(placements) == []


async def test_an_accept_racing_a_resend_cannot_use_the_old_link(
    postgres_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resend replaces the invitation's token hash, which is what ends the link it replaced."""
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    address = "nell@example.com"
    arrived = asyncio.Event()
    released = asyncio.Event()
    real_delete = OrganizationInvitationRepository.delete_by_token_hash

    async def delete_that_waits(
        self: OrganizationInvitationRepository, token_hash: str
    ) -> OrganizationInvitation | None:
        arrived.set()
        await released.wait()
        return await real_delete(self, token_hash)

    monkeypatch.setattr(OrganizationInvitationRepository, "delete_by_token_hash", delete_that_waits)

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="accept-vs-resend")
            await create_member(setup, organization, role="owner", full_name="Owner")
            invitee = await _invitee(setup, organization, email=address)
            await setup.commit()
            await _invite(setup, organization, email=address)
            await setup.commit()
            by = AcceptingPerson(email=address, user_id=invitee.id)

        async with sessions() as accepting, sessions() as resending:

            async def accept() -> InvitationNoLongerValidError | object:
                try:
                    return await _acceptance(accepting).accept(_TOKEN, by=by)
                except InvitationNoLongerValidError as refusal:
                    return refusal

            async def resend_once_the_accept_is_waiting() -> None:
                await arrived.wait()
                uow = UnitOfWork(resending)
                async with uow:
                    repositories = OrganizationRepositories.on(uow)
                    rows = await repositories.invitations.get_by_organization_and_emails(organization.id, [address])
                    assert rows
                    result = await repositories.invitations.request_send(
                        rows[0].id,
                        organization.id,
                        SendRequest(
                            expires_at=_NOW + timedelta(days=7),
                            invited_by_user_id=None,
                            next_send_at=None,
                            requested_at=_NOW,
                            token_hash=hash_token("the-resent-token"),
                        ),
                        expected_send_count=rows[0].send_count,
                    )
                    assert result.sent
                released.set()

            async with asyncio.timeout(_RACE_DEADLINE), asyncio.TaskGroup() as actors:
                accept_task = actors.create_task(accept())
                actors.create_task(resend_once_the_accept_is_waiting())
            outcome = accept_task.result()

        monkeypatch.undo()
        async with sessions() as second_try:
            accepted = await _acceptance(second_try).accept("the-resent-token", by=by)
    finally:
        await engine.dispose()

    assert isinstance(outcome, InvitationNoLongerValidError)
    assert accepted.organization_id == organization.id


async def test_an_accept_that_would_leave_no_active_owner_is_refused_and_keeps_the_link(postgres_url: str) -> None:
    """The last owner accepting an invitation offering less is refused, and the invitation survives it."""
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    address = "olive@example.com"

    try:
        async with sessions() as setup:
            organization = await create_organization(setup, slug="accept-last-owner")
            owner = await create_member(setup, organization, role="owner", full_name="Only Owner")
            owner.email = address
            setup.add(owner)
            await setup.commit()
            await _invite(setup, organization, email=address)
            await setup.commit()
            by = AcceptingPerson(email=address, user_id=owner.id)

        async with sessions() as accepting:
            with pytest.raises(MembershipUpdateError, match="at least one active owner"):
                await _acceptance(accepting).accept(_TOKEN, by=by)

        async with sessions() as check:
            survived = (
                await check.execute(
                    select(col(OrganizationInvitation.id)).where(
                        col(OrganizationInvitation.token_hash) == hash_token(_TOKEN)
                    )
                )
            ).all()
            role = (
                await check.execute(
                    select(col(OrganizationMemberRepository(check).model_class.role)).where(
                        col(OrganizationMemberRepository(check).model_class.user_id) == owner.id
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert len(survived) == 1
    assert role == "owner"


async def test_signup_is_told_the_address_an_invitation_was_sent_to(async_db: AsyncSession) -> None:
    organization = await create_organization(async_db, slug="lookup-live")
    await _invite(async_db, organization, email="ada@example.com")

    assert await _link(async_db).proof_for(_TOKEN, claimed_address="ada@example.com") is AddressAuthority.MAILBOX


async def test_signup_is_told_nothing_by_a_token_no_invitation_still_opens(async_db: AsyncSession) -> None:
    """An expired invitation, a declined one and an unknown token each prove nothing."""
    organization = await create_organization(async_db, slug="lookup-dead")
    await _invite(async_db, organization, email="expired@example.com", token="expired-token")
    await _invite(async_db, organization, email="declined@example.com", token="declined-token")
    uow = UnitOfWork(async_db)
    invitations = OrganizationInvitationRepository(uow)
    async with uow:
        expired = await invitations.get_by_token_hash(hash_token("expired-token"))
        assert expired is not None
        expired.expires_at = _NOW - timedelta(seconds=1)
        await invitations.decline_by_token_hash(hash_token("declined-token"), at=_NOW)
    lookup = _link(async_db)

    answers = [
        await lookup.proof_for("expired-token", claimed_address="expired@example.com"),
        await lookup.proof_for("declined-token", claimed_address="declined@example.com"),
        await lookup.proof_for("a-token-nobody-was-sent", claimed_address="ada@example.com"),
    ]

    assert answers == [None, None, None]


async def _domain_proven_elsewhere(session: AsyncSession, *, slug: str) -> Organization:
    """An organization whose claim on example.com is proven, so another one's links to it are suspect."""
    holder = await create_organization(session, slug=slug)
    session.add(
        OrganizationDomain(
            domain="example.com",
            organization_id=holder.id,
            verification_token=f"token-for-{slug}",
            verified_at=_NOW - timedelta(days=1),
        )
    )
    await session.flush()
    return holder


async def test_a_link_acceptance_refuses_proves_nothing_at_signup(async_db: AsyncSession) -> None:
    """One reader answers both, so a link the deployment turns away cannot go on proving an address.

    The deployment cannot deliver links, another organization has proven the address's domain, and
    this link was carried by hand.
    """
    await _domain_proven_elsewhere(async_db, slug="holds-the-domain")
    inviter = await create_organization(async_db, slug="carries-the-link")
    hand_carried = generate_shared_token()
    await _invite(async_db, inviter, email="ada@example.com", token=hand_carried)
    uow = UnitOfWork(async_db)
    link = _link(async_db, uow=uow, mail_transport="console")

    async with uow:
        opened = await link.open(hand_carried)
    proof = await link.proof_for(hand_carried, claimed_address="ada@example.com")

    assert (opened, proof) == (None, None)


async def test_a_link_that_was_emailed_is_not_held_to_another_organization_proven_domain(
    async_db: AsyncSession,
) -> None:
    """The rule is for a link an admin carried. One the deployment emailed reached the address itself."""
    await _domain_proven_elsewhere(async_db, slug="holds-it-too")
    inviter = await create_organization(async_db, slug="emailed-the-link")
    emailed = generate_token()
    await _invite(async_db, inviter, email="ada@example.com", token=emailed)

    proof = await _link(async_db, mail_transport="console").proof_for(emailed, claimed_address="ada@example.com")

    assert proof is AddressAuthority.MAILBOX


async def test_a_hand_carried_link_stays_held_to_it_after_the_deployment_can_deliver(
    async_db: AsyncSession,
) -> None:
    """The token's kind is the durable fact: it was minted because nobody could deliver it.

    Configuring a mail transport afterwards does not turn a link an admin is carrying into one the
    address received.
    """
    await _domain_proven_elsewhere(async_db, slug="holds-it-as-well")
    inviter = await create_organization(async_db, slug="could-have-emailed")
    hand_carried = generate_shared_token()
    await _invite(async_db, inviter, email="ada@example.com", token=hand_carried)

    proof = await _link(async_db).proof_for(hand_carried, claimed_address="ada@example.com")

    assert proof is None


async def test_a_token_proves_nothing_about_an_address_it_was_not_sent_to(async_db: AsyncSession) -> None:
    """The token alone is not the proof: it proves the address it was sent to, and no other."""
    organization = await create_organization(async_db, slug="proof-other-address")
    await _invite(async_db, organization, email="ada@example.com")

    assert await _link(async_db).proof_for(_TOKEN, claimed_address="bob@example.com") is None


async def test_a_shared_link_does_not_prove_the_address_an_emailed_one_proves(
    async_db: AsyncSession,
) -> None:
    """A link nobody delivered proves the inviting organization's word, which auto-join does not take."""
    organization = await create_organization(async_db, slug="proof-strength")
    hand_carried = generate_shared_token()
    emailed = generate_token()
    await _invite(async_db, organization, email="carried@example.com", token=hand_carried)
    await _invite(async_db, organization, email="mailed@example.com", token=emailed)
    lookup = _link(async_db)

    proofs = (
        await lookup.proof_for(hand_carried, claimed_address="carried@example.com"),
        await lookup.proof_for(emailed, claimed_address="mailed@example.com"),
    )

    assert proofs == (AddressAuthority.ORGANIZATION, AddressAuthority.MAILBOX)
