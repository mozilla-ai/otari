"""What an invitation link offers an invitee, and declining it."""

import uuid
from collections.abc import AsyncIterator, Collection, Sequence
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.config import GatewayConfig
from gateway.core.tokens import generate_shared_token, hash_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import (
    InvitationAddressMismatchError,
    InvitationNeedsSsoSignInError,
    InvitationNoLongerValidError,
)
from gateway.models.organizations import WorkspaceAssignment
from gateway.models.tenancy import (
    Organization,
    OrganizationDomain,
    OrganizationMemberRole,
    WorkspaceAssignmentRequest,
)
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.mail import Mailer
from gateway.services.organizations import (
    AdmittedMember,
    DeliveryRule,
    InvitationFactsReader,
    OrganizationClaimedDomains,
)
from gateway.services.organizations._acceptance import (
    Acceptance,
    AcceptingPerson,
    InvitationAcceptance,
    _require_the_invited_address,
)
from gateway.services.organizations._invitation_link import InvitationLink

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self.now = _NOW

    def __call__(self) -> datetime:
        return self.now


class _NoBlockedAddresses:
    """A blocklist that matches nothing, for a test that is not about it."""

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        del emails
        return set()


class _EverythingBlocked:
    """A blocklist that matches every address it is asked about."""

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        return set(emails)


class _Sso:
    """An SSO enforcement that refuses, or admits, whoever asks."""

    def __init__(self, *, refuses: bool) -> None:
        self._refuses = refuses
        self.asked: list[tuple[uuid.UUID, str, uuid.UUID]] = []

    async def blocks_joining(
        self, *, organization_id: uuid.UUID, role: OrganizationMemberRole, user_id: uuid.UUID
    ) -> bool:
        self.asked.append((organization_id, role, user_id))
        return self._refuses


class _RecordingAdmission:
    """Stands in for the collaborator that admits a member, recording what it was asked to do."""

    def __init__(self, member_id: uuid.UUID) -> None:
        self._member_id = member_id
        self.calls: list[tuple[uuid.UUID, str, uuid.UUID, tuple[uuid.UUID, ...]]] = []

    async def admit(
        self,
        *,
        organization_id: uuid.UUID,
        role: OrganizationMemberRole,
        user_id: uuid.UUID,
        workspaces: Sequence[WorkspaceAssignmentRequest],
    ) -> AdmittedMember:
        self.calls.append((organization_id, role, user_id, tuple(w.workspace_id for w in workspaces)))
        return AdmittedMember(created=True, member_id=self._member_id)


def _config(*, mail_transport: str = "smtp") -> GatewayConfig:
    """A deployment that can deliver links, unless given a transport that cannot."""
    return GatewayConfig(
        mail_from_email="otari@example.com",
        mail_transport=mail_transport,
        public_base_url="https://otari.example.com",
        smtp_host="smtp.example.com",
    )


class _Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.uow = UnitOfWork(session)
        self.clock = _Clock()
        self.repositories = OrganizationRepositories.on(self.uow)

    def acceptance(
        self,
        *,
        admission: object | None = None,
        blocked: object | None = None,
        mail_transport: str = "smtp",
        sso: object | None = None,
    ) -> InvitationAcceptance:
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(self.uow))
        return InvitationAcceptance(
            self.uow,
            OrganizationRepositories.on(self.uow),
            admission=admission or _RecordingAdmission(uuid.uuid4()),  # type: ignore[arg-type]
            clock=self.clock,
            link=self.link(blocked=blocked, mail_transport=mail_transport),
            facts=InvitationFactsReader(
                domains=domains,
                organizations=OrganizationRepository(self.uow),
                users=UserRepository(self.uow),
                workspaces=WorkspaceRepository(self.uow),
            ),
            sso=sso or _Sso(refuses=False),  # type: ignore[arg-type]
        )

    def link(self, *, blocked: object | None = None, mail_transport: str = "smtp") -> InvitationLink:
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(self.uow))
        blocklist = blocked if blocked is not None else _NoBlockedAddresses()
        return InvitationLink(
            self.uow,
            OrganizationRepositories.on(self.uow),
            blocked=blocklist,  # type: ignore[arg-type]
            clock=self.clock,
            delivery=DeliveryRule(domains, blocklist),  # type: ignore[arg-type]
            mailer=Mailer(_config(mail_transport=mail_transport)),
        )

    async def member(self, organization: Organization, *, full_name: str = "Joiner") -> tuple[uuid.UUID, uuid.UUID]:
        """An identity and a membership of this organization, returning both ids."""
        async with self.uow:
            user = await UserRepository(self.uow).create_local_identity(
                full_name=full_name, active_organization_id=organization.id
            )
            membership = await OrganizationMemberRepository(self.uow).create_membership(
                organization_id=organization.id, user_id=user.id, role="member"
            )
            return user.id, membership.id

    async def proven_domain_elsewhere(self, domain: str) -> None:
        """Another organization, holding a live proof of ``domain``."""
        async with self.uow:
            other = Organization(name="Other", slug=uuid.uuid4().hex)
            self._session.add(other)
            await self._session.flush()
            self._session.add(
                OrganizationDomain(
                    organization_id=other.id,
                    domain=domain,
                    verification_token="token",
                    verified_at=self.clock.now,
                )
            )
            await self._session.flush()

    async def account_for(self, email: str) -> None:
        """An account holding ``email``, so a test can prove an answer does not depend on one."""
        async with self.uow:
            await UserRepository(self.uow).create_local_identity(
                full_name="Holder", active_organization_id=uuid.uuid4(), email=email
            )

    async def organization(self, name: str = "Acme") -> Organization:
        async with self.uow:
            row = Organization(name=name, slug=uuid.uuid4().hex)
            self._session.add(row)
            await self._session.flush()
            return row

    async def invitation(
        self,
        organization: Organization,
        *,
        email: str = "ada@example.com",
        token: str = "plain-token",
        expires_in: timedelta = timedelta(days=7),
        workspaces: list[WorkspaceAssignment] | None = None,
    ) -> uuid.UUID:
        async with self.uow:
            row = await self.repositories.invitations.add(
                NewInvitation(
                    created_at=self.clock.now,
                    email=email,
                    expires_at=self.clock.now + expires_in,
                    invited_by_user_id=None,
                    next_send_at=None,
                    organization_id=organization.id,
                    role="member",
                    token_hash=hash_token(token),
                    workspace_assignments=workspaces or [],
                )
            )
            assert row is not None
            return row.id


@pytest_asyncio.fixture
async def store() -> AsyncIterator[_Store]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield _Store(session)
    finally:
        await engine.dispose()


async def test_preview_shows_the_organization_the_role_and_the_workspaces(store: _Store) -> None:
    organization = await store.organization(name="Acme")
    await store.invitation(organization)

    preview = await store.acceptance().preview("plain-token")

    assert preview.invited_email == "ada@example.com"
    assert preview.facts.organization_name == "Acme"
    assert preview.facts.role == "member"


async def test_an_expired_a_declined_and_an_unknown_link_answer_alike(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization, token="expired-token", expires_in=timedelta(days=-1))
    await store.invitation(organization, email="grace@example.com", token="declined-token")
    service = store.acceptance()
    await service.decline("declined-token")

    answers = []
    for token in ("expired-token", "declined-token", "no-such-token"):
        with pytest.raises(InvitationNoLongerValidError) as refusal:
            await service.preview(token)
        answers.append(str(refusal.value))

    assert len(set(answers)) == 1


async def test_preview_refuses_a_blocked_address(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)

    with pytest.raises(InvitationNoLongerValidError):
        await store.acceptance(blocked=_EverythingBlocked()).preview("plain-token")


async def test_decline_ends_the_link(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)
    service = store.acceptance()

    await service.decline("plain-token")

    with pytest.raises(InvitationNoLongerValidError):
        await service.preview("plain-token")


async def test_a_second_decline_answers_like_any_invalid_link(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)
    service = store.acceptance()
    await service.decline("plain-token")

    with pytest.raises(InvitationNoLongerValidError):
        await service.decline("plain-token")


async def test_a_declined_invitation_is_still_there_for_the_admin(store: _Store) -> None:
    organization = await store.organization()
    invitation_id = await store.invitation(organization)
    service = store.acceptance()

    await service.decline("plain-token")

    async with store.uow:
        row = await store.repositories.invitations.get(invitation_id)
    assert row is not None
    assert row.declined_at == _NOW


async def test_the_proven_domain_rule_is_asked_of_a_hand_carried_link_alone(store: _Store) -> None:
    """The token's kind decides it. A link this deployment emailed reached the address itself."""
    organization = await store.organization()
    hand_carried = generate_shared_token()
    await store.invitation(organization, email="ada@proven.test", token=hand_carried)
    await store.invitation(organization, email="bob@proven.test", token="emailed-token")
    await store.proven_domain_elsewhere("proven.test")

    # An emailed link is not held to another organization's proof, whether or not mail works now.
    assert (await store.acceptance().preview("emailed-token")).invited_email == "bob@proven.test"
    undelivered = store.acceptance(mail_transport="console")
    assert (await undelivered.preview("emailed-token")).invited_email == "bob@proven.test"

    # A hand-carried one is held to it, and configuring mail afterwards does not release it.
    with pytest.raises(InvitationNoLongerValidError):
        await undelivered.preview(hand_carried)
    with pytest.raises(InvitationNoLongerValidError):
        await store.acceptance().preview(hand_carried)


async def test_preview_answers_alike_whether_or_not_the_address_holds_an_account(store: _Store) -> None:
    organization = await store.organization(name="Acme")
    await store.invitation(organization, email="held@example.com", token="held-token")
    await store.invitation(organization, email="free@example.com", token="free-token")
    await store.account_for("held@example.com")
    service = store.acceptance()

    held = await service.preview("held-token")
    free = await service.preview("free-token")

    assert held.facts == free.facts


async def test_a_session_on_another_address_cannot_accept() -> None:
    with pytest.raises(InvitationAddressMismatchError, match="ada@example.com"):
        _require_the_invited_address(
            "ada@example.com", AcceptingPerson(email="grace@example.com", user_id=uuid.uuid4())
        )


async def test_the_invited_address_matches_in_any_spelling() -> None:
    _require_the_invited_address("ada@example.com", AcceptingPerson(email="  ADA@Example.com  ", user_id=uuid.uuid4()))


async def test_two_values_that_are_not_addresses_never_match() -> None:
    with pytest.raises(InvitationAddressMismatchError):
        _require_the_invited_address("not an address", AcceptingPerson(email="not an address", user_id=uuid.uuid4()))


async def test_accept_admits_the_person_on_the_terms_the_invitation_offers(store: _Store) -> None:
    organization = await store.organization(name="Acme")
    workspace_id = uuid.uuid4()
    await store.invitation(organization, workspaces=[{"workspace_id": str(workspace_id), "role": "member"}])
    user_id, member_id = await store.member(organization)
    admission = _RecordingAdmission(member_id)

    accepted = await store.acceptance(admission=admission).accept(
        "plain-token", by=AcceptingPerson(email="ada@example.com", user_id=user_id)
    )

    assert accepted == Acceptance(organization_id=organization.id, organization_name="Acme", role="member")
    assert admission.calls == [(organization.id, "member", user_id, (workspace_id,))]


async def test_accept_ends_the_link(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)
    user_id, member_id = await store.member(organization)
    service = store.acceptance(admission=_RecordingAdmission(member_id))
    by = AcceptingPerson(email="ada@example.com", user_id=user_id)

    await service.accept("plain-token", by=by)

    with pytest.raises(InvitationNoLongerValidError):
        await service.accept("plain-token", by=by)


async def test_accept_refuses_a_session_on_another_address_and_keeps_the_invitation(store: _Store) -> None:
    organization = await store.organization()
    invitation_id = await store.invitation(organization)
    user_id, member_id = await store.member(organization)
    admission = _RecordingAdmission(member_id)

    with pytest.raises(InvitationAddressMismatchError):
        await store.acceptance(admission=admission).accept(
            "plain-token", by=AcceptingPerson(email="grace@example.com", user_id=user_id)
        )

    assert admission.calls == []
    async with store.uow:
        assert await store.repositories.invitations.get(invitation_id) is not None


async def test_accept_records_who_invited_and_when_they_joined(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)
    user_id, member_id = await store.member(organization)

    await store.acceptance(admission=_RecordingAdmission(member_id)).accept(
        "plain-token", by=AcceptingPerson(email="ada@example.com", user_id=user_id)
    )

    async with store.uow:
        origin = await store.repositories.membership_origins.get(member_id)
    assert origin is not None
    assert (origin.invited_at, origin.joined_at) == (_NOW, _NOW)


async def test_accept_refuses_a_declined_link_and_leaves_it_for_the_admin(store: _Store) -> None:
    organization = await store.organization()
    invitation_id = await store.invitation(organization)
    user_id, member_id = await store.member(organization)
    service = store.acceptance(admission=_RecordingAdmission(member_id))
    await service.decline("plain-token")

    with pytest.raises(InvitationNoLongerValidError):
        await service.accept("plain-token", by=AcceptingPerson(email="ada@example.com", user_id=user_id))

    async with store.uow:
        assert await store.repositories.invitations.get(invitation_id) is not None


async def test_accept_is_refused_where_single_sign_on_does_not_admit_the_account(store: _Store) -> None:
    organization = await store.organization()
    invitation_id = await store.invitation(organization)
    user_id, member_id = await store.member(organization)
    admission = _RecordingAdmission(member_id)
    sso = _Sso(refuses=True)
    organization_id = organization.id

    with pytest.raises(InvitationNeedsSsoSignInError, match="ada@example.com"):
        await store.acceptance(admission=admission, sso=sso).accept(
            "plain-token", by=AcceptingPerson(email="ada@example.com", user_id=user_id)
        )

    assert sso.asked == [(organization_id, "member", user_id)]
    assert admission.calls == []
    async with store.uow:
        assert await store.repositories.invitations.get(invitation_id) is not None


async def test_accept_asks_single_sign_on_about_the_role_the_invitation_offers(store: _Store) -> None:
    organization = await store.organization()
    await store.invitation(organization)
    user_id, member_id = await store.member(organization)
    sso = _Sso(refuses=False)

    await store.acceptance(admission=_RecordingAdmission(member_id), sso=sso).accept(
        "plain-token", by=AcceptingPerson(email="ada@example.com", user_id=user_id)
    )

    assert sso.asked == [(organization.id, "member", user_id)]
