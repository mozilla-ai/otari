"""Sending due invitation emails, with a recording mailer and a set clock, on SQLite."""

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.api.deps import build_invitation_delivery
from gateway.core.config import GatewayConfig
from gateway.core.tokens import hash_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger as gateway_logger
from gateway.models.organizations import OrganizationInvitation
from gateway.models.tenancy import DOMAIN_PROOF_TTL, Organization, OrganizationDomain
from gateway.repositories.identity import BlockedEmailRepository
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories
from gateway.repositories.tenancy import (
    OrganizationDomainRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceRepository,
)
from gateway.services.mail import MailDelivery, Mailer, MailMessage, SmtpTransport
from gateway.services.organizations import InvitationDelivery, OrganizationClaimedDomains
from gateway.services.organizations._delivery import CLAIM_HOLD, MAX_ATTEMPTS, POLL_SECONDS, RETRY_WAITS
from gateway.services.organizations._delivery_rule import DeliveryRule
from gateway.services.organizations._invitation_facts import InvitationFactsReader

pytestmark = pytest.mark.asyncio

# The root conftest replaces the poll for every test, so the tests of the poll keep the real one.
_poll = InvitationDelivery.poll

_NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)
_ADDRESS = "grace@invitee.test"
_SEED_HASH_PREFIX = "seed-"
_DELIVERED = MailDelivery(delivered=True, transport="test")
_NOT_DELIVERED = MailDelivery(delivered=False, transport="test", reason="refused")


def _config(*, mail_transport: str = "smtp", public_base_url: str = "https://otari.example.com") -> GatewayConfig:
    """A deployment that can deliver email, unless a setting is replaced."""
    return GatewayConfig(
        mail_from_email="otari@example.com",
        mail_transport=mail_transport,
        public_base_url=public_base_url,
        smtp_host="smtp.example.com",
    )


class _Clock:
    def __init__(self) -> None:
        self.now = _NOW

    def __call__(self) -> datetime:
        return self.now


class _Blocklist:
    def __init__(self) -> None:
        self.entries: set[str] = set()

    async def blocked_among(self, emails: Collection[str]) -> set[str]:
        return {email for email in emails if email in self.entries}


class _RecordingMailer(Mailer):
    """Records each send and answers with ``result``, after running ``during_send`` when one is set."""

    def __init__(self, **settings: str) -> None:
        super().__init__(_config(**settings))
        self.sent: list[tuple[str, MailMessage]] = []
        self.result = _DELIVERED
        self.during_send: Callable[[], Awaitable[None]] | None = None

    async def send(
        self, *, to: str, message: MailMessage, reply_to: str | None = None, sender_name: str | None = None
    ) -> MailDelivery:
        assert reply_to is None
        assert sender_name is None
        self.sent.append((to, message))
        if self.during_send is not None:
            await self.during_send()
        return self.result

    def tokens(self) -> list[str]:
        """The token in the accept link of each sent email."""
        found = [re.search(r"accept-invitation\?token=([\w-]+)", message.text) for _, message in self.sent]
        return [match.group(1) for match in found if match is not None]


class _Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.uow = UnitOfWork(session)
        self.clock = _Clock()
        self.mailer = _RecordingMailer()
        self.blocklist = _Blocklist()
        self.invitations = OrganizationRepositories.on(self.uow).invitations
        self.delivery = self.delivery_with(self.mailer)

    def delivery_with(self, mailer: Mailer) -> InvitationDelivery:
        domains = OrganizationClaimedDomains(OrganizationDomainRepository(self.uow))
        return InvitationDelivery(
            self.uow,
            OrganizationRepositories.on(self.uow),
            rule=DeliveryRule(domains, self.blocklist),
            facts=InvitationFactsReader(
                domains=domains,
                organizations=OrganizationRepository(self.uow),
                users=UserRepository(self.uow),
                workspaces=WorkspaceRepository(self.uow),
            ),
            mailer=mailer,
            clock=self.clock,
        )

    async def organization(self, name: str = "Acme") -> uuid.UUID:
        async with self.uow:
            row = Organization(name=name, slug=uuid.uuid4().hex)
            self._session.add(row)
            await self._session.flush()
            return row.id

    async def proof(self, organization_id: uuid.UUID, domain: str, *, verified_at: datetime) -> None:
        async with self.uow:
            self._session.add(
                OrganizationDomain(
                    organization_id=organization_id,
                    domain=domain,
                    verification_token=uuid.uuid4().hex,
                    verified_at=verified_at,
                )
            )
            await self._session.flush()

    async def due_invitation(
        self,
        organization_id: uuid.UUID | None = None,
        *,
        email: str = _ADDRESS,
        expires_at: datetime = _NOW + timedelta(days=7),
        declined_at: datetime | None = None,
    ) -> uuid.UUID:
        """Save an invitation whose email is due now, in a new organization unless one is given."""
        if organization_id is None:
            organization_id = await self.organization()
        async with self.uow:
            row = await self.invitations.add(
                NewInvitation(
                    email=email,
                    expires_at=expires_at,
                    invited_by_user_id=None,
                    next_send_at=_NOW,
                    organization_id=organization_id,
                    role="member",
                    token_hash=f"{_SEED_HASH_PREFIX}{uuid.uuid4().hex}",
                    workspace_assignments=[],
                )
            )
            row.declined_at = declined_at
            await self._session.flush()
            return row.id

    async def invitation(self, invitation_id: uuid.UUID) -> OrganizationInvitation | None:
        self._session.expire_all()
        async with self.uow:
            return await self.invitations.get(invitation_id)

    async def make_due(self, invitation_id: uuid.UUID) -> None:
        async with self.uow:
            row = await self.invitations.get(invitation_id)
            assert row is not None
            row.next_send_at = self.clock.now
            await self._session.flush()

    async def delete(self, invitation_id: uuid.UUID) -> None:
        async with self.uow:
            row = await self.invitations.get(invitation_id)
            assert row is not None
            assert await self.invitations.delete_in_organization(invitation_id, row.organization_id)


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


@pytest.fixture(autouse=True)
def _no_email_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OTARI_MAIL_FROM_EMAIL", "OTARI_MAIL_TRANSPORT", "OTARI_PUBLIC_BASE_URL", "OTARI_SMTP_HOST"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the ``gateway`` logger, which does not propagate."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


def _warnings(logs: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in logs.records if record.levelno >= logging.WARNING]


def _assert_send_is_finished(row: OrganizationInvitation | None, *, attempts: int = 1) -> None:
    """Assert the stored state of a finished send."""
    assert row is not None
    assert row.next_send_at is None
    assert row.delivery_attempts == attempts
    assert not row.token_hash.startswith(_SEED_HASH_PREFIX)
    assert (row.send_count, row.last_send_requested_at) == (0, None)


def _assert_logs_hold_no_send_values(
    logs: pytest.LogCaptureFixture, row: OrganizationInvitation, store: _Store
) -> None:
    text = "\n".join(record.getMessage() for record in logs.records)
    for value in (_ADDRESS, "invitee.test", row.token_hash, *store.mailer.tokens()):
        assert value not in text


async def test_nothing_is_sent_when_no_email_is_due(store: _Store) -> None:
    assert await store.delivery.send_due() is False
    assert store.mailer.sent == []


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({"public_base_url": ""}, id="no address for links"),
        pytest.param({"public_base_url": "   "}, id="a blank address for links"),
        pytest.param({"mail_transport": "console"}, id="console transport"),
        pytest.param({"mail_transport": "none"}, id="no transport"),
    ],
)
async def test_nothing_is_claimed_while_the_deployment_cannot_deliver_email(
    store: _Store, logs: pytest.LogCaptureFixture, settings: dict[str, str]
) -> None:
    invitation_id = await store.due_invitation()
    mailer = _RecordingMailer(**settings)

    assert await store.delivery_with(mailer).send_due() is False

    assert mailer.sent == []
    row = await store.invitation(invitation_id)
    assert row is not None
    assert (row.next_send_at, row.delivery_attempts) == (_NOW, 0)
    assert row.token_hash.startswith(_SEED_HASH_PREFIX)
    assert logs.records == []

    assert await store.delivery.send_due() is True
    assert len(store.mailer.sent) == 1


async def test_a_delivered_email_finishes_the_send_and_carries_the_link_whose_hash_is_stored(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation()

    assert await store.delivery.send_due() is True
    assert await store.delivery.send_due() is False

    row = await store.invitation(invitation_id)
    _assert_send_is_finished(row)
    assert row is not None
    [(to, message)] = store.mailer.sent
    assert to == _ADDRESS
    [token] = store.mailer.tokens()
    assert f"https://otari.example.com/#/accept-invitation?token={token}" in message.text
    assert hash_token(token) == row.token_hash
    assert _warnings(logs) == []
    _assert_logs_hold_no_send_values(logs, row, store)


async def test_the_email_names_the_organization_and_states_when_the_invitation_expires(store: _Store) -> None:
    organization_id = await store.organization("Lovelace Labs")
    await store.due_invitation(organization_id, expires_at=datetime(2026, 6, 8, 17, 59, tzinfo=UTC))

    assert await store.delivery.send_due() is True

    [(_, message)] = store.mailer.sent
    assert "Lovelace Labs" in message.subject
    assert "The invitation expires on 8 June 2026 at 17:59 UTC." in message.text


async def test_an_email_that_is_not_delivered_is_due_again_after_each_wait_and_then_given_up(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation()
    store.mailer.result = _NOT_DELIVERED

    assert await store.delivery.send_due() is True
    first = await store.invitation(invitation_id)
    assert first is not None
    assert first.next_send_at == _NOW + timedelta(minutes=5)
    assert await store.delivery.send_due() is False

    store.clock.now = _NOW + timedelta(minutes=5)
    assert await store.delivery.send_due() is True
    second = await store.invitation(invitation_id)
    assert second is not None
    assert second.next_send_at == _NOW + timedelta(minutes=15)
    assert _warnings(logs) == []

    store.clock.now = _NOW + timedelta(minutes=15)
    assert await store.delivery.send_due() is True
    third = await store.invitation(invitation_id)
    assert third is not None
    assert third.next_send_at is None
    assert third.delivery_attempts == MAX_ATTEMPTS

    store.clock.now = _NOW + timedelta(days=1)
    assert await store.delivery.send_due() is False
    assert len(store.mailer.sent) == MAX_ATTEMPTS
    assert len(set(store.mailer.tokens())) == MAX_ATTEMPTS
    [warning] = _warnings(logs)
    assert str(invitation_id) in warning
    assert str(third.organization_id) in warning
    _assert_logs_hold_no_send_values(logs, third, store)


async def test_a_later_attempt_that_is_delivered_finishes_the_send(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation()
    store.mailer.result = _NOT_DELIVERED
    assert await store.delivery.send_due() is True

    store.mailer.result = _DELIVERED
    store.clock.now = _NOW + timedelta(minutes=5)
    assert await store.delivery.send_due() is True

    row = await store.invitation(invitation_id)
    _assert_send_is_finished(row, attempts=2)
    assert row is not None
    assert hash_token(store.mailer.tokens()[-1]) == row.token_hash
    assert _warnings(logs) == []


async def test_no_email_goes_to_an_address_at_a_domain_another_organization_has_proven(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    await store.proof(await store.organization("Other"), "invitee.test", verified_at=_NOW - timedelta(days=1))
    invitation_id = await store.due_invitation()

    assert await store.delivery.send_due() is True

    assert store.mailer.sent == []
    _assert_send_is_finished(await store.invitation(invitation_id))
    assert logs.records == []


async def test_no_email_goes_to_an_address_on_the_blocklist(store: _Store, logs: pytest.LogCaptureFixture) -> None:
    store.blocklist.entries.add(_ADDRESS)
    invitation_id = await store.due_invitation()

    assert await store.delivery.send_due() is True

    assert store.mailer.sent == []
    _assert_send_is_finished(await store.invitation(invitation_id))
    assert logs.records == []


async def test_no_email_is_sent_for_an_invitation_that_has_expired(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation(expires_at=_NOW)

    assert await store.delivery.send_due() is True

    assert store.mailer.sent == []
    _assert_send_is_finished(await store.invitation(invitation_id))
    assert logs.records == []


async def test_no_email_is_sent_for_an_invitation_that_was_declined(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation(declined_at=_NOW - timedelta(hours=1))

    assert await store.delivery.send_due() is True

    assert store.mailer.sent == []
    _assert_send_is_finished(await store.invitation(invitation_id))
    assert logs.records == []


async def test_no_email_is_sent_when_the_organization_no_longer_exists(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation(uuid.uuid4())

    assert await store.delivery.send_due() is True

    assert store.mailer.sent == []
    _assert_send_is_finished(await store.invitation(invitation_id))
    assert logs.records == []


async def test_a_finished_send_logs_nothing(store: _Store, logs: pytest.LogCaptureFixture) -> None:
    await store.due_invitation()

    assert await store.delivery.send_due() is True

    assert len(store.mailer.sent) == 1
    assert logs.records == []


async def test_an_invitation_deleted_during_the_send_is_no_error(store: _Store, logs: pytest.LogCaptureFixture) -> None:
    for result in (_DELIVERED, _NOT_DELIVERED):
        invitation_id = await store.due_invitation(email=f"{uuid.uuid4().hex}@invitee.test")
        store.mailer.result = result

        async def delete(invitation_id: uuid.UUID = invitation_id) -> None:
            await store.delete(invitation_id)

        store.mailer.during_send = delete

        assert await store.delivery.send_due() is True
        assert await store.invitation(invitation_id) is None

    assert len(store.mailer.sent) == 2
    assert _warnings(logs) == []


async def test_a_result_is_not_recorded_on_an_invitation_that_a_later_claim_holds(store: _Store) -> None:
    invitation_id = await store.due_invitation()
    store.mailer.result = _NOT_DELIVERED

    async def claim_again() -> None:
        async with store.uow:
            await store.invitations.claim_due_send(
                now=_NOW + CLAIM_HOLD, held_until=_NOW + 3 * CLAIM_HOLD, max_attempts=MAX_ATTEMPTS, token_hash="later"
            )

    store.mailer.during_send = claim_again

    assert await store.delivery.send_due() is True

    row = await store.invitation(invitation_id)
    assert row is not None
    assert (row.token_hash, row.next_send_at, row.delivery_attempts) == ("later", _NOW + 3 * CLAIM_HOLD, 2)


async def _stop_mid_send() -> None:
    raise asyncio.CancelledError


async def test_a_send_that_stops_before_it_reports_keeps_the_hash_of_its_link_and_is_claimed_again_after_the_hold(
    store: _Store,
) -> None:
    invitation_id = await store.due_invitation()
    store.mailer.during_send = _stop_mid_send

    with pytest.raises(asyncio.CancelledError):
        await store.delivery.send_due()

    stopped = await store.invitation(invitation_id)
    assert stopped is not None
    assert stopped.token_hash == hash_token(store.mailer.tokens()[0])
    assert stopped.next_send_at == _NOW + CLAIM_HOLD

    store.mailer.during_send = None
    store.clock.now = _NOW + CLAIM_HOLD - timedelta(seconds=1)
    assert await store.delivery.send_due() is False

    store.clock.now = _NOW + CLAIM_HOLD
    assert await store.delivery.send_due() is True

    row = await store.invitation(invitation_id)
    _assert_send_is_finished(row, attempts=2)
    assert row is not None
    assert row.token_hash == hash_token(store.mailer.tokens()[1])


async def test_an_invitation_whose_attempts_all_stopped_before_they_reported_is_not_claimed_again(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation()
    store.mailer.during_send = _stop_mid_send
    for _ in range(MAX_ATTEMPTS):
        with pytest.raises(asyncio.CancelledError):
            await store.delivery.send_due()
        store.clock.now += CLAIM_HOLD

    store.mailer.during_send = None
    assert await store.delivery.send_due() is False

    assert len(store.mailer.sent) == MAX_ATTEMPTS
    row = await store.invitation(invitation_id)
    assert row is not None
    assert (row.next_send_at, row.delivery_attempts) == (None, MAX_ATTEMPTS)
    assert row.token_hash == hash_token(store.mailer.tokens()[-1])
    assert _warnings(logs) == []


async def test_an_invitation_that_is_due_after_its_attempts_are_used_gets_no_email(
    store: _Store, logs: pytest.LogCaptureFixture
) -> None:
    invitation_id = await store.due_invitation()
    store.mailer.result = _NOT_DELIVERED
    for wait in (*RETRY_WAITS, timedelta(0)):
        assert await store.delivery.send_due() is True
        store.clock.now += wait
    assert len(_warnings(logs)) == 1
    await store.make_due(invitation_id)

    assert await store.delivery.send_due() is True
    assert await store.delivery.send_due() is False

    assert len(store.mailer.sent) == MAX_ATTEMPTS
    row = await store.invitation(invitation_id)
    assert row is not None
    assert (row.next_send_at, row.delivery_attempts) == (None, MAX_ATTEMPTS + 1)
    assert len(_warnings(logs)) == 1


async def test_an_email_is_delivered_when_the_proof_of_another_organization_has_expired(store: _Store) -> None:
    lapsed = _NOW - DOMAIN_PROOF_TTL - timedelta(seconds=1)
    await store.proof(await store.organization("Other"), "invitee.test", verified_at=lapsed)
    invitation_id = await store.due_invitation()

    assert await store.delivery.send_due() is True

    assert [to for to, _ in store.mailer.sent] == [_ADDRESS]
    _assert_send_is_finished(await store.invitation(invitation_id))


async def test_the_delivery_built_for_a_worker_job_reads_the_stored_blocklist_and_sends_through_the_deployment(
    store: _Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent_to: list[str] = []
    monkeypatch.setattr(SmtpTransport, "deliver", lambda self, envelope: sent_to.append(envelope.to))
    async with store.uow:
        await BlockedEmailRepository(store.uow).add(
            pattern="blocked.test", reason=None, created_by_user_id=uuid.uuid4()
        )
    organization_id = await store.organization()
    expires_at = datetime.now(UTC) + timedelta(days=7)
    blocked_id = await store.due_invitation(organization_id, email="ada@blocked.test", expires_at=expires_at)
    delivery = build_invitation_delivery(store.uow, _config())

    assert await delivery.send_due() is True
    _assert_send_is_finished(await store.invitation(blocked_id))
    assert sent_to == []

    allowed_id = await store.due_invitation(organization_id, expires_at=expires_at)
    assert await delivery.send_due() is True
    _assert_send_is_finished(await store.invitation(allowed_id))
    assert sent_to == [_ADDRESS]


@pytest.mark.parametrize("stored", ["grace@invitee.test(x.y)", "x<grace@invitee.test>"])
async def test_no_email_is_sent_for_a_stored_address_that_is_not_a_plain_address(
    store: _Store, monkeypatch: pytest.MonkeyPatch, stored: str
) -> None:
    sessions: list[object] = []
    monkeypatch.setattr("gateway.services.mail.transports.smtplib.SMTP", lambda *args, **kwargs: sessions.append(args))
    other = await store.organization("Other")
    await store.proof(other, "invitee.test", verified_at=datetime.now(UTC))
    invitation_id = await store.due_invitation(expires_at=datetime.now(UTC) + timedelta(days=7))
    async with store.uow:
        row = await store.invitations.get(invitation_id)
        assert row is not None
        row.email = stored
    delivery = build_invitation_delivery(store.uow, _config())

    assert await delivery.send_due() is True
    _assert_send_is_finished(await store.invitation(invitation_id))
    assert sessions == []


class _ScriptedDelivery:
    """Stands in for the delivery a poll builds, and stops the poll when its answers run out."""

    def __init__(self, answers: list[bool | Exception]) -> None:
        self._answers = answers
        self.calls = 0

    async def send_due(self) -> bool:
        self.calls += 1
        if not self._answers:
            raise asyncio.CancelledError
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def poll_without_a_database(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Give each poll job a Unit of Work with no session, and record each sleep without waiting."""
    sleeps: list[float] = []

    @asynccontextmanager
    async def unit_of_work() -> AsyncIterator[object]:
        yield object()

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("gateway.services.organizations._poll.create_unit_of_work", unit_of_work)
    monkeypatch.setattr("gateway.services.organizations._poll.asyncio.sleep", sleep)
    return sleeps


async def test_a_poll_sends_every_due_email_before_it_waits_again(poll_without_a_database: list[float]) -> None:
    delivery = _ScriptedDelivery([True, True, False, False])

    with pytest.raises(asyncio.CancelledError):
        await _poll(lambda uow: delivery)  # type: ignore[arg-type,return-value]

    assert delivery.calls == 5
    assert poll_without_a_database == [POLL_SECONDS, POLL_SECONDS, POLL_SECONDS]


async def test_a_poll_keeps_going_after_an_error_and_logs_only_its_class(
    poll_without_a_database: list[float], logs: pytest.LogCaptureFixture
) -> None:
    delivery = _ScriptedDelivery([TimeoutError(_ADDRESS), ValueError(_ADDRESS), False])

    with pytest.raises(asyncio.CancelledError):
        await _poll(lambda uow: delivery, interval=1.0)  # type: ignore[arg-type,return-value]

    assert delivery.calls == 4
    warnings = _warnings(logs)
    assert len(warnings) == 2
    assert "TimeoutError" in warnings[0]
    assert "ValueError" in warnings[1]
    assert _ADDRESS not in "\n".join(warnings)
