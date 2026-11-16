"""Removing expired invitations, with a set clock, on SQLite."""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger as gateway_logger
from gateway.models.tenancy import Organization
from gateway.repositories.organizations import NewInvitation, OrganizationRepositories
from gateway.services.organizations import ExpiredInvitationCleanup
from gateway.services.organizations._cleanup import EXPIRED_KEPT_FOR

pytestmark = pytest.mark.asyncio

# The root conftest replaces the poll for every test, so the tests of the poll keep the real one.
_poll = ExpiredInvitationCleanup.poll

_NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self.now = _NOW

    def __call__(self) -> datetime:
        return self.now


class _Store:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.uow = UnitOfWork(session)
        self.clock = _Clock()
        self.invitations = OrganizationRepositories.on(self.uow).invitations
        self.cleanup = ExpiredInvitationCleanup(self.uow, OrganizationRepositories.on(self.uow), clock=self.clock)

    async def organization(self, name: str = "Acme") -> uuid.UUID:
        async with self.uow:
            row = Organization(name=name, slug=uuid.uuid4().hex)
            self._session.add(row)
            await self._session.flush()
            return row.id

    async def invitation(self, *, expires_at: datetime) -> uuid.UUID:
        """Save an invitation in a new organization, expiring at ``expires_at``."""
        organization_id = await self.organization()
        async with self.uow:
            rows = await self.invitations.add_many(
                [
                    NewInvitation(
                        created_at=_NOW,
                        email=f"{uuid.uuid4().hex}@invitee.test",
                        expires_at=expires_at,
                        invited_by_user_id=None,
                        next_send_at=None,
                        organization_id=organization_id,
                        role="member",
                        token_hash=uuid.uuid4().hex,
                        workspace_assignments=[],
                    )
                ]
            )
            assert rows
            return rows[0].id

    async def exists(self, invitation_id: uuid.UUID) -> bool:
        async with self.uow:
            return await self.invitations.get(invitation_id) is not None


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


@pytest.fixture
def logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture the ``gateway`` logger, which does not propagate."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


async def test_delete_expired_keeps_an_invitation_inside_the_keep_time(store: _Store) -> None:
    invitation_id = await store.invitation(expires_at=_NOW - EXPIRED_KEPT_FOR + timedelta(days=1))

    assert await store.cleanup.delete_expired() == 0
    assert await store.exists(invitation_id) is True


async def test_delete_expired_removes_one_past_it(store: _Store, logs: pytest.LogCaptureFixture) -> None:
    invitation_id = await store.invitation(expires_at=_NOW - EXPIRED_KEPT_FOR - timedelta(days=1))

    assert await store.cleanup.delete_expired() == 1

    assert await store.exists(invitation_id) is False
    assert [record.getMessage() for record in logs.records] == ["Invitation cleanup removed 1 expired invitations"]


async def test_delete_expired_leaves_a_pending_invitation(store: _Store) -> None:
    invitation_id = await store.invitation(expires_at=_NOW + timedelta(days=7))

    assert await store.cleanup.delete_expired() == 0
    assert await store.exists(invitation_id) is True


async def test_delete_expired_respects_the_batch_size(store: _Store, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gateway.services.organizations._cleanup.CLEANUP_BATCH_SIZE", 2)
    ids = [await store.invitation(expires_at=_NOW - EXPIRED_KEPT_FOR - timedelta(days=i + 1)) for i in range(3)]

    deleted = await store.cleanup.delete_expired()

    assert deleted == 2
    survivors = [invitation_id for invitation_id in ids if await store.exists(invitation_id)]
    assert len(survivors) == 1


class _ScriptedCleanup:
    """Stands in for the cleanup a poll builds, and stops the poll when its answers run out."""

    def __init__(self, answers: list[int | Exception]) -> None:
        self._answers = answers
        self.calls = 0

    async def delete_expired(self) -> int:
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


async def test_cleanup_poll_keeps_running_after_an_error(
    poll_without_a_database: list[float], logs: pytest.LogCaptureFixture
) -> None:
    cleanup = _ScriptedCleanup([TimeoutError("boom"), ValueError("boom"), 0])

    with pytest.raises(asyncio.CancelledError):
        await _poll(lambda uow: cleanup, interval=1.0)  # type: ignore[arg-type,return-value]

    assert cleanup.calls == 4
    warnings = [record.getMessage() for record in logs.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 2
    assert "TimeoutError" in warnings[0]
    assert "ValueError" in warnings[1]


async def test_poll_runs_the_step_again_while_a_batch_fills(
    poll_without_a_database: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gateway.services.organizations._cleanup.CLEANUP_BATCH_SIZE", 2)
    cleanup = _ScriptedCleanup([2, 2, 1])

    with pytest.raises(asyncio.CancelledError):
        await _poll(lambda uow: cleanup, interval=1.0)  # type: ignore[arg-type,return-value]

    # Two full batches ran back to back before the step said a short one meant it was done.
    assert cleanup.calls == 4
