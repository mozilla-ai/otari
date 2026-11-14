"""Tests on SQLite for deleting an invitation and for claiming a due send."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.organizations import DueInvitation, NewInvitation
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository

pytestmark = pytest.mark.asyncio

_NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)
_HOLD = timedelta(minutes=2)
_HELD_UNTIL = _NOW + _HOLD
_EXPIRES_AT = datetime(2030, 1, 1, tzinfo=UTC)
_MAX_ATTEMPTS = 3


def _invitation(
    organization_id: uuid.UUID, *, email: str, token_hash: str, next_send_at: datetime | None = None
) -> NewInvitation:
    return NewInvitation(
        email=email,
        expires_at=_EXPIRES_AT,
        invited_by_user_id=None,
        next_send_at=next_send_at,
        organization_id=organization_id,
        role="member",
        token_hash=token_hash,
        workspace_assignments=[{"workspace_id": "w1", "role": "member"}],
    )


@pytest_asyncio.fixture
async def store() -> AsyncIterator[tuple[UnitOfWork, OrganizationInvitationRepository]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            uow = UnitOfWork(session)
            yield uow, OrganizationInvitationRepository(uow)
    finally:
        await engine.dispose()


async def test_deleting_an_invitation_removes_it_from_the_session_too(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store
    organization_id = uuid.uuid4()

    async with uow:
        invitation = await repository.add(_invitation(organization_id, email="ada@example.com", token_hash="t1"))

        assert await repository.delete_in_organization(invitation.id, organization_id) is True
        assert await repository.get(invitation.id) is None


async def test_a_claim_returns_the_values_of_the_invitation_and_counts_the_attempt(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store
    organization_id = uuid.uuid4()

    async with uow:
        invitation = await repository.add(
            _invitation(organization_id, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )

        first = await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="first"
        )
        second = await repository.claim_due_send(
            now=_HELD_UNTIL, held_until=_EXPIRES_AT, max_attempts=_MAX_ATTEMPTS, token_hash="second"
        )

        assert first == DueInvitation(
            attempt=1,
            declined_at=None,
            email="ada@example.com",
            expires_at=_EXPIRES_AT,
            id=invitation.id,
            invited_by_user_id=None,
            organization_id=organization_id,
            role="member",
            workspace_assignments=[{"workspace_id": "w1", "role": "member"}],
        )
        assert second is not None
        assert second.attempt == 2


async def test_a_due_invitation_is_claimed_once(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="seed", next_send_at=_NOW)
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


async def test_the_claim_of_the_last_attempt_leaves_no_send_due(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store
    later = _HELD_UNTIL + _HOLD

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )

        first = await repository.claim_due_send(now=_NOW, held_until=_HELD_UNTIL, max_attempts=2, token_hash="first")
        second = await repository.claim_due_send(now=_HELD_UNTIL, held_until=later, max_attempts=2, token_hash="second")
        third = await repository.claim_due_send(
            now=_EXPIRES_AT, held_until=_EXPIRES_AT + _HOLD, max_attempts=2, token_hash="third"
        )

        assert first is not None
        assert second is not None
        assert second.attempt == 2
        assert third is None
        assert (invitation.token_hash, invitation.delivery_attempts, invitation.next_send_at) == ("second", 2, None)


async def test_an_invitation_is_due_by_its_due_time_alone(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(now=_NOW, held_until=_HELD_UNTIL, max_attempts=1, token_hash="first")
        await repository.retry_send(invitation.id, token_hash="first", next_send_at=_HELD_UNTIL)

        again = await repository.claim_due_send(
            now=_HELD_UNTIL, held_until=_HELD_UNTIL + _HOLD, max_attempts=1, token_hash="second"
        )

        assert again is not None
        assert again.attempt == 2
        assert (invitation.token_hash, invitation.next_send_at) == ("second", None)


async def test_a_claim_that_would_hold_the_invitation_for_no_time_is_refused(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )

        with pytest.raises(ValueError, match="held"):
            await repository.claim_due_send(now=_NOW, held_until=_NOW, max_attempts=_MAX_ATTEMPTS, token_hash="claim")

        assert (invitation.token_hash, invitation.delivery_attempts, invitation.next_send_at) == ("seed", 0, _NOW)


async def test_an_invitation_with_no_send_due_is_not_claimed(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store
    organization_id = uuid.uuid4()

    async with uow:
        await repository.add(_invitation(organization_id, email="none@example.com", token_hash="none"))
        await repository.add(
            _invitation(
                organization_id,
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


async def test_the_invitation_that_has_been_due_longest_is_claimed_first(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store
    organization_id = uuid.uuid4()

    async with uow:
        newer = await repository.add(
            _invitation(organization_id, email="newer@example.com", token_hash="newer", next_send_at=_NOW)
        )
        older = await repository.add(
            _invitation(
                organization_id,
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


async def test_a_claim_replaces_the_token_hash_of_the_invitation(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="old", next_send_at=_NOW)
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


async def test_finishing_a_send_leaves_no_send_due(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="claim"
        )

        assert await repository.finish_send(invitation.id, token_hash="claim") is True
        assert invitation.next_send_at is None
        assert (
            await repository.claim_due_send(
                now=_EXPIRES_AT, held_until=_EXPIRES_AT + _HOLD, max_attempts=_MAX_ATTEMPTS, token_hash="next"
            )
            is None
        )


async def test_retrying_a_send_makes_it_due_at_the_later_time(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store
    retry_at = _NOW + timedelta(minutes=5)

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="claim"
        )

        assert await repository.retry_send(invitation.id, token_hash="claim", next_send_at=retry_at) is True
        assert invitation.next_send_at == retry_at
        assert (
            await repository.claim_due_send(
                now=retry_at - timedelta(microseconds=1),
                held_until=_EXPIRES_AT,
                max_attempts=_MAX_ATTEMPTS,
                token_hash="early",
            )
            is None
        )
        again = await repository.claim_due_send(
            now=retry_at, held_until=_EXPIRES_AT, max_attempts=_MAX_ATTEMPTS, token_hash="again"
        )
        assert again is not None
        assert again.attempt == 2


async def test_a_result_from_an_earlier_claim_changes_nothing(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store

    async with uow:
        invitation = await repository.add(
            _invitation(uuid.uuid4(), email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="earlier"
        )
        await repository.claim_due_send(
            now=_HELD_UNTIL, held_until=_EXPIRES_AT, max_attempts=_MAX_ATTEMPTS, token_hash="current"
        )

        assert await repository.finish_send(invitation.id, token_hash="earlier") is False
        assert await repository.retry_send(invitation.id, token_hash="earlier", next_send_at=_NOW) is False
        found = await repository.get_by_token_hash("current")
        assert found is not None
        assert found.next_send_at == _EXPIRES_AT


async def test_a_result_for_a_deleted_invitation_changes_nothing(
    store: tuple[UnitOfWork, OrganizationInvitationRepository],
) -> None:
    uow, repository = store
    organization_id = uuid.uuid4()

    async with uow:
        invitation = await repository.add(
            _invitation(organization_id, email="ada@example.com", token_hash="seed", next_send_at=_NOW)
        )
        await repository.claim_due_send(
            now=_NOW, held_until=_HELD_UNTIL, max_attempts=_MAX_ATTEMPTS, token_hash="claim"
        )
        await repository.delete_in_organization(invitation.id, organization_id)

        assert await repository.finish_send(invitation.id, token_hash="claim") is False
        assert await repository.retry_send(invitation.id, token_hash="claim", next_send_at=_NOW) is False
