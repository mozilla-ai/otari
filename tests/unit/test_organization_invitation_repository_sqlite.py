"""Tests on SQLite that deleting an invitation also removes it from the session."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.organizations import NewInvitation
from gateway.repositories.organizations.organization_invitation_repository import OrganizationInvitationRepository

pytestmark = pytest.mark.asyncio


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
        invitation = await repository.add(
            NewInvitation(
                email="ada@example.com",
                expires_at=datetime(2030, 1, 1, tzinfo=UTC),
                invited_by_user_id=None,
                organization_id=organization_id,
                role="member",
                token_hash="t1",
                workspace_assignments=[],
            )
        )

        assert await repository.delete_in_organization(invitation.id, organization_id) is True
        assert await repository.get(invitation.id) is None
