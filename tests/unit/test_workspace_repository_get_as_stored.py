"""A workspace is read as the database holds it, past the copy the session already has.

``WorkspaceRepository.get`` answers from the identity map, so a caller that waited on the workspace's row lock
would still hold the row a concurrent deletion removed. ``get_as_stored`` is what such a caller reads instead.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, col

import gateway.models  # noqa: F401 - registers every table on the metadata
from gateway.models.tenancy import Workspace
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def sessions(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A file rather than ``:memory:``, so a second session is a connection of its own."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'as_stored.db'}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _workspace(db: AsyncSession) -> Workspace:
    organization = await OrganizationRepository(db).create_organization(
        name="Acme", slug="acme", created_by_user_id=None
    )
    workspace = await WorkspaceRepository(db).create_workspace(
        name="Marcomms", organization_id=organization.id, created_by_user_id=None
    )
    await db.commit()
    return workspace


async def test_a_deletion_committed_elsewhere_is_read_through(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    async with sessions() as reader, sessions() as writer:
        workspace = await _workspace(reader)
        workspaces = WorkspaceRepository(reader)
        cached = await workspaces.get(workspace.id)
        assert cached is not None
        # Ends the read transaction, as the service's own commits do before it takes the lock.
        await reader.commit()

        await writer.execute(delete(Workspace).where(col(Workspace.id) == workspace.id))
        await writer.commit()

        assert await workspaces.get(workspace.id) is cached, "the session still holds the row"
        assert await workspaces.get_as_stored(workspace.id) is None, "the database no longer has it"


async def test_a_change_committed_elsewhere_is_read_through(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    async with sessions() as reader, sessions() as writer:
        workspace = await _workspace(reader)
        workspaces = WorkspaceRepository(reader)
        cached = await workspaces.get(workspace.id)
        assert cached is not None
        await reader.commit()

        renamed = await writer.get(Workspace, workspace.id)
        assert renamed is not None
        renamed.name = "Marketing"
        await writer.commit()

        stale = await workspaces.get(workspace.id)
        assert stale is not None and stale.name == "Marcomms", "the session's copy is stale"
        stored = await workspaces.get_as_stored(workspace.id)
        assert stored is not None
        assert (stored.organization_id, stored.name) == (workspace.organization_id, "Marketing")
        assert await workspaces.get_as_stored(uuid.uuid4()) is None
