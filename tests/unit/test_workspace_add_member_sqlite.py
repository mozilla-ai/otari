"""Adding a workspace member on SQLite stays one atomic step.

The pysqlite driver begins no transaction for a SELECT, so a savepoint opened after
only reads is the outermost transaction on the connection, and releasing it commits.
"""

import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session, dispose_db, init_db, reset_db
from gateway.models.tenancy import User, WorkspaceCreate, WorkspaceMember
from gateway.repositories.tenancy import OrganizationMemberRepository, OrganizationRepository, UserRepository
from gateway.services.tenancy import WorkspaceService

pytestmark = pytest.mark.asyncio


class _ListenerFailure(Exception):
    pass


class _QuietListener:
    async def member_joined(self, member: WorkspaceMember) -> None:
        return None

    async def member_removed(self, member: WorkspaceMember) -> None:
        return None

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        return None


class _FailingListener(_QuietListener):
    async def member_joined(self, member: WorkspaceMember) -> None:
        raise _ListenerFailure


@pytest_asyncio.fixture
async def sqlite_database(tmp_path: Path) -> AsyncIterator[None]:
    reset_db()
    init_db(GatewayConfig(database_url=f"sqlite+aiosqlite:///{tmp_path / 'members.db'}", auto_migrate=True))
    yield
    await dispose_db()


async def _seed() -> tuple[uuid.UUID, uuid.UUID, uuid.UUID]:
    """Return an owner, a workspace they own, and an organization member who is not in it."""
    async with create_session() as db:
        organization = await OrganizationRepository(db).create_organization(
            name="Acme", slug=f"acme-{uuid.uuid4().hex[:8]}", created_by_user_id=None
        )
        users: list[User] = []
        for full_name, role in (("Owner", "owner"), ("Joiner", "member")):
            user = await UserRepository(db).create_local_identity(
                full_name=full_name, active_organization_id=organization.id
            )
            await OrganizationMemberRepository(db).create_membership(
                organization_id=organization.id, user_id=user.id, role=role
            )
            users.append(user)
        await db.commit()
        owner_id, joiner_id = users[0].id, users[1].id
        owner = await UserRepository(db).get(owner_id)
        assert owner is not None
        workspace = await WorkspaceService(db, membership_listener=_QuietListener()).create_workspace(
            user=owner, workspace_create=WorkspaceCreate(name="Research")
        )
        return owner_id, workspace.id, joiner_id


async def test_a_listener_failure_leaves_no_membership_behind(sqlite_database: None) -> None:
    owner_id, workspace_id, joiner_id = await _seed()

    async with create_session() as db:
        owner = await UserRepository(db).get(owner_id)
        assert owner is not None
        service = WorkspaceService(db, membership_listener=_FailingListener())
        with pytest.raises(_ListenerFailure):
            await service.add_member(user=owner, workspace_id=workspace_id, user_id=joiner_id)
        await db.rollback()

    async with create_session() as db:
        joined = await db.execute(select(col(WorkspaceMember.id)).where(col(WorkspaceMember.user_id) == joiner_id))
        assert joined.all() == []
