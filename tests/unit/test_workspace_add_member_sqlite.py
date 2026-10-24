"""Adding a workspace member on SQLite stays one atomic step.

The pysqlite driver begins no transaction for a SELECT, so a savepoint opened after
only reads is the outermost transaction on the connection, and releasing it commits.
``WorkspaceMemberRepository.add`` therefore opens none: it reads the refusal from the
error, and the Unit of Work rolls the whole step back.
"""

import uuid
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import GatewayConfig
from gateway.core.database import create_session, dispose_db, init_db, reset_db
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.organizations_exceptions import WorkspaceMemberAlreadyExistsError
from gateway.models.tenancy import OrganizationMember, User, WorkspaceCreate, WorkspaceMember
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
)
from gateway.services.tenancy import WorkspaceService
from gateway.services.tenancy.membership_listener import MembershipListener

pytestmark = pytest.mark.asyncio


class _ListenerFailure(Exception):
    pass


class _QuietListener:
    async def member_joined(self, member: WorkspaceMember) -> None:
        return None

    async def member_removed(self, member: WorkspaceMember) -> None:
        return None

    async def member_role_changed(self, member: WorkspaceMember) -> None:
        return None

    async def organization_member_changed(
        self, membership: OrganizationMember, workspace_ids: Sequence[uuid.UUID]
    ) -> None:
        return None

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        return None

    async def member_lost_workspaces(self, user_id: uuid.UUID, workspace_ids: Sequence[uuid.UUID]) -> None:
        return None


class _FailingListener(_QuietListener):
    async def member_joined(self, member: WorkspaceMember) -> None:
        raise _ListenerFailure


class _ConstraintFailingListener(_QuietListener):
    """A listener whose own write is refused by a constraint that has nothing to do with the membership."""

    async def member_joined(self, member: WorkspaceMember) -> None:
        raise IntegrityError("INSERT INTO budget_ceilings", {}, Exception("UNIQUE constraint failed"))


def _workspace_service(db: AsyncSession, listener: MembershipListener) -> WorkspaceService:
    return WorkspaceService(db, uow=UnitOfWork(db), membership_listener=listener)


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
        workspace = await _workspace_service(db, _QuietListener()).create_workspace(
            user=owner, workspace_create=WorkspaceCreate(name="Research")
        )
        return owner_id, workspace.id, joiner_id


async def test_a_listener_failure_leaves_no_membership_behind(sqlite_database: None) -> None:
    owner_id, workspace_id, joiner_id = await _seed()

    async with create_session() as db:
        owner = await UserRepository(db).get(owner_id)
        assert owner is not None
        service = _workspace_service(db, _FailingListener())
        with pytest.raises(_ListenerFailure):
            await service.add_member(user=owner, workspace_id=workspace_id, user_id=joiner_id)
        await db.rollback()

    async with create_session() as db:
        joined = await db.execute(select(col(WorkspaceMember.id)).where(col(WorkspaceMember.user_id) == joiner_id))
        assert joined.all() == []


async def test_an_integrity_error_from_the_listener_surfaces_as_itself(sqlite_database: None) -> None:
    owner_id, workspace_id, joiner_id = await _seed()

    async with create_session() as db:
        owner = await UserRepository(db).get(owner_id)
        assert owner is not None
        service = _workspace_service(db, _ConstraintFailingListener())
        # Neither a duplicate member nor a deleted workspace explains it, so it is not reported as one.
        with pytest.raises(IntegrityError):
            await service.add_member(user=owner, workspace_id=workspace_id, user_id=joiner_id)
        await db.rollback()

    async with create_session() as db:
        joined = await db.execute(select(col(WorkspaceMember.id)).where(col(WorkspaceMember.user_id) == joiner_id))
        assert joined.all() == []


async def test_adding_a_member_twice_reports_a_duplicate(sqlite_database: None) -> None:
    _, workspace_id, joiner_id = await _seed()

    async with create_session() as db:
        members = WorkspaceMemberRepository(db)
        await members.add(workspace_id=workspace_id, user_id=joiner_id)
        await db.commit()

    async with create_session() as db:
        with pytest.raises(WorkspaceMemberAlreadyExistsError):
            await WorkspaceMemberRepository(db).add(workspace_id=workspace_id, user_id=joiner_id)
        await db.rollback()


async def test_a_refusal_that_is_not_a_duplicate_is_not_reported_as_one(sqlite_database: None) -> None:
    _, workspace_id, _ = await _seed()

    async with create_session() as db:
        # A user that does not exist breaks the user foreign key, not the unique pair.
        with pytest.raises(IntegrityError):
            await WorkspaceMemberRepository(db).add(workspace_id=workspace_id, user_id=uuid.uuid4())
        await db.rollback()
