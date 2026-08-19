"""Data access for workspaces and their memberships."""

import uuid
from collections.abc import Collection, Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.tenancy import (
    User,
    Workspace,
    WorkspaceCreate,
    WorkspaceMember,
    WorkspaceMemberUpdate,
    WorkspaceUpdate,
)
from gateway.repositories.base_repository import BaseRepository
from gateway.repositories.tenancy.user_repository import user_alphabetical_order


class WorkspaceRepository(BaseRepository[Workspace, WorkspaceCreate, WorkspaceUpdate]):
    """Repository for workspace rows."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, Workspace)

    async def get_by_ids(self, workspace_ids: Collection[uuid.UUID]) -> Sequence[Workspace]:
        """Return the workspaces named by a batch of ids (order unspecified).

        Lets a caller resolve only the workspaces it references instead of
        paging the organization's whole list.
        """
        if not workspace_ids:
            return []
        result = await self.db.execute(select(Workspace).where(col(Workspace.id).in_(list(workspace_ids))))
        return list(result.scalars().all())

    async def get_by_organization(
        self,
        organization_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[Sequence[Workspace], int]:
        """Return a page of an organization's workspaces, newest first, plus the total."""
        count_result = await self.db.execute(
            select(func.count()).select_from(Workspace).where(col(Workspace.organization_id) == organization_id)
        )
        count = count_result.scalar_one()

        result = await self.db.execute(
            select(Workspace)
            .where(col(Workspace.organization_id) == organization_id)
            .order_by(col(Workspace.created_at).desc(), col(Workspace.id))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all()), count

    async def get_by_organization_and_name(self, organization_id: uuid.UUID, name: str) -> Workspace | None:
        """Return an organization's workspace with this name, or None."""
        result = await self.db.execute(
            select(Workspace).where(
                col(Workspace.organization_id) == organization_id,
                col(Workspace.name) == name,
            )
        )
        return result.scalars().first()

    async def create_workspace(
        self,
        *,
        name: str,
        organization_id: uuid.UUID,
        created_by_user_id: uuid.UUID | None,
        description: str | None = None,
    ) -> Workspace:
        """Stage a new workspace."""
        workspace = Workspace(
            name=name,
            description=description,
            organization_id=organization_id,
            created_by_user_id=created_by_user_id,
        )
        self.db.add(workspace)
        await self.db.flush()
        await self.db.refresh(workspace)
        return workspace

    async def update_workspace(self, workspace: Workspace, update_data: dict[str, Any]) -> Workspace:
        """Stage an update from a plain mapping of columns."""
        workspace.sqlmodel_update(update_data)
        self.db.add(workspace)
        await self.db.flush()
        await self.db.refresh(workspace)
        return workspace

    async def delete_workspace(self, workspace: Workspace) -> None:
        """Stage a deletion. Members ride the database cascade."""
        await self.db.delete(workspace)
        await self.db.flush()


class WorkspaceMemberRepository:
    """Repository for workspace membership rows.

    Not a ``BaseRepository``: every access is keyed by the (workspace, user)
    pair rather than by the row's own id, so none of the generic helpers apply.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_workspace(
        self,
        workspace_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[Sequence[WorkspaceMember], int]:
        """Return a page of a workspace's members, directory-ordered, plus the total."""
        count_result = await self.db.execute(
            select(func.count()).select_from(WorkspaceMember).where(col(WorkspaceMember.workspace_id) == workspace_id)
        )
        count = count_result.scalar_one()

        result = await self.db.execute(
            select(WorkspaceMember)
            .join(User, col(WorkspaceMember.user_id) == col(User.id))
            .where(col(WorkspaceMember.workspace_id) == workspace_id)
            .order_by(user_alphabetical_order(), col(WorkspaceMember.id))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all()), count

    async def get_by_workspace_and_user(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> WorkspaceMember | None:
        """Return the membership joining a user to a workspace, or None."""
        result = await self.db.execute(
            select(WorkspaceMember).where(
                col(WorkspaceMember.workspace_id) == workspace_id,
                col(WorkspaceMember.user_id) == user_id,
            )
        )
        return result.scalars().first()

    async def get_active_by_workspace_and_user(
        self,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> WorkspaceMember | None:
        """Return the *active* membership joining a user to a workspace, or None.

        The any-status variant above is what the mutation paths want: adding
        someone whose membership is suspended has to see that row in order to
        revive it rather than insert beside it. Every path that asks "may this
        caller see this workspace" wants this one instead, because a suspended
        membership grants nothing.
        """
        result = await self.db.execute(
            select(WorkspaceMember).where(
                col(WorkspaceMember.workspace_id) == workspace_id,
                col(WorkspaceMember.user_id) == user_id,
                col(WorkspaceMember.status) == "active",
            )
        )
        return result.scalars().first()

    async def get_by_workspaces_and_user(
        self,
        workspace_ids: Collection[uuid.UUID],
        user_id: uuid.UUID,
    ) -> list[WorkspaceMember]:
        """Return a user's memberships across a batch of workspaces, whatever their status.

        One ``IN`` query so applying N workspace assignments costs one lookup
        rather than N. Any status, for the same reason
        ``get_by_workspace_and_user`` is: the caller revives a suspended row.
        """
        if not workspace_ids:
            return []
        result = await self.db.execute(
            select(WorkspaceMember).where(
                col(WorkspaceMember.workspace_id).in_(workspace_ids),
                col(WorkspaceMember.user_id) == user_id,
            )
        )
        return list(result.scalars().all())

    async def get_workspaces_for_user(
        self,
        *,
        user_id: uuid.UUID,
        organization_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[Sequence[WorkspaceMember], int]:
        """Return a user's *active* memberships in one organization's workspaces, plus the total.

        Suspended memberships are excluded from both the page and the count,
        so a member who was removed from a workspace stops seeing it listed.
        """
        count_result = await self.db.execute(
            select(func.count())
            .select_from(WorkspaceMember)
            .join(Workspace, col(Workspace.id) == col(WorkspaceMember.workspace_id))
            .where(
                col(WorkspaceMember.user_id) == user_id,
                col(Workspace.organization_id) == organization_id,
                col(WorkspaceMember.status) == "active",
            )
        )
        count = count_result.scalar_one()

        result = await self.db.execute(
            select(WorkspaceMember)
            .join(Workspace, col(Workspace.id) == col(WorkspaceMember.workspace_id))
            .where(
                col(WorkspaceMember.user_id) == user_id,
                col(Workspace.organization_id) == organization_id,
                col(WorkspaceMember.status) == "active",
            )
            .order_by(col(WorkspaceMember.created_at), col(WorkspaceMember.id))
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all()), count

    async def create(
        self,
        *,
        workspace_id: uuid.UUID,
        user_id: uuid.UUID,
        role: str = "member",
        status: str = "active",
    ) -> WorkspaceMember:
        """Stage a new workspace membership."""
        member = WorkspaceMember(workspace_id=workspace_id, user_id=user_id, role=role, status=status)
        self.db.add(member)
        await self.db.flush()
        await self.db.refresh(member)
        return member

    async def update(self, member: WorkspaceMember, update: WorkspaceMemberUpdate) -> WorkspaceMember:
        """Stage an update from an update schema, applying only the fields it sets."""
        member.sqlmodel_update(update.model_dump(exclude_unset=True))
        self.db.add(member)
        await self.db.flush()
        await self.db.refresh(member)
        return member

    async def delete(self, member: WorkspaceMember) -> None:
        """Stage a deletion."""
        await self.db.delete(member)
        await self.db.flush()


__all__ = ["WorkspaceMemberRepository", "WorkspaceRepository"]
