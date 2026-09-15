"""Scoped persistence and locking for provider-native file operations."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, col

from gateway.models.provider_files import ProviderAccountGeneration, ProviderFileBinding, ProviderFileOutputOperation
from gateway.models.tenancy import Organization, Workspace
from gateway.models.users import User
from gateway.repositories.base_repository import BaseRepository


class ProviderFileRepository(BaseRepository[ProviderFileBinding, SQLModel, SQLModel]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, ProviderFileBinding)

    async def lock_organization(self, organization_id: uuid.UUID) -> bool:
        return (
            await self.db.execute(
                select(col(Organization.id)).where(col(Organization.id) == organization_id).with_for_update()
            )
        ).scalar_one_or_none() is not None

    async def workspace_exists(self, workspace_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
        return (
            await self.db.execute(
                select(col(Workspace.id)).where(
                    col(Workspace.id) == workspace_id, col(Workspace.organization_id) == organization_id
                )
            )
        ).scalar_one_or_none() is not None

    async def account(self, generation_id: uuid.UUID) -> ProviderAccountGeneration | None:
        return (
            await self.db.execute(
                select(ProviderAccountGeneration)
                .where(col(ProviderAccountGeneration.id) == generation_id)
                .with_for_update()
            )
        ).scalar_one_or_none()

    async def latest_account(
        self, source: str, ref: str, organization_id: uuid.UUID
    ) -> ProviderAccountGeneration | None:
        return (
            await self.db.execute(
                select(ProviderAccountGeneration)
                .where(
                    col(ProviderAccountGeneration.organization_id) == organization_id,
                    col(ProviderAccountGeneration.credential_source) == source,
                    col(ProviderAccountGeneration.credential_ref) == ref,
                )
                .order_by(col(ProviderAccountGeneration.generation).desc())
                .limit(1)
                .with_for_update()
            )
        ).scalar_one_or_none()

    async def by_provider_id(self, generation_id: uuid.UUID, file_id: str) -> ProviderFileBinding | None:
        return (
            await self.db.execute(
                select(ProviderFileBinding).where(
                    col(ProviderFileBinding.provider_account_generation_id) == generation_id,
                    col(ProviderFileBinding.provider_file_id) == file_id,
                )
            )
        ).scalar_one_or_none()

    async def visible(
        self,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID,
        user_id: str,
        now: datetime,
        ids: list[str] | None = None,
        limit: int = 1001,
        before: tuple[datetime, uuid.UUID] | None = None,
        snapshot: datetime | None = None,
    ) -> list[ProviderFileBinding]:
        statement = (
            select(ProviderFileBinding)
            .join(ProviderAccountGeneration)
            .where(
                col(ProviderFileBinding.organization_id) == organization_id,
                col(ProviderFileBinding.workspace_id) == workspace_id,
                col(ProviderFileBinding.user_id) == user_id,
                col(ProviderFileBinding.state) == "active",
                col(ProviderFileBinding.expires_at) > now,
                col(ProviderAccountGeneration.status) == "active",
            )
        )
        if ids is not None:
            statement = statement.where(col(ProviderFileBinding.provider_file_id).in_(ids))
        if snapshot is not None:
            statement = statement.where(
                col(ProviderFileBinding.created_at) <= snapshot,
                func.coalesce(col(ProviderFileBinding.updated_at), col(ProviderFileBinding.created_at)) <= snapshot,
            )
        if before is not None:
            timestamp, identity = before
            statement = statement.where(
                or_(
                    col(ProviderFileBinding.created_at) < timestamp,
                    (col(ProviderFileBinding.created_at) == timestamp) & (col(ProviderFileBinding.id) < identity),
                )
            )
        result = await self.db.execute(
            statement.order_by(col(ProviderFileBinding.created_at).desc(), col(ProviderFileBinding.id).desc()).limit(
                limit
            )
        )
        return list(result.scalars().all())

    async def capacity(self, workspace_id: uuid.UUID, user_id: str, now: datetime) -> tuple[int, int]:
        count, size = (
            await self.db.execute(
                select(func.count(), func.coalesce(func.sum(col(ProviderFileBinding.size_bytes)), 0)).where(
                    col(ProviderFileBinding.workspace_id) == workspace_id,
                    col(ProviderFileBinding.user_id) == user_id,
                    or_(
                        col(ProviderFileBinding.state) == "active",
                        (col(ProviderFileBinding.state) == "pending_cleanup")
                        & col(ProviderFileBinding.provider_file_id).is_not(None),
                        col(ProviderFileBinding.state).in_(["pending_upload", "pending_cleanup"])
                        & (col(ProviderFileBinding.operation_deadline) > now),
                    ),
                )
            )
        ).one()
        output_count, output_size = (
            await self.db.execute(
                select(
                    func.coalesce(func.sum(col(ProviderFileOutputOperation.reserved_files)), 0),
                    func.coalesce(func.sum(col(ProviderFileOutputOperation.reserved_bytes)), 0),
                ).where(
                    col(ProviderFileOutputOperation.workspace_id) == workspace_id,
                    col(ProviderFileOutputOperation.user_id) == user_id,
                    col(ProviderFileOutputOperation.state) != "completed",
                    col(ProviderFileOutputOperation.deadline) > now,
                )
            )
        ).one()
        return int(count + output_count), int(size + output_size)

    async def revoke(
        self,
        now: datetime,
        reason: str,
        *,
        organization_id: uuid.UUID,
        generation_id: uuid.UUID | None = None,
        workspace_id: uuid.UUID | None = None,
        user_id: str | None = None,
    ) -> None:
        statement = update(ProviderFileBinding).where(
            col(ProviderFileBinding.organization_id) == organization_id,
            col(ProviderFileBinding.state).in_(["active", "pending_upload"]),
        )
        output = update(ProviderFileOutputOperation).where(
            col(ProviderFileOutputOperation.organization_id) == organization_id,
            col(ProviderFileOutputOperation.state) == "active",
        )
        if generation_id is not None:
            statement = statement.where(col(ProviderFileBinding.provider_account_generation_id) == generation_id)
            output = output.where(col(ProviderFileOutputOperation.provider_account_generation_id) == generation_id)
        if workspace_id is not None:
            statement = statement.where(col(ProviderFileBinding.workspace_id) == workspace_id)
            output = output.where(col(ProviderFileOutputOperation.workspace_id) == workspace_id)
        if user_id is not None:
            statement = statement.where(col(ProviderFileBinding.user_id) == user_id)
            output = output.where(col(ProviderFileOutputOperation.user_id) == user_id)
        await self.db.execute(statement.values(state="pending_cleanup", cleanup_reason=reason, cleanup_after=now))
        await self.db.execute(output.values(state="revoked"))
        await self.db.flush()

    async def account_busy(self, generation_id: uuid.UUID, now: datetime) -> bool:
        binding = (
            await self.db.execute(
                select(col(ProviderFileBinding.id))
                .where(
                    col(ProviderFileBinding.provider_account_generation_id) == generation_id,
                    col(ProviderFileBinding.state) != "deleted",
                    or_(
                        col(ProviderFileBinding.provider_file_id).is_not(None),
                        col(ProviderFileBinding.operation_deadline) > now,
                    ),
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        operation = (
            await self.db.execute(
                select(col(ProviderFileOutputOperation.id))
                .where(
                    col(ProviderFileOutputOperation.provider_account_generation_id) == generation_id,
                    col(ProviderFileOutputOperation.state) != "completed",
                    col(ProviderFileOutputOperation.deadline) > now,
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        return binding is not None or operation is not None

    async def active_user(self, user_id: str) -> bool:
        return (
            await self.db.execute(
                select(User.user_id).where(
                    User.user_id == user_id,
                    User.deleted_at.is_(None),
                    User.blocked.is_(False),
                )
            )
        ).scalar_one_or_none() is not None

    async def revoke_user(self, user_id: str, now: datetime) -> None:
        organizations = (
            (
                await self.db.execute(
                    select(col(ProviderFileBinding.organization_id))
                    .where(
                        col(ProviderFileBinding.user_id) == user_id,
                    )
                    .union(
                        select(col(ProviderFileOutputOperation.organization_id)).where(
                            col(ProviderFileOutputOperation.user_id) == user_id,
                        )
                    )
                    .order_by("organization_id")
                )
            )
            .scalars()
            .all()
        )
        for organization_id in organizations:
            await self.lock_organization(organization_id)
            await self.revoke(now, "user_deletion", organization_id=organization_id, user_id=user_id)

    async def lock_user(self, user_id: str) -> None:
        await self.db.execute(select(User.user_id).where(User.user_id == user_id).with_for_update())

    async def backlog(self, organization_id: uuid.UUID) -> dict[str, int]:
        now = datetime.now(UTC)
        pending = or_(
            col(ProviderFileBinding.state) == "pending_cleanup",
            (col(ProviderFileBinding.state) == "active") & (col(ProviderFileBinding.expires_at) <= now),
        )
        result = (
            await self.db.execute(
                select(
                    func.coalesce(func.sum(case((col(ProviderFileBinding.state) == "pending_upload", 1), else_=0)), 0),
                    func.coalesce(func.sum(case((pending, 1), else_=0)), 0),
                    func.coalesce(
                        func.sum(
                            case(
                                (
                                    col(ProviderFileBinding.provider_outcome_unknown)
                                    & (col(ProviderFileBinding.state) != "deleted"),
                                    1,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                    func.coalesce(func.sum(col(ProviderFileBinding.cleanup_attempts)), 0),
                ).where(col(ProviderFileBinding.organization_id) == organization_id)
            )
        ).one()
        return dict(
            zip(
                ("pending_upload", "pending_cleanup", "unknown_outcomes", "cleanup_attempts"),
                (int(value) for value in result),
                strict=True,
            )
        )
