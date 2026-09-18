"""Scoped persistence and locking for provider-native file operations."""

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel, col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.provider_files import (
    ProviderAccountGeneration,
    ProviderFileBinding,
    ProviderFileOutputOperation,
    ProviderFileRateWindow,
)
from gateway.models.provider_keys import OrgProviderKey, WorkspaceProviderKeyOverride
from gateway.models.tenancy import Organization, Workspace
from gateway.models.users import User
from gateway.repositories.base_repository import BaseRepository
from gateway.repositories.tenancy.org_provider_key_repository import WorkspaceProviderKeyOverrideRepository


class ProviderFileRepository(BaseRepository[ProviderFileBinding, SQLModel, SQLModel]):
    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, ProviderFileBinding)

    async def save(self, row: SQLModel) -> None:
        self.db.add(row)
        await self.db.flush()

    async def refresh(self, row: SQLModel) -> None:
        await self.db.refresh(row)

    async def rate_window(self, workspace_id: uuid.UUID, user_id: str, window: int) -> ProviderFileRateWindow:
        row = await self.db.get(ProviderFileRateWindow, (workspace_id, user_id))
        if row is None:
            row = ProviderFileRateWindow(workspace_id=workspace_id, user_id=user_id, window=window)
            await self.save(row)
        return row

    async def output_operation(self, operation_id: uuid.UUID) -> ProviderFileOutputOperation | None:
        return await self.db.get(ProviderFileOutputOperation, operation_id)

    async def key_candidates(
        self, organization_id: uuid.UUID, workspace_id: uuid.UUID
    ) -> list[tuple[OrgProviderKey, WorkspaceProviderKeyOverride | None]]:
        return await WorkspaceProviderKeyOverrideRepository(self.db).all_candidates(
            organization_id=organization_id, workspace_id=workspace_id
        )

    async def provider_key(self, key_id: uuid.UUID) -> OrgProviderKey | None:
        return await self.db.get(OrgProviderKey, key_id)

    async def expire_bindings(self, organization_id: uuid.UUID, now: datetime, diagnostic_seconds: int) -> None:
        await self.db.execute(
            update(ProviderFileBinding)
            .where(
                col(ProviderFileBinding.organization_id) == organization_id,
                col(ProviderFileBinding.state) == "active",
                col(ProviderFileBinding.expires_at) <= now,
            )
            .values(state="pending_cleanup", cleanup_reason="expiry", cleanup_after=now)
        )
        await self.db.execute(
            update(ProviderFileBinding)
            .where(
                col(ProviderFileBinding.organization_id) == organization_id,
                col(ProviderFileBinding.provider_file_id).is_(None),
                col(ProviderFileBinding.operation_deadline) < now - timedelta(seconds=diagnostic_seconds),
                col(ProviderFileBinding.state).in_(["pending_upload", "pending_cleanup"]),
            )
            .values(state="deleted", deleted_at=now, provider_outcome_unknown=False)
        )

    async def cleanup_account(
        self, organization_id: uuid.UUID, now: datetime, *, include_managed: bool
    ) -> ProviderAccountGeneration | None:
        due = select(col(ProviderFileBinding.provider_account_generation_id)).where(
            col(ProviderFileBinding.state) == "pending_cleanup",
            col(ProviderFileBinding.provider_file_id).is_not(None),
            col(ProviderFileBinding.cleanup_after) <= now,
        )
        return (
            await self.db.execute(
                select(ProviderAccountGeneration)
                .where(
                    col(ProviderAccountGeneration.organization_id) == organization_id,
                    col(ProviderAccountGeneration.credential_source).in_(
                        ["organization_key", "hosted_backend"] if include_managed else ["organization_key"]
                    ),
                    col(ProviderAccountGeneration.id).in_(due),
                    or_(
                        col(ProviderAccountGeneration.lease_deadline).is_(None),
                        col(ProviderAccountGeneration.lease_deadline) <= now,
                    ),
                )
                .order_by(col(ProviderAccountGeneration.id))
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()

    async def cleanup_bindings(self, generation_id: uuid.UUID, now: datetime, limit: int) -> list[ProviderFileBinding]:
        return list(
            (
                await self.db.execute(
                    select(ProviderFileBinding)
                    .where(
                        col(ProviderFileBinding.provider_account_generation_id) == generation_id,
                        col(ProviderFileBinding.state) == "pending_cleanup",
                        col(ProviderFileBinding.provider_file_id).is_not(None),
                        col(ProviderFileBinding.cleanup_after) <= now,
                    )
                    .order_by(col(ProviderFileBinding.cleanup_after), col(ProviderFileBinding.id))
                    .limit(min(20, max(1, limit)))
                )
            ).scalars()
        )

    async def leased_account(self, organization_id: uuid.UUID, lease_id: uuid.UUID) -> ProviderAccountGeneration | None:
        return (
            await self.db.execute(
                select(ProviderAccountGeneration)
                .where(
                    col(ProviderAccountGeneration.organization_id) == organization_id,
                    col(ProviderAccountGeneration.lease_id) == lease_id,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()

    async def leased_bindings(self, generation_id: uuid.UUID, lease_id: uuid.UUID) -> list[ProviderFileBinding]:
        return list(
            (
                await self.db.execute(
                    select(ProviderFileBinding).where(
                        col(ProviderFileBinding.provider_account_generation_id) == generation_id,
                        col(ProviderFileBinding.lease_id) == lease_id,
                    )
                )
            ).scalars()
        )

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
        *,
        provider: str = "anthropic",
        purpose: str | None = None,
        ascending: bool = False,
        reverse_cursor: bool = False,
        provider_order: bool = False,
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
                col(ProviderAccountGeneration.provider) == provider,
            )
        )
        if ids is not None:
            statement = statement.where(col(ProviderFileBinding.provider_file_id).in_(ids))
        if purpose is not None:
            statement = statement.where(col(ProviderFileBinding.purpose) == purpose)
        if snapshot is not None:
            statement = statement.where(
                col(ProviderFileBinding.created_at) <= snapshot,
                func.coalesce(col(ProviderFileBinding.updated_at), col(ProviderFileBinding.created_at)) <= snapshot,
            )
        timestamp_column = (
            func.coalesce(col(ProviderFileBinding.provider_created_at), col(ProviderFileBinding.created_at))
            if provider_order
            else col(ProviderFileBinding.created_at)
        )
        id_column = col(ProviderFileBinding.id)
        if before is not None:
            timestamp, identity = before
            later = ascending != reverse_cursor
            statement = statement.where(
                or_(
                    timestamp_column > timestamp if later else timestamp_column < timestamp,
                    (timestamp_column == timestamp) & (id_column > identity if later else id_column < identity),
                )
            )
        ordering = (timestamp_column, id_column)
        result = await self.db.execute(
            statement.order_by(*(column.asc() if ascending else column.desc() for column in ordering)).limit(limit)
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
