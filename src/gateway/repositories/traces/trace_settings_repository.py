"""Data access for ``workspace_trace_settings``. Flushes, never commits."""

import uuid
from datetime import UTC, datetime
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.traces import WorkspaceTraceSettings
from gateway.repositories.base_repository import BaseRepository


class TraceSettingsRepository(BaseRepository[WorkspaceTraceSettings, Never, Never]):
    """Read and set how much of its requests' content each workspace keeps."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, WorkspaceTraceSettings)

    async def get(self, workspace_id: uuid.UUID) -> WorkspaceTraceSettings | None:
        return await self.db.get(WorkspaceTraceSettings, workspace_id)

    async def set_admin_content_access(
        self, workspace_id: uuid.UUID, allowed: bool, *, updated_by: uuid.UUID | None
    ) -> None:
        row = await self.db.get(WorkspaceTraceSettings, workspace_id)
        if row is None:
            self.db.add(
                WorkspaceTraceSettings(
                    workspace_id=workspace_id, admin_content_access=allowed, updated_by_user_id=updated_by
                )
            )
        else:
            row.admin_content_access = allowed
            row.updated_by_user_id = updated_by
            row.updated_at = datetime.now(UTC)
        await self.db.flush()

    async def content_capture(self, workspace_id: uuid.UUID) -> str | None:
        result = await self.db.execute(
            select(WorkspaceTraceSettings.content_capture).where(WorkspaceTraceSettings.workspace_id == workspace_id)
        )
        return result.scalar_one_or_none()

    async def set_content_capture(self, workspace_id: uuid.UUID, level: str, *, updated_by: uuid.UUID | None) -> None:
        row = await self.db.get(WorkspaceTraceSettings, workspace_id)
        if row is None:
            self.db.add(
                WorkspaceTraceSettings(workspace_id=workspace_id, content_capture=level, updated_by_user_id=updated_by)
            )
        else:
            row.content_capture = level
            row.updated_by_user_id = updated_by
            row.updated_at = datetime.now(UTC)
        await self.db.flush()
