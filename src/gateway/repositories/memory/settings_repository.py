"""Data access for ``memory_workspace_settings``. Flushes, never commits.

Not fenced: the table holds switches, never content, and the sweeps read it across workspaces.
"""

import uuid
from typing import Never

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.memory import MemoryWorkspaceSettings, MemoryWritePolicy
from gateway.repositories.base_repository import BaseRepository


class SettingsRepository(BaseRepository[MemoryWorkspaceSettings, Never, Never]):
    """Read and change a workspace's memory switches."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, MemoryWorkspaceSettings)

    async def of_workspace(self, workspace_id: uuid.UUID) -> MemoryWorkspaceSettings | None:
        """The workspace's switches, or None while it has never set them, which is memory off."""
        return await self.db.get(MemoryWorkspaceSettings, workspace_id)

    async def put(
        self,
        workspace_id: uuid.UUID,
        *,
        enabled: bool,
        workspace_scope_enabled: bool,
        write_policy: MemoryWritePolicy,
        hide_external: bool,
        retention_days: int | None,
    ) -> MemoryWorkspaceSettings:
        """Stage the workspace's switches, creating its row the first time."""
        row = await self.of_workspace(workspace_id)
        if row is None:
            row = MemoryWorkspaceSettings(workspace_id=workspace_id)
            self.db.add(row)
        row.enabled = enabled
        row.workspace_scope_enabled = workspace_scope_enabled
        row.write_policy = write_policy
        row.hide_external = hide_external
        row.retention_days = retention_days
        await self.db.flush()
        return row
