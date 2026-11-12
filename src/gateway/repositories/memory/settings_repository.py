"""Data access for ``memory_workspace_settings``. Flushes, never commits.

Not fenced: the table holds switches, never content, and the sweeps read it across workspaces.
"""

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import select

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
        organization_id: uuid.UUID,
        enabled: bool,
        workspace_scope_enabled: bool,
        write_policy: MemoryWritePolicy,
        offer_to_every_request: bool,
        hide_external: bool,
        retention_days: int | None,
        auto_recall: bool,
        recall_min_similarity: float,
        recall_max_entries: int,
        rerank: tuple[bool, float],
        experiment: tuple[int, datetime | None, datetime | None],
    ) -> MemoryWorkspaceSettings:
        """Stage the workspace's switches, creating its row the first time.

        ``rerank`` is whether recall checks relevance, and the score an entry must reach. ``experiment`` is the
        savings experiment's holdout, start and end.
        """
        row = await self.of_workspace(workspace_id)
        if row is None:
            row = MemoryWorkspaceSettings(workspace_id=workspace_id, organization_id=organization_id)
            self.db.add(row)
        row.organization_id = organization_id
        row.enabled = enabled
        row.workspace_scope_enabled = workspace_scope_enabled
        row.write_policy = write_policy
        row.offer_to_every_request = offer_to_every_request
        row.hide_external = hide_external
        row.retention_days = retention_days
        row.auto_recall = auto_recall
        row.recall_min_similarity = recall_min_similarity
        row.recall_max_entries = recall_max_entries
        row.recall_rerank, row.recall_min_relevance = rerank
        row.experiment_holdout_percent, row.experiment_started_at, row.experiment_ended_at = experiment
        await self.db.flush()
        return row

    async def enabled_workspaces(self) -> Sequence[tuple[uuid.UUID, uuid.UUID]]:
        """Every workspace that has memory on itself, with its organization, whose own switch the caller checks."""
        result = await self.db.execute(
            select(MemoryWorkspaceSettings.workspace_id, MemoryWorkspaceSettings.organization_id)
            .where(MemoryWorkspaceSettings.enabled.is_(True))
            .order_by(MemoryWorkspaceSettings.workspace_id)
        )
        return [(workspace_id, organization_id) for workspace_id, organization_id in result.all()]

    async def all(self) -> Sequence[MemoryWorkspaceSettings]:
        """Every workspace that has set its switches, on or off: retention runs for both."""
        result = await self.db.execute(select(MemoryWorkspaceSettings).order_by(MemoryWorkspaceSettings.workspace_id))
        return result.scalars().all()
