"""The base of every memory repository whose table is fenced by workspace.

On PostgreSQL the memory tables are under forced row-level security: a row is
visible only while ``otari.memory_workspace`` names its workspace. Every query
here names the workspace twice, then: in the setting, for the transaction, and in
its own filter. Either alone keeps a tenant's rows from another; the setting is
what still does when a filter is forgotten. SQLite has neither the setting nor the
policies, so there the filter is the bound.
"""

import uuid
from typing import Generic, Never, TypeVar

from sqlalchemy import text

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.base_repository import BaseRepository

FencedModel = TypeVar("FencedModel")

WORKSPACE_SETTING = "otari.memory_workspace"


class FencedRepository(BaseRepository[FencedModel, Never, Never], Generic[FencedModel]):
    """A repository over one fenced table, whose every operation names its workspace."""

    def __init__(self, uow: UnitOfWork, model_class: type[FencedModel]) -> None:
        super().__init__(uow, model_class)

    async def _fence(self, workspace_id: uuid.UUID) -> None:
        """Name ``workspace_id`` as the one this transaction's memory rows may come from."""
        if self.db.get_bind().dialect.name != "postgresql":
            return
        # Local to the transaction, so it ends with the Unit of Work's block and never reaches a pooled connection.
        await self.db.execute(
            text("SELECT set_config(:setting, :workspace_id, true)"),
            {"setting": WORKSPACE_SETTING, "workspace_id": str(workspace_id)},
        )
