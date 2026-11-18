"""Data access for ``workspace_mcp_server_overrides``: each workspace's settings on an organization's MCP servers.

The only module that constructs or queries the table. Writes flush and leave
the commit to the caller.
"""

import uuid
from collections.abc import Collection
from typing import Any, Never

from sqlalchemy import ColumnElement, delete, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceMcpServer, WorkspaceMcpServerOverride
from gateway.repositories.base_repository import BaseRepository


class WorkspaceMcpServerOverrideRepository(BaseRepository[WorkspaceMcpServerOverride, Never, Never]):
    """Read and stage override rows, always addressed by workspace and server."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceMcpServerOverride)

    async def find(self, *, workspace_id: uuid.UUID, server_id: uuid.UUID) -> WorkspaceMcpServerOverride | None:
        """A workspace's override of one server, or ``None`` where it inherits."""
        found = await self.db.execute(
            select(WorkspaceMcpServerOverride).where(
                WorkspaceMcpServerOverride.workspace_id == workspace_id,
                WorkspaceMcpServerOverride.server_id == server_id,
            )
        )
        return found.scalar_one_or_none()

    async def for_workspace(
        self, workspace_id: uuid.UUID, server_ids: Collection[uuid.UUID]
    ) -> dict[uuid.UUID, WorkspaceMcpServerOverride]:
        """A workspace's overrides of these servers, by server id; a server it inherits has none."""
        if not server_ids:
            return {}
        rows = await self.db.execute(
            select(WorkspaceMcpServerOverride).where(
                WorkspaceMcpServerOverride.workspace_id == workspace_id,
                WorkspaceMcpServerOverride.server_id.in_(list(server_ids)),
            )
        )
        return {override.server_id: override for override in rows.scalars().all()}

    async def for_server(self, server_id: uuid.UUID) -> list[WorkspaceMcpServerOverride]:
        """Every workspace's override of one server, oldest first."""
        rows = await self.db.execute(
            select(WorkspaceMcpServerOverride)
            .where(WorkspaceMcpServerOverride.server_id == server_id)
            .order_by(WorkspaceMcpServerOverride.created_at, WorkspaceMcpServerOverride.id)
        )
        return list(rows.scalars().all())

    async def attaching_workspace_ids(self) -> frozenset[uuid.UUID]:
        """Every workspace whose own settings attach one of its organization's servers to every request."""
        found = await self.db.execute(
            select(WorkspaceMcpServerOverride.workspace_id)
            .where(WorkspaceMcpServerOverride.attach_by_default.is_(True))
            .distinct()
        )
        return frozenset(found.scalars())

    async def save(
        self,
        override: WorkspaceMcpServerOverride | None,
        *,
        workspace_id: uuid.UUID,
        server: WorkspaceMcpServer,
        organization_tools: list[str] | None,
        enabled: bool,
        everyone_tools: list[str] | None,
        attach_by_default: bool | None,
    ) -> WorkspaceMcpServerOverride | None:
        """Stage a workspace's settings on an organization's ``server``: a new override, or ``override`` changed.

        Settings that are all the defaults are no override: the one passed in is removed, and ``None`` returned.
        The keys refuse a workspace of another organization than the server's.

        Only the settings that differ from ``override`` as the caller read it (the defaults, where it read none)
        are written, so a setting another writer stored meanwhile is kept. The caller's workspace lock is what
        orders two writers on PostgreSQL; on SQLite, which takes no row lock, this is.

        Raises:
            ValueError: ``server`` is one workspace's own, which no other workspace reaches, or ``override`` is
                another workspace's or another server's.
        """
        if server.workspace_id is not None:
            raise ValueError(f"MCP server {server.id} is a workspace's own; only an organization's takes overrides")
        if override is not None and (override.workspace_id != workspace_id or override.server_id != server.id):
            raise ValueError("The override passed in is another workspace's or another server's")
        read = _settings(override)
        wanted = {
            "organization_tools": organization_tools,
            "enabled": enabled,
            "everyone_tools": everyone_tools,
            "attach_by_default": attach_by_default,
        }
        changed = {name: value for name, value in wanted.items() if value != read[name]}
        if override is None:
            if wanted == _DEFAULTS:
                return None
            # Inserts nothing where another writer's first override landed meanwhile; that one is changed below.
            insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
            await self.db.execute(
                insert(WorkspaceMcpServerOverride)
                .values(
                    workspace_id=workspace_id, organization_id=server.organization_id, server_id=server.id, **wanted
                )
                .on_conflict_do_nothing(
                    index_elements=[WorkspaceMcpServerOverride.workspace_id, WorkspaceMcpServerOverride.server_id]
                )
            )
            override = await self._current(workspace_id, server.id)
            if override is None:
                raise LookupError("the MCP server override is missing after its insert")
        # Each pass is one other writer's change landing between this read and this write; a few settle it.
        for _ in range(_MAX_SAVE_PASSES):
            current = _settings(override)
            if {**current, **changed} != _DEFAULTS:
                for name, value in changed.items():
                    setattr(override, name, value)
                await self.db.flush()
                return override
            # Removed only while the settings this caller left alone are still the defaults it read.
            kept: list[ColumnElement[bool]] = [_is_default(name) for name in _DEFAULTS if name not in changed]
            removed = await self.db.execute(
                delete(WorkspaceMcpServerOverride)
                .where(WorkspaceMcpServerOverride.id == override.id, *kept)
                .execution_options(synchronize_session=False)
            )
            if getattr(removed, "rowcount", 0) == 1:
                self.db.expunge(override)
                return None
            override = await self._current(workspace_id, server.id)
            if override is None:
                return None
        raise RuntimeError("the MCP server override kept changing under this write")

    async def _current(self, workspace_id: uuid.UUID, server_id: uuid.UUID) -> WorkspaceMcpServerOverride | None:
        """The override as stored now, replacing what this session last read of it."""
        found = await self.db.execute(
            select(WorkspaceMcpServerOverride)
            .where(
                WorkspaceMcpServerOverride.workspace_id == workspace_id,
                WorkspaceMcpServerOverride.server_id == server_id,
            )
            .execution_options(populate_existing=True)
        )
        return found.scalar_one_or_none()

    async def remove(self, override: WorkspaceMcpServerOverride) -> None:
        """Stage an override's removal."""
        await self.db.delete(override)
        await self.db.flush()


_MAX_SAVE_PASSES = 5

# The settings of a workspace that has no override.
_DEFAULTS: dict[str, Any] = {
    "organization_tools": None,
    "enabled": True,
    "everyone_tools": None,
    "attach_by_default": None,
}


def _settings(override: WorkspaceMcpServerOverride | None) -> dict[str, Any]:
    if override is None:
        return dict(_DEFAULTS)
    return {name: getattr(override, name) for name in _DEFAULTS}


def _is_default(name: str) -> ColumnElement[bool]:
    column = getattr(WorkspaceMcpServerOverride, name)
    condition: ColumnElement[bool] = column.is_(True) if _DEFAULTS[name] is True else column.is_(None)
    return condition
