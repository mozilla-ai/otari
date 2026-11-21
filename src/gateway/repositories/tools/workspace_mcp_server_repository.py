"""Data access for ``workspace_mcp_servers``.

New reads of the table go here. The tenancy and Playground services that read and write it directly predate this
module and are on the architecture check's baseline until they move.
"""

import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import ColumnElement, and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import WorkspaceMcpServer
from gateway.repositories.base_repository import BaseRepository

# The whitespace a stored URL may carry around it: Python's ``str.strip`` drops these too.
_WHITESPACE = " \t\n\r\f\v"


class WorkspaceMcpServerRepository(BaseRepository[WorkspaceMcpServer, Never, Never]):
    """Read stored MCP server rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WorkspaceMcpServer)

    async def find(
        self, server_id: uuid.UUID, *, workspace_id: uuid.UUID | None = None, lock: bool = False
    ) -> WorkspaceMcpServer | None:
        """One stored server by id, or ``None``; ``lock`` holds the row until the caller commits.

        With ``workspace_id``, a server of another workspace is ``None`` too.
        """
        query = select(WorkspaceMcpServer).where(WorkspaceMcpServer.id == server_id)
        if workspace_id is not None:
            query = query.where(WorkspaceMcpServer.workspace_id == workspace_id)
        if lock:
            query = query.with_for_update().execution_options(populate_existing=True)
        return (await self.db.execute(query)).scalar_one_or_none()

    async def find_many(self, server_ids: Collection[uuid.UUID]) -> list[WorkspaceMcpServer]:
        """The stored servers these ids name."""
        if not server_ids:
            return []
        rows = await self.db.execute(select(WorkspaceMcpServer).where(WorkspaceMcpServer.id.in_(list(server_ids))))
        return list(rows.scalars().all())

    @staticmethod
    def _reached_from(workspace_id: uuid.UUID, organization_id: uuid.UUID | None) -> ColumnElement[bool]:
        """A workspace's own servers, and with ``organization_id`` its organization's, before its settings on them."""
        own = WorkspaceMcpServer.workspace_id == workspace_id
        if organization_id is None:
            return own
        return or_(
            own,
            and_(WorkspaceMcpServer.workspace_id.is_(None), WorkspaceMcpServer.organization_id == organization_id),
        )

    async def ids_in_workspace(
        self, workspace_id: uuid.UUID, organization_id: uuid.UUID | None = None
    ) -> list[uuid.UUID]:
        """The ids of a workspace's stored servers, and with ``organization_id`` of its organization's."""
        return list(
            (
                await self.db.execute(
                    select(WorkspaceMcpServer.id).where(self._reached_from(workspace_id, organization_id))
                )
            )
            .scalars()
            .all()
        )

    async def ids_of_organization(self, organization_id: uuid.UUID) -> list[uuid.UUID]:
        """The ids of an organization's own servers, the ones no workspace owns."""
        rows = await self.db.execute(
            select(WorkspaceMcpServer.id).where(
                WorkspaceMcpServer.organization_id == organization_id, WorkspaceMcpServer.workspace_id.is_(None)
            )
        )
        return list(rows.scalars().all())

    async def ids_in_workspaces(self, workspace_ids: Collection[uuid.UUID]) -> list[uuid.UUID]:
        """The ids of these workspaces' stored servers."""
        if not workspace_ids:
            return []
        return list(
            (
                await self.db.execute(
                    select(WorkspaceMcpServer.id).where(WorkspaceMcpServer.workspace_id.in_(list(workspace_ids)))
                )
            )
            .scalars()
            .all()
        )

    async def for_request(
        self,
        workspace_id: uuid.UUID,
        organization_id: uuid.UUID,
        server_ids: Collection[uuid.UUID],
        *,
        attached: bool,
    ) -> list[WorkspaceMcpServer]:
        """The servers a workspace reaches with these ids, and with ``attached`` each it might attach, oldest first.

        Every organization server is one it might attach, since the workspace's own setting can attach it.
        """
        wanted: ColumnElement[bool] = WorkspaceMcpServer.id.in_(list(server_ids))
        if attached:
            wanted = or_(
                wanted, WorkspaceMcpServer.attach_by_default.is_(True), WorkspaceMcpServer.workspace_id.is_(None)
            )
        query = (
            select(WorkspaceMcpServer)
            .where(self._reached_from(workspace_id, organization_id), wanted)
            .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
        )
        return list((await self.db.execute(query)).scalars().all())

    async def enabled_in_workspace(
        self, workspace_id: uuid.UUID, organization_id: uuid.UUID | None = None
    ) -> list[WorkspaceMcpServer]:
        """A workspace's enabled servers, and with ``organization_id`` its organization's, oldest first."""
        query = (
            select(WorkspaceMcpServer)
            .where(self._reached_from(workspace_id, organization_id), WorkspaceMcpServer.enabled.is_(True))
            .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
        )
        return list((await self.db.execute(query)).scalars().all())

    async def in_workspace(self, workspace_id: uuid.UUID, organization_id: uuid.UUID) -> list[WorkspaceMcpServer]:
        """A workspace's servers and its organization's, enabled or not, oldest first."""
        query = (
            select(WorkspaceMcpServer)
            .where(self._reached_from(workspace_id, organization_id))
            .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
        )
        return list((await self.db.execute(query)).scalars().all())

    async def enabled_of_organization(self, organization_id: uuid.UUID) -> list[WorkspaceMcpServer]:
        """An organization's own enabled servers, oldest first."""
        query = (
            select(WorkspaceMcpServer)
            .where(
                WorkspaceMcpServer.organization_id == organization_id,
                WorkspaceMcpServer.workspace_id.is_(None),
                WorkspaceMcpServer.enabled.is_(True),
            )
            .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
        )
        return list((await self.db.execute(query)).scalars().all())

    async def name_in_use(
        self, organization_id: uuid.UUID, name: str, *, of_workspace: bool, except_id: uuid.UUID | None
    ) -> bool:
        """Whether a server ``name`` would collide with another one a workspace of the organization reaches.

        A workspace's server (``of_workspace``) collides with an organization server of that name, its own
        workspace's being the unique constraint's to refuse. An organization server collides with any server
        of the organization of that name, every workspace reaching it.
        """
        query = select(WorkspaceMcpServer.id).where(
            WorkspaceMcpServer.organization_id == organization_id, WorkspaceMcpServer.name == name
        )
        if of_workspace:
            query = query.where(WorkspaceMcpServer.workspace_id.is_(None))
        if except_id is not None:
            query = query.where(WorkspaceMcpServer.id != except_id)
        return (await self.db.execute(query.limit(1))).first() is not None

    async def attached_workspace_ids(self) -> frozenset[uuid.UUID]:
        """Every workspace that has an enabled server of its own attached by default."""
        found = await self.db.execute(
            select(WorkspaceMcpServer.workspace_id)
            .where(
                WorkspaceMcpServer.attach_by_default.is_(True),
                WorkspaceMcpServer.enabled.is_(True),
                WorkspaceMcpServer.workspace_id.is_not(None),
            )
            .distinct()
        )
        return frozenset(workspace_id for workspace_id in found.scalars() if workspace_id is not None)

    async def attached_organization_ids(self) -> frozenset[uuid.UUID]:
        """Every organization with an enabled server of its own attached by default."""
        found = await self.db.execute(
            select(WorkspaceMcpServer.organization_id)
            .where(
                WorkspaceMcpServer.attach_by_default.is_(True),
                WorkspaceMcpServer.enabled.is_(True),
                WorkspaceMcpServer.workspace_id.is_(None),
            )
            .distinct()
        )
        return frozenset(found.scalars())

    async def has_url(self, workspace_id: uuid.UUID, url: str) -> bool:
        """Whether a workspace stores a server at ``url``, a trailing slash and surrounding whitespace aside.

        SQL ``trim`` takes spaces alone, so the whitespace a stored URL can carry (a tab, a line
        break) is named; ``ltrim`` and ``rtrim`` with a set read the same on PostgreSQL and SQLite.
        """
        bare = url.strip().rstrip("/")
        stored = func.rtrim(func.ltrim(WorkspaceMcpServer.url, _WHITESPACE), _WHITESPACE + "/")
        rows = await self.db.execute(
            select(WorkspaceMcpServer.id)
            .where(WorkspaceMcpServer.workspace_id == workspace_id, stored == bare)
            .limit(1)
        )
        return rows.first() is not None

    async def url_page(self, workspace_id: uuid.UUID, *, skip: int, limit: int) -> dict[uuid.UUID, str]:
        """A page of a workspace's stored servers' URLs, by server id, oldest first as the server list pages."""
        rows = await self.db.execute(
            select(WorkspaceMcpServer.id, WorkspaceMcpServer.url)
            .where(WorkspaceMcpServer.workspace_id == workspace_id)
            .order_by(WorkspaceMcpServer.created_at, WorkspaceMcpServer.id)
            .offset(skip)
            .limit(limit)
        )
        return {row.id: row.url for row in rows}

    async def urls_in_workspace(self, workspace_id: uuid.UUID) -> dict[uuid.UUID, str]:
        """Each of a workspace's stored servers' URL, by server id."""
        rows = await self.db.execute(
            select(WorkspaceMcpServer.id, WorkspaceMcpServer.url).where(WorkspaceMcpServer.workspace_id == workspace_id)
        )
        return {row.id: row.url for row in rows}

    async def forget_secrets(
        self, organization_ids: Collection[uuid.UUID], workspace_ids: Collection[uuid.UUID]
    ) -> None:
        """Stage clearing the bearer token of every server in these workspaces or of these organizations."""
        if not workspace_ids and not organization_ids:
            return
        await self.db.execute(
            update(WorkspaceMcpServer)
            .where(
                or_(
                    WorkspaceMcpServer.workspace_id.in_(list(workspace_ids)),
                    WorkspaceMcpServer.organization_id.in_(list(organization_ids)),
                )
            )
            .values(encrypted_token=None)
            .execution_options(synchronize_session=False)
        )
