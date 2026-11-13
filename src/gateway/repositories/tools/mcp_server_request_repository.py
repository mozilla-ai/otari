"""Data access for ``mcp_server_requests``: members asking for a catalog server. Flushes, never commits."""

import uuid
from collections.abc import Collection
from datetime import datetime
from typing import Never

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import McpServerRequest, McpServerRequestStatus
from gateway.repositories.base_repository import BaseRepository

_PENDING = McpServerRequestStatus.PENDING


class McpServerRequestConflict(Exception):
    """The requester already has a pending request for that catalog server in the workspace."""


class McpServerRequestRepository(BaseRepository[McpServerRequest, Never, Never]):
    """Read and stage request rows. Every read names the workspace it is for."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpServerRequest)

    async def add(
        self,
        *,
        workspace_id: uuid.UUID,
        catalog_name: str,
        title: str,
        url: str,
        requester_id: uuid.UUID,
        note: str | None,
    ) -> McpServerRequest:
        """Stage a pending request.

        Raises:
            McpServerRequestConflict: the requester's pending request for that server already exists.
        """
        row = McpServerRequest(
            workspace_id=workspace_id,
            catalog_name=catalog_name,
            title=title,
            url=url,
            requester_id=requester_id,
            note=note,
        )
        self.db.add(row)
        try:
            await self.db.flush()
        except IntegrityError:
            # The step ends here, and its block rolls the transaction back.
            raise McpServerRequestConflict from None
        return row

    async def has_pending(self, workspace_id: uuid.UUID, catalog_name: str, requester_id: uuid.UUID) -> bool:
        """Whether the requester's request for that catalog server is waiting in the workspace."""
        rows = await self.db.execute(
            select(McpServerRequest.id).where(
                McpServerRequest.workspace_id == workspace_id,
                McpServerRequest.catalog_name == catalog_name,
                McpServerRequest.requester_id == requester_id,
                McpServerRequest.status == _PENDING,
            )
        )
        return rows.first() is not None

    async def find(
        self, workspace_id: uuid.UUID, request_id: uuid.UUID, *, lock: bool = False
    ) -> McpServerRequest | None:
        """One request of a workspace, or ``None`` when the id names none there.

        With ``lock`` the row is locked, as :meth:`pending_for_server` locks the rows an answer changes.
        """
        query = select(McpServerRequest).where(
            McpServerRequest.id == request_id, McpServerRequest.workspace_id == workspace_id
        )
        if lock:
            query = query.with_for_update().execution_options(populate_existing=True)
        return (await self.db.execute(query)).scalar_one_or_none()

    async def pending_servers(self, workspace_id: uuid.UUID, *, skip: int, limit: int) -> list[str]:
        """The catalog names a workspace's pending requests ask for, the one waited for longest first."""
        first = func.min(McpServerRequest.created_at)
        rows = await self.db.execute(
            select(McpServerRequest.catalog_name)
            .where(McpServerRequest.workspace_id == workspace_id, McpServerRequest.status == _PENDING)
            .group_by(McpServerRequest.catalog_name)
            .order_by(first, McpServerRequest.catalog_name)
            .offset(skip)
            .limit(limit)
        )
        return list(rows.scalars().all())

    async def pending_in_workspace(
        self, workspace_id: uuid.UUID, *, catalog_names: Collection[str], per_server: int
    ) -> list[McpServerRequest]:
        """A workspace's pending requests for these catalog servers, the first ``per_server`` of each, oldest first.

        Bounded per server, so one server's many requests never crowd another out of the page.
        """
        if not catalog_names:
            return []
        waiting = (
            McpServerRequest.workspace_id == workspace_id,
            McpServerRequest.status == _PENDING,
            McpServerRequest.catalog_name.in_(list(catalog_names)),
        )
        ranked = (
            select(
                McpServerRequest.id,
                func.row_number()
                .over(
                    partition_by=McpServerRequest.catalog_name,
                    order_by=(McpServerRequest.created_at, McpServerRequest.id),
                )
                .label("rank"),
            )
            .where(*waiting)
            .subquery()
        )
        rows = await self.db.execute(
            select(McpServerRequest)
            .join(ranked, ranked.c.id == McpServerRequest.id)
            .where(ranked.c.rank <= per_server)
            .order_by(McpServerRequest.created_at, McpServerRequest.id)
        )
        return list(rows.scalars().all())

    async def pending_for_name(
        self, workspace_id: uuid.UUID, catalog_name: str, *, skip: int, limit: int
    ) -> list[McpServerRequest]:
        """A page of the pending requests for one catalog server, oldest first."""
        rows = await self.db.execute(
            select(McpServerRequest)
            .where(
                McpServerRequest.workspace_id == workspace_id,
                McpServerRequest.status == _PENDING,
                McpServerRequest.catalog_name == catalog_name,
            )
            .order_by(McpServerRequest.created_at, McpServerRequest.id)
            .offset(skip)
            .limit(limit)
        )
        return list(rows.scalars().all())

    async def count_pending_by_server(self, workspace_id: uuid.UUID, catalog_names: Collection[str]) -> dict[str, int]:
        """How many requests wait for each of these catalog servers, in one query."""
        if not catalog_names:
            return {}
        rows = await self.db.execute(
            select(McpServerRequest.catalog_name, func.count())
            .where(
                McpServerRequest.workspace_id == workspace_id,
                McpServerRequest.status == _PENDING,
                McpServerRequest.catalog_name.in_(list(catalog_names)),
            )
            .group_by(McpServerRequest.catalog_name)
        )
        return {name: count for name, count in rows.all()}

    async def count_pending_in_workspace(self, workspace_id: uuid.UUID) -> int:
        counted = (
            select(func.count())
            .select_from(McpServerRequest)
            .where(McpServerRequest.workspace_id == workspace_id, McpServerRequest.status == _PENDING)
        )
        return (await self.db.execute(counted)).scalar_one()

    async def pending_for_server(
        self, workspace_id: uuid.UUID, *, catalog_name: str | None = None, urls: Collection[str] = ()
    ) -> list[McpServerRequest]:
        """A workspace's pending requests for one catalog server, by its name or by any of ``urls``."""
        if catalog_name is None and not urls:
            return []
        matches = (
            McpServerRequest.catalog_name == catalog_name
            if catalog_name is not None
            else McpServerRequest.url.in_(list(urls))
        )
        rows = await self.db.execute(
            select(McpServerRequest)
            .where(McpServerRequest.workspace_id == workspace_id, McpServerRequest.status == _PENDING, matches)
            .order_by(McpServerRequest.created_at, McpServerRequest.id)
            .with_for_update()
        )
        return list(rows.scalars().all())

    async def for_requester(
        self,
        workspace_id: uuid.UUID,
        requester_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int,
    ) -> list[McpServerRequest]:
        """A requester's pending requests in a workspace, and the answers they have not acknowledged, newest first.

        An approval whose server is gone is left out: there is nothing left to point at.
        """
        unacknowledged = McpServerRequest.acknowledged_at.is_(None)
        rows = await self.db.execute(
            select(McpServerRequest)
            .where(
                McpServerRequest.workspace_id == workspace_id,
                McpServerRequest.requester_id == requester_id,
                (McpServerRequest.status == _PENDING)
                | ((McpServerRequest.status == McpServerRequestStatus.DECLINED) & unacknowledged)
                | (
                    (McpServerRequest.status == McpServerRequestStatus.APPROVED)
                    & unacknowledged
                    & McpServerRequest.server_id.is_not(None)
                ),
            )
            .order_by(McpServerRequest.created_at.desc(), McpServerRequest.id)
            .offset(skip)
            .limit(limit)
        )
        return list(rows.scalars().all())

    async def count_for_requester(
        self, workspace_id: uuid.UUID, requester_id: uuid.UUID, *, since: datetime | None = None
    ) -> int:
        """How many requests the requester made in a workspace since then; with no ``since``, how many are pending."""
        counted = (
            select(func.count())
            .select_from(McpServerRequest)
            .where(McpServerRequest.workspace_id == workspace_id, McpServerRequest.requester_id == requester_id)
        )
        if since is None:
            counted = counted.where(McpServerRequest.status == _PENDING)
        else:
            counted = counted.where(McpServerRequest.created_at >= since)
        return (await self.db.execute(counted)).scalar_one()

    async def withdraw_for_requester(
        self, requester_id: uuid.UUID, workspace_ids: Collection[uuid.UUID], *, at: datetime
    ) -> None:
        """Withdraw every pending request the requester holds in these workspaces."""
        if not workspace_ids:
            return
        await self.db.execute(
            update(McpServerRequest)
            .where(
                McpServerRequest.requester_id == requester_id,
                McpServerRequest.workspace_id.in_(list(workspace_ids)),
                McpServerRequest.status == _PENDING,
            )
            .values(status=McpServerRequestStatus.WITHDRAWN, decided_at=at, updated_at=at)
            .execution_options(synchronize_session=False)
        )

    async def acknowledge_declined(
        self,
        workspace_id: uuid.UUID,
        *,
        at: datetime,
        catalog_name: str | None = None,
        urls: Collection[str] = (),
        requester_id: uuid.UUID | None = None,
    ) -> None:
        """Stop showing a workspace's declined requests for one server, named by ``catalog_name`` or by ``urls``.

        Every requester's, or one's when ``requester_id`` is given. The rows stay.
        """
        if catalog_name is None and not urls:
            return
        matches = (
            McpServerRequest.catalog_name == catalog_name
            if catalog_name is not None
            else McpServerRequest.url.in_(list(urls))
        )
        statement = update(McpServerRequest).where(
            McpServerRequest.workspace_id == workspace_id,
            McpServerRequest.status == McpServerRequestStatus.DECLINED,
            McpServerRequest.acknowledged_at.is_(None),
            matches,
        )
        if requester_id is not None:
            statement = statement.where(McpServerRequest.requester_id == requester_id)
        await self.db.execute(
            statement.values(acknowledged_at=at, updated_at=at).execution_options(synchronize_session=False)
        )

    async def save(self) -> None:
        """Flush changes made to loaded rows."""
        await self.db.flush()
