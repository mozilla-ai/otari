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
        try:
            # A savepoint, so losing to the unique index leaves the caller's transaction usable.
            async with self.db.begin_nested():
                self.db.add(row)
                await self.db.flush()
        except IntegrityError:
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

    async def find(self, workspace_id: uuid.UUID, request_id: uuid.UUID) -> McpServerRequest | None:
        """One request of a workspace, or ``None`` when the id names none there."""
        rows = await self.db.execute(
            select(McpServerRequest).where(
                McpServerRequest.id == request_id, McpServerRequest.workspace_id == workspace_id
            )
        )
        return rows.scalar_one_or_none()

    async def pending_in_workspace(self, workspace_id: uuid.UUID, *, limit: int) -> list[McpServerRequest]:
        """A workspace's pending requests, oldest first."""
        rows = await self.db.execute(
            select(McpServerRequest)
            .where(McpServerRequest.workspace_id == workspace_id, McpServerRequest.status == _PENDING)
            .order_by(McpServerRequest.created_at, McpServerRequest.id)
            .limit(limit)
        )
        return list(rows.scalars().all())

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
