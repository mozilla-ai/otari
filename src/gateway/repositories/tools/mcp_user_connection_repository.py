"""Data access for the sign-ins a stored MCP server's calls carry.

``mcp_user_connections`` holds each user's own tokens for a per-user server, and
``mcp_shared_connections`` the one account a manager signed in to for everyone on a shared one.
The only module that constructs or queries either table. Writes flush and leave
the commit to the caller.
"""

import uuid
from collections.abc import Callable, Collection
from datetime import UTC, datetime
from typing import Never

from sqlalchemy import ColumnElement, case, delete, event, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, SessionTransaction

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import McpSharedConnection, McpUserConnection
from gateway.repositories.base_repository import BaseRepository

# The most connections one removal reads to revoke; the rest are counted.
REVOCABLE_PAGE = 500


class McpUserConnectionRepository(BaseRepository[McpUserConnection, Never, Never]):
    """Read and stage connection rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpUserConnection)

    async def find(self, server_id: uuid.UUID, user_id: str, *, lock: bool = False) -> McpUserConnection | None:
        """A user's connection to a server, or ``None``.

        The row is read as it is committed, not as this session loaded it earlier, and ``lock`` holds it
        until the caller commits, so a caller that waited on another's refresh sees the tokens it stored.
        """
        query = (
            select(McpUserConnection)
            .where(McpUserConnection.server_id == server_id, McpUserConnection.user_id == user_id)
            .execution_options(populate_existing=True)
        )
        if lock:
            query = query.with_for_update()
        return (await self.db.execute(query)).scalar_one_or_none()

    async def connected_server_ids(self, user_id: str, server_ids: Collection[uuid.UUID]) -> set[uuid.UUID]:
        """Which of these servers the user has a connection to."""
        if not server_ids:
            return set()
        rows = await self.db.execute(
            select(McpUserConnection.server_id).where(
                McpUserConnection.user_id == user_id, McpUserConnection.server_id.in_(list(server_ids))
            )
        )
        return set(rows.scalars().all())

    async def for_user(
        self, user_id: str, server_ids: Collection[uuid.UUID], *, skip: int, limit: int
    ) -> list[McpUserConnection]:
        """A page of a user's connections to these servers, oldest first."""
        if not server_ids:
            return []
        rows = await self.db.execute(
            select(McpUserConnection)
            .where(McpUserConnection.user_id == user_id, McpUserConnection.server_id.in_(list(server_ids)))
            .order_by(McpUserConnection.created_at, McpUserConnection.id)
            .offset(skip)
            .limit(limit)
        )
        return list(rows.scalars().all())

    async def count_for_user(self, user_id: str, server_ids: Collection[uuid.UUID]) -> int:
        """How many of these servers a user is connected to."""
        if not server_ids:
            return 0
        counted = (
            select(func.count())
            .select_from(McpUserConnection)
            .where(McpUserConnection.user_id == user_id, McpUserConnection.server_id.in_(list(server_ids)))
        )
        return (await self.db.execute(counted)).scalar_one()

    async def for_server(
        self, server_id: uuid.UUID, *, skip: int, limit: int, first_user_id: str | None = None
    ) -> list[McpUserConnection]:
        """A page of a server's connections, ``first_user_id``'s first when it has one, then oldest first."""
        rows = await self.db.execute(
            select(McpUserConnection)
            .where(McpUserConnection.server_id == server_id)
            .order_by(
                case((McpUserConnection.user_id == first_user_id, 0), else_=1),
                McpUserConnection.created_at,
                McpUserConnection.id,
            )
            .offset(skip)
            .limit(limit)
        )
        return list(rows.scalars().all())

    async def revocable(
        self,
        *,
        server_ids: Collection[uuid.UUID] | None = None,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[McpUserConnection], int]:
        """Up to ``limit`` connections whose authorization server can be asked to end them, and how many there are.

        Narrowed to these servers, to this user, or both. A removal that ends more than ``limit``
        revokes the first ``limit`` and counts the rest, rather than holding them all in memory.
        """
        if server_ids is not None and not server_ids:
            return [], 0
        limit = limit or REVOCABLE_PAGE
        conditions: list[ColumnElement[bool]] = [McpUserConnection.revocation_endpoint.is_not(None)]
        if server_ids is not None:
            conditions.append(McpUserConnection.server_id.in_(list(server_ids)))
        if user_id is not None:
            conditions.append(McpUserConnection.user_id == user_id)
        rows = await self.db.execute(
            select(McpUserConnection)
            .where(*conditions)
            .order_by(McpUserConnection.created_at, McpUserConnection.id)
            .limit(limit)
        )
        found = list(rows.scalars().all())
        if len(found) < limit:
            return found, len(found)
        total = (
            await self.db.execute(select(func.count()).select_from(McpUserConnection).where(*conditions))
        ).scalar_one()
        return found, total

    def after_commit(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once the transaction this repository writes in commits, and never if it rolls back.

        For work that must not happen before the removal it follows is durable, such as asking an
        authorization server to end tokens, from code that does not own the commit.

        SQLAlchemy fires both events for savepoints too, so each is narrowed. ``after_commit`` counts
        only the outermost commit, when no savepoint is open. A rollback counts only when it undoes
        the transaction the callback was registered in, or one around it; a savepoint beside it
        rolling back leaves it pending. The listeners stay on the session until it closes, inert
        once settled, rather than being removed from inside their own dispatch.
        """
        session = self.db.sync_session
        scope = session.get_nested_transaction() or session.get_transaction()
        settled = False

        def committed(committing: Session) -> None:
            nonlocal settled
            if settled or committing.get_nested_transaction() is not None:
                return
            settled = True
            callback()

        def rolled_back(_session: Session, previous: SessionTransaction) -> None:
            nonlocal settled
            transaction = scope
            while transaction is not None:
                if transaction is previous:
                    settled = True
                    return
                transaction = transaction.parent

        event.listen(session, "after_commit", committed)
        event.listen(session, "after_soft_rollback", rolled_back)

    async def count_for_server(self, server_id: uuid.UUID) -> int:
        """How many users are connected to a server."""
        counted = select(func.count()).select_from(McpUserConnection).where(McpUserConnection.server_id == server_id)
        return (await self.db.execute(counted)).scalar_one()

    async def counts_for_servers(self, server_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, int]:
        """How many users are connected to each of these servers that has any, in one query."""
        if not server_ids:
            return {}
        rows = await self.db.execute(
            select(McpUserConnection.server_id, func.count())
            .where(McpUserConnection.server_id.in_(list(server_ids)))
            .group_by(McpUserConnection.server_id)
        )
        return {server_id: count for server_id, count in rows.all()}

    async def put(
        self,
        server_id: uuid.UUID,
        user_id: str,
        *,
        encrypted_access_token: str,
        encrypted_refresh_token: str | None,
        expires_at: datetime | None,
        scope: str | None,
        client_id: str,
        client_auth_method: str,
        token_endpoint: str,
        revocation_endpoint: str | None,
        resource: str | None,
        server_url: str,
    ) -> McpUserConnection:
        """Stage a user's connection to a server, replacing the tokens of one they already hold.

        Two sign-ins by one user can finish together. The insert turns into an update on the unique
        ``(server_id, user_id)`` constraint, so the later sign-in is the one the row keeps, and the row
        is read back locked until the caller commits.
        """
        fields = {
            "encrypted_access_token": encrypted_access_token,
            "encrypted_refresh_token": encrypted_refresh_token,
            "expires_at": expires_at,
            "scope": scope,
            "client_id": client_id,
            "client_auth_method": client_auth_method,
            "token_endpoint": token_endpoint,
            "revocation_endpoint": revocation_endpoint,
            "resource": resource,
            "server_url": server_url,
        }
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        await self.db.execute(
            insert(McpUserConnection)
            .values(server_id=server_id, user_id=user_id, **fields)
            .on_conflict_do_update(
                index_elements=[McpUserConnection.server_id, McpUserConnection.user_id],
                set_={**fields, "updated_at": datetime.now(UTC)},
            )
        )
        await self.db.flush()
        stored = await self.find(server_id, user_id, lock=True)
        if stored is None:
            raise RuntimeError("an upserted MCP connection could not be read back")
        return stored

    async def find_shared(self, server_id: uuid.UUID, *, lock: bool = False) -> McpSharedConnection | None:
        """A shared server's account, or ``None``; read and locked as :meth:`find` does."""
        query = (
            select(McpSharedConnection)
            .where(McpSharedConnection.server_id == server_id)
            .execution_options(populate_existing=True)
        )
        if lock:
            query = query.with_for_update()
        return (await self.db.execute(query)).scalar_one_or_none()

    async def shared_for_servers(self, server_ids: Collection[uuid.UUID]) -> dict[uuid.UUID, McpSharedConnection]:
        """The shared account of each of these servers that has one, by server."""
        if not server_ids:
            return {}
        rows = await self.db.execute(
            select(McpSharedConnection).where(McpSharedConnection.server_id.in_(list(server_ids)))
        )
        return {row.server_id: row for row in rows.scalars().all()}

    async def put_shared(
        self,
        server_id: uuid.UUID,
        *,
        connected_by: uuid.UUID | None,
        encrypted_access_token: str,
        encrypted_refresh_token: str | None,
        expires_at: datetime | None,
        scope: str | None,
        client_id: str,
        client_auth_method: str,
        token_endpoint: str,
        revocation_endpoint: str | None,
        resource: str | None,
        server_url: str,
    ) -> McpSharedConnection:
        """Stage a shared server's account, in place of the one it holds.

        The caller holds the server's row lock, so two sign-ins for one server store one after the other.
        """
        existing = await self.find_shared(server_id, lock=True)
        if existing is not None:
            await self.db.delete(existing)
            await self.db.flush()
        created = McpSharedConnection(
            server_id=server_id,
            connected_by=connected_by,
            encrypted_access_token=encrypted_access_token,
            encrypted_refresh_token=encrypted_refresh_token,
            expires_at=expires_at,
            scope=scope,
            client_id=client_id,
            client_auth_method=client_auth_method,
            token_endpoint=token_endpoint,
            revocation_endpoint=revocation_endpoint,
            resource=resource,
            server_url=server_url,
        )
        self.db.add(created)
        await self.db.flush()
        return created

    async def remove_shared_for_server(self, server_id: uuid.UUID) -> None:
        """Stage the removal of a shared server's account."""
        await self.db.execute(delete(McpSharedConnection).where(McpSharedConnection.server_id == server_id))

    async def store_tokens(
        self,
        connection: McpUserConnection | McpSharedConnection,
        *,
        encrypted_access_token: str,
        encrypted_refresh_token: str | None,
        expires_at: datetime | None,
        scope: str | None,
    ) -> None:
        """Stage the tokens a refresh returned on a connection."""
        connection.encrypted_access_token = encrypted_access_token
        connection.encrypted_refresh_token = encrypted_refresh_token
        connection.expires_at = expires_at
        connection.scope = scope
        await self.db.flush()

    async def remove(self, connection: McpUserConnection | McpSharedConnection) -> None:
        """Stage one connection's removal."""
        await self.db.delete(connection)
        await self.db.flush()

    async def remove_for_user(self, user_id: str) -> None:
        """Stage the removal of every connection a user holds, on any server.

        A user is soft-deleted, so the foreign key's cascade never fires; a user recreated under the
        same ID must not find the deleted one's sign-ins waiting for it.
        """
        await self.db.execute(delete(McpUserConnection).where(McpUserConnection.user_id == user_id))

    async def remove_for_user_on_servers(self, user_id: str, server_ids: Collection[uuid.UUID]) -> None:
        """Stage the removal of a user's connections to these servers."""
        if not server_ids:
            return
        await self.db.execute(
            delete(McpUserConnection).where(
                McpUserConnection.user_id == user_id, McpUserConnection.server_id.in_(list(server_ids))
            )
        )

    async def remove_for_server(self, server_id: uuid.UUID) -> None:
        """Stage the removal of every connection to a server."""
        await self.db.execute(delete(McpUserConnection).where(McpUserConnection.server_id == server_id))
