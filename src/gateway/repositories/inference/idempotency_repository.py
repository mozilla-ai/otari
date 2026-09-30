"""Data access for the idempotency keys completion requests claim."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Never, cast

from sqlalchemy import delete, func, or_, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.inference import IdempotencyRecord, IdempotencyState
from gateway.models.users import User
from gateway.repositories.base_repository import BaseRepository


class IdempotencyRepository(BaseRepository[IdempotencyRecord, Never, Never]):
    """Claim, complete and reclaim idempotency keys in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, IdempotencyRecord)

    async def get_database_time(self) -> datetime:
        """Return the database's current time, so every gateway times leases by the same clock.

        The time is read when asked, not when the open transaction began.
        SQLite serves one host, so its gateway's own clock is that clock.
        """
        if dialect_name(self.db) != "postgresql":
            return datetime.now(UTC)
        return cast("datetime", (await self.db.execute(select(func.clock_timestamp()))).scalar_one())

    async def get_caller(self, user_id: str) -> User | None:
        """Return the user a key is claimed for, or None when no such user exists or it was deleted."""
        result = await self.db.execute(select(User).where(User.user_id == user_id, User.deleted_at.is_(None)))
        return result.scalar_one_or_none()

    async def insert_claim(self, values: dict[str, Any]) -> bool:
        """Insert a fresh ``in_progress`` claim, returning False when the key is already held.

        ``ON CONFLICT DO NOTHING`` rather than catching the unique violation, so
        losing the race leaves the transaction usable.
        """
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        statement = (
            insert(IdempotencyRecord)
            .values(**values)
            .on_conflict_do_nothing(index_elements=[IdempotencyRecord.scope, IdempotencyRecord.idempotency_key])
        )
        result = cast("CursorResult[Any]", await self.db.execute(statement))
        await self.db.flush()
        return bool(result.rowcount)

    async def find(self, scope: str, idempotency_key: str) -> IdempotencyRecord | None:
        """Return the key's current row, read fresh rather than from the session's identity map."""
        result = await self.db.execute(
            select(IdempotencyRecord)
            .where(IdempotencyRecord.scope == scope, IdempotencyRecord.idempotency_key == idempotency_key)
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def take_over(self, seen: IdempotencyRecord, *, now: datetime, values: dict[str, Any]) -> bool:
        """Replace the claim this caller saw, returning whether this caller got it.

        The claim is replaced only while it is still as seen: the same claim, in the same state,
        and for a running claim, with its lease still lapsed at ``now``.
        A claim that completed or renewed since it was read is left alone, and so is one another retry took.
        """
        unchanged = [
            IdempotencyRecord.scope == seen.scope,
            IdempotencyRecord.idempotency_key == seen.idempotency_key,
            IdempotencyRecord.claim_token == seen.claim_token,
            IdempotencyRecord.state == seen.state,
        ]
        if seen.state == IdempotencyState.IN_PROGRESS:
            unchanged.append(IdempotencyRecord.locked_until <= now)
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(update(IdempotencyRecord).where(*unchanged).values(**values)),
        )
        await self.db.flush()
        return bool(result.rowcount)

    async def extend_lease(
        self, scope: str, idempotency_key: str, *, claim_token: str, locked_until: datetime
    ) -> bool:
        """Push out the lease on the claim this caller still holds, returning whether it still holds it."""
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(IdempotencyRecord)
                .where(
                    IdempotencyRecord.scope == scope,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                    IdempotencyRecord.claim_token == claim_token,
                    IdempotencyRecord.state == IdempotencyState.IN_PROGRESS,
                )
                .values(locked_until=locked_until)
            ),
        )
        await self.db.flush()
        return bool(result.rowcount)

    async def complete(
        self,
        scope: str,
        idempotency_key: str,
        *,
        claim_token: str,
        status_code: int,
        response_body: str,
        response_headers: dict[str, str],
        expires_at: datetime,
    ) -> bool:
        """Store the response on the claim this caller still holds."""
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(IdempotencyRecord)
                .where(
                    IdempotencyRecord.scope == scope,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                    IdempotencyRecord.claim_token == claim_token,
                    IdempotencyRecord.state == IdempotencyState.IN_PROGRESS,
                )
                .values(
                    state=IdempotencyState.COMPLETED,
                    status_code=status_code,
                    response_body=response_body,
                    response_headers=response_headers,
                    expires_at=expires_at,
                )
            ),
        )
        await self.db.flush()
        return bool(result.rowcount)

    async def release(self, scope: str, idempotency_key: str, *, claim_token: str) -> None:
        """Delete the claim this caller still holds, leaving a replacement claim alone."""
        await self.db.execute(
            delete(IdempotencyRecord).where(
                IdempotencyRecord.scope == scope,
                IdempotencyRecord.idempotency_key == idempotency_key,
                IdempotencyRecord.claim_token == claim_token,
                IdempotencyRecord.state == IdempotencyState.IN_PROGRESS,
            )
        )
        await self.db.flush()

    async def delete_expired(self, now: datetime, *, limit: int) -> int:
        """Delete up to ``limit`` of the oldest rows whose retention has passed and whose claim is no longer live.

        Rows another sweep holds are skipped, so gateways sweeping at once delete different rows.
        """
        expired = (
            select(IdempotencyRecord.scope, IdempotencyRecord.idempotency_key)
            .where(
                IdempotencyRecord.expires_at <= now,
                or_(
                    IdempotencyRecord.state == IdempotencyState.COMPLETED,
                    IdempotencyRecord.locked_until <= now,
                ),
            )
            .order_by(IdempotencyRecord.expires_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                delete(IdempotencyRecord).where(
                    tuple_(IdempotencyRecord.scope, IdempotencyRecord.idempotency_key).in_(expired)
                )
            ),
        )
        await self.db.flush()
        return result.rowcount
