"""Data access for the idempotency keys completion requests claim."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.inference import IdempotencyRecord, IdempotencyState
from gateway.repositories.base_repository import BaseRepository


class IdempotencyRepository(BaseRepository[IdempotencyRecord, Never, Never]):
    """Claim, complete and reclaim idempotency keys in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, IdempotencyRecord)

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

    async def take_over(self, scope: str, idempotency_key: str, *, previous_token: str, values: dict[str, Any]) -> bool:
        """Replace the claim ``previous_token`` names, returning whether this caller got it.

        One conditional update, so two retries racing for an abandoned or expired
        key cannot both win.
        """
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                update(IdempotencyRecord)
                .where(
                    IdempotencyRecord.scope == scope,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                    IdempotencyRecord.claim_token == previous_token,
                )
                .values(**values)
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

    async def delete_expired(self, now: datetime) -> int:
        """Delete every row whose retention has passed and whose claim is no longer live."""
        result = cast(
            "CursorResult[Any]",
            await self.db.execute(
                delete(IdempotencyRecord).where(
                    IdempotencyRecord.expires_at <= now,
                    or_(
                        IdempotencyRecord.state == IdempotencyState.COMPLETED,
                        IdempotencyRecord.locked_until <= now,
                    ),
                )
            ),
        )
        await self.db.flush()
        return result.rowcount
