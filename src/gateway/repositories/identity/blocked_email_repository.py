import uuid
from collections.abc import Collection
from typing import Never

from sqlalchemy import delete, false, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import BlockedEmailEntryExistsError
from gateway.models.identity import BlockedEmail
from gateway.repositories.base_repository import BaseRepository

# The advisory lock a blocklist check holds shared and a new block holds exclusive, so neither commits around the other.
_BLOCKLIST_LOCK_CLASS = 0x6F74_626C  # "otbl"


class BlockedEmailRepository(BaseRepository[BlockedEmail, Never, Never]):
    """Query the deployment's email blocklist."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, BlockedEmail)

    async def list_page(self, *, skip: int, limit: int) -> tuple[list[BlockedEmail], int]:
        """Return a page of entries, newest first, and how many there are in all."""
        rows = await self.db.execute(
            select(BlockedEmail).order_by(BlockedEmail.created_at.desc(), BlockedEmail.id).offset(skip).limit(limit)
        )
        count = await self.db.scalar(select(func.count()).select_from(BlockedEmail))
        return list(rows.scalars().all()), int(count or 0)

    async def lock_shared(self) -> None:
        """Hold the blocklist lock shared until the transaction ends, so no block commits around this one.

        Holders do not wait for each other. A no-op on SQLite, which admits one writer at a time anyway.
        """
        if dialect_name(self.db) == "postgresql":
            await self.db.execute(select(func.pg_advisory_xact_lock_shared(_BLOCKLIST_LOCK_CLASS, 0)))

    async def lock_exclusive(self) -> None:
        """Hold the blocklist lock exclusive until the transaction ends, waiting out every shared holder.

        On SQLite, a write that matches no row instead: it takes the one write lock, so a sign-in cannot commit
        an account between this block's lookup and its commit.
        """
        if dialect_name(self.db) == "postgresql":
            await self.db.execute(select(func.pg_advisory_xact_lock(_BLOCKLIST_LOCK_CLASS, 0)))
        else:
            await self.db.execute(delete(BlockedEmail).where(false()))

    async def patterns_among(self, patterns: Collection[str]) -> set[str]:
        """Return the members of ``patterns`` that are blocked."""
        if not patterns:
            return set()
        result = await self.db.execute(select(BlockedEmail.pattern).where(BlockedEmail.pattern.in_(patterns)))
        return set(result.scalars().all())

    async def add(self, *, pattern: str, reason: str | None, created_by_user_id: uuid.UUID) -> BlockedEmail:
        """Stage a new entry.

        Raises:
            BlockedEmailEntryExistsError: If ``pattern`` is already blocked.

        """
        entry = BlockedEmail(pattern=pattern, reason=reason, created_by_user_id=created_by_user_id)
        try:
            # A savepoint, so a duplicate leaves the rest of the transaction usable.
            async with self.db.begin_nested():
                self.db.add(entry)
                await self.db.flush()
        except IntegrityError as error:
            raise BlockedEmailEntryExistsError(pattern) from error
        return entry

    async def remove(self, entry: BlockedEmail) -> None:
        """Stage the removal of an entry."""
        await self.db.delete(entry)
        await self.db.flush()
