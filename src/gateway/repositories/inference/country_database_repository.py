"""Data access for the country file stored in the database. Flushes, never commits."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import defer

from gateway.core.sql import dialect_name
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.inference import StoredCountryDatabase
from gateway.repositories.base_repository import BaseRepository

_ROW_ID = 1


class CountryDatabaseRepository(BaseRepository[StoredCountryDatabase, Never, Never]):
    """Read, replace and remove the one stored country file."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, StoredCountryDatabase)

    async def find_summary(self) -> StoredCountryDatabase | None:
        """The stored file's metadata, its bytes left unread: reading ``content`` raises."""
        result = await self.db.execute(
            select(StoredCountryDatabase)
            .options(defer(StoredCountryDatabase.content, raiseload=True))
            .execution_options(populate_existing=True)
        )
        return result.scalar_one_or_none()

    async def find_sha256(self) -> str | None:
        """The stored file's SHA-256, or None when no file is stored."""
        return (await self.db.execute(select(StoredCountryDatabase.sha256))).scalar_one_or_none()

    async def find(self) -> StoredCountryDatabase | None:
        """The stored file, bytes included."""
        result = await self.db.execute(select(StoredCountryDatabase).execution_options(populate_existing=True))
        return result.scalar_one_or_none()

    async def replace(
        self, *, content: bytes, database_type: str, built_at: datetime, sha256: str, uploaded_at: datetime
    ) -> None:
        """Store ``content`` in place of any file stored before.

        An upsert, so two uploads at once leave the later one rather than a key conflict.
        """
        values: dict[str, Any] = {
            "id": _ROW_ID,
            "content": content,
            "database_type": database_type,
            "built_at": built_at,
            "size_bytes": len(content),
            "sha256": sha256,
            "uploaded_at": uploaded_at,
        }
        insert = postgresql_insert if dialect_name(self.db) == "postgresql" else sqlite_insert
        statement = insert(StoredCountryDatabase).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=[StoredCountryDatabase.id],
            set_={key: value for key, value in values.items() if key != "id"},
        )
        await self.db.execute(statement)
        await self.db.flush()

    async def remove(self) -> bool:
        """Remove the stored file, returning whether there was one."""
        result = cast("CursorResult[Any]", await self.db.execute(delete(StoredCountryDatabase)))
        await self.db.flush()
        return bool(result.rowcount)
