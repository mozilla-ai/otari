"""``search_tool_credentials`` rows: the search and fetch instances stored through the dashboard.

Flushes, never commits. The plaintext key never lands in a column: the caller
hands this repository the ciphertext.
"""

from collections.abc import Sequence
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy import CursorResult, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import SearchToolCredential
from gateway.repositories.base_repository import BaseRepository


class SearchToolConflict(Exception):
    """The database refused a flush on a unique key.

    The repository's name for that ``IntegrityError``, so the service that maps
    it to a domain error imports no database library.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class SearchToolRepository(BaseRepository[SearchToolCredential, BaseModel, BaseModel]):
    """Stored search and fetch instances."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        """Bind the repository to a session or a unit of work."""
        super().__init__(db, SearchToolCredential)

    async def list_committed(self) -> Sequence[SearchToolCredential]:
        """Every stored row as committed, including one this session already holds.

        ``populate_existing``: sessions use ``expire_on_commit=False``, so without
        it a row still in the identity map, as after a rotation on this session,
        would come back with the values it was loaded with.
        """
        stmt = select(SearchToolCredential).execution_options(populate_existing=True)
        return (await self.db.execute(stmt)).scalars().all()

    async def list_by_kind(self, kind: str | None = None) -> Sequence[SearchToolCredential]:
        """Every stored row, or every one of ``kind``, ordered by name."""
        stmt = select(SearchToolCredential).order_by(SearchToolCredential.name)
        if kind is not None:
            stmt = stmt.where(SearchToolCredential.kind == kind)
        return (await self.db.execute(stmt)).scalars().all()

    async def get_for_update(self, name: str) -> SearchToolCredential | None:
        """The row called ``name``, locked ``FOR UPDATE``, or None.

        So a version check and the write it guards run under the same row lock.
        """
        stmt = select(SearchToolCredential).where(SearchToolCredential.name == name).with_for_update()
        return (await self.db.execute(stmt)).scalar_one_or_none()

    async def insert(self, **fields: Any) -> SearchToolCredential:
        """Stage a new row, flushed with whatever the transaction staged before it.

        Raises:
            SearchToolConflict: a row of that name already exists, or a write
                staged before it collided with one committed meanwhile.
        """
        row = SearchToolCredential(**fields)
        self.db.add(row)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise SearchToolConflict(row.name) from exc
        await self.db.refresh(row)
        return row

    async def list_encrypted(self) -> Sequence[SearchToolCredential]:
        """Every row holding a key, for a key rotation."""
        stmt = select(SearchToolCredential).where(SearchToolCredential.encrypted_api_key.is_not(None))
        return (await self.db.execute(stmt)).scalars().all()

    async def swap_ciphertext(self, name: str, *, expected: str, replacement: str) -> bool:
        """Replace a row's key ciphertext if it still holds ``expected``.

        A conditional UPDATE, so an edit committed while a rotation ran is not
        overwritten. Returns whether the row was written.
        """
        result = cast(
            CursorResult[Any],
            await self.db.execute(
                update(SearchToolCredential)
                .where(SearchToolCredential.name == name, SearchToolCredential.encrypted_api_key == expected)
                .values(encrypted_api_key=replacement)
                .execution_options(synchronize_session=False)
            ),
        )
        return result.rowcount == 1
