"""``search_tool_credentials`` rows: the search and fetch instances stored through the dashboard."""

from collections.abc import Sequence

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import SearchToolCredential
from gateway.repositories.base_repository import BaseRepository


class SearchToolRepository(BaseRepository[SearchToolCredential, BaseModel, BaseModel]):
    """Stored search and fetch instances.

    It holds the overlay's read. The writes are still in
    ``search_tool_store_service``, which moving the tools domain replaces.
    """

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
