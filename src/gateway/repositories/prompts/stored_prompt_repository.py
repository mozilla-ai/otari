"""Data access for stored prompts. Flushes, never commits."""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.prompts import StoredPrompt
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.prompts import PromptCreate, PromptUpdate


class PromptNameTaken(Exception):
    """The unique constraint refused a name another writer stored first."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


class StoredPromptRepository(BaseRepository[StoredPrompt, PromptCreate, PromptUpdate]):
    """Repository for ``stored_prompts`` rows."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, StoredPrompt)

    async def list_all(self) -> Sequence[StoredPrompt]:
        """Every stored prompt, in name order."""
        result = await self.db.execute(
            select(StoredPrompt).order_by(StoredPrompt.name).execution_options(populate_existing=True)
        )
        return result.scalars().all()

    async def get_by_id(self, prompt_id: uuid.UUID) -> StoredPrompt | None:
        """Return the prompt with this id, or None."""
        result = await self.db.execute(
            select(StoredPrompt).where(StoredPrompt.id == prompt_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> StoredPrompt | None:
        """Return the prompt with this name, or None."""
        result = await self.db.execute(
            select(StoredPrompt).where(StoredPrompt.name == name)
        )
        return result.scalar_one_or_none()

    async def add(self, row: StoredPrompt) -> StoredPrompt:
        """Stage a new prompt, flushed so the primary key answers.

        Raises:
            PromptNameTaken: a prompt with this name is already stored.
        """
        self.db.add(row)
        try:
            await self.db.flush()
        except IntegrityError as exc:
            raise PromptNameTaken(row.name) from exc
        await self.db.refresh(row)
        return row

    async def create_new(self, name: str, content: str) -> StoredPrompt:
        """Create and stage a new StoredPrompt row.

        Raises:
            PromptNameTaken: a prompt with this name is already stored.
        """
        return await self.add(StoredPrompt(name=name, content=content))
