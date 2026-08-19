"""Generic CRUD for the reconciled control plane's repositories.

The async counterpart of the platform's ``BaseRepository``
(`otari-ai` `backend/app/repositories/base.py`), converted on arrival: it takes
the gateway's ``AsyncSession`` and every operation awaits. The method surface
and the transaction contract are unchanged, so a rehomed repository inherits
the same helpers it did on the platform side.

**The transaction contract.** Every write here ``flush()``es and never commits:
staging a change makes it visible to the rest of the transaction (so a service
can read back a generated primary key) while leaving the commit boundary with
the service, which is the only layer that knows whether a unit of work is
complete. A repository that committed would break every multi-step service
operation into separately-durable pieces.
"""

from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import class_mapper
from sqlmodel import SQLModel

ModelType = TypeVar("ModelType", bound=SQLModel)
CreateSchemaType = TypeVar("CreateSchemaType", bound=SQLModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=SQLModel)


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Common CRUD operations over one SQLModel table.

    Repositories are pure data access: no business logic, no authorization, and
    no commits (see the module docstring).

    Type Parameters:
        ModelType: The SQLModel table class.
        CreateSchemaType: The schema describing a creation payload.
        UpdateSchemaType: The schema describing an update payload.
    """

    def __init__(self, db: AsyncSession, model_class: type[ModelType]):
        """Bind the repository to a session and the table it serves.

        Args:
            db: The active async session.
            model_class: The SQLModel class for this repository.
        """
        self.db = db
        self.model_class = model_class

    async def get(self, entity_id: Any) -> ModelType | None:
        """Return one entity by primary key, or None."""
        return await self.db.get(self.model_class, entity_id)

    async def get_all(self, *, skip: int = 0, limit: int = 100) -> list[ModelType]:
        """Return a page of entities, ordered by primary key.

        The order is not cosmetic. ``OFFSET``/``LIMIT`` over an unordered query
        is undefined, so two pages of the same unchanged table may repeat a row
        and omit another, and every repository inheriting this pages that way.
        The primary key is the one column every model here has and the one that
        never ties.
        """
        primary_key = class_mapper(self.model_class).primary_key
        result = await self.db.execute(select(self.model_class).order_by(*primary_key).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def create(self, obj_in: CreateSchemaType) -> ModelType:
        """Stage a new entity and return it with its generated values.

        Args:
            obj_in: The creation schema carrying the new entity's data.
        """
        db_obj = self.model_class(**obj_in.model_dump())
        self.db.add(db_obj)
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def update(self, db_obj: ModelType, obj_in: UpdateSchemaType | dict[str, Any]) -> ModelType:
        """Stage an update to an existing entity.

        Only fields the caller actually set are applied, so a partial update
        cannot silently reset a column to its schema default.

        Args:
            db_obj: The entity to update.
            obj_in: An update schema, or a plain mapping of new values.
        """
        update_data = obj_in if isinstance(obj_in, dict) else obj_in.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            setattr(db_obj, field, value)

        self.db.add(db_obj)
        await self.db.flush()
        await self.db.refresh(db_obj)
        return db_obj

    async def delete(self, db_obj: ModelType) -> None:
        """Stage a deletion."""
        await self.db.delete(db_obj)
        await self.db.flush()

    async def count(self) -> int:
        """Count the table's rows without loading them."""
        result = await self.db.execute(select(func.count()).select_from(self.model_class))
        return result.scalar_one()


__all__ = ["BaseRepository"]
