"""Data access for the reconciled control plane's identities."""

import uuid

from sqlalchemy import func, nulls_last, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlmodel import col

from gateway.models.tenancy import User, UserBase, UserCreate
from gateway.repositories.base_repository import BaseRepository


def user_alphabetical_order() -> ColumnElement[str]:
    """Case-insensitive sort key for queries joined to ``User``.

    Sorts by full name, falling back to the email address for identities
    without one (or with a whitespace-only one), so a member roster reads
    top-to-bottom the way a directory would. A local identity has neither, and
    sorts last.

    ``nulls_last()`` is what makes that last sentence true on both engines:
    PostgreSQL puts NULLs last in an ascending sort and SQLite puts them first,
    so an email-less operator would otherwise head the roster on the engine the
    OSS base ships by default.
    """
    return nulls_last(func.lower(func.coalesce(func.nullif(func.trim(col(User.full_name)), ""), col(User.email))))


class UserRepository(BaseRepository[User, UserCreate, UserBase]):
    """Repository for identity rows."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, User)

    async def get_by_email(self, email: str) -> User | None:
        """Return the identity with this email address, or None."""
        result = await self.db.execute(select(User).where(col(User.email) == email))
        return result.scalars().first()

    async def create_local_identity(
        self,
        *,
        full_name: str | None,
        active_organization_id: uuid.UUID,
        is_superuser: bool = False,
    ) -> User:
        """Stage an email-less local identity.

        A standalone operator (and, after M4's backfill, every re-parented
        gateway user) is an operator-defined label rather than a sign-in
        address, so the row is stored with no email. The nullable column
        tolerates that where a create schema requiring an address would not.
        """
        user = User(
            email=None,
            full_name=full_name,
            is_active=True,
            is_superuser=is_superuser,
            active_organization_id=active_organization_id,
        )
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def set_active_organization(self, user: User, organization_id: uuid.UUID) -> User:
        """Stage a change of the identity's active organization."""
        user.active_organization_id = organization_id
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user


__all__ = ["UserRepository", "user_alphabetical_order"]
