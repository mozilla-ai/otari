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
        email: str | None = None,
        is_active: bool = True,
        is_superuser: bool = False,
    ) -> User:
        """Stage a local identity, claimable later.

        A standalone operator (and, after M4's backfill, every re-parented
        gateway user) is an operator-defined label rather than a sign-in
        address, so the row is stored with no email by default; the nullable
        column tolerates that where a create schema requiring an address would
        not. An identity an admin adds by address carries it from the start, as
        the handle the claim flow will match on, but it is unverified and grants
        nothing until that flow exists.

        ``is_active`` is a parameter rather than a constant because the M5
        in-place upgrade needs it: the reconciliation spec maps a soft-deleted
        gateway user onto a deactivated identity so its history stays
        resolvable, and the platform's own backfill passes
        ``is_active=row.deleted_at is None`` for exactly that. A helper that can
        only produce active rows cannot carry out that migration.

        ``default_organization_id`` is stamped with the same organization, and
        that is a **deliberate divergence**, not parity. The platform stamps it
        on its signup paths but not in its own ``create_local_identity``, so its
        re-parenting backfill leaves it NULL. Stamping is the safer default
        here: the hosted edition resolves an identity's offered-credit owner
        through that column and reads NULL as nobody, so an unstamped identity
        silently forfeits the anchor the column exists to hold, and nothing in
        this edition reads it to notice. The cost is that the two editions
        disagree on this column for identically-shaped rows, which belongs in
        the reconciliation ledger rather than being found during a cutover.
        """
        user = User(
            email=email,
            full_name=full_name,
            is_active=is_active,
            is_superuser=is_superuser,
            active_organization_id=active_organization_id,
            default_organization_id=active_organization_id,
        )
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def get_by_active_organization(self, organization_id: uuid.UUID) -> list[User]:
        """Return every identity currently pointed at this organization.

        The organization-delete path reads this: ``active_organization_id`` is
        NOT NULL with no delete rule, so those rows have to be repointed before
        the organization can go.
        """
        result = await self.db.execute(select(User).where(col(User.active_organization_id) == organization_id))
        return list(result.scalars().all())

    async def set_active_organization(self, user: User, organization_id: uuid.UUID) -> User:
        """Stage a change of the identity's active organization."""
        user.active_organization_id = organization_id
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user


__all__ = ["UserRepository", "user_alphabetical_order"]
