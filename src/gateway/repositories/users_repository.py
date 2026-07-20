from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import User

# The owner a key falls back to when it is created without a user_id (the API's
# convenience path, and the first-run bootstrap key). One shared, visible,
# budgetable user rather than a throwaway per key, so nothing is untracked and the
# operator can cap all such keys with a single budget on this user.
DEFAULT_USER_ID = "default"


async def get_active_user(db: AsyncSession, user_id: str, *, for_update: bool = False) -> User | None:
    """Query for a non-deleted user by user_id."""

    stmt = select(User).where(User.user_id == user_id, User.deleted_at.is_(None))
    if for_update:
        stmt = stmt.with_for_update()
    result = await db.execute(stmt)
    return result.scalar_one_or_none()


async def get_or_create_default_user(db: AsyncSession) -> User:
    """Return the shared ``default`` user, creating (or reviving) it if needed.

    The caller is responsible for committing; the row is added to the session but
    not flushed here, matching how the keys route batches its writes.
    """
    existing = (await db.execute(select(User).where(User.user_id == DEFAULT_USER_ID))).scalar_one_or_none()
    if existing is not None:
        if existing.deleted_at is not None:
            existing.deleted_at = None
        return existing
    user = User(user_id=DEFAULT_USER_ID, alias="Default")
    db.add(user)
    return user
