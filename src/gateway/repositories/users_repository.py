from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
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


async def _revive(user: User) -> User:
    """Clear a soft delete so the shared default owner is usable again."""
    if user.deleted_at is not None:
        user.deleted_at = None
    return user


async def get_or_create_default_user(db: AsyncSession) -> User:
    """Return the shared ``default`` user, creating (or reviving) it if needed.

    The caller still owns the final commit. The insert goes through a SAVEPOINT so
    that losing a race to a concurrent creator rolls back only this row, not
    whatever the caller has already staged: ``user_id`` is the primary key, so
    without this the loser's commit would raise and surface as a 500 for a request
    that should simply have reused the row the winner just created.
    """
    existing = (await db.execute(select(User).where(User.user_id == DEFAULT_USER_ID))).scalar_one_or_none()
    if existing is not None:
        return await _revive(existing)

    user = User(user_id=DEFAULT_USER_ID, alias="Default")
    try:
        async with db.begin_nested():
            db.add(user)
        return user
    except IntegrityError:
        # Someone else inserted it between our select and our flush; adopt theirs.
        winner = (await db.execute(select(User).where(User.user_id == DEFAULT_USER_ID))).scalar_one_or_none()
        if winner is None:
            raise
        return await _revive(winner)
