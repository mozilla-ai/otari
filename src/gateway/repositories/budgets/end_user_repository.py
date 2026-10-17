from collections.abc import Collection
from datetime import datetime
from typing import Never

from sqlalchemy import ColumnElement, func, select, update
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.users import User
from gateway.repositories.base_repository import BaseRepository


class EndUserRepository(BaseRepository[User, Never, Never]):
    """Query and stage the end users a service key bills, in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, User)

    async def find(self, owner_user_id: str, external_id: str) -> User | None:
        """Return the owner's end user named ``external_id``, soft-deleted or not."""
        result = await self.db.execute(
            select(User).where(User.parent_user_id == owner_user_id, User.external_id == external_id)
        )
        return result.scalar_one_or_none()

    async def page(
        self,
        owner_user_id: str,
        *,
        external_id: str | None,
        budget_id: str | None,
        blocked: bool | None,
        skip: int,
        limit: int,
        created_before: datetime | None = None,
    ) -> list[User]:
        """Return a page of the owner's live end users, oldest first, narrowed by id, budget, blocked state and age."""
        result = await self.db.execute(
            select(User)
            .where(
                *_live_end_users(owner_user_id, external_id=external_id, budget_id=budget_id, blocked=blocked),
                *([User.created_at < created_before] if created_before is not None else []),
            )
            .order_by(User.created_at, User.user_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_matching(
        self, owner_user_id: str, *, external_id: str | None, budget_id: str | None, blocked: bool | None
    ) -> int:
        """Count the owner's live end users that ``page`` would list across every page."""
        result = await self.db.execute(
            select(func.count())
            .select_from(User)
            .where(*_live_end_users(owner_user_id, external_id=external_id, budget_id=budget_id, blocked=blocked))
        )
        return int(result.scalar_one())

    async def count_by_budget(self, owner_user_id: str, budget_ids: list[str]) -> dict[str, tuple[int, int]]:
        """Count the owner's live end users on each of ``budget_ids``: (all of them, the blocked ones)."""
        if not budget_ids:
            return {}
        result = await self.db.execute(
            select(User.budget_id, func.count(), func.count().filter(User.blocked.is_(True)))
            .where(
                *_live_end_users(owner_user_id, external_id=None, budget_id=None, blocked=None),
                User.budget_id.in_(budget_ids),
            )
            .group_by(User.budget_id)
        )
        return {str(budget_id): (int(total), int(blocked)) for budget_id, total, blocked in result.all()}

    async def soft_delete(self, owner_user_id: str, user_ids: Collection[str], deleted_at: datetime) -> int:
        """Mark these live end users of the owner deleted, and return how many were.

        An id that is not a live end user of this owner is skipped, so a caller
        can never delete another owner's end user or a user that is not one.
        """
        if not user_ids:
            return 0
        result = await self.db.execute(
            update(User)
            .where(
                User.user_id.in_(list(user_ids)),
                *_live_end_users(owner_user_id, external_id=None, budget_id=None, blocked=None),
            )
            .values(deleted_at=deleted_at)
            .execution_options(synchronize_session=False)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def add(self, user: User) -> bool:
        """Stage a new end user, returning False when a concurrent request created it first.

        The insert runs in a SAVEPOINT, so losing that race rolls back this row alone.
        """
        try:
            async with self.db.begin_nested():
                self.db.add(user)
        except IntegrityError:
            return False
        return True


def _live_end_users(
    owner_user_id: str, *, external_id: str | None, budget_id: str | None, blocked: bool | None
) -> list[ColumnElement[bool]]:
    conditions = [
        User.parent_user_id == owner_user_id,
        User.external_id.is_not(None),
        User.deleted_at.is_(None),
    ]
    if external_id is not None:
        conditions.append(User.external_id == external_id)
    if budget_id is not None:
        conditions.append(User.budget_id == budget_id)
    if blocked is not None:
        conditions.append(User.blocked == blocked)
    return conditions
