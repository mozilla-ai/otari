from typing import Never

from sqlalchemy import select
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
