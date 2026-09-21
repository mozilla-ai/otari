"""Persistence for atomic request-plane user revocation."""

import uuid
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.api_keys import APIKey
from gateway.models.users import User
from gateway.repositories.users_repository import in_organization


class AttributionUserRepository:
    def __init__(self, uow: UnitOfWork) -> None:
        self.uow = uow

    @property
    def db(self) -> AsyncSession:
        return session_for(self.uow)

    async def lock_active(self, user_id: str, organization_id: uuid.UUID | None) -> User | None:
        statement = select(User).where(User.user_id == user_id, User.deleted_at.is_(None)).with_for_update()
        if organization_id is not None:
            statement = statement.where(in_organization(organization_id))
        return (await self.db.execute(statement)).scalar_one_or_none()

    async def soft_delete(self, user: User, now: datetime) -> None:
        await self.db.execute(
            update(APIKey)
            .where(APIKey.user_id == user.user_id)
            .values(is_active=False)
            .execution_options(synchronize_session=False)
        )
        user.deleted_at = now
        await self.db.flush()
