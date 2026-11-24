"""Data access for the personal access tokens a person holds."""

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import delete, select, update
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import PersonalAccessToken
from gateway.repositories.base_repository import BaseRepository


class PersonalAccessTokenRepository(BaseRepository[PersonalAccessToken, Never, Never]):
    """Query and stage personal access tokens. Writes flush and never commit."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, PersonalAccessToken)

    async def add(
        self,
        *,
        user_id: uuid.UUID,
        name: str,
        token_hash: str,
        token_suffix: str,
        created_at: datetime,
        expires_at: datetime | None,
    ) -> PersonalAccessToken:
        """Stage a token for an account."""
        token = PersonalAccessToken(
            user_id=user_id,
            name=name,
            token_hash=token_hash,
            token_suffix=token_suffix,
            created_at=created_at,
            expires_at=expires_at,
        )
        self.db.add(token)
        await self.db.flush()
        return token

    async def list_for_user(self, user_id: uuid.UUID) -> Sequence[PersonalAccessToken]:
        """Every token one account holds, newest first."""
        result = await self.db.execute(
            select(PersonalAccessToken)
            .where(col(PersonalAccessToken.user_id) == user_id)
            .order_by(col(PersonalAccessToken.created_at).desc())
        )
        return result.scalars().all()

    async def delete_for_user(self, user_id: uuid.UUID, token_id: uuid.UUID) -> bool:
        """Stage the removal of one of an account's tokens; whether it held one by that ID."""
        result = await self.db.execute(
            delete(PersonalAccessToken).where(
                col(PersonalAccessToken.id) == token_id, col(PersonalAccessToken.user_id) == user_id
            )
        )
        return bool(getattr(result, "rowcount", 0))

    async def find_by_hash(self, token_hash: str) -> PersonalAccessToken | None:
        """The token stored under this hash."""
        result = await self.db.execute(
            select(PersonalAccessToken).where(col(PersonalAccessToken.token_hash) == token_hash)
        )
        return result.scalar_one_or_none()

    async def touch(self, token_id: uuid.UUID, now: datetime) -> None:
        """Stage the token's last use."""
        await self.db.execute(
            update(PersonalAccessToken).where(col(PersonalAccessToken.id) == token_id).values(last_used_at=now)
        )
