"""Data access for the personal access tokens a person holds."""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import PersonalAccessToken
from gateway.repositories.base_repository import BaseRepository

_DELETE_CHUNK = 1000


class PersonalAccessTokenRepository(BaseRepository[PersonalAccessToken, Never, Never]):
    """Query and stage personal access tokens. Writes flush and never commit."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, PersonalAccessToken)

    async def add(
        self,
        *,
        user_id: uuid.UUID,
        name: str,
        token_hash: str,
        token_suffix: str,
        created_at: datetime,
        expires_at: datetime,
        organization_id: uuid.UUID | None,
        client_id: str | None = None,
        access_expires_at: datetime | None = None,
        refresh_token_hash: str | None = None,
    ) -> PersonalAccessToken:
        """Stage a token for an account."""
        token = PersonalAccessToken(
            user_id=user_id,
            name=name,
            token_hash=token_hash,
            token_suffix=token_suffix,
            created_at=created_at,
            expires_at=expires_at,
            organization_id=organization_id,
            client_id=client_id,
            access_expires_at=access_expires_at,
            refresh_token_hash=refresh_token_hash,
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

    async def delete_expired_for_user(self, user_id: uuid.UUID, now: datetime) -> None:
        """Stage the removal of the account's tokens that have ended."""
        await self.db.execute(
            delete(PersonalAccessToken).where(
                col(PersonalAccessToken.user_id) == user_id, col(PersonalAccessToken.expires_at) <= now
            )
        )

    async def delete_all_for_users(self, user_ids: Collection[uuid.UUID]) -> None:
        """Stage the removal of every token the accounts in ``user_ids`` hold, in chunks under the bind limit."""
        ids = list(user_ids)
        for start in range(0, len(ids), _DELETE_CHUNK):
            chunk = ids[start : start + _DELETE_CHUNK]
            await self.db.execute(delete(PersonalAccessToken).where(col(PersonalAccessToken.user_id).in_(chunk)))

    async def delete_all(self) -> None:
        """Stage the removal of every token on the deployment."""
        await self.db.execute(delete(PersonalAccessToken))

    async def delete_by_hash(self, token_hash: str) -> None:
        """Stage the removal of the token, or the sign-in whose refresh token, is stored under this hash."""
        await self.db.execute(
            delete(PersonalAccessToken).where(
                (col(PersonalAccessToken.token_hash) == token_hash)
                | (col(PersonalAccessToken.refresh_token_hash) == token_hash)
            )
        )

    async def find_by_refresh_hash_locked(self, refresh_token_hash: str) -> PersonalAccessToken | None:
        """The sign-in this refresh token renews, locked so one renewal wins (PostgreSQL only)."""
        result = await self.db.execute(
            select(PersonalAccessToken)
            .where(col(PersonalAccessToken.refresh_token_hash) == refresh_token_hash)
            .with_for_update()
        )
        return result.scalar_one_or_none()

    async def delete_by_previous_refresh_hash(self, refresh_token_hash: str) -> bool:
        """Stage the end of the sign-in this rotated-out refresh token belonged to; whether there was one."""
        result = await self.db.execute(
            delete(PersonalAccessToken).where(
                col(PersonalAccessToken.previous_refresh_token_hash) == refresh_token_hash
            )
        )
        return bool(getattr(result, "rowcount", 0))

    async def rotate(
        self,
        token_id: uuid.UUID,
        *,
        token_hash: str,
        token_suffix: str,
        access_expires_at: datetime,
        refresh_token_hash: str,
        previous_refresh_token_hash: str,
    ) -> None:
        """Stage a sign-in's new access and refresh tokens in place of the current ones."""
        await self.db.execute(
            update(PersonalAccessToken)
            .where(col(PersonalAccessToken.id) == token_id)
            .values(
                token_hash=token_hash,
                token_suffix=token_suffix,
                access_expires_at=access_expires_at,
                refresh_token_hash=refresh_token_hash,
                previous_refresh_token_hash=previous_refresh_token_hash,
            )
        )

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
