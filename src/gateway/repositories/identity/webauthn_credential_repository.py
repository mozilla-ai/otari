"""Data access for the passkeys an identity has registered."""

import uuid
from typing import Never

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import WebAuthnCredential
from gateway.repositories.base_repository import BaseRepository


class WebAuthnCredentialRepository(BaseRepository[WebAuthnCredential, Never, Never]):
    """Query and stage passkeys. Writes flush and never commit."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WebAuthnCredential)

    async def exists_for_user(self, user_id: uuid.UUID, *, rp_id: str) -> bool:
        """Whether this identity holds a passkey usable under ``rp_id``.

        Scoped to one relying party, because an authenticator will not offer a credential
        registered under a different one.
        """
        result = await self.db.execute(
            select(col(WebAuthnCredential.id))
            .where(col(WebAuthnCredential.user_id) == user_id, col(WebAuthnCredential.rp_id) == rp_id)
            .limit(1)
        )
        return result.first() is not None

    async def delete_for_user(self, user_id: uuid.UUID) -> None:
        """Stage the removal of every passkey one identity holds, on every relying party."""
        await self.db.execute(delete(WebAuthnCredential).where(col(WebAuthnCredential.user_id) == user_id))
