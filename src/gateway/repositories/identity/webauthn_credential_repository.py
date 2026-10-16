"""Data access for the passkeys an identity has registered."""

import uuid
from typing import Never

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import WebAuthnCredential
from gateway.repositories.base_repository import BaseRepository


class WebAuthnCredentialRepository(BaseRepository[WebAuthnCredential, Never, Never]):
    """Query and stage passkeys. Writes flush and never commit."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, WebAuthnCredential)

    async def delete_for_user(self, user_id: uuid.UUID) -> None:
        """Stage the removal of every passkey one identity holds, on every relying party."""
        await self.db.execute(delete(WebAuthnCredential).where(col(WebAuthnCredential.user_id) == user_id))
