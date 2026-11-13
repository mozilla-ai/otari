"""Atomic delivery of OAuth app grants across workers."""

import uuid
from datetime import datetime
from typing import Never

from sqlalchemy import delete, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import OAuthAppGrant
from gateway.repositories.base_repository import BaseRepository


class OAuthAppGrantRepository(BaseRepository[OAuthAppGrant, Never, Never]):
    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, OAuthAppGrant)

    async def issue(
        self,
        *,
        code_hash: str,
        client_id: str,
        redirect_uri: str,
        challenge: str,
        user_id: uuid.UUID,
        workspace_id: uuid.UUID,
        scopes: list[str],
        expires_at: datetime,
    ) -> None:
        """Stage an approval without minting or retaining a credential."""
        self.db.add(
            OAuthAppGrant(
                code_hash=code_hash,
                client_id=client_id,
                redirect_uri=redirect_uri,
                challenge=challenge,
                user_id=user_id,
                workspace_id=workspace_id,
                scopes=scopes,
                expires_at=expires_at,
            )
        )
        await self.db.flush()

    async def prune_expired(self, now: datetime) -> None:
        """Remove stale approvals within the successful authorization transaction."""
        await self.db.execute(delete(OAuthAppGrant).where(OAuthAppGrant.expires_at <= now))

    async def find_bound(
        self, *, code_hash: str, client_id: str, redirect_uri: str, challenge: str, now: datetime
    ) -> OAuthAppGrant | None:
        """Read the approved scope before the service takes identity and workspace locks."""
        result = await self.db.execute(
            select(OAuthAppGrant).where(
                OAuthAppGrant.code_hash == code_hash,
                OAuthAppGrant.client_id == client_id,
                OAuthAppGrant.redirect_uri == redirect_uri,
                OAuthAppGrant.challenge == challenge,
                OAuthAppGrant.expires_at > now,
            )
        )
        return result.scalar_one_or_none()

    async def consume(self, code_hash: str, now: datetime) -> bool:
        """Consume only once, after authority locks, without reversing parent/cascade lock order."""
        result = await self.db.execute(
            delete(OAuthAppGrant)
            .where(OAuthAppGrant.code_hash == code_hash, OAuthAppGrant.expires_at > now)
            .returning(OAuthAppGrant.code_hash)
        )
        return result.scalar_one_or_none() is not None
