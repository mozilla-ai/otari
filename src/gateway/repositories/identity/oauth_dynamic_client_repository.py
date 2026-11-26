"""Data access for OAuth clients that registered themselves."""

from datetime import datetime
from typing import Never

from sqlalchemy import delete, update
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.identity import OAuthDynamicClient
from gateway.repositories.base_repository import BaseRepository


class OAuthDynamicClientRepository(BaseRepository[OAuthDynamicClient, Never, Never]):
    """Query and stage self-registered clients. Writes flush and never commit."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, OAuthDynamicClient)

    async def add(self, *, client_id: str, client_name: str, redirect_uris: list[str], created_at: datetime) -> None:
        """Stage a registration."""
        self.db.add(
            OAuthDynamicClient(
                client_id=client_id, client_name=client_name, redirect_uris=redirect_uris, created_at=created_at
            )
        )
        await self.db.flush()

    async def mark_authorized(self, client_id: str, now: datetime) -> None:
        """Stage the first approval of a client, which keeps it from being pruned."""
        await self.db.execute(
            update(OAuthDynamicClient)
            .where(col(OAuthDynamicClient.client_id) == client_id, col(OAuthDynamicClient.authorized_at).is_(None))
            .values(authorized_at=now)
        )

    async def prune_unauthorized(self, before: datetime) -> None:
        """Stage the removal of every registration nobody approved since it was made before ``before``."""
        await self.db.execute(
            delete(OAuthDynamicClient).where(
                col(OAuthDynamicClient.authorized_at).is_(None), col(OAuthDynamicClient.created_at) < before
            )
        )
