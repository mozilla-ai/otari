import uuid
from collections.abc import Sequence
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKey
from gateway.repositories.base_repository import BaseRepository


class ApiKeyRepository(BaseRepository[APIKey, Never, Never]):
    """Query API keys in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, APIKey)

    async def get_workspace_id_for_key(self, key_id: str) -> uuid.UUID | None:
        """Return the ID of the workspace that owns a key, or None when no key has that ID."""
        result = await self.db.execute(select(APIKey.workspace_id).where(APIKey.id == key_id))
        return result.scalar_one_or_none()

    async def get_key_ids_in_workspaces(self, workspace_ids: Sequence[uuid.UUID]) -> list[str]:
        """Return the IDs of the keys in these workspaces, in no particular order."""
        result = await self.db.execute(select(APIKey.id).where(APIKey.workspace_id.in_(workspace_ids)))
        return list(result.scalars().all())
