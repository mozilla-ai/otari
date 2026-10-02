import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Never

from sqlalchemy import select
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import Workspace
from gateway.models.users import User
from gateway.repositories.base_repository import BaseRepository


@dataclass(frozen=True)
class KeyHolding:
    """Who holds a key and where it sits, as the rows stand.

    ``owner`` is the ``users`` row ``user_id`` names, deleted or not, and ``None``
    when the key names no user or the row is gone. ``organization_id`` is
    ``None`` when the key's workspace resolves to no organization.
    """

    user_id: str | None
    workspace_id: uuid.UUID
    organization_id: uuid.UUID | None
    owner: User | None


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

    async def get_holding(self, key_id: str) -> KeyHolding | None:
        """Return a key's owner and its workspace's organization in one read, or None when no key has that ID."""
        result = await self.db.execute(
            select(APIKey.user_id, APIKey.workspace_id, col(Workspace.organization_id), User)
            .outerjoin(Workspace, col(Workspace.id) == APIKey.workspace_id)
            .outerjoin(User, User.user_id == APIKey.user_id)
            .where(APIKey.id == key_id)
        )
        row = result.one_or_none()
        if row is None:
            return None
        user_id, workspace_id, organization_id, owner = row
        return KeyHolding(user_id=user_id, workspace_id=workspace_id, organization_id=organization_id, owner=owner)
