import uuid
from collections.abc import Sequence
from typing import Never

from sqlalchemy import select
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import Workspace
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

    async def lock(self, key_id: str) -> bool:
        """Lock a key until the caller's transaction ends, returning whether it still exists."""
        result = await self.db.execute(select(APIKey.id).where(APIKey.id == key_id).with_for_update())
        return result.scalar_one_or_none() is not None

    async def lock_in_organization(
        self, key_id: str, organization_id: uuid.UUID, *, owner_user_id: str | None = None
    ) -> APIKey | None:
        """Lock a visible key in this organization, optionally restricted to its owner.

        Internal keys are not managed through the key surface. Only the key row is locked,
        so revoking one key does not serialize writes to every key in its workspace.
        """
        statement = (
            select(APIKey)
            .join(Workspace, col(Workspace.id) == APIKey.workspace_id)
            .where(
                APIKey.id == key_id,
                col(Workspace.organization_id) == organization_id,
                APIKey.internal_secret.is_(None),
            )
            .with_for_update(of=APIKey)
            .execution_options(populate_existing=True)
        )
        if owner_user_id is not None:
            statement = statement.where(APIKey.user_id == owner_user_id)
        result = await self.db.execute(statement)
        return result.scalar_one_or_none()
