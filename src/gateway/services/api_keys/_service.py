import uuid
from collections.abc import Sequence

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.api_keys_exceptions import ApiKeyNotFoundError
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.services.api_keys._listener import ApiKeyDeletionListener


class ApiKeyService:
    """Look up and revoke API keys.

    Lookups and locks run in the caller's Unit of Work block. Deletion opens its own
    block and needs a listener; a service built without one is for reads and locks only.
    """

    def __init__(
        self, uow: UnitOfWork, keys: ApiKeyRepository, *, deletion_listener: ApiKeyDeletionListener | None = None
    ) -> None:
        self._uow = uow
        self._keys = keys
        self._deletion_listener = deletion_listener

    async def get_workspace_id_for_key(self, key_id: str) -> uuid.UUID | None:
        """Return the ID of the workspace that owns a key, or None when no key has that ID."""
        return await self._keys.get_workspace_id_for_key(key_id)

    async def get_key_ids_in_workspaces(self, workspace_ids: Sequence[uuid.UUID]) -> list[str]:
        """Return the IDs of the keys in these workspaces, in no particular order."""
        return await self._keys.get_key_ids_in_workspaces(workspace_ids)

    async def lock(self, key_id: str) -> bool:
        """Hold a key's row lock in the caller's transaction, if the key still exists."""
        return await self._keys.lock(key_id)

    async def delete(self, *, key_id: str, organization_id: uuid.UUID, owner_user_id: str | None = None) -> None:
        """Revoke a visible key and remove dependent state in one transaction.

        Ceiling creation holds the same key lock until it commits. If it commits first,
        the listener sweeps the new ceiling; if deletion wins, the creator finds no key.
        """
        if self._deletion_listener is None:
            msg = "API-key deletion requires a deletion listener"
            raise RuntimeError(msg)
        async with self._uow:
            key = await self._keys.lock_in_organization(key_id, organization_id, owner_user_id=owner_user_id)
            if key is None:
                raise ApiKeyNotFoundError(key_id)
            await self._deletion_listener.key_deleted(key_id)
            await self._keys.delete(key)
