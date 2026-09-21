import uuid
from collections.abc import Sequence

from gateway.repositories.api_keys import ApiKeyRepository


class ApiKeyService:
    """Answer questions about API keys and the workspaces that hold them.

    Each method runs in the caller's Unit of Work block and raises ``OutsideUnitOfWorkError`` outside one.
    """

    def __init__(self, keys: ApiKeyRepository) -> None:
        self._keys = keys

    async def get_workspace_id_for_key(self, key_id: str) -> uuid.UUID | None:
        """Return the ID of the workspace that owns a key, or None when no key has that ID."""
        return await self._keys.get_workspace_id_for_key(key_id)

    async def get_key_ids_in_workspaces(self, workspace_ids: Sequence[uuid.UUID]) -> list[str]:
        """Return the IDs of the keys in these workspaces, in no particular order."""
        return await self._keys.get_key_ids_in_workspaces(workspace_ids)
