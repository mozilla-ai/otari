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

    async def keys_defaulting_end_users_to(self, budget_id: str) -> list[str]:
        """Name the keys that start a new end user on this budget when a request names none."""
        return await self._keys.names_defaulting_end_users_to(budget_id)

    async def forget_end_user_budget(self, budget_id: str) -> None:
        """Take a budget off every key's list of end-user budgets."""
        await self._keys.remove_end_user_budget(budget_id)
