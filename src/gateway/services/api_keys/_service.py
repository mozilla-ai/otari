import uuid
from collections.abc import Sequence

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.schemas.api_keys import KeyIdentity


class ApiKeyService:
    """Answer questions about API keys and the workspaces that hold them.

    The lookups run in the caller's Unit of Work block and raise ``OutsideUnitOfWorkError`` outside one.
    ``identify`` is a business step of its own and opens a block, which joins the caller's where one is open.
    """

    def __init__(self, uow: UnitOfWork, keys: ApiKeyRepository) -> None:
        self._uow = uow
        self._keys = keys

    async def get_workspace_id_for_key(self, key_id: str) -> uuid.UUID | None:
        """Return the ID of the workspace that owns a key, or None when no key has that ID."""
        return await self._keys.get_workspace_id_for_key(key_id)

    async def get_key_ids_in_workspaces(self, workspace_ids: Sequence[uuid.UUID]) -> list[str]:
        """Return the IDs of the keys in these workspaces, in no particular order."""
        return await self._keys.get_key_ids_in_workspaces(workspace_ids)

    async def identify(self, key_id: str) -> KeyIdentity | None:
        """Say who owns a verified key and which workspace and organization it belongs to.

        ``None`` when the key's owner is deleted or blocked, or the key is gone: a
        key whose owner may not use it identifies nobody. Blocking a user is an
        operator's kill switch, and the budget gate already refuses every request
        such a key makes, so a service asking whose the key is gets the same answer.
        A key with no owner is not refused, because there is no owner to block.
        """
        async with self._uow:
            holding = await self._keys.get_holding(key_id)
        if holding is None:
            return None
        if holding.user_id is not None:
            owner = holding.owner
            if owner is None or owner.deleted_at is not None or owner.blocked:
                return None
        return KeyIdentity(
            api_key_id=key_id,
            user_id=holding.user_id,
            workspace_id=str(holding.workspace_id),
            organization_id=None if holding.organization_id is None else str(holding.organization_id),
        )
