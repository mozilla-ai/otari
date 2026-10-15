import uuid
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime, timedelta

from gateway.core.settings.api_keys import ApiKeyConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKey
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.services.api_keys._declared import apply_declared_key


class ApiKeyService:
    """Answer questions about API keys and the workspaces that hold them.

    Each method runs in the caller's Unit of Work block and raises ``OutsideUnitOfWorkError`` outside one,
    except the sweep, which is a worker job's whole step and opens its own.
    """

    def __init__(self, keys: ApiKeyRepository, uow: UnitOfWork) -> None:
        self._keys = keys
        self._uow = uow

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

    async def apply_declared_key(
        self,
        config_name: str,
        spec: ApiKeyConfig,
        *,
        owner: str,
        declared_names: Collection[str],
        end_user_budget_ids: list[str] | None,
        workspace_id: uuid.UUID,
        fingerprint: Callable[[str], str],
    ) -> str:
        """Create the key config.yml declares under ``config_name``, or write the declared values over it.

        Returns the key's id. ``owner`` is the id of the user it bills to; ``end_user_budget_ids`` is the declared list
        once checked; ``fingerprint`` is the bound key format's, and a new key lands in ``workspace_id``.
        """
        return await apply_declared_key(
            self._keys,
            config_name,
            spec,
            owner=owner,
            declared_names=declared_names,
            end_user_budget_ids=end_user_budget_ids,
            workspace_id=workspace_id,
            fingerprint=fingerprint,
        )

    async def release_undeclared_keys(self, declared_names: Collection[str]) -> list[str]:
        """Stop marking as declared every key config.yml no longer declares, and return their former names.

        The keys themselves stay, with their secrets and settings.
        """
        return await self._keys.release_config_names(declared_names)

    async def add_end_user_key(
        self,
        *,
        parent: APIKey,
        user_id: str,
        key_hash: str,
        key_prefix: str,
        key_suffix: str,
        key_name: str | None,
        expires_at: datetime,
        allowed_models: list[str] | None,
        tags: dict[str, str] | None,
    ) -> APIKey:
        """Stage a key ``parent`` mints for its end user ``user_id``, in ``parent``'s workspace."""
        return await self._keys.add_child(
            parent=parent,
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            key_suffix=key_suffix,
            key_name=key_name,
            expires_at=expires_at,
            allowed_models=allowed_models,
            tags=tags,
        )

    async def list_end_user_keys(
        self, parent_id: str, *, user_id: str | None, include_inactive: bool, skip: int, limit: int
    ) -> list[APIKey]:
        """Return a page of the keys ``parent_id`` minted, oldest first; unless asked, only those that still work."""
        return await self._keys.list_children(
            parent_id,
            user_id=user_id,
            active_at=None if include_inactive else datetime.now(UTC),
            skip=skip,
            limit=limit,
        )

    async def deactivate_end_user_key(self, parent_id: str, key_id: str) -> APIKey | None:
        """Deactivate the key ``parent_id`` minted under ``key_id`` and return it, or None when there is none."""
        return await self._keys.deactivate_child(parent_id, key_id)

    async def is_end_user_key(self, key_id: str) -> bool:
        """Whether the key with this id was minted by a service key for one of its end users."""
        return await self._keys.is_child(key_id)

    async def sweep_expired_end_user_keys(self, retention_sec: int) -> int:
        """Delete the minted keys that expired more than ``retention_sec`` ago, and say how many."""
        async with self._uow:
            return await self._keys.delete_children_expired_before(datetime.now(UTC) - timedelta(seconds=retention_sec))
