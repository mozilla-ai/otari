import uuid
from collections.abc import Collection, Sequence
from typing import Never

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import APIKey
from gateway.repositories.base_repository import BaseRepository
from gateway.repositories.users_repository import get_or_create_attribution_user, get_or_create_default_user


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

    async def names_defaulting_end_users_to(self, budget_id: str) -> list[str]:
        """Name each key that starts its end users on ``budget_id`` by default, by its name or else its id."""
        result = await self.db.execute(
            select(APIKey.key_name, APIKey.id).where(APIKey.end_user_budget_id == budget_id).order_by(APIKey.id)
        )
        return [name or key_id for name, key_id in result.all()]

    async def remove_end_user_budget(self, budget_id: str) -> None:
        """Take ``budget_id`` off every key's ``end_user_budget_ids``.

        Filtered here rather than in SQL, which has no portable JSON containment
        test; only service keys carry a list, and there are few of them.
        """
        result = await self.db.execute(select(APIKey).where(APIKey.end_user_budget_ids.is_not(None)))
        for key in result.scalars():
            if key.end_user_budget_ids and budget_id in key.end_user_budget_ids:
                key.end_user_budget_ids = [listed for listed in key.end_user_budget_ids if listed != budget_id]

    async def get_by_config_name(self, config_name: str) -> APIKey | None:
        """Return the key config.yml declares under this name, or None."""
        result = await self.db.execute(select(APIKey).where(APIKey.config_name == config_name))
        return result.scalar_one_or_none()

    async def get_by_hash(self, key_hash: str) -> APIKey | None:
        """Return the key whose secret hashes to ``key_hash``, or None."""
        result = await self.db.execute(select(APIKey).where(APIKey.key_hash == key_hash))
        return result.scalar_one_or_none()

    async def add_declared_if_absent(
        self,
        *,
        config_name: str,
        workspace_id: uuid.UUID,
        user_id: str,
        key_hash: str,
        key_prefix: str,
        key_suffix: str,
    ) -> bool:
        """Stage a declared key, returning False when a row already holds its name or its secret.

        The insert runs in a SAVEPOINT, so losing that race to another replica rolls back this row alone.
        """
        try:
            async with self.db.begin_nested():
                self.db.add(
                    APIKey(
                        id=str(uuid.uuid4()),
                        config_name=config_name,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        key_hash=key_hash,
                        key_prefix=key_prefix,
                        key_suffix=key_suffix,
                        key_name=config_name,
                        metadata_={},
                    )
                )
        except IntegrityError:
            return False
        return True

    async def owner_user_id(self, user_id: str | None) -> str:
        """Return the id of the user a declared key bills to, creating the user, or reviving it, when needed.

        No id names the shared ``default`` user, as a key minted without an owner does.
        """
        if user_id is None:
            return (await get_or_create_default_user(self.db)).user_id
        return (await get_or_create_attribution_user(self.db, user_id=user_id, alias=f"User {user_id}")).user_id

    async def release_config_names(self, keep: Collection[str]) -> list[str]:
        """Clear the config name of every key config.yml no longer declares, returning the names cleared."""
        result = await self.db.execute(
            select(APIKey).where(APIKey.config_name.is_not(None), APIKey.config_name.not_in(list(keep)))
        )
        released: list[str] = []
        for key in result.scalars():
            released.append(str(key.config_name))
            key.config_name = None
        await self.db.flush()
        return released
