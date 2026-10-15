import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Never

from sqlalchemy import delete, or_, select
from sqlalchemy.exc import IntegrityError

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

    async def add_child(
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
        """Stage a key ``parent`` mints for its end user ``user_id``, in the parent's workspace."""
        key = APIKey(
            id=str(uuid.uuid4()),
            workspace_id=parent.workspace_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            key_suffix=key_suffix,
            key_name=key_name,
            user_id=user_id,
            expires_at=expires_at,
            allowed_models=allowed_models,
            capture_agent_telemetry=parent.capture_agent_telemetry,
            parent_key_id=parent.id,
            parent=parent,
            tags=tags,
            metadata_={},
        )
        self.db.add(key)
        await self.db.flush()
        return key

    async def list_children(
        self,
        parent_id: str,
        *,
        user_id: str | None,
        active_at: datetime | None,
        skip: int,
        limit: int,
    ) -> list[APIKey]:
        """Return a page of the keys ``parent_id`` minted, oldest first.

        ``user_id`` narrows them to one end user's; ``active_at`` to those active and unexpired then.
        """
        statement = select(APIKey).where(APIKey.parent_key_id == parent_id)
        if user_id is not None:
            statement = statement.where(APIKey.user_id == user_id)
        if active_at is not None:
            statement = statement.where(
                APIKey.is_active.is_(True), or_(APIKey.expires_at.is_(None), APIKey.expires_at > active_at)
            )
        result = await self.db.execute(statement.order_by(APIKey.created_at, APIKey.id).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def deactivate_child(self, parent_id: str, key_id: str) -> APIKey | None:
        """Deactivate the key ``parent_id`` minted under this id and return it, or None when there is none."""
        result = await self.db.execute(select(APIKey).where(APIKey.id == key_id, APIKey.parent_key_id == parent_id))
        key = result.scalar_one_or_none()
        if key is not None:
            key.is_active = False
            await self.db.flush()
        return key

    async def delete_children_expired_before(self, cutoff: datetime) -> int:
        """Delete every minted key that expired before ``cutoff``, returning how many went."""
        result = await self.db.execute(
            delete(APIKey)
            .where(APIKey.parent_key_id.is_not(None), APIKey.expires_at < cutoff)
            .execution_options(synchronize_session=False)
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def is_child(self, key_id: str) -> bool:
        """Whether the key with this id was minted by another key."""
        result = await self.db.execute(select(APIKey.parent_key_id).where(APIKey.id == key_id))
        return result.scalar_one_or_none() is not None
