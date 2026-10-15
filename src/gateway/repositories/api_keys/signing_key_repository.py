import uuid
from typing import Never

from sqlalchemy import delete, func, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.api_keys import EndUserSigningKey
from gateway.repositories.base_repository import BaseRepository


class SigningKeyRepository(BaseRepository[EndUserSigningKey, Never, Never]):
    """Query the public keys service keys sign end-user tokens with, in the open block of a Unit of Work."""

    def __init__(self, uow: UnitOfWork) -> None:
        super().__init__(uow, EndUserSigningKey)

    async def count_for(self, api_key_id: str) -> int:
        """How many signing keys the key with this id holds."""
        result = await self.db.execute(
            select(func.count()).select_from(EndUserSigningKey).where(EndUserSigningKey.api_key_id == api_key_id)
        )
        return int(result.scalar_one())

    async def add(self, *, api_key_id: str, public_key: str, name: str | None) -> EndUserSigningKey:
        """Stage a signing key for ``api_key_id`` under a new kid."""
        key = EndUserSigningKey(id=str(uuid.uuid4()), api_key_id=api_key_id, public_key=public_key, name=name)
        self.db.add(key)
        await self.db.flush()
        return key

    async def list_for(self, api_key_id: str) -> list[EndUserSigningKey]:
        """The signing keys the key with this id holds, oldest first."""
        result = await self.db.execute(
            select(EndUserSigningKey)
            .where(EndUserSigningKey.api_key_id == api_key_id)
            .order_by(EndUserSigningKey.created_at, EndUserSigningKey.id)
        )
        return list(result.scalars().all())

    async def find(self, kid: str) -> EndUserSigningKey | None:
        """The signing key with this kid, whichever key holds it."""
        result = await self.db.execute(select(EndUserSigningKey).where(EndUserSigningKey.id == kid))
        return result.scalar_one_or_none()

    async def delete_for(self, api_key_id: str, kid: str) -> bool:
        """Delete the signing key ``api_key_id`` holds under this kid, saying whether there was one."""
        result = await self.db.execute(
            delete(EndUserSigningKey)
            .where(EndUserSigningKey.id == kid, EndUserSigningKey.api_key_id == api_key_id)
            .execution_options(synchronize_session=False)
        )
        return bool(getattr(result, "rowcount", 0))
