"""``sandbox_credentials`` rows: the hosted sandbox provider keys stored through the dashboard."""

from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tools import SandboxCredential
from gateway.repositories.base_repository import BaseRepository


class SandboxCredentialRepository(BaseRepository[SandboxCredential, BaseModel, BaseModel]):
    """One stored key per sandbox provider. Writes take ciphertext: the plaintext key never reaches a column."""

    def __init__(self, db: AsyncSession | UnitOfWork):
        super().__init__(db, SandboxCredential)

    async def get_committed(self, provider: str) -> SandboxCredential | None:
        """The provider's row as committed, even one this session already holds.

        ``populate_existing`` for the reason ``SearchToolRepository.list_committed`` gives.
        """
        stmt = (
            select(SandboxCredential)
            .where(SandboxCredential.provider == provider)
            .execution_options(populate_existing=True)
        )
        return (await self.db.execute(stmt)).scalars().first()

    async def put(self, provider: str, *, encrypted_api_key: str, last4: str) -> SandboxCredential:
        """Store the provider's key, replacing the one it had."""
        row = await self.get(provider)
        if row is None:
            row = SandboxCredential(provider=provider)
            self.db.add(row)
        row.encrypted_api_key = encrypted_api_key
        row.last4 = last4
        row.updated_at = datetime.now(UTC)
        await self.db.flush()
        return row

    async def remove(self, provider: str) -> bool:
        """Delete the provider's stored key; whether there was one."""
        row = await self.get(provider)
        if row is None:
            return False
        await self.delete(row)
        return True
