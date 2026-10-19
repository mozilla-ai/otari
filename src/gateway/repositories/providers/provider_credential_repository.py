"""Data access for the deployment providers stored through the dashboard (``provider_credentials``).

Read and removed here by the move onto hosted providers. The runtime overlay
that merges these rows over ``config.providers`` still reads them through
``services/provider_store_service.py``.

Built on the Unit of Work. Flushes, never commits.
"""

from collections.abc import Sequence
from typing import Never

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import ProviderCredential
from gateway.repositories.base_repository import BaseRepository

# Rows an operator added by hand, so a full listing stays small enough never to page.
_ALL_CREDENTIALS = 1000


class ProviderCredentialRepository(BaseRepository[ProviderCredential, Never, Never]):
    """Repository for `provider_credentials` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, ProviderCredential)

    async def list_all(self) -> Sequence[ProviderCredential]:
        """Every stored deployment provider by instance name, bounded."""
        result = await self.db.execute(
            select(ProviderCredential).order_by(ProviderCredential.instance).limit(_ALL_CREDENTIALS)
        )
        return result.scalars().all()

    async def delete_row(self, row: ProviderCredential) -> None:
        await self.db.delete(row)
        await self.db.flush()
