"""Data access for the provider endpoints a workspace or a user owns.

Built on the Unit of Work. Flushes, never commits.
"""

import uuid
from collections.abc import Collection, Sequence
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import ProviderEndpoint
from gateway.repositories.base_repository import BaseRepository
from gateway.schemas.providers import ProviderEndpointCreateRequest, ProviderEndpointUpdateRequest


class EndpointNameConflict(Exception):
    """A unique constraint refused a name the owner already uses.

    Raised in place of ``IntegrityError`` because the service that maps it to a
    domain error may not import a database library.
    """


class ProviderEndpointRepository(
    BaseRepository[ProviderEndpoint, ProviderEndpointCreateRequest, ProviderEndpointUpdateRequest]
):
    """Repository for `provider_endpoints` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, ProviderEndpoint)

    async def list_all(self) -> Sequence[ProviderEndpoint]:
        """Every endpoint, for the dispatch cache's refresh."""
        return (await self.db.execute(select(ProviderEndpoint))).scalars().all()

    async def list_page(
        self,
        *,
        workspace_id: uuid.UUID | None,
        user_id: str | None,
        skip: int,
        limit: int,
    ) -> tuple[Sequence[ProviderEndpoint], int]:
        """One page of endpoints, narrowed by owner when given, and the total that matches."""
        conditions = []
        if workspace_id is not None:
            conditions.append(ProviderEndpoint.workspace_id == workspace_id)
        if user_id is not None:
            conditions.append(ProviderEndpoint.user_id == user_id)
        rows = (
            (
                await self.db.execute(
                    select(ProviderEndpoint)
                    .where(*conditions)
                    .order_by(ProviderEndpoint.name, ProviderEndpoint.id)
                    .offset(skip)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        total = (
            await self.db.execute(select(func.count()).select_from(ProviderEndpoint).where(*conditions))
        ).scalar_one()
        return rows, total

    async def insert(self, **fields: Any) -> ProviderEndpoint:
        """Stage a new endpoint.

        Raises:
            EndpointNameConflict: the owner already has an endpoint of that name.
        """
        row = ProviderEndpoint(**fields)
        self.db.add(row)
        await self._flush_or_conflict()
        return row

    async def apply(self, row: ProviderEndpoint, changes: dict[str, Any]) -> ProviderEndpoint:
        """Stage changes to an endpoint.

        Raises:
            EndpointNameConflict: a rename collided with another of the owner's endpoints.
        """
        for field, value in changes.items():
            setattr(row, field, value)
        await self._flush_or_conflict()
        return row

    async def forget_secrets(
        self, organization_ids: Collection[uuid.UUID], workspace_ids: Collection[uuid.UUID]
    ) -> None:
        """Stage clearing the API key and default request fields of every endpoint in these workspaces."""
        if not workspace_ids:
            return
        await self.db.execute(
            update(ProviderEndpoint)
            .where(ProviderEndpoint.workspace_id.in_(list(workspace_ids)))
            .values(encrypted_api_key=None, last4=None, encrypted_default_params=None)
            .execution_options(synchronize_session=False)
        )

    async def _flush_or_conflict(self) -> None:
        try:
            await self.db.flush()
        except IntegrityError:
            raise EndpointNameConflict from None
