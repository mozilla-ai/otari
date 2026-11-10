"""Data access for what model servers say about the models they serve. Flushes, never commits."""

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.catalog import ServedModel
from gateway.repositories.base_repository import BaseRepository


class ServedModelRepository(BaseRepository[ServedModel, BaseModel, BaseModel]):
    """Repository for ``served_models`` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, ServedModel)

    async def list_all(self) -> Sequence[ServedModel]:
        """Every row, for the per-worker copy's refresh."""
        return (await self.db.execute(select(ServedModel))).scalars().all()

    async def replace(
        self,
        provider: str,
        models: Mapping[str, Mapping[str, Any]],
        *,
        organization_id: uuid.UUID | None = None,
        org_provider_key_id: uuid.UUID | None = None,
    ) -> None:
        """Make ``models`` (model id to column values) one owner's whole set, in a SAVEPOINT.

        The owner is an organization's key when ``org_provider_key_id`` is given,
        and otherwise the deployment's provider of that name. Two workers can
        record the same listing at once; the one whose insert loses rolls back
        its own SAVEPOINT alone and keeps the winner's rows, which describe the
        same server.
        """
        owned = (
            ServedModel.org_provider_key_id == org_provider_key_id
            if org_provider_key_id is not None
            else (ServedModel.provider == provider) & ServedModel.org_provider_key_id.is_(None)
        )
        try:
            async with self.db.begin_nested():
                await self.db.execute(delete(ServedModel).where(owned))
                self.db.add_all(
                    ServedModel(
                        provider=provider,
                        model=model,
                        organization_id=organization_id,
                        org_provider_key_id=org_provider_key_id,
                        **values,
                    )
                    for model, values in models.items()
                )
        except IntegrityError:
            pass
