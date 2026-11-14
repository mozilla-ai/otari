"""Data access for the model metadata an operator or an organization set. Flushes, never commits."""

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel
from sqlalchemy import delete, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.catalog import ModelMetadataOverride
from gateway.repositories.base_repository import BaseRepository


def _owner(organization_id: uuid.UUID | None) -> Any:
    """The rows ``organization_id`` holds, or the deployment's where it is ``None``."""
    return (
        ModelMetadataOverride.organization_id == organization_id
        if organization_id is not None
        else ModelMetadataOverride.organization_id.is_(None)
    )


def _owned(provider: str, model: str, organization_id: uuid.UUID | None) -> Any:
    return (
        (ModelMetadataOverride.provider == provider) & (ModelMetadataOverride.model == model) & _owner(organization_id)
    )


class ModelMetadataOverrideRepository(BaseRepository[ModelMetadataOverride, BaseModel, BaseModel]):
    """Repository for ``model_metadata_overrides`` rows."""

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, ModelMetadataOverride)

    async def list_all(self) -> Sequence[ModelMetadataOverride]:
        """Every row, for the per-worker copy's refresh."""
        return (await self.db.execute(select(ModelMetadataOverride))).scalars().all()

    async def put(
        self, provider: str, model: str, organization_id: uuid.UUID | None, values: Mapping[str, Any]
    ) -> ModelMetadataOverride:
        """Make ``values`` the owner's override of ``provider:model``, replacing any it had."""
        row = (
            await self.db.execute(select(ModelMetadataOverride).where(_owned(provider, model, organization_id)))
        ).scalar_one_or_none()
        if row is None:
            row = ModelMetadataOverride(provider=provider, model=model, organization_id=organization_id)
            self.db.add(row)
        for name, value in values.items():
            setattr(row, name, value)
        await self.db.flush()
        return row

    async def remove_provider(self, provider: str, organization_id: uuid.UUID | None) -> None:
        """Drop every override the owner holds under ``provider``."""
        await self.db.execute(
            delete(ModelMetadataOverride).where((ModelMetadataOverride.provider == provider) & _owner(organization_id))
        )

    async def remove(self, provider: str, model: str, organization_id: uuid.UUID | None) -> bool:
        """Drop the owner's override of ``provider:model``. Whether there was one."""
        result = await self.db.execute(delete(ModelMetadataOverride).where(_owned(provider, model, organization_id)))
        return bool(getattr(result, "rowcount", 0))
