"""Data access for ``mcp_catalog_featured``. Flushes, never commits."""

import uuid
from typing import Any, Never

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.mcp_catalog import McpCatalogFeatured
from gateway.repositories.base_repository import BaseRepository


class McpCatalogFeaturedRepository(BaseRepository[McpCatalogFeatured, Never, Never]):
    """Repository for the organizations' featured shelves. Every read names its organization."""

    def __init__(self, db: AsyncSession | UnitOfWork) -> None:
        super().__init__(db, McpCatalogFeatured)

    async def list_for_organization(self, organization_id: uuid.UUID) -> list[McpCatalogFeatured]:
        """An organization's shelf, by title."""
        result = await self.db.execute(
            select(McpCatalogFeatured)
            .where(McpCatalogFeatured.organization_id == organization_id)
            .order_by(func.lower(McpCatalogFeatured.title), McpCatalogFeatured.name)
        )
        return list(result.scalars().all())

    async def get_in_organization(
        self, organization_id: uuid.UUID, featured_id: uuid.UUID
    ) -> McpCatalogFeatured | None:
        """One entry of this organization's shelf, or None, also for another organization's entry."""
        result = await self.db.execute(
            select(McpCatalogFeatured).where(
                McpCatalogFeatured.id == featured_id, McpCatalogFeatured.organization_id == organization_id
            )
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, organization_id: uuid.UUID, name: str) -> McpCatalogFeatured | None:
        """This organization's entry for a catalog name, or None."""
        result = await self.db.execute(
            select(McpCatalogFeatured).where(
                McpCatalogFeatured.organization_id == organization_id, McpCatalogFeatured.name == name
            )
        )
        return result.scalar_one_or_none()

    async def count_for_organization(self, organization_id: uuid.UUID) -> int:
        """How many servers an organization features."""
        result = await self.db.execute(
            select(func.count())
            .select_from(McpCatalogFeatured)
            .where(McpCatalogFeatured.organization_id == organization_id)
        )
        return result.scalar_one()

    async def add(self, organization_id: uuid.UUID, name: str, copy: dict[str, Any]) -> McpCatalogFeatured | None:
        """Stage a new entry holding ``copy`` of the catalog server called ``name``, flushed so its id answers.

        None when the organization already features that name, which two requests at once can both miss
        before either inserts. The insert runs in a SAVEPOINT, so the loser's transaction goes on.
        """
        row = McpCatalogFeatured(organization_id=organization_id, name=name, **copy)
        try:
            async with self.db.begin_nested():
                self.db.add(row)
        except IntegrityError:
            return None
        return row
