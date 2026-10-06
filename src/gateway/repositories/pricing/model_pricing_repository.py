"""Data access for the deployment price list, and the organization overrides above it.

Built on the Unit of Work. Flushes, never commits. Every statement is
dialect-neutral, because the chain and the OSS edition run on SQLite as well as
PostgreSQL: a newest-version read is a ``MAX`` subquery rather than
``DISTINCT ON``.
"""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Any, Never, cast

from sqlalchemy import ColumnElement, CursorResult, and_, delete, func, select

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.pricing import ModelPricing, OrganizationModelPricing
from gateway.repositories.base_repository import BaseRepository

# A key list long enough to exceed SQLite's default limit on bind parameters in
# one statement is a realistic page of one provider's models.
_KEY_CHUNK = 500


class ModelPricingRepository(BaseRepository[ModelPricing, Never, Never]):
    """Repository for `model_pricing` rows, and the overrides a sweep takes with them.

    A version is written by the service, which owns the inherit-forward rule a
    new version follows.
    """

    def __init__(self, uow: UnitOfWork):
        super().__init__(uow, ModelPricing)

    async def keys_with_a_price(self, model_keys: Collection[str]) -> set[str]:
        """The keys among ``model_keys`` that have at least one stored version."""
        keys = sorted(set(model_keys))
        priced: set[str] = set()
        for start in range(0, len(keys), _KEY_CHUNK):
            chunk = keys[start : start + _KEY_CHUNK]
            result = await self.db.execute(
                select(ModelPricing.model_key).where(ModelPricing.model_key.in_(chunk)).distinct()
            )
            priced |= set(result.scalars().all())
        return priced

    async def latest_versions(
        self, model_keys: Collection[str], *, as_of: datetime | None = None
    ) -> dict[str, ModelPricing]:
        """The newest stored version of each key that has one.

        ``as_of`` narrows the answer to versions already in effect; ``None``
        takes the newest version whatever its date. One row per key, chosen in
        SQL, because ``model_pricing`` is a version series.
        """
        keys = sorted(set(model_keys))
        latest: dict[str, ModelPricing] = {}
        for start in range(0, len(keys), _KEY_CHUNK):
            chunk = keys[start : start + _KEY_CHUNK]
            conditions: list[ColumnElement[bool]] = [ModelPricing.model_key.in_(chunk)]
            if as_of is not None:
                conditions.append(ModelPricing.effective_at <= as_of)
            newest = (
                select(
                    ModelPricing.model_key.label("model_key"),
                    func.max(ModelPricing.effective_at).label("effective_at"),
                )
                .where(*conditions)
                .group_by(ModelPricing.model_key)
                .subquery()
            )
            result = await self.db.execute(
                select(ModelPricing).join(
                    newest,
                    and_(
                        ModelPricing.model_key == newest.c.model_key,
                        ModelPricing.effective_at == newest.c.effective_at,
                    ),
                )
            )
            for row in result.scalars():
                latest[row.model_key] = row
        return latest

    def add_all(self, rows: Sequence[ModelPricing]) -> None:
        """Stage new versions."""
        if rows:
            self.db.add_all(rows)

    async def flush(self) -> None:
        """Flush staged versions so a read in the same block sees them."""
        await self.db.flush()

    async def all_keys(self) -> list[tuple[str, int]]:
        """Every key on the list, with how many versions each holds.

        The whole table: one row per rate change per model, which stays in the
        low thousands even on a deployment that has re-imported a catalog for
        years.
        """
        result = await self.db.execute(
            select(ModelPricing.model_key, func.count())
            .group_by(ModelPricing.model_key)
            .order_by(ModelPricing.model_key)
        )
        return [(model_key, versions) for model_key, versions in result.all()]

    async def overrides_for_keys(self, model_keys: Sequence[str]) -> list[tuple[uuid.UUID, uuid.UUID, str]]:
        """Every organization override above ``model_keys``, as ``(id, organization_id, model_key)``.

        Bounded by the overrides tenants hold on the keys given, which is what
        a caller then decides about one by one.
        """
        found: list[tuple[uuid.UUID, uuid.UUID, str]] = []
        for start in range(0, len(model_keys), _KEY_CHUNK):
            chunk = list(model_keys[start : start + _KEY_CHUNK])
            result = await self.db.execute(
                select(
                    OrganizationModelPricing.id,
                    OrganizationModelPricing.organization_id,
                    OrganizationModelPricing.model_key,
                ).where(OrganizationModelPricing.model_key.in_(chunk))
            )
            found.extend((row_id, organization_id, key) for row_id, organization_id, key in result.all())
        return found

    async def delete_keys(self, model_keys: Sequence[str], override_ids: Sequence[uuid.UUID]) -> tuple[int, int]:
        """Delete every version of each key, and the overrides named by id.

        The overrides go first: an override is a commitment over a price-list
        entry, so a moment where the entry is gone and the override is not
        would price a model the deployment no longer lists. Returns the rows
        removed from each table.
        """
        prices = 0
        overrides = 0
        for start in range(0, len(override_ids), _KEY_CHUNK):
            ids = list(override_ids[start : start + _KEY_CHUNK])
            deleted_overrides = cast(
                CursorResult[Any],
                await self.db.execute(delete(OrganizationModelPricing).where(OrganizationModelPricing.id.in_(ids))),
            )
            overrides += deleted_overrides.rowcount
        for start in range(0, len(model_keys), _KEY_CHUNK):
            keys = list(model_keys[start : start + _KEY_CHUNK])
            deleted_prices = cast(
                CursorResult[Any], await self.db.execute(delete(ModelPricing).where(ModelPricing.model_key.in_(keys)))
            )
            prices += deleted_prices.rowcount
        await self.db.flush()
        return prices, overrides
