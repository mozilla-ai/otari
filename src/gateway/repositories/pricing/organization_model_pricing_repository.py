"""Data access for an organization's own rates, and the deployment list beneath them.

Serves the surface that offers models on an organization's provider key, which
needs to say what each offered model currently costs and to move the rates it
seeded itself. Reads only the two lower rungs of
`services.pricing_service.find_model_pricing`'s ladder; the genai-prices
fallback is not a table and is resolved by that module.

Built on the Unit of Work. Flushes, never commits.
"""

import uuid
from collections.abc import Collection, Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.pricing import ModelPricing, OrganizationModelPricing

# The bound `services.pricing_service` chunks its own key lookups at, for the
# same reason: a key list long enough to exceed SQLite's default limit on bind
# parameters in one statement is a realistic page of one provider's models.
_KEY_CHUNK = 500


class OrganizationModelPricingRepository:
    """Repository for `organization_model_pricing` rows, and reads of `model_pricing`."""

    def __init__(self, uow: UnitOfWork):
        self._uow = uow

    @property
    def db(self) -> AsyncSession:
        """The session this repository's operations run on.

        Raises:
            OutsideUnitOfWorkError: no block of the unit of work is open.
        """
        return session_for(self._uow)

    async def applicable_rows(
        self,
        organization_id: uuid.UUID,
        model_keys: Collection[str],
        as_of: datetime,
    ) -> dict[str, OrganizationModelPricing]:
        """The organization's row covering ``as_of`` for each key that has one.

        The same half-open rule `services.pricing_service` resolves with:
        ``effective_from`` inclusive, ``effective_to`` exclusive, and within one
        key the newest applicable period wins. Returned as the stored rows rather
        than as resolved rates, because the caller edits them in place and reads
        their ``origin`` to tell a rate it seeded from one an admin set.
        """
        keys = sorted(set(model_keys))
        applicable: dict[str, OrganizationModelPricing] = {}
        for start in range(0, len(keys), _KEY_CHUNK):
            chunk = keys[start : start + _KEY_CHUNK]
            result = await self.db.execute(
                select(OrganizationModelPricing)
                .where(
                    col(OrganizationModelPricing.organization_id) == organization_id,
                    col(OrganizationModelPricing.model_key).in_(chunk),
                    col(OrganizationModelPricing.effective_from) <= as_of,
                )
                .order_by(
                    col(OrganizationModelPricing.model_key),
                    col(OrganizationModelPricing.effective_from),
                )
            )
            for row in result.scalars():
                if row.effective_to is not None and row.effective_to <= as_of:
                    continue
                # Ordered oldest-first, so the last applicable period seen for a
                # key is the newest one, which is the one that applies.
                applicable[row.model_key] = row
        return applicable

    async def deployment_rows(self, model_keys: Collection[str], as_of: datetime) -> dict[str, ModelPricing]:
        """The deployment price list's current version for each key that has one.

        ``model_pricing`` is a version series keyed ``(model_key, effective_at)``,
        so the row that applies is the newest version at or before ``as_of``.
        """
        keys = sorted(set(model_keys))
        current: dict[str, ModelPricing] = {}
        for start in range(0, len(keys), _KEY_CHUNK):
            chunk = keys[start : start + _KEY_CHUNK]
            result = await self.db.execute(
                select(ModelPricing)
                .where(
                    col(ModelPricing.model_key).in_(chunk),
                    col(ModelPricing.effective_at) <= as_of,
                )
                .order_by(col(ModelPricing.model_key), col(ModelPricing.effective_at))
            )
            for row in result.scalars():
                current[row.model_key] = row
        return current

    def add_all(self, rows: Sequence[OrganizationModelPricing]) -> None:
        """Stage new rate rows. The caller owns the transaction."""
        if rows:
            self.db.add_all(rows)

    async def flush(self) -> None:
        """Push staged changes so the database's constraints answer before the commit."""
        await self.db.flush()
