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

from pydantic import BaseModel
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.pricing import ModelPricing, OrganizationModelPricing
from gateway.repositories.base_repository import BaseRepository

# The bound `services.pricing_service` chunks its own key lookups at, for the
# same reason: a key list long enough to exceed SQLite's default limit on bind
# parameters in one statement is a realistic page of one provider's models.
_KEY_CHUNK = 500


class OrganizationModelPricingRepository(BaseRepository[OrganizationModelPricing, BaseModel, BaseModel]):
    """Repository for `organization_model_pricing` rows, and reads of `model_pricing`.

    The create and update parameters are unconstrained because this repository
    exposes neither: a rate is written through `OrganizationPricingService`,
    which owns the overlap rule that decides whether a period may exist at all.
    """

    def __init__(self, db: AsyncSession | UnitOfWork):
        """Bind to a unit of work, or to a session for a service still in the older shape."""
        super().__init__(db, OrganizationModelPricing)

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
                    # The half-open period, stated in SQL rather than walked in
                    # Python: this table grows a row per model per period, so an
                    # expired one filtered here is a row the database never
                    # sends. ``effective_to`` is exclusive, so a period ending
                    # exactly at ``as_of`` no longer applies.
                    or_(
                        col(OrganizationModelPricing.effective_to).is_(None),
                        col(OrganizationModelPricing.effective_to) > as_of,
                    ),
                )
                .order_by(
                    col(OrganizationModelPricing.model_key),
                    col(OrganizationModelPricing.effective_from),
                )
            )
            for row in result.scalars():
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
            # One row per key, chosen in SQL. ``model_pricing`` is a version
            # series, so a model repriced often carries a row per version and
            # loading them all to keep the last is a table that grows with the
            # deployment's history rather than with the page being drawn.
            newest = (
                select(
                    col(ModelPricing.model_key).label("model_key"),
                    func.max(col(ModelPricing.effective_at)).label("effective_at"),
                )
                .where(
                    col(ModelPricing.model_key).in_(chunk),
                    col(ModelPricing.effective_at) <= as_of,
                )
                .group_by(col(ModelPricing.model_key))
                .subquery()
            )
            result = await self.db.execute(
                select(ModelPricing).join(
                    newest,
                    and_(
                        col(ModelPricing.model_key) == newest.c.model_key,
                        col(ModelPricing.effective_at) == newest.c.effective_at,
                    ),
                )
            )
            for row in result.scalars():
                current[row.model_key] = row
        return current

    def add_all(self, rows: Sequence[OrganizationModelPricing]) -> None:
        """Stage new rate rows. The caller owns the transaction."""
        if rows:
            self.db.add_all(rows)

    async def page_for_organization(
        self,
        organization_id: uuid.UUID,
        *,
        model_key: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[OrganizationModelPricing], int]:
        """One page of an organization's stored rates, and the total matching.

        Ordered by key then newest period, so paging is stable. The count is the
        total rather than the length of the page, because that is what tells a
        caller whether to ask for another.
        """
        where = [col(OrganizationModelPricing.organization_id) == organization_id]
        if model_key is not None:
            where.append(col(OrganizationModelPricing.model_key) == model_key)
        total = (
            await self.db.execute(select(func.count()).select_from(OrganizationModelPricing).where(*where))
        ).scalar_one()
        rows = (
            await self.db.execute(
                select(OrganizationModelPricing)
                .where(*where)
                .order_by(
                    col(OrganizationModelPricing.model_key),
                    col(OrganizationModelPricing.effective_from).desc(),
                )
                .offset(skip)
                .limit(limit)
            )
        ).scalars().all()
        return list(rows), int(total)

    async def flush(self) -> None:
        """Push staged changes so the database's constraints answer before the commit."""
        await self.db.flush()
