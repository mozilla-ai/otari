"""Data access for the deployment price list, and the organization overrides above it.

Serves the surface that offers models on the deployment's own hosted
providers, which needs to say what each offered model is currently served at,
store a rate, and take a model's rates off the catalog when nothing serves it.

Built on the Unit of Work. Flushes, never commits. The pricing tables are
declarative ``Base`` tables, so their columns are referenced directly, and
``col()`` wraps only the SQLModel key table joined for the override rule.
Every statement is dialect-neutral: the chain and the OSS edition run on SQLite as well as
PostgreSQL, so a newest-version read is a ``MAX`` subquery rather than
``DISTINCT ON``, and a prefix match is ``LIKE`` rather than ``split_part``.
"""

from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel
from sqlalchemy import ColumnElement, CursorResult, and_, delete, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.pricing import ModelPricing, OrganizationModelPricing
from gateway.models.provider_keys import OrgProviderKey
from gateway.repositories.base_repository import BaseRepository

# The bound `services.pricing_service` chunks its own key lookups at, for the
# same reason: a key list long enough to exceed SQLite's default limit on bind
# parameters in one statement is a realistic page of one provider's models.
_KEY_CHUNK = 500


class ModelPricingRepository(BaseRepository[ModelPricing, BaseModel, BaseModel]):
    """Repository for `model_pricing` rows, and the overrides a sweep takes with them.

    The create and update parameters are unconstrained because a version is
    written through `DeploymentPricingService`, which owns the inherit-forward
    rule a new version follows.
    """

    def __init__(self, db: AsyncSession | UnitOfWork):
        """Bind to a unit of work, or to a session for a read outside one."""
        super().__init__(db, ModelPricing)

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

        ``as_of`` narrows the answer to versions already in effect, which is
        what a reader of the current rate wants; ``None`` takes the newest
        version whatever its date, which is what the seeding bookkeeping
        compares its timestamp against. One row per key, chosen in SQL, because
        ``model_pricing`` is a version series and loading every version to keep
        the last grows with the deployment's history rather than with the page
        being drawn.
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

    async def version_at(self, model_key: str, effective_at: datetime) -> ModelPricing | None:
        """The version of ``model_key`` at exactly ``effective_at``, for a write landing on the same instant."""
        result = await self.db.execute(
            select(ModelPricing).where(
                ModelPricing.model_key == model_key,
                ModelPricing.effective_at == effective_at,
            )
        )
        return result.scalar_one_or_none()

    def add_all(self, rows: Sequence[ModelPricing]) -> None:
        """Stage new versions. The caller owns the transaction."""
        if rows:
            self.db.add_all(rows)

    async def flush(self) -> None:
        """Flush staged versions so a read in the same block sees them."""
        await self.db.flush()

    async def all_keys(self) -> list[tuple[str, int]]:
        """Every key in the deployment price list, with how many versions it holds.

        The whole table, because it is the deployment's price list rather than
        a tenant's: one row per rate change per model, which stays in the low
        thousands even on a deployment that has re-imported a catalog for years.
        """
        result = await self.db.execute(
            select(ModelPricing.model_key, func.count())
            .group_by(ModelPricing.model_key)
            .order_by(ModelPricing.model_key)
        )
        return [(model_key, versions) for model_key, versions in result.all()]

    @staticmethod
    def _doomed_override(model_keys: Sequence[str]) -> ColumnElement[bool]:
        """The overrides above ``model_keys`` that a sweep takes with them.

        One predicate for the preview's count and the apply's delete, so the two
        cannot disagree. An override is spared when its own organization holds a
        live key for the provider: the deployment's rate is going because no
        hosted provider serves the model, but that organization reaches it on
        its own key and this is its rate on it. Live means unarchived, whether
        or not the stored secret decrypts here: a key this process cannot read
        is still a tenant's key.
        """
        own_key = exists(
            select(col(OrgProviderKey.id)).where(
                col(OrgProviderKey.organization_id) == OrganizationModelPricing.organization_id,
                col(OrgProviderKey.archived_at).is_(None),
                OrganizationModelPricing.model_key.like(col(OrgProviderKey.provider) + ":%"),
            )
        )
        return OrganizationModelPricing.model_key.in_(model_keys) & ~own_key

    async def count_doomed_overrides(self, model_keys: Sequence[str]) -> int:
        """How many organization overrides :meth:`delete_keys` would remove above ``model_keys``.

        For a preview that must report the same number without writing. Takes
        the canonical spelling, for the reason that method gives.
        """
        total = 0
        for start in range(0, len(model_keys), _KEY_CHUNK):
            chunk = list(model_keys[start : start + _KEY_CHUNK])
            result = await self.db.execute(
                select(func.count()).select_from(OrganizationModelPricing).where(self._doomed_override(chunk))
            )
            total += result.scalar_one()
        return total

    async def delete_keys(self, model_keys: Sequence[str], override_keys: Sequence[str]) -> tuple[int, int]:
        """Delete every version of each key, and the organization overrides above them.

        Two key lists because the two tables are spelled differently. A price
        row holds whatever spelling was written, the legacy ``provider/model``
        included, while an override is normalized before it is stored
        (``organization_pricing_service``), so matching overrides on the price
        row's own key would leave a slash-spelled model's override behind.

        Returns the rows removed from each. The overrides go first: an override
        is a commitment over a price-list entry, so a moment where the entry is
        gone and the override is not would price a model the deployment no
        longer lists. The exception is an organization's rate on a model it
        reaches on its own key, which :meth:`_doomed_override` spares. The
        caller owns the transaction.
        """
        prices = 0
        overrides = 0
        for start in range(0, len(override_keys), _KEY_CHUNK):
            chunk = list(override_keys[start : start + _KEY_CHUNK])
            result = cast(
                CursorResult[Any],
                await self.db.execute(delete(OrganizationModelPricing).where(self._doomed_override(chunk))),
            )
            overrides += result.rowcount
        for start in range(0, len(model_keys), _KEY_CHUNK):
            chunk = list(model_keys[start : start + _KEY_CHUNK])
            result = cast(
                CursorResult[Any], await self.db.execute(delete(ModelPricing).where(ModelPricing.model_key.in_(chunk)))
            )
            prices += result.rowcount
        await self.db.flush()
        return prices, overrides
