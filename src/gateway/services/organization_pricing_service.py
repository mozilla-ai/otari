"""An organization's own model rates, above the deployment price list.

The write half of per-organization pricing. The read half is
`services.pricing_service.find_model_pricing`, which consults these rows ahead
of ``model_pricing`` and the genai-prices dataset when it is given an
organization.

Two rules live here rather than in the route, because both have to hold for
every writer and one of them cannot be expressed in the schema at all:

- **A period may not overlap another for the same key.** ``model_pricing`` is a
  version series where a later row shadows an earlier one; an override is a
  commitment for a period, so two periods covering one instant is refused. The
  natural enforcement is a PostgreSQL ``EXCLUDE`` over a range type, and SQLite,
  which the OSS edition ships by default, has neither exclusion constraints nor
  range types. So the rule is checked here and the schema holds the part both
  engines can (a unique index on the period start); see
  `models.entities.OrganizationModelPricing` for the race that leaves.
- **Only a management role may write.** Rates decide what every member of the
  organization is billed, so this is the same owner-or-admin gate the rest of the
  organization surface uses, delegated to ``OrganizationService`` rather than
  re-deriving membership here.

Periods are half-open, ``[effective_from, effective_to)``. Two adjacent periods
may therefore share an instant (one ends exactly where the next begins) without
overlapping, which is what lets an operator retire a rate and start the next one
at the same timestamp with no gap and no conflict. The resolution query applies
the identical rule, so what is storable and what is resolvable cannot disagree.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import OrganizationModelPricing
from gateway.models.tenancy import User as TenancyUser
from gateway.services.pricing_service import normalize_effective_at
from gateway.services.tenancy.errors import (
    OrganizationPricingNotFoundError,
    OrganizationPricingOverlapError,
    TenancyValidationError,
)
from gateway.services.tenancy.organization_service import OrganizationService


@dataclass(frozen=True)
class PricingOverrideInput:
    """The rates and period a caller asked to store.

    A dataclass rather than the request model, so the service does not depend on
    the API layer (the architecture check forbids it) and so the route stays the
    only place that knows the wire shape.

    Carries no ``model_key``. A create takes one alongside this, and an update
    cannot change it, so putting it here would mean handing the update path a
    value it must ignore.
    """

    input_price_per_million: float
    output_price_per_million: float
    cache_read_price_per_million: float | None
    cache_write_price_per_million: float | None
    cache_write_1h_price_per_million: float | None
    pricing_tiers: list[dict[str, object]]
    effective_from: datetime
    effective_to: datetime | None


def _describe_period(effective_from: datetime, effective_to: datetime | None) -> str:
    """A period rendered for an error message a human has to act on."""
    start = effective_from.isoformat()
    if effective_to is None:
        return f"from {start} onwards"
    return f"{start} to {effective_to.isoformat()}"


class OrganizationPricingService:
    """Read and write the caller's organization's pricing overrides."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.organizations = OrganizationService(db)

    async def _writable_organization_id(self, user: TenancyUser) -> uuid.UUID:
        """The caller's organization, having checked they may change its rates."""
        organization = await self.organizations.get_active_organization_for_user(user)
        await self.organizations.require_active_organization_management_access(
            user=user,
            organization=organization,
        )
        return organization.id

    async def _readable_organization_id(self, user: TenancyUser) -> uuid.UUID:
        """The caller's organization, for a read.

        No management gate: a member may see what their own requests are priced
        at. The rates are not a secret from the people being billed at them, and
        the same reasoning already applies to ``GET /v1/pricing``, which any API
        key may read.
        """
        organization = await self.organizations.get_active_organization_for_user(user)
        return organization.id

    async def raise_if_overlapping(
        self,
        *,
        organization_id: uuid.UUID,
        model_key: str,
        effective_from: datetime,
        effective_to: datetime | None,
        exclude_id: uuid.UUID | None = None,
    ) -> None:
        """Refuse a period that covers an instant this key is already priced for.

        Public because it is the rule this surface exists to enforce, and it is
        asserted directly over every arrangement of two periods in
        ``tests/unit/test_organization_pricing_resolution.py``. Reaching it only
        through a create or an update would mean proving the rule through the role
        gate and the identity resolver, which is not what is under test.

        Two half-open periods overlap when each starts before the other ends. An
        open-ended period (``effective_to`` NULL) ends at infinity, so its side of
        the test is simply dropped rather than compared against a sentinel date:
        an unbounded period ends after every existing start, unconditionally.
        """
        stmt = select(OrganizationModelPricing).where(
            OrganizationModelPricing.organization_id == organization_id,
            OrganizationModelPricing.model_key == model_key,
            # The existing row ends after the candidate starts. NULL is open
            # ended, which always satisfies this.
            or_(
                OrganizationModelPricing.effective_to.is_(None),
                OrganizationModelPricing.effective_to > effective_from,
            ),
        )
        if effective_to is not None:
            # And the candidate ends after the existing row starts. Omitted when
            # the candidate is open ended, because then it is always true; a
            # Python ``None`` check inside ``or_()`` would be a literal, not a
            # predicate, and would quietly collapse the whole clause.
            stmt = stmt.where(OrganizationModelPricing.effective_from < effective_to)
        if exclude_id is not None:
            stmt = stmt.where(OrganizationModelPricing.id != exclude_id)

        clash = (await self.db.execute(stmt.limit(1))).scalar_one_or_none()
        if clash is not None:
            raise OrganizationPricingOverlapError(
                model_key,
                _describe_period(clash.effective_from, clash.effective_to),
            )

    async def list_for_caller(self, user: TenancyUser) -> list[OrganizationModelPricing]:
        """Every override in the caller's organization, newest period first."""
        organization_id = await self._readable_organization_id(user)
        stmt = (
            select(OrganizationModelPricing)
            .where(OrganizationModelPricing.organization_id == organization_id)
            .order_by(
                OrganizationModelPricing.model_key,
                OrganizationModelPricing.effective_from.desc(),
            )
        )
        return list((await self.db.execute(stmt)).scalars().all())

    async def create_for_caller(
        self,
        user: TenancyUser,
        model_key: str,
        override: PricingOverrideInput,
    ) -> OrganizationModelPricing:
        """Store a new override, refusing one that overlaps an existing period."""
        organization_id = await self._writable_organization_id(user)
        effective_from = normalize_effective_at(override.effective_from)
        effective_to = normalize_effective_at(override.effective_to) if override.effective_to else None
        validate_period(effective_from, effective_to)

        await self.raise_if_overlapping(
            organization_id=organization_id,
            model_key=model_key,
            effective_from=effective_from,
            effective_to=effective_to,
        )

        row = OrganizationModelPricing(
            organization_id=organization_id,
            model_key=model_key,
            input_price_per_million=override.input_price_per_million,
            output_price_per_million=override.output_price_per_million,
            cache_read_price_per_million=override.cache_read_price_per_million,
            cache_write_price_per_million=override.cache_write_price_per_million,
            cache_write_1h_price_per_million=override.cache_write_1h_price_per_million,
            pricing_tiers=override.pricing_tiers,
            effective_from=effective_from,
            effective_to=effective_to,
        )
        self.db.add(row)
        await self.db.flush()
        return row

    async def _owned_row(self, organization_id: uuid.UUID, pricing_id: uuid.UUID) -> OrganizationModelPricing:
        """One override, scoped to the organization so another tenant's is a 404."""
        row = (
            await self.db.execute(
                select(OrganizationModelPricing).where(
                    OrganizationModelPricing.id == pricing_id,
                    OrganizationModelPricing.organization_id == organization_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise OrganizationPricingNotFoundError(pricing_id)
        return row

    async def replace_for_caller(
        self,
        user: TenancyUser,
        pricing_id: uuid.UUID,
        override: PricingOverrideInput,
    ) -> OrganizationModelPricing:
        """Rewrite an override in place.

        Editing is retroactive, deliberately and visibly: the row's period is what
        resolution reads, so changing a rate re-prices every *future* request in
        that period and leaves already-settled usage rows alone. Settled cost is
        stored on the usage row, not recomputed from pricing, so history does not
        move under an edit.

        ``model_key`` is immutable here. Changing it would silently retire the
        override for one model and create one for another, which is two operations
        an operator should perform as two.
        """
        organization_id = await self._writable_organization_id(user)
        row = await self._owned_row(organization_id, pricing_id)

        effective_from = normalize_effective_at(override.effective_from)
        effective_to = normalize_effective_at(override.effective_to) if override.effective_to else None
        validate_period(effective_from, effective_to)

        await self.raise_if_overlapping(
            organization_id=organization_id,
            model_key=row.model_key,
            effective_from=effective_from,
            effective_to=effective_to,
            exclude_id=row.id,
        )

        row.input_price_per_million = override.input_price_per_million
        row.output_price_per_million = override.output_price_per_million
        row.cache_read_price_per_million = override.cache_read_price_per_million
        row.cache_write_price_per_million = override.cache_write_price_per_million
        row.cache_write_1h_price_per_million = override.cache_write_1h_price_per_million
        row.pricing_tiers = override.pricing_tiers
        row.effective_from = effective_from
        row.effective_to = effective_to
        await self.db.flush()
        return row

    async def delete_for_caller(self, user: TenancyUser, pricing_id: uuid.UUID) -> None:
        """Remove an override.

        The model falls straight back to the deployment price list from the next
        request; usage already settled under the override keeps the cost it was
        billed, for the same reason an edit is not retroactive over history.
        """
        organization_id = await self._writable_organization_id(user)
        row = await self._owned_row(organization_id, pricing_id)
        await self.db.delete(row)
        await self.db.flush()


def validate_period(effective_from: datetime, effective_to: datetime | None) -> None:
    """Refuse a period that could never apply to any instant.

    The same rule as the table's CHECK, applied here so the caller gets a 400
    naming the problem rather than a 500 from an integrity error. Equality is
    refused along with inversion: a zero-width period resolves for nothing, so
    storing one is silently the same as storing nothing.
    """
    if effective_to is not None and effective_to <= effective_from:
        raise TenancyValidationError("effective_to must be after effective_from")


__all__ = [
    "OrganizationPricingService",
    "PricingOverrideInput",
    "validate_period",
]
