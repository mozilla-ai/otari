"""An organization's own model rates, above the deployment price list.

The write half of per-organization pricing. The read half is
`services.pricing_service.find_model_pricing`, which consults these rows ahead
of ``model_pricing`` and the genai-prices dataset when it is given an
organization.

Three rules live here rather than in the route, because each has to hold for
every writer and none of them can be expressed in the schema at all:

- **A period may not overlap another for the same key.** ``model_pricing`` is a
  version series where a later row shadows an earlier one; an override is a
  commitment for a period, so two periods covering one instant is refused. The
  natural enforcement is a PostgreSQL ``EXCLUDE`` over a range type, and SQLite,
  which the OSS edition ships by default, has neither exclusion constraints nor
  range types. So the rule is checked here and the schema holds the part both
  engines can (a unique index on the period start); see
  `models.pricing.OrganizationModelPricing` for the race that leaves.
- **Only a management role may write.** Rates decide what every member of the
  organization is billed, so this is the same owner-or-admin gate the rest of the
  organization surface uses, delegated to ``OrganizationService`` rather than
  re-deriving membership here.
- **A deployment-supplied model is not the organization's to re-price.** A key
  addressed through one of ``config.providers``' instances dispatches on the
  deployment's own credential, so the deployment settles its upstream bill and
  owns its rate. A bare ``provider:model`` key gets the same refusal when a
  workspace lacks a usable BYO key and the bound ``ModelProviderPort`` would
  serve it on a deployment-owned hosted credential. See
  :meth:`OrganizationPricingService.raise_if_deployment_supplied`.

Periods are half-open, ``[effective_from, effective_to)``. Two adjacent periods
may therefore share an instant (one ends exactly where the next begins) without
overlapping, which is what lets an operator retire a rate and start the next one
at the same timestamp with no gap and no conflict. The resolution query applies
the identical rule, so what is storable and what is resolvable cannot disagree.
"""

import asyncio
import uuid
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.exceptions import TenancyValidationError
from gateway.exceptions.pricing_exceptions import (
    OrganizationPricingManagedModelError,
    OrganizationPricingNotFoundError,
    OrganizationPricingOverlapError,
)
from gateway.models.money import to_usd, to_usd_or_none
from gateway.models.pricing import API_ORIGIN, ModelPricing, OrganizationModelPricing, PriceSource
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.model_provider_port import HostedAccessDeniedError, ModelProviderPort
from gateway.repositories.pricing import OrganizationModelPricingRepository
from gateway.services.pricing_service import (
    default_model_pricing,
    normalize_effective_at,
    override_as_model_pricing,
)
from gateway.services.provider_kwargs import is_deployment_instance_key, split_selector
from gateway.services.tenancy.deployment_user_service import DeploymentUserService
from gateway.services.tenancy.org_provider_key_service import OrgProviderKeyService
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
    unit: str = "tokens"


def _describe_period(effective_from: datetime, effective_to: datetime | None) -> str:
    """A period rendered for an error message a human has to act on."""
    start = effective_from.isoformat()
    if effective_to is None:
        return f"from {start} onwards"
    return f"{start} to {effective_to.isoformat()}"



@dataclass(frozen=True)
class EffectiveRate:
    """One rung's answer: the rate, and which rung it came from."""

    source: PriceSource
    rates: ModelPricing
    row: OrganizationModelPricing | None
    """The organization's own row, when that rung answered, so a caller can edit it."""


def _rates_of(row: OrganizationModelPricing | ModelPricing) -> ModelPricing:
    """The rate columns of either table as one shape."""
    if isinstance(row, ModelPricing):
        return row
    return override_as_model_pricing(row)


def _resolve_defaults(model_keys: Sequence[str], as_of: datetime) -> dict[str, ModelPricing]:
    """Community default rates for several keys. Synchronous; run off the loop."""
    resolved: dict[str, ModelPricing] = {}
    for model_key in model_keys:
        provider, _, model = model_key.partition(":")
        default = default_model_pricing(provider or None, model or model_key, as_of)
        if default is not None:
            resolved[model_key] = default
    return resolved


class OrganizationPricingService:
    """Read and write the caller's organization's pricing overrides."""

    def __init__(
        self,
        db: AsyncSession,
        config: GatewayConfig,
        *,
        model_provider: ModelProviderPort | None,
    ):
        """Bind the request's session, provider map, and hosted-credential port.

        ``model_provider`` has no default, so a caller cannot drop the hosted-credential check by accident.
        """
        self.db = db
        self.config = config
        self.organizations = OrganizationService(db, membership_listener=None)
        self.provider_keys = OrgProviderKeyService(db)
        self.model_provider = model_provider
        self.rows = OrganizationModelPricingRepository(db)

    # ------------------------------------------------------------------
    # The ladder, in batch
    # ------------------------------------------------------------------

    async def rates_in_effect(
        self, organization_id: uuid.UUID, model_keys: Collection[str], as_of: datetime
    ) -> dict[str, EffectiveRate]:
        """What this organization is charged for each key, and which rung says so.

        The batch form of `pricing_service.find_model_pricing`'s order, which is
        the order a request is metered by: the organization's own row, then the
        deployment price list, then the community dataset. A caller pricing a
        page of models at once reads it here rather than restating the order,
        because a second statement of it is a second answer to what a request
        costs, and the two drift.

        A key nothing prices is absent from the result rather than present with
        an empty rate, so "unpriced" is one check at the call site.
        """
        stored = await self.rows.applicable_rows(organization_id, model_keys, as_of)
        deployment = await self.rows.deployment_rows(model_keys, as_of)
        effective: dict[str, EffectiveRate] = {}
        unpriced: list[str] = []
        for model_key in model_keys:
            if (row := stored.get(model_key)) is not None:
                effective[model_key] = EffectiveRate("organization", _rates_of(row), row)
            elif (deployment_row := deployment.get(model_key)) is not None:
                effective[model_key] = EffectiveRate("deployment", _rates_of(deployment_row), None)
            else:
                unpriced.append(model_key)
        for model_key, default in (await self.community_defaults(unpriced, as_of)).items():
            effective[model_key] = EffectiveRate("defaults", _rates_of(default), None)
        return effective

    async def community_defaults(self, model_keys: Collection[str], as_of: datetime) -> dict[str, ModelPricing]:
        """The dataset's own rate for each key, off the event loop.

        Resolution is synchronous and walks the dataset once per model, so a
        page of them is a thread hop rather than a stall. Deliberately not gated
        on ``default_pricing_enabled``: that switch governs the silent
        billing-time fallback, and a caller here is answering "what would this
        cost", or storing a rate an admin asked for.
        """
        wanted = sorted(set(model_keys))
        if not wanted:
            return {}
        return await asyncio.to_thread(_resolve_defaults, wanted, as_of)

    async def stage_seeded_rates(self, rows: Sequence[OrganizationModelPricing]) -> None:
        """Stage rates copied from the dataset on an organization's behalf.

        Staged rather than committed: the caller's unit of work owns the
        boundary. No overlap check, because a seeded row is only ever written
        for a key :meth:`rates_in_effect` just reported as unpriced.
        """
        if not rows:
            return
        self.rows.add_all(list(rows))
        await self.rows.flush()


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

    async def raise_if_deployment_supplied(
        self,
        user: TenancyUser,
        model_key: str,
        organization_id: uuid.UUID,
    ) -> None:
        """Refuse a rate for a model this deployment, not this organization, pays for.

        Public for the reason :meth:`raise_if_overlapping` is: it is one of the
        rules this surface exists to enforce, and asserting it through a create
        would prove it through the role gate and the identity resolver instead.

        The deployment operator is exempt because they are the party the rule
        protects. On a standalone deployment that identity is also the single
        organization's administrator, so nothing there changes; on a control plane
        serving tenants who did not pay for the upstream capacity, an override on a
        deployment-supplied instance would let a tenant name its own cost basis,
        and a zero would make the model free and spend no budget.

        Gotcha: this runs at write time only.
        A stored override still applies if the organization later loses BYO coverage.
        """
        if await DeploymentUserService(self.db).has_administration_access(user):
            return
        if model_key in await self.deployment_supplied_keys(organization_id, [model_key]):
            raise OrganizationPricingManagedModelError(model_key)

    async def deployment_supplied_keys(
        self, organization_id: uuid.UUID, model_keys: Collection[str]
    ) -> set[str]:
        """Which of ``model_keys`` the deployment pays the upstream bill for.

        A ``config.providers`` instance always does. A bare key does when a
        workspace lacks a usable BYO key and the port would serve it on a hosted
        credential, and a port refusal counts too, because the model still runs
        on a deployment-owned upstream.

        Asked in a batch because the organization's BYO providers have to be
        resolved before the port can be asked at all, and that is one query for
        an answer that does not change between models. The port call stays per
        model: whether a hosted credential serves one is a question about that
        model.

        Two callers want different things from the same answer.
        :meth:`raise_if_deployment_supplied` refuses an override for such a
        model. The offered-models surface skips seeding a rate for one and offers
        it anyway, since it prices from the deployment's list instead.
        """
        if not model_keys:
            return set()
        byo = await self.provider_keys.get_byo_providers(organization_id=organization_id)
        return {
            model_key
            for model_key in model_keys
            if await self._is_deployment_supplied(organization_id, model_key, byo)
        }

    async def _is_deployment_supplied(
        self, organization_id: uuid.UUID, model_key: str, byo_providers: Collection[str]
    ) -> bool:
        """The rule itself, over a BYO set the caller has already resolved."""
        if is_deployment_instance_key(self.config, model_key):
            return True
        if self.model_provider is None:
            return False
        split = split_selector(model_key)
        if split is None:
            return False
        provider, model = split
        # Gotcha: keep this before the port call. The port's contract only covers a candidate no BYO key serves.
        if provider in byo_providers:
            return False
        try:
            credential = await self.model_provider.resolve_hosted_credential(
                organization_id=organization_id,
                workspace_id=None,
                provider=provider,
                model=model,
            )
        except HostedAccessDeniedError:
            return True
        return credential is not None

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

    async def list_for_caller(
        self,
        user: TenancyUser,
        *,
        model_key: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[list[OrganizationModelPricing], int]:
        """One page of the caller's organization's overrides, and the total.

        Paged rather than whole: the table grows a row per model per period, so a
        long-lived organization accumulates them and an unbounded read would get
        slower forever. Ordered by key then newest period, so paging is stable.

        ``model_key`` narrows to one model, which is what an editor for that
        model needs: it has to see every period stored for it, both to open on
        the one in force and to refuse a new one that would overlap. Taking the
        first page of the whole table instead would answer that correctly only
        while the organization's overrides fit in one page, then quietly start
        opening a create form over a rate that already exists.

        The count is the total matching rows, not the length of the page, because
        that is what tells a client whether to ask for another one.
        """
        organization_id = await self._readable_organization_id(user)
        return await self.rows.page_for_organization(
            organization_id, model_key=model_key, skip=skip, limit=limit
        )

    async def create_for_caller(
        self,
        user: TenancyUser,
        model_key: str,
        override: PricingOverrideInput,
    ) -> OrganizationModelPricing:
        """Store a new override.

        Refused for a model the deployment supplies the credential for, and for a
        period that overlaps one already stored for this key.
        """
        organization_id = await self._writable_organization_id(user)
        await self.raise_if_deployment_supplied(user, model_key, organization_id)
        effective_from = normalize_effective_at(override.effective_from)
        effective_to = None if override.effective_to is None else normalize_effective_at(override.effective_to)
        validate_period(effective_from, effective_to)
        validate_rates(override)

        await self.raise_if_overlapping(
            organization_id=organization_id,
            model_key=model_key,
            effective_from=effective_from,
            effective_to=effective_to,
        )

        row = OrganizationModelPricing(
            organization_id=organization_id,
            model_key=model_key,
            input_price_per_million=to_usd(override.input_price_per_million),
            output_price_per_million=to_usd(override.output_price_per_million),
            cache_read_price_per_million=to_usd_or_none(override.cache_read_price_per_million),
            cache_write_price_per_million=to_usd_or_none(override.cache_write_price_per_million),
            cache_write_1h_price_per_million=to_usd_or_none(override.cache_write_1h_price_per_million),
            pricing_tiers=override.pricing_tiers,
            effective_from=effective_from,
            effective_to=effective_to,
            unit=override.unit,
            origin=API_ORIGIN,
        )
        self.db.add(row)
        await self._flush_or_conflict(organization_id, model_key, effective_from)
        return row

    async def _flush_or_conflict(
        self,
        organization_id: uuid.UUID,
        model_key: str,
        effective_from: datetime,
        *,
        exclude_id: uuid.UUID | None = None,
    ) -> None:
        """Flush, mapping the unique-index race onto the overlap conflict.

        The overlap check above is what refuses an overlapping period, and it is
        enough single-threaded. Two writers racing on the same period both pass it,
        and the unique index on ``(organization_id, model_key, effective_from)``
        refuses the second. That refusal arrives here, at ``flush``, not at the
        route's ``commit``, so without this it escapes as a 500 and the caller is
        told to retry something that was really a conflict.

        Which ``IntegrityError`` it was decides the answer, so the row is read
        back rather than assumed. A row already occupying this period start is the
        race, and its *own* period is what the message names; anything else (a
        CHECK violation, a foreign key) is not a conflict anybody caused and
        propagates unchanged. The ``except`` stays broad because the error's
        constraint name is dialect-specific, which is how
        ``OrganizationService.create_active_organization_member_for_user`` handles
        the same problem.

        ``exclude_id`` is what keeps that read-back honest on the update path. The
        rollback restores the row being rewritten to its stored period, so an
        update that keeps its period and fails the flush for some *other* reason
        would find itself and report a 409 naming the caller's own override.
        Excluding it mirrors ``raise_if_overlapping``'s parameter of the same name,
        and for the same reason.

        The rollback is required rather than tidy: a failed flush leaves the
        session unusable, so anything the caller did next would raise
        ``PendingRollbackError`` and mask this.
        """
        try:
            await self.db.flush()
        except IntegrityError:
            await self.db.rollback()
            occupant = select(OrganizationModelPricing).where(
                OrganizationModelPricing.organization_id == organization_id,
                OrganizationModelPricing.model_key == model_key,
                OrganizationModelPricing.effective_from == effective_from,
            )
            if exclude_id is not None:
                occupant = occupant.where(OrganizationModelPricing.id != exclude_id)
            clash = (await self.db.execute(occupant)).scalar_one_or_none()
            if clash is None:
                raise
            raise OrganizationPricingOverlapError(
                model_key,
                _describe_period(clash.effective_from, clash.effective_to),
            ) from None

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
        await self.raise_if_deployment_supplied(user, row.model_key, organization_id)

        effective_from = normalize_effective_at(override.effective_from)
        effective_to = None if override.effective_to is None else normalize_effective_at(override.effective_to)
        validate_period(effective_from, effective_to)
        validate_rates(override)

        await self.raise_if_overlapping(
            organization_id=organization_id,
            model_key=row.model_key,
            effective_from=effective_from,
            effective_to=effective_to,
            exclude_id=row.id,
        )

        row.input_price_per_million = to_usd(override.input_price_per_million)
        row.output_price_per_million = to_usd(override.output_price_per_million)
        row.cache_read_price_per_million = to_usd_or_none(override.cache_read_price_per_million)
        row.cache_write_price_per_million = to_usd_or_none(override.cache_write_price_per_million)
        row.cache_write_1h_price_per_million = to_usd_or_none(override.cache_write_1h_price_per_million)
        row.pricing_tiers = override.pricing_tiers
        row.effective_from = effective_from
        row.effective_to = effective_to
        row.unit = override.unit
        # The rate is somebody's choice now, whatever it was before. A row the
        # offered-models surface seeded carries ``seed`` and its refresh moves it
        # to each day's community default; leaving that marking on a rate an
        # admin has just set would have the next refresh overwrite it.
        row.origin = API_ORIGIN
        await self._flush_or_conflict(organization_id, row.model_key, effective_from, exclude_id=row.id)
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


def validate_rates(override: PricingOverrideInput) -> None:
    """Refuse a negative rate before it reaches the table's CHECK.

    The route already bounds all five with ``Field(ge=0)``, so nothing over HTTP
    arrives here negative. This exists for the other callers: the service is the
    boundary the entity contract is stated at, and a direct call (a test, a future
    importer, an overlay) would otherwise surface a negative rate as an
    ``IntegrityError`` at flush, which reads as an internal fault rather than as
    the refusal it is. Names the field, because "a rate is negative" is not
    actionable when there are five of them.
    """
    rates = {
        "input_price_per_million": override.input_price_per_million,
        "output_price_per_million": override.output_price_per_million,
        "cache_read_price_per_million": override.cache_read_price_per_million,
        "cache_write_price_per_million": override.cache_write_price_per_million,
        "cache_write_1h_price_per_million": override.cache_write_1h_price_per_million,
    }
    for field, value in rates.items():
        if value is not None and value < 0:
            raise TenancyValidationError(f"{field} must be non-negative; got {value}")


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
    "EffectiveRate",
    "OrganizationPricingService",
    "PricingOverrideInput",
    "validate_period",
    "validate_rates",
]
