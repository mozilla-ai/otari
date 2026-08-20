"""Per-organization rate overrides: how they resolve, and when they are refused.

Driven against SQLite rather than the integration suite's PostgreSQL, for two
reasons. It is the engine the OSS edition ships by default, so an override that
resolved only on PostgreSQL would be broken for most self-hosters and nothing
else covers that. And the overlap rule is enforced in the service precisely
*because* SQLite cannot express it as a constraint, so SQLite is where that check
has to be shown working.

The HTTP surface (the role gate, the statuses, the envelope) is covered in
`tests/integration/test_organization_pricing_routes.py`, which needs a real
request and a real identity.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the shared metadata)
from gateway.models.entities import ModelPricing, OrganizationModelPricing
from gateway.models.tenancy import Organization
from gateway.services.organization_pricing_service import (
    OrganizationPricingService,
    validate_period,
)
from gateway.services.pricing_service import find_model_pricing
from gateway.services.tenancy.errors import (
    OrganizationPricingOverlapError,
    TenancyValidationError,
)

T = TypeVar("T")

_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
_MODEL_KEY = "openai:gpt-4o"
_DEPLOYMENT_INPUT_RATE = 10.0
_OVERRIDE_INPUT_RATE = 2.5


def _run(scenario: Callable[[AsyncSession], Awaitable[T]]) -> T:
    """Create the schema on a fresh in-memory SQLite database and run one scenario."""

    async def main() -> T:
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(SQLModel.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            async with session_factory() as session:
                return await scenario(session)
        finally:
            # aiosqlite runs each connection on its own thread, so an undisposed
            # engine leaks one per call.
            await engine.dispose()

    return asyncio.run(main())


async def _organization(session: AsyncSession, slug: str = "acme") -> uuid.UUID:
    organization = Organization(name=slug.title(), slug=slug)
    session.add(organization)
    await session.flush()
    return organization.id


async def _deployment_price(session: AsyncSession, *, rate: float = _DEPLOYMENT_INPUT_RATE) -> None:
    """The one price list a deployment has today, for the model under test."""
    session.add(
        ModelPricing(
            model_key=_MODEL_KEY,
            effective_at=_NOW - timedelta(days=30),
            input_price_per_million=rate,
            output_price_per_million=rate * 2,
        )
    )
    await session.flush()


async def _override(
    session: AsyncSession,
    organization_id: uuid.UUID,
    *,
    rate: float = _OVERRIDE_INPUT_RATE,
    effective_from: datetime | None = None,
    effective_to: datetime | None = None,
    model_key: str = _MODEL_KEY,
) -> OrganizationModelPricing:
    row = OrganizationModelPricing(
        organization_id=organization_id,
        model_key=model_key,
        input_price_per_million=rate,
        output_price_per_million=rate * 2,
        effective_from=effective_from or (_NOW - timedelta(days=1)),
        effective_to=effective_to,
        pricing_tiers=[],
    )
    session.add(row)
    await session.flush()
    return row


def test_an_override_wins_over_the_deployment_price_list() -> None:
    """The first line of the definition of done: the organization's rate applies."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _deployment_price(session)
        await _override(session, organization_id)

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=organization_id)

        assert pricing is not None
        assert pricing.input_price_per_million == _OVERRIDE_INPUT_RATE

    _run(scenario)


def test_without_an_organization_the_deployment_row_is_unchanged() -> None:
    """A deployment that never creates an override prices exactly as it did.

    The third line of the definition of done, and the reason ``organization_id``
    defaults to ``None``: every deployment-wide caller keeps reading the
    deployment's own list.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _deployment_price(session)
        await _override(session, organization_id)

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW)

        assert pricing is not None
        assert pricing.input_price_per_million == _DEPLOYMENT_INPUT_RATE

    _run(scenario)


def test_an_organization_without_an_override_reads_the_deployment_row() -> None:
    """Passing an organization is not the same as having an override."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _deployment_price(session)

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=organization_id)

        assert pricing is not None
        assert pricing.input_price_per_million == _DEPLOYMENT_INPUT_RATE

    _run(scenario)


def test_another_organizations_override_does_not_apply() -> None:
    """The tenant boundary: an override prices one organization's requests only."""

    async def scenario(session: AsyncSession) -> None:
        theirs = await _organization(session, "theirs")
        mine = await _organization(session, "mine")
        await _deployment_price(session)
        await _override(session, theirs)

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=mine)

        assert pricing is not None
        assert pricing.input_price_per_million == _DEPLOYMENT_INPUT_RATE

    _run(scenario)


def test_an_override_outside_its_period_does_not_apply() -> None:
    """A rate that has expired falls back rather than lingering."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _deployment_price(session)
        await _override(
            session,
            organization_id,
            effective_from=_NOW - timedelta(days=10),
            effective_to=_NOW - timedelta(days=5),
        )

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=organization_id)

        assert pricing is not None
        assert pricing.input_price_per_million == _DEPLOYMENT_INPUT_RATE

    _run(scenario)


def test_the_period_end_is_exclusive_and_the_start_inclusive() -> None:
    """Half-open, which is what lets two periods meet without overlapping."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _deployment_price(session)
        boundary = _NOW
        await _override(
            session,
            organization_id,
            rate=1.0,
            effective_from=boundary - timedelta(days=1),
            effective_to=boundary,
        )
        await _override(session, organization_id, rate=2.0, effective_from=boundary)

        at_boundary = await find_model_pricing(
            session, "openai", "gpt-4o", as_of=boundary, organization_id=organization_id
        )
        just_before = await find_model_pricing(
            session,
            "openai",
            "gpt-4o",
            as_of=boundary - timedelta(seconds=1),
            organization_id=organization_id,
        )

        assert at_boundary is not None and at_boundary.input_price_per_million == 2.0
        assert just_before is not None and just_before.input_price_per_million == 1.0

    _run(scenario)


def test_an_override_resolves_for_the_legacy_slash_key() -> None:
    """The override matches on the same candidate keys the deployment row does."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _override(session, organization_id, model_key="openai/gpt-4o")

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=organization_id)

        assert pricing is not None
        assert pricing.input_price_per_million == _OVERRIDE_INPUT_RATE

    _run(scenario)


def test_a_resolved_override_is_never_persisted() -> None:
    """It resolves into a transient ``ModelPricing``, not a row to be flushed.

    The same contract ``default_model_pricing`` has. A resolution that attached to
    the session would write a deployment-wide price row on the next flush, which
    is how one organization's negotiated rate would leak to every other.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _override(session, organization_id)
        await session.commit()

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=organization_id)
        assert pricing is not None
        assert pricing not in session
        await session.commit()

        stored = (await session.execute(ModelPricing.__table__.select())).all()
        assert stored == []

    _run(scenario)


def test_a_resolved_override_carries_an_aware_timestamp() -> None:
    """``effective_at`` reads back UTC-aware even on SQLite.

    ``DateTime(timezone=True)`` is a no-op there, so without the explicit stamp a
    caller comparing this against an aware timestamp would raise instead of
    compare.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _override(session, organization_id)

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=organization_id)

        assert pricing is not None
        assert pricing.effective_at.tzinfo is not None
        assert pricing.effective_at.utcoffset() == timedelta(0)

    _run(scenario)


def test_an_overlapping_period_is_refused_naming_the_period_it_hits() -> None:
    """The second line of the definition of done: a conflict, not a shadow."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        service = OrganizationPricingService(session)
        await _override(
            session,
            organization_id,
            effective_from=_NOW - timedelta(days=10),
            effective_to=_NOW + timedelta(days=10),
        )

        with pytest.raises(OrganizationPricingOverlapError) as caught:
            await service.raise_if_overlapping(
                organization_id=organization_id,
                model_key=_MODEL_KEY,
                effective_from=_NOW,
                effective_to=_NOW + timedelta(days=1),
            )

        assert _MODEL_KEY in caught.value.message
        assert caught.value.status_code == 409

    _run(scenario)


@pytest.mark.parametrize(
    ("effective_from", "effective_to", "overlaps"),
    [
        # The stored period is [day 0, day 10).
        pytest.param(-5, 5, True, id="candidate straddles the start"),
        pytest.param(5, 15, True, id="candidate straddles the end"),
        pytest.param(2, 8, True, id="candidate inside"),
        pytest.param(-5, 15, True, id="candidate encloses"),
        pytest.param(0, 10, True, id="identical"),
        pytest.param(-5, 0, False, id="candidate ends where the stored period begins"),
        pytest.param(10, 20, False, id="candidate begins where the stored period ends"),
        pytest.param(-10, -5, False, id="candidate entirely before"),
        pytest.param(15, 20, False, id="candidate entirely after"),
        pytest.param(20, None, False, id="open ended, starting after"),
        pytest.param(5, None, True, id="open ended, starting inside"),
    ],
)
def test_the_overlap_rule_on_every_arrangement_of_two_periods(
    effective_from: int, effective_to: int | None, overlaps: bool
) -> None:
    """Both boundary cases are the interesting ones: touching is not overlapping."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        service = OrganizationPricingService(session)
        origin = _NOW
        await _override(
            session,
            organization_id,
            effective_from=origin,
            effective_to=origin + timedelta(days=10),
        )

        check = service.raise_if_overlapping(
            organization_id=organization_id,
            model_key=_MODEL_KEY,
            effective_from=origin + timedelta(days=effective_from),
            effective_to=(origin + timedelta(days=effective_to)) if effective_to is not None else None,
        )
        if overlaps:
            with pytest.raises(OrganizationPricingOverlapError):
                await check
        else:
            await check

    _run(scenario)


def test_an_open_ended_stored_period_overlaps_everything_after_it() -> None:
    """A NULL end is infinity, not a missing value to be skipped."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        service = OrganizationPricingService(session)
        await _override(session, organization_id, effective_from=_NOW)

        with pytest.raises(OrganizationPricingOverlapError):
            await service.raise_if_overlapping(
                organization_id=organization_id,
                model_key=_MODEL_KEY,
                effective_from=_NOW + timedelta(days=365),
                effective_to=None,
            )

    _run(scenario)


def test_a_different_model_key_is_not_an_overlap() -> None:
    """The rule is per key: two models may share a period freely."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        service = OrganizationPricingService(session)
        await _override(session, organization_id, effective_from=_NOW, model_key="openai:gpt-4o")

        await service.raise_if_overlapping(
            organization_id=organization_id,
            model_key="anthropic:claude-sonnet-5",
            effective_from=_NOW,
            effective_to=None,
        )

    _run(scenario)


def test_editing_a_row_does_not_overlap_itself() -> None:
    """Without the exclusion an edit that keeps the period would refuse itself."""

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        service = OrganizationPricingService(session)
        row = await _override(session, organization_id, effective_from=_NOW)

        await service.raise_if_overlapping(
            organization_id=organization_id,
            model_key=_MODEL_KEY,
            effective_from=_NOW,
            effective_to=None,
            exclude_id=row.id,
        )

    _run(scenario)


@pytest.mark.parametrize(
    ("offset_days", "refused"),
    [
        pytest.param(1, False, id="ends after it starts"),
        pytest.param(0, True, id="zero width"),
        pytest.param(-1, True, id="ends before it starts"),
    ],
)
def test_a_period_must_end_after_it_starts(offset_days: int, refused: bool) -> None:
    """A 400 from the service, so it never reaches the table's CHECK as a 500.

    Zero width is refused with inversion: a period covering no instant resolves
    for nothing, so storing one is silently the same as storing nothing.
    """
    effective_to = _NOW + timedelta(days=offset_days)
    if refused:
        with pytest.raises(TenancyValidationError) as caught:
            validate_period(_NOW, effective_to)
        assert caught.value.status_code == 400
    else:
        validate_period(_NOW, effective_to)


def test_an_open_ended_period_is_always_valid() -> None:
    """No end is the common case, not a missing value to be rejected."""
    validate_period(_NOW, None)
