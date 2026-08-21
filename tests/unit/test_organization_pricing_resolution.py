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
from gateway.services.pricing_service import find_model_pricing, price_tool_calls
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

    ``expire_all`` is what makes this a real assertion. Without it the ``select``
    below returns the instance still in the identity map, carrying the aware value
    it was *constructed* from, so the SQLite round-trip never happens and the test
    passes whatever the column type is. Expiring forces the read.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _override(session, organization_id)
        await session.commit()
        session.expire_all()

        pricing = await find_model_pricing(session, "openai", "gpt-4o", as_of=_NOW, organization_id=organization_id)

        assert pricing is not None
        assert pricing.effective_at.tzinfo is not None
        assert pricing.effective_at.utcoffset() == timedelta(0)

    _run(scenario)


def test_the_stored_period_reads_back_aware_on_sqlite() -> None:
    """The column type, asserted on the engine the OSS edition ships.

    This is the wire shape, not the cost path: ``OrganizationModelPricingPublic``
    serializes these columns straight out, and a browser parses an offset-less
    date-time as *local*, which would shift the period every time the Edit dialog
    round-tripped it. ``UtcDateTime`` is what keeps the offset; a plain
    ``DateTime(timezone=True)`` is a no-op here and this test fails against it.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        stored = await _override(
            session,
            organization_id,
            effective_from=_NOW,
            effective_to=_NOW + timedelta(days=1),
        )
        # Read the id before expiring: afterwards touching the stale instance
        # would trigger a lazy refresh outside the async context.
        stored_id = stored.id
        await session.commit()
        session.expire_all()

        row = await session.get(OrganizationModelPricing, stored_id)

        assert row is not None
        for column, value in (
            ("effective_from", row.effective_from),
            ("effective_to", row.effective_to),
            ("created_at", row.created_at),
            ("updated_at", row.updated_at),
        ):
            assert value is not None, column
            assert value.tzinfo is not None, f"{column} read back naive"
            assert value.utcoffset() == timedelta(0), column
        # The instant survives the round trip, not just the awareness.
        assert row.effective_from == _NOW

    _run(scenario)


def test_a_naive_period_is_refused_rather_than_stored_ambiguously() -> None:
    """``UtcDateTime`` refuses a naive write instead of guessing a zone.

    The engines disagree about what a naive value means, so storing one is how a
    period ends up hours off with nothing to show for it.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        session.add(
            OrganizationModelPricing(
                organization_id=organization_id,
                model_key=_MODEL_KEY,
                input_price_per_million=1.0,
                output_price_per_million=2.0,
                effective_from=datetime(2026, 8, 20, 12, 0),  # noqa: DTZ001
                pricing_tiers=[],
            )
        )
        with pytest.raises(Exception, match="timezone-aware"):
            await session.flush()

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


def test_the_tool_gate_and_the_settlement_agree_on_both_key_spellings() -> None:
    """A tool priced only under the legacy slash key must still settle at that rate.

    The require-pricing gate resolves through ``find_model_pricing``, which tries
    ``otari:web_search`` and ``otari/web_search``. The batched settlement lookup
    matched only the canonical form, so an override stored under the slash
    spelling admitted the request and then charged zero: the tool ran, the row
    recorded no cost. Writes normalize now, which stops new rows landing that way,
    but the read side is what closes it for a key whose prefix normalization
    cannot resolve.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        await _override(
            session,
            organization_id,
            rate=1_000_000.0,  # the per-request convention: stored rate / 1e6 = $1.00
            model_key="otari/web_search",
            effective_from=_NOW - timedelta(days=1),
        )

        gate = await find_model_pricing(
            session, "otari", "web_search", as_of=_NOW, use_defaults=False, organization_id=organization_id
        )
        total, lines, unpriced = await price_tool_calls(
            session, {"web_search": 1}, as_of=_NOW, organization_id=organization_id
        )

        assert gate is not None, "the gate admits the request"
        assert unpriced == [], "so the settlement must not call it unpriced"
        assert total == 1.0
        assert lines[0]["cost"] == 1.0

    _run(scenario)


def test_both_lookup_paths_pick_the_canonical_row_when_both_spellings_exist() -> None:
    """The tool path and the model path must not disagree about which row wins.

    ``find_model_pricing`` prefers the canonical ``otari:tool`` over the legacy
    ``otari/tool`` one key at a time. The batched tool lookup assigns into a dict
    keyed on the tool, so without the same preference in its ORDER BY the winner
    was whichever spelling happened to carry the later period, and the same tool
    could be gated on one rate and settled at another.
    """

    async def scenario(session: AsyncSession) -> None:
        organization_id = await _organization(session)
        # The legacy row is deliberately the *newer* period, so an ordering that
        # only considers time would pick it.
        await _override(
            session,
            organization_id,
            rate=9_000_000.0,
            model_key="otari/web_search",
            effective_from=_NOW - timedelta(hours=1),
        )
        await _override(
            session,
            organization_id,
            rate=1_000_000.0,
            model_key="otari:web_search",
            effective_from=_NOW - timedelta(days=1),
        )

        via_model_path = await find_model_pricing(
            session, "otari", "web_search", as_of=_NOW, use_defaults=False, organization_id=organization_id
        )
        total, _lines, unpriced = await price_tool_calls(
            session, {"web_search": 1}, as_of=_NOW, organization_id=organization_id
        )

        assert via_model_path is not None
        assert unpriced == []
        # Both resolve the canonical row: 1_000_000 / 1e6 == $1.00 per call.
        assert via_model_path.input_price_per_million == 1_000_000.0
        assert total == 1.0

    _run(scenario)


def test_the_deployment_price_list_also_prefers_the_canonical_tool_key() -> None:
    """``_tool_rates`` resolves two tables, and both need the same preference.

    The override statement got it first; this covers the ``model_pricing`` half,
    which is the likelier one in practice because that table is what an operator
    re-imports and it predates key normalization. Ordering on ``effective_at``
    alone let the legacy spelling win whenever it carried the later row, while
    ``find_model_pricing`` gated on the canonical one: admitted at one rate,
    settled at another.
    """

    async def scenario(session: AsyncSession) -> None:
        # The legacy row is deliberately the newer one, so a time-only ordering
        # picks it.
        session.add(
            ModelPricing(
                model_key="otari/web_search",
                effective_at=_NOW - timedelta(hours=1),
                input_price_per_million=9_000_000.0,
                output_price_per_million=0.0,
            )
        )
        session.add(
            ModelPricing(
                model_key="otari:web_search",
                effective_at=_NOW - timedelta(days=1),
                input_price_per_million=1_000_000.0,
                output_price_per_million=0.0,
            )
        )
        await session.flush()

        gate = await find_model_pricing(session, "otari", "web_search", as_of=_NOW, use_defaults=False)
        total, _lines, unpriced = await price_tool_calls(session, {"web_search": 1}, as_of=_NOW)

        assert gate is not None
        assert unpriced == []
        # Both resolve the canonical row: 1_000_000 / 1e6 == $1.00 per call.
        assert gate.input_price_per_million == 1_000_000.0
        assert total == 1.0

    _run(scenario)
