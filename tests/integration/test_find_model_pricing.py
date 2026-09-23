"""Tests for the shared find_model_pricing helper."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.db import ModelPricing
from gateway.services.pricing_service import configure_default_pricing, find_model_pricing


@pytest.mark.asyncio
async def test_find_pricing_colon_format(async_db: AsyncSession) -> None:
    """Test lookup with canonical colon-separated key."""
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4",
            effective_at=datetime(2025, 1, 1, tzinfo=UTC),
            input_price_per_million=30.0,
            output_price_per_million=60.0,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.input_price_per_million == 30.0


@pytest.mark.asyncio
async def test_find_pricing_legacy_slash_fallback(async_db: AsyncSession) -> None:
    """Test fallback to legacy slash-separated key when colon key is missing."""
    async_db.add(
        ModelPricing(
            model_key="openai/gpt-4",
            effective_at=datetime(2025, 1, 1, tzinfo=UTC),
            input_price_per_million=30.0,
            output_price_per_million=60.0,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "openai/gpt-4"


@pytest.mark.asyncio
async def test_find_pricing_no_provider(async_db: AsyncSession) -> None:
    """Test lookup without a provider uses model name directly."""
    async_db.add(
        ModelPricing(
            model_key="gpt-4",
            effective_at=datetime(2025, 1, 1, tzinfo=UTC),
            input_price_per_million=30.0,
            output_price_per_million=60.0,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, None, "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "gpt-4"


@pytest.mark.asyncio
async def test_find_pricing_not_found(async_db: AsyncSession) -> None:
    """Test that None is returned when no pricing exists."""
    pricing = await find_model_pricing(async_db, "openai", "nonexistent-model")
    assert pricing is None


@pytest.mark.asyncio
async def test_find_pricing_colon_preferred_over_slash(async_db: AsyncSession) -> None:
    """Test that colon format is returned when both formats exist."""
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4",
            effective_at=datetime(2025, 1, 1, tzinfo=UTC),
            input_price_per_million=30.0,
            output_price_per_million=60.0,
        )
    )
    async_db.add(
        ModelPricing(
            model_key="openai/gpt-4",
            effective_at=datetime(2025, 1, 1, tzinfo=UTC),
            input_price_per_million=10.0,
            output_price_per_million=20.0,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "openai:gpt-4"
    assert pricing.input_price_per_million == 30.0


@pytest.mark.asyncio
async def test_find_pricing_defaults_to_now(async_db: AsyncSession) -> None:
    """When as_of is omitted, the latest effective price is returned."""

    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4",
            effective_at=datetime(2025, 1, 1, tzinfo=UTC),
            input_price_per_million=10.0,
            output_price_per_million=20.0,
        )
    )
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4",
            effective_at=datetime(2025, 2, 1, tzinfo=UTC),
            input_price_per_million=30.0,
            output_price_per_million=60.0,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.input_price_per_million == 30.0


@pytest.mark.asyncio
async def test_find_pricing_future_prices_ignored(async_db: AsyncSession) -> None:
    """Future-effective pricing should not be returned for present lookups.

    Default pricing is off by default (see the autouse reset fixture), so the
    DB-only lookup is exercised in isolation here.
    """

    future_effective = datetime.now(UTC) + timedelta(days=10)
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4",
            effective_at=future_effective,
            input_price_per_million=99.0,
            output_price_per_million=199.0,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4")
    assert pricing is None


@pytest.mark.asyncio
async def test_find_pricing_with_explicit_as_of(async_db: AsyncSession) -> None:
    """Providing as_of returns the matching historical price."""

    older = datetime(2025, 1, 1, tzinfo=UTC)
    newer = datetime(2025, 2, 1, tzinfo=UTC)
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4",
            effective_at=older,
            input_price_per_million=5.0,
            output_price_per_million=10.0,
        )
    )
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4",
            effective_at=newer,
            input_price_per_million=12.0,
            output_price_per_million=24.0,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4", as_of=older)
    assert pricing is not None
    assert pricing.input_price_per_million == 5.0


@pytest.mark.asyncio
async def test_find_pricing_falls_back_to_genai_defaults(async_db: AsyncSession) -> None:
    """With no DB row and defaults enabled, a well-known model is priced."""
    configure_default_pricing(True)
    pricing = await find_model_pricing(async_db, "openai", "gpt-4o")

    assert pricing is not None
    assert pricing.model_key == "openai:gpt-4o"
    assert pricing.input_price_per_million > 0
    assert pricing.output_price_per_million > 0

    # The default is a lookup result, not a stored row: nothing is persisted.
    count = (await async_db.execute(select(func.count()).select_from(ModelPricing))).scalar_one()
    assert count == 0


@pytest.mark.asyncio
async def test_find_pricing_skips_genai_defaults_when_asked(async_db: AsyncSession) -> None:
    """use_defaults=False suppresses the fallback for keys that are not models.

    The genai-prices lookup also tries a provider-agnostic match on the bare
    name, so a non-model key (a search tool named after a real model) would
    otherwise inherit that model's per-million-token rate.
    """
    configure_default_pricing(True)
    assert await find_model_pricing(async_db, "exa", "gpt-4o", use_defaults=False) is None


@pytest.mark.asyncio
async def test_find_pricing_db_overrides_genai_defaults(async_db: AsyncSession) -> None:
    """An explicit DB price for a known model wins over the genai-prices default."""
    configure_default_pricing(True)
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4o",
            effective_at=datetime(2025, 1, 1, tzinfo=UTC),
            input_price_per_million=0.123,
            output_price_per_million=0.456,
        )
    )
    await async_db.commit()

    pricing = await find_model_pricing(async_db, "openai", "gpt-4o")
    assert pricing is not None
    assert pricing.input_price_per_million == Decimal("0.123")
    assert pricing.output_price_per_million == Decimal("0.456")


@pytest.mark.asyncio
async def test_find_pricing_defaults_can_be_disabled(async_db: AsyncSession) -> None:
    """default_pricing=False restores the DB-only (fail-closed) behavior."""
    configure_default_pricing(False)

    pricing = await find_model_pricing(async_db, "openai", "gpt-4o")
    assert pricing is None


@pytest.mark.asyncio
async def test_the_batch_ladder_answers_what_settlement_answers(async_db: AsyncSession) -> None:
    """The two statements of the ladder must not drift apart.

    `find_model_pricing` is the order a request is metered by, one model at a
    time. `OrganizationPricingService.rates_in_effect` is the same order in a
    batch, because the offered-models surface prices a page at once and asking
    per model would be a query per model. Two implementations of one order is
    the arrangement this pins: a rung added, reordered or re-spelled in either
    has to be done in both, and this fails until it is.
    """
    from gateway.core.config import GatewayConfig
    from gateway.models.money import to_usd
    from gateway.models.pricing import API_ORIGIN, OrganizationModelPricing
    from gateway.repositories.tenancy import OrganizationRepository
    from gateway.services.organization_pricing_service import OrganizationPricingService
    from gateway.services.pricing_service import normalize_effective_at

    organization = await OrganizationRepository(async_db).create_organization(
        name="Ladder", slug="ladder", created_by_user_id=None
    )
    organization_id = organization.id
    as_of = normalize_effective_at(None)
    configure_default_pricing(True)

    # `openai:gpt-4o` is priced at *both* stored rungs, which is what makes the
    # order decide rather than merely the lookup: a ladder that consulted the
    # deployment first would answer 9.0 where settlement answers 1.0. The other
    # two keys cover a rung each on their own.
    async_db.add(
        OrganizationModelPricing(
            organization_id=organization_id,
            model_key="openai:gpt-4o",
            input_price_per_million=to_usd(1.0),
            output_price_per_million=to_usd(2.0),
            effective_from=as_of - timedelta(days=1),
            origin=API_ORIGIN,
        )
    )
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4o",
            effective_at=as_of - timedelta(days=1),
            input_price_per_million=9.0,
            output_price_per_million=18.0,
        )
    )
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4o-mini",
            effective_at=as_of - timedelta(days=1),
            input_price_per_million=3.0,
            output_price_per_million=4.0,
        )
    )
    await async_db.commit()

    keys = ["openai:gpt-4o", "openai:gpt-4o-mini", "anthropic:claude-sonnet-4"]
    service = OrganizationPricingService(async_db, GatewayConfig(), model_provider=None)
    batch = await service.rates_in_effect(organization_id, keys, as_of)

    for model_key in keys:
        provider, _, model = model_key.partition(":")
        settled = await find_model_pricing(async_db, provider, model, as_of=as_of, organization_id=organization_id)
        rung = batch.get(model_key)
        assert (rung is None) == (settled is None), model_key
        if rung is None or settled is None:
            continue
        assert float(rung.rates.input_price_per_million) == float(settled.input_price_per_million), model_key
        assert float(rung.rates.output_price_per_million) == float(settled.output_price_per_million), model_key

    # The organization's own rate, not the deployment's 9.0 for the same key.
    assert batch["openai:gpt-4o"].source == "organization"
    assert float(batch["openai:gpt-4o"].rates.input_price_per_million) == 1.0
    assert batch["openai:gpt-4o-mini"].source == "deployment"
    assert batch["anthropic:claude-sonnet-4"].source == "defaults"


@pytest.mark.asyncio
async def test_resolve_pricing_reports_the_rung_that_answered(async_db: AsyncSession) -> None:
    """``resolve_model_pricing`` names the same rung ``find_model_pricing`` stops on."""
    from gateway.models.money import to_usd
    from gateway.models.pricing import API_ORIGIN, OrganizationModelPricing
    from gateway.repositories.tenancy import OrganizationRepository
    from gateway.services.pricing_service import normalize_effective_at, resolve_model_pricing

    organization = await OrganizationRepository(async_db).create_organization(
        name="Sources", slug="sources", created_by_user_id=None
    )
    as_of = normalize_effective_at(None)
    configure_default_pricing(True)
    async_db.add(
        OrganizationModelPricing(
            organization_id=organization.id,
            model_key="openai:gpt-4o",
            input_price_per_million=to_usd(1.0),
            output_price_per_million=to_usd(2.0),
            effective_from=as_of - timedelta(days=1),
            origin=API_ORIGIN,
        )
    )
    async_db.add(
        ModelPricing(
            model_key="openai:gpt-4o-mini",
            effective_at=as_of - timedelta(days=1),
            input_price_per_million=3.0,
            output_price_per_million=4.0,
        )
    )
    await async_db.commit()

    async def source_of(provider: str, model: str) -> str | None:
        resolved = await resolve_model_pricing(
            async_db, provider, model, as_of=as_of, organization_id=organization.id
        )
        return resolved.source if resolved is not None else None

    assert await source_of("openai", "gpt-4o") == "organization"
    assert await source_of("openai", "gpt-4o-mini") == "deployment"
    assert await source_of("anthropic", "claude-sonnet-4") == "defaults"
    assert await source_of("openai", "nonexistent-model") is None
