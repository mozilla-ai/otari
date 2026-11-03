"""A request prices each model it bills once: admission, a top-up and settlement share it."""

import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.repositories.pricing import OrganizationModelPricingRepository
from gateway.services import pricing_service
from gateway.services.pricing_service import RequestPrices


@pytest.mark.asyncio
async def test_a_model_is_resolved_once_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated pricing of the same provider/model reuses one organization-scoped lookup."""
    resolved: Any = SimpleNamespace(pricing=object())
    lookup = AsyncMock(return_value=resolved)
    monkeypatch.setattr(pricing_service, "resolve_model_pricing", lookup)
    organization = uuid.uuid4()
    prices = RequestPrices(organization, Mock(spec=OrganizationModelPricingRepository))
    db: Any = object()

    assert await prices.resolve(db, "openai", "gpt-5") is resolved
    assert await prices.resolve(db, "openai", "gpt-5") is resolved

    lookup.assert_awaited_once_with(db, "openai", "gpt-5", organization_id=organization, deployment_paid=False)


@pytest.mark.asyncio
async def test_each_model_is_resolved_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct provider/model pairs retain separate cached pricing results."""
    by_model: dict[tuple[str, str], Any] = {
        ("openai", "gpt-5"): SimpleNamespace(pricing=object()),
        ("together", "llama"): SimpleNamespace(pricing=object()),
    }
    lookup = AsyncMock(side_effect=lambda _db, provider, model, **_: by_model[(provider, model)])
    monkeypatch.setattr(pricing_service, "resolve_model_pricing", lookup)
    prices = RequestPrices(None, Mock(spec=OrganizationModelPricingRepository))

    assert await prices.resolve(None, "openai", "gpt-5") is by_model[("openai", "gpt-5")]  # type: ignore[arg-type]
    assert await prices.resolve(None, "together", "llama") is by_model[("together", "llama")]  # type: ignore[arg-type]
    assert lookup.await_count == 2


@pytest.mark.asyncio
async def test_an_unpriced_model_is_remembered_as_unpriced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing price is cached so later phases do not repeat the lookup."""
    lookup = AsyncMock(return_value=None)
    monkeypatch.setattr(pricing_service, "resolve_model_pricing", lookup)
    prices = RequestPrices(None, Mock(spec=OrganizationModelPricingRepository))

    assert await prices.resolve(None, "local", "free") is None  # type: ignore[arg-type]
    assert await prices.resolve(None, "local", "free") is None  # type: ignore[arg-type]
    assert lookup.await_count == 1


@pytest.mark.parametrize(
    ("unit", "missing"),
    [("token", False), ("request", False), ("image", False), ("second", True), ("minute", True)],
)
def test_a_rate_in_a_unit_with_no_completion_meter_is_pricing_required(unit: str, *, missing: bool) -> None:
    from decimal import Decimal

    from gateway.models.pricing import ModelPricing

    rate = ModelPricing(
        model_key="openai:gpt-4o", input_price_per_million=Decimal(1), output_price_per_million=Decimal(1), unit=unit
    )

    assert pricing_service.pricing_required_but_missing(rate, require_pricing=True) is missing
    assert not pricing_service.pricing_required_but_missing(rate, require_pricing=False)


def test_no_rate_is_pricing_required_only_when_pricing_is_required() -> None:
    assert pricing_service.pricing_required_but_missing(None, require_pricing=True)
    assert not pricing_service.pricing_required_but_missing(None, require_pricing=False)


def test_the_refusal_names_the_unit_no_completion_meter_reads() -> None:
    detail = pricing_service.no_pricing_error_detail("openai:gpt-4o", unit="second")

    assert detail.startswith("Model 'openai:gpt-4o' is priced per second, which no completion meter reads yet")
    assert "No pricing is configured" not in detail


@pytest.mark.asyncio
async def test_a_candidate_in_a_unit_with_no_completion_meter_estimates_as_unpriced() -> None:
    """Its estimate would be nothing, and nothing ranks cheapest; ``None`` keeps it behind every priced candidate."""
    from decimal import Decimal

    from gateway.models.pricing import ModelPricing
    from gateway.services.pricing_service import ResolvedPricing

    prices = RequestPrices(None, Mock(spec=OrganizationModelPricingRepository))
    tokens = ModelPricing(model_key="openai:a", input_price_per_million=Decimal(1), output_price_per_million=Decimal(1))
    seconds = ModelPricing(
        model_key="openai:b", input_price_per_million=Decimal(1), output_price_per_million=Decimal(1), unit="second"
    )
    prices._resolved[("openai", "a", False)] = ResolvedPricing(tokens, "deployment", "openai:a")  # noqa: SLF001
    prices._resolved[("openai", "b", False)] = ResolvedPricing(seconds, "deployment", "openai:b")  # noqa: SLF001
    prices._resolved[("openai", "c", False)] = None  # noqa: SLF001

    costs = await prices.estimate_candidates([("openai", "a"), ("openai", "b"), ("openai", "c")], lambda _: Decimal(2))

    assert costs == {"openai:a": Decimal(2), "openai:b": None, "openai:c": None}
