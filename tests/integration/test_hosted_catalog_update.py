"""The catalog update, against a real database: prices first, then every provider's models.

The price half is a fake ``PriceUpdates``, because what it does to the snapshot
is covered by the pricing refresh tests; here it records the order of the calls,
and approving swaps in the community rates the new prices would bring. Model
discovery and the community dataset are stubbed as in the hosted provider tests.
"""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from any_llm.types.model import Model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import build_hosted_provider_service
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.providers_exceptions import HostedCatalogPricesChangedError
from gateway.models.pricing import ModelPricing
from gateway.models.providers import HostedProviderModel
from gateway.schemas.providers import HostedProviderCreateRequest
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.pricing_refresh_service import PendingSnapshotChanged, PricingRefreshError, PricingRefreshPreview
from gateway.services.pricing_service import configure_default_pricing
from gateway.services.providers import HostedCatalogUpdate
from gateway.services.secret_box import generate_secret_key

pytestmark = pytest.mark.asyncio

KEY = "sk-live-hosted-1234"
DIAL = "gateway.services.providers._hosted_provider_service.test_provider_credentials"
DATASET = "gateway.services.pricing_service.default_model_pricing"
DIGEST = "d" * 64


@pytest.fixture(autouse=True)
def _secret_key_and_default_pricing(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    configure_default_pricing(True)
    yield
    configure_default_pricing(False)


def _discovery(monkeypatch: pytest.MonkeyPatch, *models: str) -> None:
    async def _stub(impl_name: str, **_: object) -> ProviderDiscovery:
        return ProviderDiscovery(
            provider=impl_name, models=[Model(id=m, object="model", created=0, owned_by=impl_name) for m in models]
        )

    monkeypatch.setattr(DIAL, _stub)


def _defaults(monkeypatch: pytest.MonkeyPatch, rates: dict[str, tuple[str, str]]) -> None:
    def _stub(
        provider: str | None, model: str, as_of: object, *, implementation: str | None = None
    ) -> ModelPricing | None:
        if model not in rates:
            return None
        input_rate, output_rate = rates[model]
        return ModelPricing(
            model_key=f"{provider}:{model}",
            input_price_per_million=Decimal(input_rate),
            output_price_per_million=Decimal(output_rate),
            pricing_tiers=[],
            unit="tokens",
        )

    monkeypatch.setattr(DATASET, _stub)


def _preview() -> PricingRefreshPreview:
    return PricingRefreshPreview(
        fetched_at=datetime.now(UTC),
        added_count=1,
        changed_count=0,
        removed_count=0,
        changes=[],
        changes_truncated=False,
        digest=DIGEST,
    )


class FakePrices:
    """Records each price step; ``on_approve`` stands in for the snapshot the approval installs."""

    def __init__(self, *, on_approve: Callable[[], None] = lambda: None) -> None:
        self.calls: list[str] = []
        self.on_approve = on_approve
        self.claimed = True
        self.fetch_fails = False
        self.poll_fails = False
        self.approve_raises: Exception | None = None
        self.approve_returns = True

    async def claim(self, interval_seconds: float) -> bool:
        self.calls.append("claim")
        return self.claimed

    async def fetch(self) -> PricingRefreshPreview | None:
        self.calls.append("fetch")
        if self.fetch_fails:
            raise PricingRefreshError("unreachable")
        return _preview()

    async def poll(self, policy: str) -> str:
        self.calls.append(f"poll:{policy}")
        if self.poll_fails:
            raise PricingRefreshError("unreachable")
        if policy == "auto":
            self.on_approve()
            return "applied"
        return "pending"

    async def approve(self, digest: str) -> bool:
        self.calls.append(f"approve:{digest}")
        if self.approve_raises is not None:
            raise self.approve_raises
        if self.approve_returns:
            self.on_approve()
        return self.approve_returns


def _update(db: AsyncSession, prices: FakePrices) -> HostedCatalogUpdate:
    return HostedCatalogUpdate(build_hosted_provider_service(UnitOfWork(db), config=GatewayConfig()), prices)


async def _provider(db: AsyncSession, monkeypatch: pytest.MonkeyPatch, *models: str) -> None:
    _discovery(monkeypatch, *models)
    service = build_hosted_provider_service(UnitOfWork(db), config=GatewayConfig())
    await service.create_provider(HostedProviderCreateRequest(provider="openai", api_key=KEY))


async def _offered(db: AsyncSession) -> dict[str, bool]:
    rows = (await db.execute(select(HostedProviderModel).where(HostedProviderModel.provider == "openai"))).scalars()
    return {row.model: row.enabled for row in rows}


async def test_approving_prices_serves_a_new_model_they_price_in_the_same_update(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case this exists for: a snapshot the old prices did not cover."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices(on_approve=lambda: _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-5-1017": ("3", "12")}))

    outcome = await _update(async_db, prices).apply(digest=DIGEST, cleanup=False)

    assert prices.calls == [f"approve:{DIGEST}"]
    assert outcome.prices_applied is True
    assert [provider.added for provider in outcome.catalog.providers] == [["gpt-5-1017"]]
    assert await _offered(async_db) == {"gpt-4o": True, "gpt-5-1017": True}


async def test_without_a_digest_the_update_refreshes_models_and_changes_no_price(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices()

    outcome = await _update(async_db, prices).apply(digest=None, cleanup=False)

    assert prices.calls == []
    assert outcome.prices_applied is False
    assert await _offered(async_db) == {"gpt-4o": True, "gpt-5-1017": False}


@pytest.mark.parametrize("failure", ["changed", "gone"])
async def test_prices_that_changed_since_the_preview_refuse_the_update_and_touch_no_model(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices()
    if failure == "changed":
        prices.approve_raises = PendingSnapshotChanged("replaced")
    else:
        prices.approve_returns = False

    with pytest.raises(HostedCatalogPricesChangedError):
        await _update(async_db, prices).apply(digest=DIGEST, cleanup=False)

    assert await _offered(async_db) == {"gpt-4o": True}


async def test_the_preview_holds_the_prices_and_lists_new_models_without_offering_them(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices()

    preview = await _update(async_db, prices).preview(cleanup=False)

    assert prices.calls == ["fetch"]
    assert preview.prices is not None
    assert (preview.prices.digest, preview.prices.added_count, preview.price_error) == (DIGEST, 1, None)
    assert [provider.added for provider in preview.catalog.providers] == [["gpt-5-1017"]]
    assert preview.catalog.removed_count == 0
    assert await _offered(async_db) == {"gpt-4o": True}


async def test_a_price_fetch_that_fails_still_previews_the_model_lists(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices()
    prices.fetch_fails = True

    preview = await _update(async_db, prices).preview(cleanup=False)

    assert preview.prices is None
    assert preview.price_error is not None
    assert [provider.added for provider in preview.catalog.providers] == [["gpt-5-1017"]]


async def test_the_cleanup_removes_a_price_nothing_serves_only_when_asked(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    async_db.add(
        ModelPricing(
            model_key="openai:retired",
            effective_at=datetime(2026, 1, 1, tzinfo=UTC),
            input_price_per_million=Decimal("1"),
            output_price_per_million=Decimal("2"),
            pricing_tiers=[],
            unit="tokens",
            origin="config",
        )
    )
    await async_db.commit()
    update = _update(async_db, FakePrices())

    kept = await update.apply(digest=None, cleanup=False)
    preview = await update.preview(cleanup=True)
    removed = await update.apply(digest=None, cleanup=True)

    assert kept.catalog.removed == []
    assert preview.catalog.removed == ["openai:retired"]
    assert removed.catalog.removed == ["openai:retired"]
    retired = select(ModelPricing).where(ModelPricing.model_key == "openai:retired")
    assert (await async_db.execute(retired)).first() is None


async def test_a_scheduled_auto_update_applies_prices_before_it_refreshes_models(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices(on_approve=lambda: _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-5-1017": ("3", "12")}))

    outcome = await _update(async_db, prices).run_scheduled("auto", 3600)

    assert prices.calls == ["claim", "poll:auto"]
    assert outcome == "prices applied; 1 models added; 0 providers not reached"
    assert await _offered(async_db) == {"gpt-4o": True, "gpt-5-1017": True}


async def test_a_scheduled_review_update_holds_the_prices_and_still_refreshes_models(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New models arrive at the prices in force, and one they do not cover waits for the approval."""
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices()

    outcome = await _update(async_db, prices).run_scheduled("review", 3600)

    assert outcome == "prices pending; 1 models added; 0 providers not reached"
    assert await _offered(async_db) == {"gpt-4o": True, "gpt-5-1017": False}


async def test_a_scheduled_update_another_worker_claimed_does_nothing(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices()
    prices.claimed = False

    assert await _update(async_db, prices).run_scheduled("auto", 3600) is None
    assert prices.calls == ["claim"]
    assert await _offered(async_db) == {"gpt-4o": True}


async def test_a_scheduled_update_whose_price_fetch_fails_still_refreshes_models(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    _defaults(monkeypatch, {"gpt-4o": ("2.5", "10"), "gpt-5-1017": ("3", "12")})
    await _provider(async_db, monkeypatch, "gpt-4o")
    _discovery(monkeypatch, "gpt-4o", "gpt-5-1017")
    prices = FakePrices()
    prices.poll_fails = True

    outcome = await _update(async_db, prices).run_scheduled("auto", 3600)

    assert outcome == "prices unreachable; 1 models added; 0 providers not reached"
    assert await _offered(async_db) == {"gpt-4o": True, "gpt-5-1017": True}
