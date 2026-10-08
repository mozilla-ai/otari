"""The scheduled models.dev price check: what each policy does with what it fetches."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

import gateway.services.pricing_refresh_service as refresh
from gateway.core.config import GatewayConfig
from gateway.services.pricing_refresh_service import PricingRefreshPreview


def _preview(changed: int, *, needs_review: bool = False) -> PricingRefreshPreview:
    return PricingRefreshPreview(
        fetched_at=datetime.now(UTC),
        added_count=0,
        changed_count=changed,
        removed_count=0,
        changes=[],
        changes_truncated=False,
        digest="abc",
        needs_review=needs_review,
        review_reason="priced models fall from 10 to 1" if needs_review else None,
    )


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    prepare = AsyncMock(return_value=_preview(changed=2))
    confirm = AsyncMock(return_value=True)
    reject = AsyncMock(return_value=True)
    monkeypatch.setattr(refresh, "prepare_price_refresh", prepare)
    monkeypatch.setattr(refresh, "confirm_price_refresh", confirm)
    monkeypatch.setattr(refresh, "reject_price_refresh", reject)
    return {"prepare": prepare, "confirm": confirm, "reject": reject}


@pytest.mark.asyncio
async def test_review_holds_an_update_for_an_operator(stubs: dict[str, AsyncMock]) -> None:
    session = AsyncMock(spec=AsyncSession)

    assert await refresh.poll_price_updates(session, "review") == "pending"

    stubs["prepare"].assert_awaited_once_with(session, None, reuse_within=0.0)
    stubs["confirm"].assert_not_awaited()
    stubs["reject"].assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_applies_an_update_and_says_who_did(stubs: dict[str, AsyncMock]) -> None:
    session = AsyncMock(spec=AsyncSession)

    assert await refresh.poll_price_updates(session, "auto") == "applied"

    stubs["confirm"].assert_awaited_once_with(session, accepted_by="schedule", digest="abc")


@pytest.mark.asyncio
async def test_auto_leaves_an_implausible_update_pending(stubs: dict[str, AsyncMock]) -> None:
    stubs["prepare"].return_value = _preview(changed=2, needs_review=True)
    session = AsyncMock(spec=AsyncSession)

    assert await refresh.poll_price_updates(session, "auto") == "pending"

    stubs["confirm"].assert_not_awaited()
    stubs["reject"].assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unchanged_fetch_leaves_nothing_pending(stubs: dict[str, AsyncMock]) -> None:
    # Under either policy: a pending row equal to the active one would make the
    # dashboard offer a review with nothing in it.
    stubs["prepare"].return_value = _preview(changed=0)
    session = AsyncMock(spec=AsyncSession)

    assert await refresh.poll_price_updates(session, "auto") == "unchanged"

    stubs["reject"].assert_awaited_once_with(session, digest="abc")
    stubs["confirm"].assert_not_awaited()


def test_the_policy_is_validated_at_load() -> None:
    with pytest.raises(ValueError):
        GatewayConfig(master_key="k", pricing_refresh="nightly")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GatewayConfig(master_key="k", pricing_refresh_interval_seconds=10)
    assert GatewayConfig(master_key="k").pricing_refresh == "manual"
    assert GatewayConfig(master_key="k").public_catalog is False
    assert GatewayConfig(master_key="k").public_catalog_rate_limit_per_minute == 60
    with pytest.raises(ValueError):
        GatewayConfig(master_key="k", public_catalog_rate_limit_per_minute=0)


@pytest.mark.parametrize("name", ["otari", "hosted"])
def test_a_reserved_instance_name_is_refused(name: str) -> None:
    # The instance check runs from the config loader, after the model is built.
    config = GatewayConfig(master_key="k", providers={name: {"api_key": "x"}})
    with pytest.raises(ValueError, match="reserved"):
        config.validate_provider_instances()
