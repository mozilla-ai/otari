"""Tests for explicit models.dev price snapshot refreshes."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import gateway.services.pricing_refresh_service as refresh
from gateway.models.pricing import PricingSnapshot, PricingSnapshotHistory
from gateway.services.pricing import active_generations, bundled_generation, current_index


def _catalog(input_rate: float = 1, *, extra: bool = False) -> dict[str, Any]:
    models: dict[str, Any] = {
        "model": {"id": "model", "name": "Model", "cost": {"input": input_rate, "output": 2}, "limit": {"context": 8}}
    }
    if extra:
        models["other"] = {"id": "other", "cost": {"input": 3, "output": 4}}
    return {"test": {"id": "test", "name": "Test", "doc": "https://example.test", "models": models}}


def _raw(input_rate: float = 1, *, extra: bool = False) -> str:
    return json.dumps(_catalog(input_rate, extra=extra))


@pytest_asyncio.fixture
async def session(tmp_path: Path) -> AsyncIterator[AsyncSession]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'prices.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(PricingSnapshot.__table__.create)  # type: ignore[attr-defined]
        await conn.run_sync(PricingSnapshotHistory.__table__.create)  # type: ignore[attr-defined]
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        yield db
    await engine.dispose()


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The catalog the fetch returns, and the arguments it was called with."""
    state: dict[str, Any] = {"catalog": _catalog(), "calls": []}

    async def fetch(**kwargs: Any) -> dict[str, Any]:
        state["calls"].append(kwargs)
        catalog: dict[str, Any] = state["catalog"]
        return catalog

    monkeypatch.setattr(refresh, "fetch_models_dev_document", fetch)
    return state


def test_parse_snapshot_refuses_what_is_not_a_catalog() -> None:
    for raw in ("not json", "[]", "{}", '{"p": {"id": "p", "models": {}}}'):
        with pytest.raises(ValueError, match="Invalid models.dev snapshot"):
            refresh._parse_snapshot(raw)


@pytest.mark.asyncio
async def test_a_fetch_is_held_for_review_and_confirmation_activates_it(
    session: AsyncSession, upstream: dict[str, Any]
) -> None:
    bundled = current_index()
    preview = await refresh.prepare_price_refresh(session)

    assert preview.removed_count > 0
    assert preview.added_count == 1
    assert len(preview.changes) == 100
    assert preview.changes_truncated
    assert current_index() is bundled
    pending = await session.get(PricingSnapshot, refresh.MODELS_DEV_PENDING_SOURCE)
    assert pending is not None
    assert json.loads(pending.snapshot)["test"]["models"]["model"]["limit"] == {"context": 8}

    assert await refresh.confirm_price_refresh(session) is True

    assert current_index().get("test", "model") is not None
    assert len(active_generations()) == 1
    assert await session.get(PricingSnapshot, refresh.MODELS_DEV_PENDING_SOURCE) is None
    active = await session.get(PricingSnapshot, refresh.MODELS_DEV_SOURCE)
    assert active is not None and active.snapshot == pending.snapshot
    history = (await session.execute(select(PricingSnapshotHistory))).scalars().all()
    assert [(row.source, row.accepted_by, row.model_count) for row in history] == [("models.dev", "operator", 1)]
    assert await refresh.confirm_price_refresh(session) is False


@pytest.mark.asyncio
async def test_the_stored_snapshot_is_trimmed_to_the_price_fields(
    session: AsyncSession, upstream: dict[str, Any]
) -> None:
    upstream["catalog"]["test"]["models"]["model"].update(
        description="long text", modalities={"input": ["text"]}, experimental={"modes": {}}
    )
    await refresh.prepare_price_refresh(session)

    pending = await session.get(PricingSnapshot, refresh.MODELS_DEV_PENDING_SOURCE)
    assert pending is not None
    assert set(json.loads(pending.snapshot)["test"]["models"]["model"]) == {"name", "cost", "limit"}


@pytest.mark.asyncio
async def test_the_preview_reports_changed_rates_against_the_active_snapshot(
    session: AsyncSession, upstream: dict[str, Any]
) -> None:
    await refresh.prepare_price_refresh(session)
    await refresh.confirm_price_refresh(session)

    upstream["catalog"] = _catalog(input_rate=9, extra=True)
    preview = await refresh.prepare_price_refresh(session)

    assert (preview.added_count, preview.changed_count, preview.removed_count) == (1, 1, 0)
    assert [(c.model_key, c.change) for c in preview.changes] == [("test:model", "changed"), ("test:other", "added")]
    assert not preview.changes_truncated

    pending = await refresh.preview_pending_refresh(session)
    assert pending is not None and pending.changed_count == 1
    assert await refresh.reject_price_refresh(session) is True
    assert await refresh.preview_pending_refresh(session) is None


@pytest.mark.asyncio
async def test_a_fetch_failure_is_a_refresh_error(session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(**_: Any) -> dict[str, Any]:
        raise RuntimeError("offline")

    monkeypatch.setattr(refresh, "fetch_models_dev_document", fail)
    with pytest.raises(refresh.PricingRefreshError):
        await refresh.prepare_price_refresh(session)


@pytest.mark.asyncio
async def test_a_catalog_without_models_is_not_stored(session: AsyncSession, upstream: dict[str, Any]) -> None:
    upstream["catalog"] = {"p": {"id": "p", "models": {}}}
    with pytest.raises(refresh.PricingRefreshError):
        await refresh.prepare_price_refresh(session)
    assert await session.get(PricingSnapshot, refresh.MODELS_DEV_PENDING_SOURCE) is None


@pytest.mark.asyncio
async def test_a_catalog_pricing_no_model_is_not_stored(session: AsyncSession, upstream: dict[str, Any]) -> None:
    upstream["catalog"] = {"p": {"id": "p", "models": {"m": {"id": "m", "name": "M"}}}}
    with pytest.raises(refresh.PricingRefreshError, match="prices no models"):
        await refresh.prepare_price_refresh(session)
    assert await session.get(PricingSnapshot, refresh.MODELS_DEV_PENDING_SOURCE) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [-1, 2_000_000])
async def test_an_implausible_rate_rejects_the_whole_catalog(
    session: AsyncSession, upstream: dict[str, Any], rate: float
) -> None:
    upstream["catalog"]["test"]["models"]["other"] = {"id": "other", "cost": {"input": 1, "output": rate}}
    with pytest.raises(refresh.PricingRefreshError, match="implausible"):
        await refresh.prepare_price_refresh(session)
    assert await session.get(PricingSnapshot, refresh.MODELS_DEV_PENDING_SOURCE) is None


@pytest.mark.asyncio
async def test_an_implausible_tier_rate_rejects_the_whole_catalog(
    session: AsyncSession, upstream: dict[str, Any]
) -> None:
    upstream["catalog"]["test"]["models"]["model"]["cost"]["tiers"] = [
        {"tier": {"type": "context", "size": 100}, "input": float("inf")}
    ]
    with pytest.raises(refresh.PricingRefreshError):
        await refresh.prepare_price_refresh(session)


@pytest.mark.asyncio
async def test_a_large_drop_is_stored_for_review_and_flagged(session: AsyncSession, upstream: dict[str, Any]) -> None:
    preview = await refresh.prepare_price_refresh(session)

    assert preview.needs_review is True
    assert preview.review_reason is not None and "priced models fall" in preview.review_reason
    assert await session.get(PricingSnapshot, refresh.MODELS_DEV_PENDING_SOURCE) is not None
    assert await refresh.confirm_price_refresh(session) is True


@pytest.mark.asyncio
async def test_a_small_change_needs_no_review(session: AsyncSession, upstream: dict[str, Any]) -> None:
    await refresh.prepare_price_refresh(session)
    await refresh.confirm_price_refresh(session)
    upstream["catalog"] = _catalog(input_rate=9, extra=True)

    preview = await refresh.prepare_price_refresh(session)

    assert preview.needs_review is False
    assert preview.review_reason is None


@pytest.mark.asyncio
async def test_failed_persistence_keeps_the_active_generations(
    session: AsyncSession, upstream: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    await refresh.prepare_price_refresh(session)
    bundled = active_generations()

    async def broken_commit() -> None:
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(session, "commit", broken_commit)
    with pytest.raises(refresh.PricingRefreshError):
        await refresh.confirm_price_refresh(session)

    assert active_generations() == bundled


@pytest.mark.asyncio
async def test_startup_restores_the_accepted_history(session: AsyncSession, upstream: dict[str, Any]) -> None:
    await refresh.prepare_price_refresh(session)
    await refresh.confirm_price_refresh(session)
    upstream["catalog"] = _catalog(input_rate=5)
    await refresh.prepare_price_refresh(session)
    await refresh.confirm_price_refresh(session)

    refresh.reset_price_refresh_state()
    assert active_generations() == (bundled_generation(),)
    await refresh.load_persisted_price_snapshot(session)

    generations = active_generations()
    assert len(generations) == 2
    assert generations[0].effective_at < generations[1].effective_at
    old, new = (g.index.get("test", "model") for g in generations)
    assert old is not None and new is not None
    assert (str(old.input), str(new.input)) == ("1", "5")


@pytest.mark.asyncio
async def test_refresher_applies_a_snapshot_accepted_on_another_worker(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A confirm served by a sibling worker propagates here on the next tick, once."""

    session.add(PricingSnapshot(source=refresh.MODELS_DEV_SOURCE, snapshot=_raw()))
    await session.commit()

    await refresh.refresh_price_snapshot(session)

    assert current_index().get("test", "model") is not None
    assert refresh._applied_snapshot_raw == _raw()

    monkeypatch.setattr(
        refresh,
        "_apply_persisted_snapshots",
        lambda *_: pytest.fail("an unchanged snapshot must not be re-applied"),
    )
    await refresh.refresh_price_snapshot(session)


@pytest.mark.asyncio
async def test_an_invalid_persisted_snapshot_leaves_the_bundled_prices(session: AsyncSession) -> None:
    session.add(PricingSnapshot(source=refresh.MODELS_DEV_SOURCE, snapshot="{}"))
    await session.commit()

    await refresh.load_persisted_price_snapshot(session)

    assert active_generations() == (bundled_generation(),)


@pytest.mark.asyncio
async def test_the_history_keeps_only_the_newest_snapshots(
    session: AsyncSession, upstream: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(refresh, "PRICING_SNAPSHOT_HISTORY_KEEP", 2)
    for rate in (1, 2, 3):
        upstream["catalog"] = _catalog(input_rate=rate)
        await refresh.prepare_price_refresh(session)
        await refresh.confirm_price_refresh(session)

    assert len(await refresh.list_accepted_snapshots(session)) == 2


@pytest.mark.asyncio
async def test_a_poll_claim_is_honored_for_one_interval(session: AsyncSession) -> None:
    assert await refresh.claim_poll_tick(session, 3600) is True
    assert await refresh.claim_poll_tick(session, 3600) is False
    row = await session.get(PricingSnapshot, refresh.MODELS_DEV_POLL_CLAIM_SOURCE)
    assert row is not None
