"""The stored models.dev catalog: the one download each interval, and every worker serving it."""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import gateway.services.pricing_refresh_service as refresh
from gateway.core.config import GatewayConfig
from gateway.models.pricing import PricingSnapshot, PricingSnapshotHistory
from gateway.services import model_catalog_service as mcs
from gateway.types.models_dev import ModelsDevCatalog, catalog_document, parse_catalog

Sessions = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def _catalog(description: str = "A model.", input_rate: float = 1) -> ModelsDevCatalog:
    model = {"description": description, "cost": {"input": input_rate, "output": 2}, "limit": {"context": 8}}
    return parse_catalog({"test": {"id": "test", "models": {"model": model}}})


@pytest_asyncio.fixture
async def sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Sessions]:
    """Sessions on one SQLite file, which the refresher's own sessions open too."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'catalog.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(PricingSnapshot.__table__.create)  # type: ignore[attr-defined]
        await conn.run_sync(PricingSnapshotHistory.__table__.create)  # type: ignore[attr-defined]
    factory = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(refresh, "create_session", factory)
    refresh.reset_catalog_refresh_state()
    yield factory
    refresh.reset_catalog_refresh_state()
    await engine.dispose()


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What models.dev serves: a catalog, or the error a download raises. ``downloads`` counts the downloads."""
    state: dict[str, Any] = {"catalog": _catalog(), "error": None, "downloads": 0}

    async def download() -> ModelsDevCatalog:
        state["downloads"] += 1
        if state["error"] is not None:
            raise state["error"]
        catalog: ModelsDevCatalog = state["catalog"]
        return catalog

    monkeypatch.setattr(mcs, "_download", download)
    return state


def _new_worker() -> None:
    """Forget what this process serves, as a sibling worker or a restart has nothing in memory."""
    refresh.reset_catalog_refresh_state()


async def _row(sessions: Sessions, source: str) -> PricingSnapshot | None:
    async with sessions() as session:
        row: PricingSnapshot | None = await session.get(PricingSnapshot, source)
        return row


CONFIG = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=86400)


async def _a_day_later(sessions: Sessions) -> None:
    """Age the download claim and the served catalog past the interval, so the next tick downloads."""
    two_days_ago = datetime.now(UTC) - timedelta(days=2)
    async with sessions() as session:
        claim = await session.get(PricingSnapshot, refresh.MODELS_DEV_CATALOG_CLAIM_SOURCE)
        assert claim is not None
        claim.updated_at = two_days_ago
        await session.commit()
    mcs._cache.fetched_at = two_days_ago


@pytest.mark.asyncio
async def test_one_worker_downloads_and_every_worker_serves_the_stored_catalog(
    sessions: Sessions, upstream: dict[str, Any]
) -> None:
    assert await refresh.download_models_dev_catalog_if_due(CONFIG)
    # A sibling worker claims nothing within the interval, so it does not download.
    _new_worker()
    assert not await refresh.download_models_dev_catalog_if_due(CONFIG)
    assert upstream["downloads"] == 1

    async with sessions() as session:
        assert await refresh.load_stored_models_dev_catalog(session)
        # Read once: the row has not changed since.
        assert not await refresh.load_stored_models_dev_catalog(session)
    assert mcs.cached_models_dev_catalog() == upstream["catalog"]
    row = await _row(sessions, refresh.MODELS_DEV_CATALOG_SOURCE)
    assert row is not None
    assert json.loads(row.snapshot) == json.loads(json.dumps(catalog_document(upstream["catalog"])))


@pytest.mark.asyncio
async def test_after_a_restart_with_no_network_the_stored_catalog_serves(
    sessions: Sessions, upstream: dict[str, Any]
) -> None:
    await refresh.download_models_dev_catalog_if_due(CONFIG)
    _new_worker()
    upstream["error"] = mcs.ModelsDevFetchError("unreachable")

    async with sessions() as session:
        await refresh.load_stored_models_dev_catalog(session)

    assert mcs.cached_models_dev_catalog() == upstream["catalog"]
    assert await mcs.load_models_dev_catalog(CONFIG, serve_stale=True) == upstream["catalog"]


@pytest.mark.asyncio
async def test_an_unchanged_download_does_not_write_the_row_again(sessions: Sessions, upstream: dict[str, Any]) -> None:
    async with sessions() as session:
        assert await refresh.store_models_dev_catalog(session, upstream["catalog"])
        assert not await refresh.store_models_dev_catalog(session, _catalog())
        assert await refresh.store_models_dev_catalog(session, _catalog("A better model."))


@pytest.mark.asyncio
async def test_a_failed_download_keeps_the_stored_catalog_and_retries_within_a_minute(
    sessions: Sessions, upstream: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    await refresh.download_models_dev_catalog_if_due(CONFIG)
    stored = await _row(sessions, refresh.MODELS_DEV_CATALOG_SOURCE)
    assert stored is not None
    await _a_day_later(sessions)
    upstream["error"] = mcs.ModelsDevFetchError("models.dev v2 catalog fetch failed: unreachable")

    assert not await refresh.download_models_dev_catalog_if_due(CONFIG)

    after = await _row(sessions, refresh.MODELS_DEV_CATALOG_SOURCE)
    assert after is not None and after.snapshot == stored.snapshot
    assert "keeping the stored copy" in caplog.text
    retried = await _row(sessions, refresh.MODELS_DEV_CATALOG_CLAIM_SOURCE)
    assert retried is not None
    due_in = retried.updated_at.replace(tzinfo=UTC) + timedelta(seconds=mcs.catalog_refresh_interval(CONFIG))
    assert timedelta(0) < due_in - datetime.now(UTC) <= timedelta(seconds=refresh.CATALOG_RETRY_SECONDS)


@pytest.mark.asyncio
async def test_a_stored_catalog_this_version_cannot_read_is_skipped_once(
    sessions: Sessions, caplog: pytest.LogCaptureFixture
) -> None:
    mcs.serve_catalog(_catalog(), datetime.now(UTC))
    async with sessions() as session:
        session.add(PricingSnapshot(source=refresh.MODELS_DEV_CATALOG_SOURCE, snapshot='{"_meta": {"format": 99}}'))
        await session.commit()
        assert not await refresh.load_stored_models_dev_catalog(session)
        assert not await refresh.load_stored_models_dev_catalog(session)

    assert caplog.text.count("Ignoring the stored models.dev catalog") == 1
    assert mcs.cached_models_dev_catalog() == _catalog()


@pytest.mark.asyncio
async def test_a_fact_change_applies_without_review_and_leaves_the_prices_alone(
    sessions: Sessions, upstream: dict[str, Any]
) -> None:
    await refresh.download_models_dev_catalog_if_due(CONFIG)
    await _a_day_later(sessions)
    upstream["catalog"] = _catalog("A better model.", input_rate=5)

    assert await refresh.download_models_dev_catalog_if_due(CONFIG)

    _new_worker()
    async with sessions() as session:
        await refresh.load_stored_models_dev_catalog(session)
        sources = set((await session.execute(select(PricingSnapshot.source))).scalars())
    served = mcs.cached_models_dev_catalog()
    assert served is not None
    model = served.model_for("test", "model")
    assert model is not None and model.description == "A better model."
    # No price row was written or held for review: the new price waits for a price refresh.
    assert sources == {refresh.MODELS_DEV_CATALOG_SOURCE, refresh.MODELS_DEV_CATALOG_CLAIM_SOURCE}


@pytest.mark.asyncio
async def test_without_models_dev_metadata_the_refresher_serves_the_stored_catalog_and_downloads_nothing(
    sessions: Sessions, upstream: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    async with sessions() as session:
        await refresh.store_models_dev_catalog(session, upstream["catalog"])
    _new_worker()

    async def stop(_seconds: float) -> None:
        raise _Stop

    monkeypatch.setattr("gateway.services.pricing_refresh_service.asyncio.sleep", stop)
    with pytest.raises(_Stop):
        await refresh.run_catalog_refresher(GatewayConfig(models_dev_metadata=False))

    assert upstream["downloads"] == 0
    assert mcs.cached_models_dev_catalog() == upstream["catalog"]


class _Stop(Exception):
    """Ends the refresher loop after one tick."""


@pytest.mark.asyncio
async def test_the_price_check_on_another_worker_reuses_the_stored_download(
    sessions: Sessions, upstream: dict[str, Any]
) -> None:
    await refresh.download_models_dev_catalog_if_due(CONFIG)
    _new_worker()
    async with sessions() as session:
        await refresh.load_stored_models_dev_catalog(session)
        preview = await refresh.prepare_price_refresh(session, CONFIG, reuse_within=3600)

    assert upstream["downloads"] == 1
    assert preview.added_count == 1


@pytest.mark.asyncio
async def test_a_tick_with_no_download_due_writes_nothing(
    sessions: Sessions, upstream: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    await refresh.download_models_dev_catalog_if_due(CONFIG)
    claims: list[str] = []

    async def claim(_session: AsyncSession, _interval: float, source: str) -> bool:
        claims.append(source)
        return False

    monkeypatch.setattr(refresh, "claim_poll_tick", claim)

    assert not await refresh.download_models_dev_catalog_if_due(CONFIG)
    assert claims == []


@pytest.mark.asyncio
async def test_a_database_error_in_a_tick_is_logged_without_its_statement(
    sessions: Sessions, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def fail(_session: AsyncSession) -> bool:
        raise OperationalError(
            "UPDATE pricing_snapshots SET snapshot=%(snapshot)s", {"snapshot": "the-payload"}, Exception("closed")
        )

    async def stop(_seconds: float) -> None:
        raise _Stop

    monkeypatch.setattr(refresh, "load_stored_models_dev_catalog", fail)
    monkeypatch.setattr("gateway.services.pricing_refresh_service.asyncio.sleep", stop)
    with pytest.raises(_Stop):
        await refresh.run_catalog_refresher(CONFIG)

    assert "models.dev catalog refresh failed (OperationalError" in caplog.text
    assert "the-payload" not in caplog.text
    assert "UPDATE pricing_snapshots" not in caplog.text
