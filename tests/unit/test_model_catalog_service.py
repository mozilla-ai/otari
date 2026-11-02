"""Unit tests for the models.dev catalog loader and the metadata it serves (no network)."""

import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from gateway.core.config import GatewayConfig
from gateway.services import model_catalog_service as mcs
from gateway.services.catalog import load_catalog_metadata, metadata_entry
from gateway.services.model_catalog_service import ModelCatalogEntry, build_metadata_map
from gateway.types.models_dev import ModelsDevCatalog, parse_catalog


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    mcs.clear_catalog_cache()


def _config(providers: dict[str, dict[str, Any]]) -> GatewayConfig:
    return GatewayConfig(master_key="test", providers=providers)


GPT_4O = {
    "id": "gpt-4o",
    "name": "GPT-4o",
    "description": "General reasoning model.",
    "family": "gpt",
    "canonical_model_id": "openai/gpt-4o",
    "reasoning": False,
    "tool_call": True,
    "structured_output": True,
    "attachment": True,
    "temperature": True,
    "knowledge": "2023-09",
    "release_date": "2024-05-13",
    "last_updated": "2024-08-06",
    "modalities": {"input": ["text", "image", "pdf"], "output": ["text"]},
    "open_weights": False,
    "limit": {"context": 128000, "output": 16384},
    "cost": {"input": 2.5, "output": 10, "cache_read": 1.25},
}

CATALOG = parse_catalog({"openai": {"id": "openai", "name": "OpenAI", "models": {"gpt-4o": GPT_4O}}})
EMPTY = parse_catalog({"openai": {}})


def _entry(model: dict[str, Any]) -> ModelCatalogEntry:
    return ModelCatalogEntry.from_model(parse_catalog({"p": {"models": {"m": model}}}).providers["p"].models["m"])


def test_the_entry_reads_all_fields() -> None:
    entry = _entry(GPT_4O)
    assert entry.name == "GPT-4o"
    assert entry.canonical_model_id == "openai/gpt-4o"
    assert entry.input_modalities == ["text", "image", "pdf"]
    assert entry.output_modalities == ["text"]
    assert entry.tool_call is True
    assert entry.reasoning is False
    assert entry.structured_output is True
    assert entry.context_window == 128000
    assert entry.max_output_tokens == 16384
    assert entry.knowledge_cutoff == "2023-09"
    assert entry.release_date == "2024-05-13"
    assert entry.cost_input == 2.5
    assert entry.cost_output == 10.0
    assert entry.deprecated is False


def test_the_entry_tolerates_missing_fields() -> None:
    entry = _entry({"id": "bare"})
    assert entry.name is None
    assert entry.canonical_model_id is None
    assert entry.input_modalities == []
    assert entry.context_window is None
    assert entry.tool_call is False
    assert entry.deprecated is False
    assert (entry.cost_input, entry.cost_output) == (None, None)


def test_the_entry_reads_a_zero_or_negative_limit_as_unknown() -> None:
    """models.dev writes 0 where it does not know a limit; a client must not read that as no tokens."""
    entry = _entry({"limit": {"context": 0, "output": -1}})

    assert (entry.context_window, entry.max_output_tokens) == (None, None)


def test_the_entry_marks_deprecated_status() -> None:
    assert _entry({"id": "old", "status": "deprecated"}).deprecated is True


def test_build_metadata_map_keys_by_instance() -> None:
    config = _config({"openai": {"api_key": "sk-x"}})
    result = build_metadata_map(config, CATALOG)
    assert set(result) == {"openai:gpt-4o"}
    assert result["openai:gpt-4o"].tool_call is True


def test_build_metadata_map_uses_provider_type_for_named_instance() -> None:
    # A named instance backed by openai joins to openai's models under its own key.
    config = _config({"my-oai": {"provider_type": "openai", "api_key": "sk-x"}})
    result = build_metadata_map(config, CATALOG)
    assert set(result) == {"my-oai:gpt-4o"}


def test_build_metadata_map_skips_unknown_providers() -> None:
    config = _config({"mystery": {"api_key": "x"}})
    assert build_metadata_map(config, CATALOG) == {}


def test_build_metadata_map_empty_when_catalog_missing() -> None:
    config = _config({"openai": {"api_key": "sk-x"}})
    assert build_metadata_map(config, None) == {}


@pytest.mark.asyncio
async def test_stale_catalog_is_served_without_refetching() -> None:
    """A dashboard read must not pay the 15s models.dev fetch timeout."""
    mcs.clear_catalog_cache()
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=86400)

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=EMPTY)) as fetch:
        await mcs.load_models_dev_catalog(config, force=True)
        assert fetch.await_count == 1
        # Age the entry past its TTL; a stale read still answers from cache.
        mcs._cache.at = time.monotonic() - 200_000
        result = await mcs.load_models_dev_catalog(config, serve_stale=True)
        assert fetch.await_count == 1

    assert result is EMPTY


@pytest.mark.asyncio
async def test_stale_read_without_a_cache_entry_still_fetches() -> None:
    """A cold worker fetches rather than reporting metadata unavailable."""
    mcs.clear_catalog_cache()
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=86400)

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=EMPTY)) as fetch:
        result = await mcs.load_models_dev_catalog(config, serve_stale=True)

    assert fetch.await_count == 1
    assert result is EMPTY


@pytest.mark.asyncio
async def test_stale_read_retries_a_failed_fetch_on_the_negative_ttl() -> None:
    """A failed fetch must not be pinned for the refresher's whole interval.

    The refresh cadence is the *success* cadence (``models_dev_cache_ttl_seconds``,
    a day by default). Serving a failure at any age would leave the dashboard
    without enrichment until the next tick, and ``/api/v1/models/metadata`` has no
    ``refresh`` flag to escape it. The 60s negative TTL still governs a failure.
    """
    mcs.clear_catalog_cache()
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=86400)

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=None)) as failing:
        assert await mcs.load_models_dev_catalog(config, force=True) is None
        assert failing.await_count == 1

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=EMPTY)) as recovered:
        # Inside the negative TTL the failure is still served, as before.
        assert await mcs.load_models_dev_catalog(config, serve_stale=True) is None
        assert recovered.await_count == 0
        # Past it, a read retries rather than waiting out the refresh interval.
        mcs._cache.at = time.monotonic() - (mcs._NEGATIVE_TTL_SECONDS + 1)
        assert await mcs.load_models_dev_catalog(config, serve_stale=True) is EMPTY
        assert recovered.await_count == 1


@pytest.mark.asyncio
async def test_a_failed_fetch_keeps_the_last_good_catalog() -> None:
    """A models.dev outage must not take the context windows and capabilities away with it."""
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=86400)
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
        await mcs.load_models_dev_catalog(config, force=True)

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=None)) as failing:
        assert await mcs.load_models_dev_catalog(config, force=True) is CATALOG
        assert await mcs.load_models_dev_catalog(config, serve_stale=True) is CATALOG
        assert mcs.cached_models_dev_catalog(config) is CATALOG
        assert failing.await_count == 1
        # The failure still governs the retry: past the negative TTL a read fetches again.
        mcs._cache.at = time.monotonic() - (mcs._NEGATIVE_TTL_SECONDS + 1)
        assert await mcs.load_models_dev_catalog(config, serve_stale=True) is CATALOG
        assert failing.await_count == 2


def test_background_catalog_requires_caching_and_enrichment() -> None:
    """Both existing knobs still mean what they say; there is no third one."""
    assert mcs.background_catalog_enabled(GatewayConfig(models_dev_metadata=False)) is False
    assert (
        mcs.background_catalog_enabled(GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=0)) is False
    )
    assert (
        mcs.background_catalog_enabled(GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=60)) is True
    )


def test_metadata_entry_reads_a_gemini_listing_id_as_the_bare_one() -> None:
    """Gemini lists ``models/<id>``; models.dev keys the bare id under ``google``."""
    catalog = parse_catalog(
        {"google": {"models": {"gemini-3.5-flash-lite": {"limit": {"context": 1048576, "output": 65536}}}}}
    )

    entry = metadata_entry(catalog, "gemini", "models/gemini-3.5-flash-lite")

    assert entry is not None
    assert (entry.context_window, entry.max_output_tokens) == (1048576, 65536)
    assert metadata_entry(catalog, "gemini", "gemini-3.5-flash-lite") is not None
    assert metadata_entry(catalog, "gemini", "models/gemini-9") is None
    assert metadata_entry(None, "gemini", "gemini-3.5-flash-lite") is None


@pytest.mark.parametrize(
    "catalog",
    [
        pytest.param({"openai": ["gpt-4o"]}, id="provider-not-an-object"),
        pytest.param({"openai": {"models": ["gpt-4o"]}}, id="models-not-an-object"),
        pytest.param({"openai": {}}, id="no-models"),
        pytest.param({"openai": {"models": {"gpt-4o": "GPT-4o"}}}, id="model-not-an-object"),
    ],
)
def test_metadata_entry_reads_a_malformed_catalog_as_no_entry(catalog: dict[str, Any]) -> None:
    """models.dev is fetched JSON, so a shape it was not expected to have describes no model rather than raising."""
    assert metadata_entry(parse_catalog(catalog), "openai", "gpt-4o") is None


@pytest.mark.asyncio
async def test_catalog_metadata_fetches_only_when_a_read_may() -> None:
    """A request-time read fetches on a cold cache; a read off the request path never does."""
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=60)

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)) as fetch:
        assert await load_catalog_metadata(config, cached_only=True) is None
        fetch.assert_not_called()

        assert await load_catalog_metadata(config) is CATALOG
        fetch.assert_awaited_once()

        assert await load_catalog_metadata(config, cached_only=True) is CATALOG
    assert await load_catalog_metadata(GatewayConfig(models_dev_metadata=False)) is None


@pytest.mark.asyncio
async def test_the_download_reads_the_body_into_a_catalog() -> None:
    """The one parser runs on the way in, so every reader gets the typed catalog and never the raw JSON."""

    class _Response:
        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self) -> Any:
            yield b'{"openai": {"id": "openai", "models": {"gpt-4o": {"name": "GPT-4o"}}}}'

        async def __aenter__(self) -> "_Response":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def stream(self, *args: Any, **kwargs: Any) -> _Response:
            return _Response()

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    with patch("httpx.AsyncClient", _Client):
        catalog = await mcs._download()

    assert isinstance(catalog, ModelsDevCatalog)
    assert catalog.providers["openai"].models["gpt-4o"].name == "GPT-4o"


@pytest.mark.asyncio
async def test_the_shared_fetch_fills_the_metadata_cache_and_is_reused() -> None:
    mcs.clear_catalog_cache()
    with patch.object(mcs, "_download", new=AsyncMock(return_value=EMPTY)) as download:
        assert await mcs.fetch_models_dev_document(fill_cache=True) is EMPTY
        assert await mcs.fetch_models_dev_document(reuse_within=60) is EMPTY
        assert download.await_count == 1

        # A catalog refresher tick right after reuses it too, instead of fetching again.
        config = GatewayConfig(master_key="k", models_dev_metadata=True)
        assert await mcs.load_models_dev_catalog(config, force=True, reuse_within=60) is EMPTY
        assert download.await_count == 1
    mcs.clear_catalog_cache()


@pytest.mark.asyncio
async def test_the_shared_fetch_leaves_the_cache_alone_unless_asked_and_raises_on_failure() -> None:
    mcs.clear_catalog_cache()
    with patch.object(mcs, "_download", new=AsyncMock(return_value=EMPTY)):
        await mcs.fetch_models_dev_document()
    assert mcs._cache.ok is False

    with (
        patch.object(mcs, "_download", new=AsyncMock(side_effect=mcs.ModelsDevFetchError("down"))),
        pytest.raises(mcs.ModelsDevFetchError),
    ):
        await mcs.fetch_models_dev_document()
    mcs.clear_catalog_cache()
