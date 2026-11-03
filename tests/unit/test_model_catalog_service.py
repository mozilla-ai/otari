"""Unit tests for the models.dev catalog loader and the metadata it serves (no network)."""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gateway.core.config import GatewayConfig
from gateway.log_config import logger as gateway_logger
from gateway.services import model_catalog_service as mcs
from gateway.services.catalog import load_catalog_metadata, metadata_entry
from gateway.services.model_catalog_service import ModelCatalogEntry, build_metadata_map
from gateway.types.models_dev import MODELS_DEV_SOURCES, ModelsDevCatalog, parse_catalog, parse_catalog_v2

# The real fetch and fallback, before the shared conftest turns both off.
_REAL_FETCH = mcs._fetch
_REAL_FALLBACK = mcs._fallback


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


def _v2_entry(model: dict[str, Any]) -> ModelCatalogEntry:
    return ModelCatalogEntry.from_model(parse_catalog_v2({"p": {"models": {"m": model}}}).providers["p"].models["m"])


def test_the_entry_names_the_routes_its_protocols_map_to_in_models_dev_order() -> None:
    api: dict[str, Any] = {"generate-content": {}, "chat-completions": {}, "interactions": {}, "responses": {}}

    assert _v2_entry({"api": api}).endpoints == ["/chat/completions", "/responses"]


def test_a_v2_entry_with_no_api_block_has_no_endpoints() -> None:
    assert _v2_entry({}).endpoints == []


def test_a_v1_entry_has_no_endpoints() -> None:
    assert _entry(GPT_4O).endpoints == []


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
async def test_a_read_the_refresher_keeps_warm_never_fetches() -> None:
    """A dashboard read must not pay the 15s models.dev fetch timeout, not even on a cold worker."""
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=86400)

    with (
        patch.object(mcs, "_fallback", return_value=EMPTY),
        patch.object(mcs, "_download", new=AsyncMock(return_value=CATALOG)) as download,
    ):
        assert await mcs.load_models_dev_catalog(config, serve_stale=True) is EMPTY
        await mcs.fetch_models_dev_document(fill_cache=True)
        mcs._cache.fetched_at = datetime.now(UTC) - timedelta(days=30)
        assert await mcs.load_models_dev_catalog(config, serve_stale=True) is CATALOG

    assert download.await_count == 1


@pytest.mark.asyncio
async def test_without_a_refresher_a_read_fetches_and_holds_a_failure_briefly() -> None:
    """With caching off (TTL 0) every read fetches, but a failure is held for the negative TTL."""
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=0)

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=None)) as failing:
        assert await mcs.load_models_dev_catalog(config) is None
        assert await mcs.load_models_dev_catalog(config) is None
        assert failing.await_count == 1

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=EMPTY)) as recovered:
        mcs._cache.at = time.monotonic() - (mcs._NEGATIVE_TTL_SECONDS + 1)
        assert await mcs.load_models_dev_catalog(config) is EMPTY
        assert await mcs.load_models_dev_catalog(config) is EMPTY
        assert recovered.await_count == 2


@pytest.mark.asyncio
async def test_a_failed_fetch_keeps_the_last_good_catalog() -> None:
    """A models.dev outage must not take the context windows and capabilities away with it."""
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=0)
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)):
        await mcs.load_models_dev_catalog(config)

    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=None)) as failing:
        assert await mcs.load_models_dev_catalog(config) is CATALOG
        assert await mcs.load_models_dev_catalog(config, serve_stale=True) is CATALOG
        assert mcs.cached_models_dev_catalog() is CATALOG
        assert failing.await_count == 1


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
    """Only a read with caching off fetches; one the refresher keeps warm, or one off the request path, never does."""
    with patch.object(mcs, "_fetch", new=AsyncMock(return_value=CATALOG)) as fetch:
        warm = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=60)
        assert await load_catalog_metadata(warm) is None
        assert await load_catalog_metadata(warm, cached_only=True) is None
        fetch.assert_not_called()

        uncached = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=0)
        assert await load_catalog_metadata(uncached, cached_only=True) is None
        fetch.assert_not_called()
        assert await load_catalog_metadata(uncached) is CATALOG
        fetch.assert_awaited_once()

        assert await load_catalog_metadata(warm, cached_only=True) is CATALOG
        # Without models_dev_metadata nothing is fetched, and the catalog already held is still served.
        assert await load_catalog_metadata(GatewayConfig(models_dev_metadata=False)) is CATALOG
        fetch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False], ids=["metadata-on", "metadata-off"])
async def test_the_bundled_catalog_serves_until_a_download_succeeds(enabled: bool) -> None:
    config = GatewayConfig(models_dev_metadata=enabled, models_dev_cache_ttl_seconds=0)

    with (
        patch.object(mcs, "_fallback", return_value=EMPTY),
        patch.object(mcs, "_fetch", new=AsyncMock(return_value=None)) as fetch,
    ):
        assert mcs.cached_models_dev_catalog() is EMPTY
        assert await load_catalog_metadata(config) is EMPTY
        assert fetch.await_count == int(enabled)

        fetch.return_value = CATALOG
        mcs.clear_catalog_cache()
        assert await load_catalog_metadata(config) is (CATALOG if enabled else EMPTY)
        assert mcs.cached_models_dev_catalog() is (CATALOG if enabled else EMPTY)


def test_the_fallback_is_the_bundled_catalog() -> None:
    catalog = _REAL_FALLBACK()

    assert catalog is not None
    assert catalog.model_for("openai", "gpt-4o") is not None


V2_URL, V1_URL = (source.url for source in MODELS_DEV_SOURCES)

# A document each reader reads the same, so either URL serves it.
SERVED = {"openai": {"id": "openai", "models": {"gpt-4o": {"name": "GPT-4o"}}}}


@contextmanager
def _serving(v2: object = None, v1: object = None) -> Iterator[list[str]]:
    """models.dev answering each document with its body as JSON, or with an HTTP error where it has none.

    Yields the URLs requested, in order.
    """
    requested: list[str] = []
    bodies = {V2_URL: v2, V1_URL: v1}

    class _Response:
        def __init__(self, body: object) -> None:
            self.body = body

        def raise_for_status(self) -> None:
            if self.body is None:
                raise httpx.HTTPError("503 Service Unavailable")

        async def aiter_bytes(self) -> Any:
            yield json.dumps(self.body).encode()

        async def __aenter__(self) -> "_Response":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def stream(self, method: str, url: str, **kwargs: Any) -> _Response:
            requested.append(url)
            return _Response(bodies[url])

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    with patch("httpx.AsyncClient", _Client):
        yield requested


@contextmanager
def _logs(caplog: pytest.LogCaptureFixture) -> Iterator[None]:
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger=gateway_logger.name)
    try:
        yield
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.mark.asyncio
async def test_the_download_reads_v2_into_a_catalog(caplog: pytest.LogCaptureFixture) -> None:
    """The one parser runs on the way in, so every reader gets the typed catalog and never the raw JSON."""
    model = {"name": "GPT-4o", "type": "chat", "api": {"responses": {"base_url": "https://api.openai.com/v1"}}}
    with _serving(v2={"openai": {"models": {"gpt-4o": model}}}) as requested, _logs(caplog):
        catalog = await mcs._download()

    assert isinstance(catalog, ModelsDevCatalog)
    read = catalog.providers["openai"].models["gpt-4o"]
    assert (read.name, read.api[0].protocol) == ("GPT-4o", "responses")
    assert requested == [V2_URL]
    assert caplog.records == []


@pytest.mark.parametrize(
    "v2",
    [
        pytest.param(None, id="unreachable"),
        pytest.param({"openai": {"models": ["gpt-4o"]}}, id="refused"),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_v2_download_falls_back_to_v1_with_one_warning(
    caplog: pytest.LogCaptureFixture, v2: object
) -> None:
    with _serving(v2=v2, v1=SERVED) as requested, _logs(caplog):
        catalog = await mcs._download()

    assert catalog.providers["openai"].models["gpt-4o"].api == ()
    assert requested == [V2_URL, V1_URL]
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert caplog.records[0].getMessage().startswith("models.dev v2 catalog ")
    assert caplog.text.rstrip().endswith("; reading the models.dev v1 catalog instead")


@pytest.mark.asyncio
async def test_a_download_with_a_few_damaged_models_is_kept_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    models = {f"m{n}": {"name": f"M{n}"} for n in range(9)} | {"bad": {"limit": {"context": "big"}}}
    with _serving(v2={"p": {"models": models}}), _logs(caplog):
        catalog = await mcs._download()

    assert len(catalog.providers["p"].models) == 10
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert "v2 catalog read with 1 unreadable values in 1 of 10 models; p.models.bad.limit.context: expected" in (
        caplog.text
    )


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(["openai"], id="not-an-object"),
        pytest.param({"openai": {"models": ["gpt-4o"]}}, id="no-models"),
        pytest.param({"p": {"models": {"a": {}, "b": {"name": 3}, "c": "C"}}}, id="mostly-damaged"),
    ],
)
@pytest.mark.asyncio
async def test_a_download_of_the_wrong_shape_is_refused(body: object) -> None:
    with _serving(v2=body, v1=body), pytest.raises(mcs.ModelsDevRefusedError, match="models.dev v1 catalog refused"):
        await mcs._download()


@pytest.mark.asyncio
async def test_a_refused_download_logs_one_error_and_keeps_the_last_good_catalog(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcs, "_fetch", _REAL_FETCH)
    config = GatewayConfig(models_dev_metadata=True, models_dev_cache_ttl_seconds=0)
    with _serving(v2=SERVED):
        good = await mcs.load_models_dev_catalog(config)
    assert good is not None

    refused = {"openai": {"models": ["gpt-4o"]}}
    with _serving(v2=refused, v1=refused), _logs(caplog):
        assert await mcs.load_models_dev_catalog(config) is good

    assert [record.levelname for record in caplog.records] == ["WARNING", "ERROR"]
    assert "openai.models: expected an object; keeping the last good copy" in caplog.records[1].getMessage()


@pytest.mark.asyncio
async def test_the_shared_fetch_fills_the_metadata_cache_and_is_reused() -> None:
    mcs.clear_catalog_cache()
    with patch.object(mcs, "_download", new=AsyncMock(return_value=EMPTY)) as download:
        assert await mcs.fetch_models_dev_document(fill_cache=True) is EMPTY
        assert await mcs.fetch_models_dev_document(reuse_within=60) is EMPTY
        assert download.await_count == 1
        assert mcs.cached_models_dev_catalog() is EMPTY
    mcs.clear_catalog_cache()


@pytest.mark.asyncio
async def test_a_stored_catalog_is_reused_by_when_it_was_downloaded() -> None:
    """A catalog another worker downloaded an hour ago is reused within a day, and not within ten minutes."""
    mcs.serve_catalog(CATALOG, datetime.now(UTC) - timedelta(hours=1))
    with patch.object(mcs, "_download", new=AsyncMock(return_value=EMPTY)) as download:
        assert await mcs.fetch_models_dev_document(reuse_within=86400) is CATALOG
        download.assert_not_awaited()
        assert await mcs.fetch_models_dev_document(reuse_within=600) is EMPTY
        download.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_shared_fetch_leaves_the_cache_alone_unless_asked_and_raises_on_failure() -> None:
    mcs.clear_catalog_cache()
    with patch.object(mcs, "_download", new=AsyncMock(return_value=EMPTY)):
        await mcs.fetch_models_dev_document()
    assert mcs._cache.data is None

    with (
        patch.object(mcs, "_download", new=AsyncMock(side_effect=mcs.ModelsDevFetchError("down"))),
        pytest.raises(mcs.ModelsDevFetchError),
    ):
        await mcs.fetch_models_dev_document()
    mcs.clear_catalog_cache()
