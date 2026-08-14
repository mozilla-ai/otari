"""Unit tests for the stored search-tool overlay merged over config search tools."""

import time
from collections.abc import Iterator
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.models.entities import SearchToolCredential
from gateway.services import search_tool_store_service as store
from gateway.services.search_backend import resolve_search_tool
from gateway.services.search_tool_store_service import (
    apply_to_config,
    config_file_search_tools,
    reset_search_tool_cache,
)
from gateway.services.secret_box import (
    SecretDecryptionError,
    encrypt_secret,
    generate_secret_key,
)


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_search_tool_cache()
    yield
    reset_search_tool_cache()


def _prime(overlay: dict[str, dict[str, Any]]) -> None:
    """Stand in for a database load without needing a session."""
    store._cache.clear()
    store._cache.update(overlay)
    store._cached_at = time.monotonic()


def test_config_tools_untouched_when_no_stored() -> None:
    config = GatewayConfig(search_tools={"exa": {"provider": "exa", "api_key": "k"}})
    assert apply_to_config(config) == set()
    assert config.search_tools == {"exa": {"provider": "exa", "api_key": "k"}}


def test_stored_tool_is_added_alongside_config() -> None:
    config = GatewayConfig(search_tools={"exa": {"provider": "exa", "api_key": "k"}})
    _prime({"local": {"provider": "searxng", "api_base": "http://searxng:8080"}})
    assert apply_to_config(config) == set()
    assert config.search_tools["exa"] == {"provider": "exa", "api_key": "k"}
    assert config.search_tools["local"] == {"provider": "searxng", "api_base": "http://searxng:8080"}


def test_stored_tool_shadows_config_of_same_name() -> None:
    config = GatewayConfig(search_tools={"exa": {"provider": "exa", "api_key": "config-key"}})
    _prime({"exa": {"provider": "exa", "api_key": "stored-key"}})
    assert apply_to_config(config) == {"exa"}
    assert config.search_tools["exa"]["api_key"] == "stored-key"


def test_removed_stored_row_restores_config_on_reapply() -> None:
    config = GatewayConfig(search_tools={"exa": {"provider": "exa", "api_key": "config-key"}})
    _prime({"exa": {"provider": "exa", "api_key": "stored-key"}})
    apply_to_config(config)
    # Simulate the row being deleted: cache empties, overlay re-applied.
    store._cache.clear()
    apply_to_config(config)
    assert config.search_tools["exa"]["api_key"] == "config-key"


def test_cache_reset_does_not_bake_overlay_into_baseline() -> None:
    # The overlay must never become the baseline, or a deleted stored row would
    # be permanent. Same regression the provider overlay guards against.
    config = GatewayConfig(search_tools={"exa": {"provider": "exa", "api_key": "config-key"}})
    _prime({"exa": {"provider": "exa", "api_key": "stored-key"}})
    apply_to_config(config)
    reset_search_tool_cache()
    apply_to_config(config)
    assert config.search_tools["exa"]["api_key"] == "config-key"


def test_config_file_search_tools_strips_the_overlay() -> None:
    config = GatewayConfig(search_tools={"exa": {"provider": "exa", "api_key": "k"}})
    _prime({"local": {"provider": "searxng", "api_base": "http://searxng:8080"}})
    apply_to_config(config)
    assert set(config_file_search_tools(config)) == {"exa"}


def test_config_file_search_tools_before_any_overlay() -> None:
    config = GatewayConfig(search_tools={"exa": {"provider": "exa", "api_key": "k"}})
    assert set(config_file_search_tools(config)) == {"exa"}


def test_stored_tool_resolves_on_the_request_path() -> None:
    """The point of the overlay: a dashboard-added tool is dispatchable."""
    config = GatewayConfig()
    _prime({"local": {"provider": "searxng", "api_base": "http://searxng:8080"}})
    apply_to_config(config)
    tool = resolve_search_tool(config, "local")
    assert tool.provider == "searxng"
    assert tool.api_base == "http://searxng:8080"


def test_stored_searxng_tool_inherits_web_search_url() -> None:
    """A stored tool with no api_base falls back the same way a config one does."""
    config = GatewayConfig(web_search_url="http://searxng:8080")
    _prime({"local": {"provider": "searxng"}})
    apply_to_config(config)
    assert resolve_search_tool(config, "local").api_base == "http://searxng:8080"


def test_row_to_entry_decrypts_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = SearchToolCredential(
        name="exa-search",
        provider="exa",
        encrypted_api_key=encrypt_secret("exa-live"),
        last4="live",
        options={},
    )
    assert store._row_to_entry(row) == {"provider": "exa", "api_key": "exa-live"}


def test_row_to_entry_carries_base_timeout_and_options() -> None:
    row = SearchToolCredential(
        name="local",
        provider="searxng",
        api_base="http://searxng:8080",
        timeout_seconds=12.5,
        options={"engines": "brave"},
    )
    assert store._row_to_entry(row) == {
        "provider": "searxng",
        "api_base": "http://searxng:8080",
        "timeout": 12.5,
        "options": {"engines": "brave"},
    }


def test_row_to_entry_raises_when_key_cannot_be_decrypted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    row = SearchToolCredential(name="exa", provider="exa", encrypted_api_key=encrypt_secret("k"), options={})
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    with pytest.raises(SecretDecryptionError):
        store._row_to_entry(row)
