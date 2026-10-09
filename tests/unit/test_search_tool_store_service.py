"""Unit tests for the stored search-tool overlay merged over config search tools."""

import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import (
    BUILTIN_FETCH,
    effective_fetch_instances,
    effective_search_instances,
    warn_about_instances,
)
from gateway.models.tools import SearchToolCredential
from gateway.services import search_tool_store_service as store
from gateway.services.search_backend import resolve_search_tool
from gateway.services.search_tool_store_service import (
    apply_to_config,
    config_file_fetch_tools,
    config_file_search_tools,
    refresh_tool_instances,
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


# --------------------------------------------------------------------------- #
# Fetch rows, and what the overlay reports
# --------------------------------------------------------------------------- #

_T0 = datetime(2026, 10, 9, tzinfo=UTC)


def _row(name: str, kind: str = "search", *, at: datetime = _T0, **fields: Any) -> SearchToolCredential:
    return SearchToolCredential(name=name, kind=kind, provider="fake", updated_at=at, **{"options": {}, **fields})


def _logged(caplog: pytest.LogCaptureFixture, config: GatewayConfig, rows: list[SearchToolCredential]) -> str:
    caplog.set_level(logging.INFO, logger="gateway")
    caplog.clear()
    store._overlay_rows(config, rows, report_changes=True)
    return "\n".join(record.getMessage() for record in caplog.records)


def test_row_to_entry_carries_a_search_rows_fetch_tool() -> None:
    assert store._row_to_entry(_row("exa", fetch_tool="exa-fetch")) == {"provider": "fake", "fetch_tool": "exa-fetch"}
    assert store._row_to_entry(_row("exa-fetch", "fetch", fetch_tool="ignored")) == {"provider": "fake"}


def test_fetch_rows_overlay_the_fetch_map_and_search_rows_the_search_map() -> None:
    config = GatewayConfig(
        search_tools={"from-file": {"provider": "fake"}}, fetch_tools={"file-fetch": {"provider": "fake"}}
    )
    shadowed = store._overlay_rows(
        config, [_row("stored"), _row("stored-fetch", "fetch"), _row("file-fetch", "fetch")], report_changes=False
    )
    assert set(config.search_tools) == {"from-file", "stored"}
    assert set(config.fetch_tools) == {"file-fetch", "stored-fetch"}
    assert shadowed == {"file-fetch"}
    assert set(config_file_fetch_tools(config)) == {"file-fetch"}
    assert set(effective_fetch_instances(config)) == {BUILTIN_FETCH, "file-fetch", "stored-fetch"}


def test_a_stored_fetch_row_named_like_a_search_instance_is_left_out_with_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rule 4: the fetch entry is the one refused, here a stored row meeting a configured search entry."""
    config = GatewayConfig(search_tools={"shared": {"provider": "fake"}})
    logged = _logged(caplog, config, [_row("shared", "fetch")])
    assert "shared" in config.search_tools
    assert "shared" not in effective_fetch_instances(config)
    assert "fetch_tools.shared is left out: a search instance has the same name" in logged


def test_a_stored_search_row_leaves_out_a_configured_fetch_entry_of_its_name(caplog: pytest.LogCaptureFixture) -> None:
    """Rule 4 the other way round: the stored search row keeps its name, the configured fetch entry goes."""
    config = GatewayConfig(fetch_tools={"shared": {"provider": "fake"}})
    caplog.set_level(logging.ERROR, logger="gateway")
    store._overlay_rows(config, [_row("shared")], report_changes=True)
    warn_about_instances(config, (), ["shared"])
    assert "shared" in effective_search_instances(config)
    assert "shared" not in effective_fetch_instances(config)
    assert any("fetch_tools.shared is left out" in record.getMessage() for record in caplog.records)


def test_a_stored_search_row_named_builtin_fetch_loads_beside_the_built_in_fetcher(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig()
    logged = _logged(caplog, config, [_row(BUILTIN_FETCH)])
    assert BUILTIN_FETCH in effective_search_instances(config)
    assert effective_fetch_instances(config)[BUILTIN_FETCH].provider == "builtin"
    assert f"search_tools.{BUILTIN_FETCH} breaks the instance rules and loads anyway" in logged


def test_a_stored_fetch_row_that_breaks_the_name_rules_is_left_out_with_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig()
    logged = _logged(caplog, config, [_row("a:b", "fetch"), _row("None", "fetch")])
    assert set(effective_fetch_instances(config)) == {BUILTIN_FETCH}
    assert "fetch_tools.a:b is left out: its name contains ':'" in logged
    assert "fetch_tools.None is left out: its name is reserved" in logged


def test_a_rows_problems_are_reported_once_per_version(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig()
    stored = _row("odd", options={"bogus": 1})
    assert "search_tools.odd breaks the instance rules" in _logged(caplog, config, [stored])
    assert _logged(caplog, config, [stored]) == ""
    edited = _row("odd", at=_T0 + timedelta(seconds=1), options={"bogus": 1})
    assert "option 'bogus' is not one the provider knows" in _logged(caplog, config, [edited])


def test_startup_reports_no_instance_rules_of_its_own(caplog: pytest.LogCaptureFixture) -> None:
    """The lifespan's own check reports them once the settings have loaded too."""
    config = GatewayConfig()
    caplog.set_level(logging.INFO, logger="gateway")
    store._overlay_rows(config, [_row("odd", options={"bogus": 1})], report_changes=False)
    assert not [record for record in caplog.records if "breaks the instance rules" in record.getMessage()]
    # And the next refresh does not report what startup's check already did.
    assert _logged(caplog, config, [_row("odd", options={"bogus": 1})]) == ""


def test_a_dangling_default_is_reported_once_per_value(caplog: pytest.LogCaptureFixture) -> None:
    config = GatewayConfig(web_fetch_default_tool="gone")
    caplog.set_level(logging.WARNING, logger="gateway")

    def reported() -> list[str]:
        caplog.clear()
        store._report_dangling_references(config)
        return [record.getMessage() for record in caplog.records]

    assert reported() == [
        "web_fetch_default_tool names no fetch instance (gone), so builtin_fetch is the fetch default."
    ]
    assert reported() == []
    config.web_fetch_default_tool = "also-gone"
    assert reported() == [
        "web_fetch_default_tool names no fetch instance (also-gone), so builtin_fetch is the fetch default."
    ]


def test_a_fetch_tool_left_dangling_by_a_deleted_fetch_instance_is_reported_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = GatewayConfig()
    store._overlay_rows(config, [_row("s", fetch_tool="f"), _row("f", "fetch")], report_changes=False)
    caplog.set_level(logging.WARNING, logger="gateway")
    store._report_dangling_references(config)
    assert caplog.records == []
    # 'f' is deleted; 's' itself has not changed.
    store._overlay_rows(config, [_row("s", fetch_tool="f")], report_changes=True)
    for _ in range(2):
        store._report_dangling_references(config)
    assert [record.getMessage() for record in caplog.records] == [
        "search_tools.s.fetch_tool names no fetch instance (f), so the fetch default enriches its results."
    ]


def test_a_refresh_does_not_undo_a_write_applied_after_its_read() -> None:
    """Only a value changed in the database since the last read is applied, never one this worker set since."""
    config = GatewayConfig()
    store._apply_settings(config, {"web_search_url": "http://old:8080"})
    assert config.web_search_url == "http://old:8080"
    # A PATCH on this worker commits and applies a new value while the next
    # refresh's read, which still saw the old one, is in flight.
    config.web_search_url = "http://new:8080"
    store._apply_settings(config, {"web_search_url": "http://old:8080"})
    assert config.web_search_url == "http://new:8080"


class _Session:
    """Stands in for a session: the refresh's reads are replaced below."""


@pytest.mark.asyncio
async def test_the_refresh_reads_the_rows_then_the_settings_and_applies_both(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rows first, so a default and the rows it names reach this worker together."""
    reads: list[str] = []

    async def load_rows(_db: object) -> list[SearchToolCredential]:
        reads.append("rows")
        return [_row("first"), _row("second")]

    async def load_overrides(_db: object, *, report_invalid: bool) -> dict[str, Any]:
        assert not report_invalid, "startup reported them; a refresh would repeat it every time"
        reads.append("settings")
        return {"web_search_default_tool": "first", "web_search_url": None}

    monkeypatch.setattr(store, "_load_rows", load_rows)
    monkeypatch.setattr(store, "load_overrides", load_overrides)
    caplog.set_level(logging.INFO, logger="gateway")
    config = GatewayConfig()
    await refresh_tool_instances(_Session(), config)  # type: ignore[arg-type]
    assert reads == ["rows", "settings"]
    assert set(config.search_tools) == {"first", "second"}
    assert config.web_search_default_tool == "first"
    applied = [record.getMessage() for record in caplog.records if "Applied tool setting" in record.getMessage()]
    # Only what changed, and by key: web_search_url was already unset.
    assert applied == ["Applied tool setting web_search_default_tool, changed since this worker last read it"]
