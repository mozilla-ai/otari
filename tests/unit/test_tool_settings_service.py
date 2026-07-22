"""Unit tests for the editable tool/guardrail settings service."""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.runtime_settings_service import SettingValue
from gateway.services.tool_settings_service import (
    _parse,
    _serialize,
    apply_override,
    effective_value,
    validate_url,
    validate_value,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://searxng:8080",  # docker-compose private sidecar
        "http://guardrails:8000",
        "http://localhost:8080",  # loopback
        "http://127.0.0.1:9",
        "https://search.example.com",
        "https://host:8443/path?q=1",
    ],
)
def test_validate_url_accepts_operator_urls_including_private_and_loopback(url: str) -> None:
    # The operator is trusted; deny-private would reject the bundled sidecars,
    # which is the primary thing this page configures.
    assert validate_url(url) == url


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "gopher://x", "ftp://host/f", "not-a-url", "http://", "://nohost"],
)
def test_validate_url_rejects_non_web_or_hostless(url: str) -> None:
    with pytest.raises(ValueError):
        validate_url(url)


def test_validate_value_url_field_structural() -> None:
    assert validate_value("web_search_url", "http://searxng:8080") == "http://searxng:8080"
    with pytest.raises(ValueError):
        validate_value("sandbox_url", "file:///etc/passwd")


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_validate_value_blank_clears_any_field(blank: SettingValue) -> None:
    for key in ("web_search_url", "web_search_engines", "web_search_max_results", "web_search_extract"):
        assert validate_value(key, blank) is None


def test_validate_value_int_bounds_and_bool_rejection() -> None:
    assert validate_value("web_search_max_results", 5) == 5
    with pytest.raises(ValueError):
        validate_value("web_search_max_results", 0)  # ge=1
    with pytest.raises(ValueError):
        validate_value("web_search_max_results", True)  # bool is not an int here
    with pytest.raises(ValueError):
        validate_value("web_search_max_results", "5")  # wrong type


def test_validate_value_bool_field() -> None:
    assert validate_value("web_search_extract", True) is True
    assert validate_value("web_search_extract", False) is False
    with pytest.raises(ValueError):
        validate_value("web_search_extract", "true")


def test_validate_value_unknown_key() -> None:
    with pytest.raises(ValueError):
        validate_value("not_a_tool_setting", "x")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("web_search_url", "http://searxng:8080"),
        ("web_search_engines", "google,bing"),
        ("web_search_max_results", 7),
        ("web_search_extract", False),
        ("web_search_extract", True),
        ("sandbox_url", None),
    ],
)
def test_serialize_parse_round_trip(key: str, value: SettingValue) -> None:
    assert _parse(key, _serialize(value)) == value


def test_effective_value_prefers_config_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = GatewayConfig(sandbox_url="http://from-config:1234")
    assert effective_value(config, "sandbox_url") == "http://from-config:1234"

    # When config is unset, fall back to the OTARI_ env var (coerced per type).
    config2 = GatewayConfig()
    config2.web_search_max_results = None
    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "9")
    assert effective_value(config2, "web_search_max_results") == 9

    config2.web_search_extract = None
    monkeypatch.setenv("OTARI_WEB_SEARCH_EXTRACT", "false")
    assert effective_value(config2, "web_search_extract") is False


def test_effective_value_false_bool_not_lost() -> None:
    # A stored False must not be treated as "unset" and fall through to env.
    config = GatewayConfig()
    config.web_search_extract = False
    assert effective_value(config, "web_search_extract") is False


def test_apply_override_mutates_config() -> None:
    config = GatewayConfig()
    apply_override(config, "web_search_url", "http://new:8080")
    assert config.web_search_url == "http://new:8080"
    apply_override(config, "web_search_url", None)
    assert config.web_search_url is None
