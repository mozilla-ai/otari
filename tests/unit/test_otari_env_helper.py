import pytest

from gateway.core.env import otari_env


def test_otari_env_reads_otari_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://searx:8080")
    monkeypatch.delenv("GATEWAY_WEB_SEARCH_URL", raising=False)
    assert otari_env("WEB_SEARCH_URL") == "http://searx:8080"


def test_otari_env_falls_back_to_legacy_gateway_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://legacy:8080")
    assert otari_env("WEB_SEARCH_URL") == "http://legacy:8080"


def test_otari_env_prefers_otari_over_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://new:8080")
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://legacy:8080")
    assert otari_env("WEB_SEARCH_URL") == "http://new:8080"


def test_otari_env_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_TOOLS_HEADER", raising=False)
    monkeypatch.delenv("GATEWAY_TOOLS_HEADER", raising=False)
    assert otari_env("TOOLS_HEADER") is None
    assert otari_env("TOOLS_HEADER", "x-default") == "x-default"
