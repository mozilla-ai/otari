import pytest

from gateway.core.env import otari_env


def test_otari_env_reads_otari_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://searx:8080")
    assert otari_env("WEB_SEARCH_URL") == "http://searx:8080"


def test_otari_env_ignores_legacy_gateway_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """The legacy GATEWAY_ fallback was removed: only OTARI_ is read."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    monkeypatch.setenv("GATEWAY_WEB_SEARCH_URL", "http://legacy:8080")
    assert otari_env("WEB_SEARCH_URL") is None


def test_otari_env_returns_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTARI_TOOLS_HEADER", raising=False)
    assert otari_env("TOOLS_HEADER") is None
    assert otari_env("TOOLS_HEADER", "x-default") == "x-default"
