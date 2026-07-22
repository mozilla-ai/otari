"""Endpoint tests for /v1/tool-settings (sqlite-backed TestClient)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app

AUTH = {"Authorization": "Bearer sk-test-master"}


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'tool-settings-test.db'}",
        master_key="sk-test-master",
        **overrides,
    )
    return TestClient(create_app(config))


def _fields(body: dict[str, Any]) -> dict[str, Any]:
    return {f["key"]: f for f in body["fields"]}


def test_requires_master_key(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/v1/tool-settings").status_code == 401
        assert client.get("/v1/tool-settings", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_get_reports_effective_values(tmp_path: Path) -> None:
    with _client(tmp_path, sandbox_url="http://sandbox:8000", web_search_max_results=7) as client:
        body = client.get("/v1/tool-settings", headers=AUTH).json()
    fields = _fields(body)
    assert fields["sandbox_url"]["value"] == "http://sandbox:8000"
    assert fields["sandbox_url"]["service"] == "sandbox"
    assert fields["sandbox_url"]["type"] == "url"
    assert fields["web_search_max_results"]["value"] == 7
    assert fields["web_search_extract"]["type"] == "bool"


def test_get_redacts_url_password(tmp_path: Path) -> None:
    with _client(tmp_path, guardrails_url="https://user:secret@guardrails:8000") as client:
        body = client.get("/v1/tool-settings", headers=AUTH).json()
    value = _fields(body)["guardrails_url"]["value"]
    assert "secret" not in value
    assert "***" in value


def test_patch_persists_and_hot_applies(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.patch(
            "/v1/tool-settings",
            headers=AUTH,
            json={"web_search_url": "http://searxng:8080", "web_search_max_results": 10},
        )
        assert resp.status_code == 200
        # The running config was mutated (hot-apply) and the GET reflects it.
        app_config: GatewayConfig = client.app.state.config  # type: ignore[attr-defined]
        assert app_config.web_search_url == "http://searxng:8080"
        assert app_config.web_search_max_results == 10
        fields = _fields(client.get("/v1/tool-settings", headers=AUTH).json())
        assert fields["web_search_url"]["value"] == "http://searxng:8080"


def test_patch_accepts_bundled_sidecar_urls(tmp_path: Path) -> None:
    # T1: the primary use case. Private/loopback sidecar URLs must be settable;
    # a deny-private gate here would break the default docker-compose deployment.
    with _client(tmp_path) as client:
        resp = client.patch(
            "/v1/tool-settings",
            headers=AUTH,
            json={
                "web_search_url": "http://searxng:8080",
                "sandbox_url": "http://sandbox:8000",
                "guardrails_url": "http://localhost:8000",
            },
        )
    assert resp.status_code == 200


def test_patch_rejects_non_web_scheme(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.patch("/v1/tool-settings", headers=AUTH, json={"sandbox_url": "file:///etc/passwd"})
    assert resp.status_code == 422
    # Nothing was stored.
    with _client(tmp_path) as client:
        assert _fields(client.get("/v1/tool-settings", headers=AUTH).json())["sandbox_url"]["value"] is None


def test_patch_rejects_out_of_bounds_max_results(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        # ge=1 is enforced by the request model (422) before it even reaches the service.
        assert client.patch("/v1/tool-settings", headers=AUTH, json={"web_search_max_results": 0}).status_code == 422


def test_patch_clear_falls_back_and_survives_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Model a deployment configured via env: OTARI_WEB_SEARCH_URL is what a real
    # deployment sets (or what YAML is bridged into), and it is what the read path
    # falls back to when an override is cleared. (A directly-constructed config
    # value with no env twin has nothing to fall back to; that is not a real path.)
    db = f"sqlite:///{tmp_path / 'clear-test.db'}"
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://env-default:8080")
    with TestClient(create_app(GatewayConfig(database_url=db, master_key="sk-test-master"))) as client:
        # Override it, then clear it back to the configured default.
        client.patch("/v1/tool-settings", headers=AUTH, json={"web_search_url": "http://override:9999"})
        assert client.app.state.config.web_search_url == "http://override:9999"  # type: ignore[attr-defined]
        client.patch("/v1/tool-settings", headers=AUTH, json={"web_search_url": None})
        # Cleared: the read path falls back to the configured env value, not "nothing".
        fields = _fields(client.get("/v1/tool-settings", headers=AUTH).json())
        assert fields["web_search_url"]["value"] == "http://env-default:8080"

    # Restart: the cleared override ("") is re-applied as None; the read path again
    # falls back to the configured env value.
    with TestClient(create_app(GatewayConfig(database_url=db, master_key="sk-test-master"))) as client2:
        fields2 = _fields(client2.get("/v1/tool-settings", headers=AUTH).json())
        assert fields2["web_search_url"]["value"] == "http://env-default:8080"


def test_test_endpoint_reports_reachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200

    async def fake_get(self: Any, url: str) -> _Resp:  # noqa: ARG001
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    with _client(tmp_path) as client:
        resp = client.post(
            "/v1/tool-settings/web_search/test", headers=AUTH, json={"url": "http://searxng:8080"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "200" in body["reason"]


def test_test_endpoint_reports_unreachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get(self: Any, url: str) -> Any:  # noqa: ARG001
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    with _client(tmp_path) as client:
        resp = client.post("/v1/tool-settings/sandbox/test", headers=AUTH, json={"url": "http://localhost:1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "unreachable" in body["reason"]


def test_test_endpoint_rejects_unsafe_url(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.post("/v1/tool-settings/sandbox/test", headers=AUTH, json={"url": "file:///etc/passwd"})
    assert resp.status_code == 422


def test_test_endpoint_unknown_service(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.post("/v1/tool-settings/bogus/test", headers=AUTH, json={"url": "http://x:8080"})
    assert resp.status_code == 404


@pytest.fixture
def _hybrid_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    yield


def test_tool_settings_not_mounted_in_hybrid_mode(tmp_path: Path, _hybrid_env: None) -> None:
    config = GatewayConfig(
        mode="hybrid",
        master_key="sk-test-master",
        platform={"base_url": "https://otari.ai"},
    )
    with TestClient(create_app(config)) as client:
        # Standalone-only: the management route is not registered in hybrid mode.
        assert client.get("/v1/tool-settings", headers=AUTH).status_code == 404
