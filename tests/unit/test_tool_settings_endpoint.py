"""Endpoint tests for /api/v1/tool-settings (sqlite-backed TestClient)."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig
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
        assert client.get(f"{API_ROOT}/tool-settings").status_code == 401
        assert client.get(f"{API_ROOT}/tool-settings", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_get_reports_effective_values(tmp_path: Path) -> None:
    with _client(tmp_path, sandbox_url="http://sandbox:8000", web_search_max_results=7) as client:
        body = client.get(f"{API_ROOT}/tool-settings", headers=AUTH).json()
    fields = _fields(body)
    assert fields["sandbox_url"]["value"] == "http://sandbox:8000"
    assert fields["sandbox_url"]["service"] == "sandbox"
    assert fields["sandbox_url"]["type"] == "url"
    assert fields["web_search_max_results"]["value"] == 7
    assert fields["web_search_extract"]["type"] == "bool"


def test_get_redacts_url_password(tmp_path: Path) -> None:
    with _client(tmp_path, guardrails_url="https://user:secret@guardrails:8000") as client:
        body = client.get(f"{API_ROOT}/tool-settings", headers=AUTH).json()
    value = _fields(body)["guardrails_url"]["value"]
    assert "secret" not in value
    assert "***" in value


def test_patch_persists_and_hot_applies(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.patch(
            f"{API_ROOT}/tool-settings",
            headers=AUTH,
            json={"web_search_url": "http://searxng:8080", "web_search_max_results": 10},
        )
        assert resp.status_code == 200
        # The running config was mutated (hot-apply) and the GET reflects it.
        app_config: GatewayConfig = client.app.state.config  # type: ignore[attr-defined]
        assert app_config.web_search_url == "http://searxng:8080"
        assert app_config.web_search_max_results == 10
        fields = _fields(client.get(f"{API_ROOT}/tool-settings", headers=AUTH).json())
        assert fields["web_search_url"]["value"] == "http://searxng:8080"


def test_patch_accepts_bundled_sidecar_urls(tmp_path: Path) -> None:
    # T1: the primary use case. Private/loopback sidecar URLs must be settable;
    # a deny-private gate here would break the default docker-compose deployment.
    with _client(tmp_path) as client:
        resp = client.patch(
            f"{API_ROOT}/tool-settings",
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
        resp = client.patch(f"{API_ROOT}/tool-settings", headers=AUTH, json={"sandbox_url": "file:///etc/passwd"})
    assert resp.status_code == 422
    # Nothing was stored.
    with _client(tmp_path) as client:
        assert _fields(client.get(f"{API_ROOT}/tool-settings", headers=AUTH).json())["sandbox_url"]["value"] is None


def test_patch_rejects_out_of_bounds_max_results(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        # ge=1 is enforced by the request model (422) before it even reaches the service.
        assert (
            client.patch(f"{API_ROOT}/tool-settings", headers=AUTH, json={"web_search_max_results": 0}).status_code
            == 422
        )


def test_patch_clear_falls_back_and_survives_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Model a deployment configured via env: OTARI_WEB_SEARCH_URL is what a real
    # deployment sets (or what YAML is bridged into), and it is what the read path
    # falls back to when an override is cleared. (A directly-constructed config
    # value with no env twin has nothing to fall back to; that is not a real path.)
    db = f"sqlite:///{tmp_path / 'clear-test.db'}"
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://env-default:8080")
    with TestClient(create_app(GatewayConfig(database_url=db, master_key="sk-test-master"))) as client:
        # Override it, then clear it back to the configured default.
        client.patch(f"{API_ROOT}/tool-settings", headers=AUTH, json={"web_search_url": "http://override:9999"})
        assert client.app.state.config.web_search_url == "http://override:9999"  # type: ignore[attr-defined]
        client.patch(f"{API_ROOT}/tool-settings", headers=AUTH, json={"web_search_url": None})
        # Cleared: the read path falls back to the configured env value, not "nothing".
        fields = _fields(client.get(f"{API_ROOT}/tool-settings", headers=AUTH).json())
        assert fields["web_search_url"]["value"] == "http://env-default:8080"

    # Restart: the cleared override ("") is re-applied as None; the read path again
    # falls back to the configured env value.
    with TestClient(create_app(GatewayConfig(database_url=db, master_key="sk-test-master"))) as client2:
        fields2 = _fields(client2.get(f"{API_ROOT}/tool-settings", headers=AUTH).json())
        assert fields2["web_search_url"]["value"] == "http://env-default:8080"


def test_test_endpoint_reports_reachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status_code = 200

    async def fake_get(self: Any, url: str) -> _Resp:  # noqa: ARG001
        return _Resp()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    with _client(tmp_path) as client:
        resp = client.post(
            f"{API_ROOT}/tool-settings/web_search/test", headers=AUTH, json={"url": "http://searxng:8080"}
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
        resp = client.post(f"{API_ROOT}/tool-settings/sandbox/test", headers=AUTH, json={"url": "http://localhost:1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert "unreachable" in body["reason"]


def test_test_endpoint_rejects_unsafe_url(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.post(f"{API_ROOT}/tool-settings/sandbox/test", headers=AUTH, json={"url": "file:///etc/passwd"})
    assert resp.status_code == 422


def test_test_endpoint_unknown_service(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        resp = client.post(f"{API_ROOT}/tool-settings/bogus/test", headers=AUTH, json={"url": "http://x:8080"})
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
        assert client.get(f"{API_ROOT}/tool-settings", headers=AUTH).status_code == 404
        # And the catalog read with it, since it is mounted on the same router.
        assert client.get(f"{API_ROOT}/tool-settings/guardrails/profiles", headers=AUTH).status_code == 404


def test_patch_persists_the_sandbox_image(tmp_path: Path) -> None:
    """The deployment's own image is an operator setting like the URL beside it (#740).

    The workspace allow-list is deliberately *not* one: it is the supply-chain
    gate a workspace policy is checked against, so it stays config/env-only and
    a PATCH naming it is rejected rather than quietly ignored.
    """
    with _client(tmp_path) as client:
        patched = client.patch(
            f"{API_ROOT}/tool-settings",
            json={"sandbox_session_image": "mzdotai/otari-sandbox-container:latest"},
            headers=AUTH,
        )
        assert patched.status_code == 200, patched.text
        fields = _fields(client.get(f"{API_ROOT}/tool-settings", headers=AUTH).json())

    assert fields["sandbox_session_image"]["value"] == "mzdotai/otari-sandbox-container:latest"
    assert fields["sandbox_session_image"]["service"] == "sandbox"
    assert "sandbox_allowed_session_images" not in fields


def _stub_guardrails_service(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Answer the guardrails service's ``GET /profiles`` from ``handler``.

    Through the transport rather than by patching a method, because the catalog
    streams the body to cap its size and so calls no single request method.
    """
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient  # captured before patching, to avoid recursion

    def factory(*_args: object, **_kwargs: object) -> httpx.AsyncClient:
        return real_async_client(transport=transport)

    monkeypatch.setattr("gateway.services.guardrail_catalog.httpx.AsyncClient", factory)


def test_guardrail_profiles_lists_what_the_service_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://anyguardrails:8000/profiles"
        return httpx.Response(200, json=[{"name": "house-policy", "guardrail_name": "any_llm"}])

    _stub_guardrails_service(monkeypatch, handler)
    with _client(tmp_path, guardrails_url="http://anyguardrails:8000") as client:
        resp = client.get(f"{API_ROOT}/tool-settings/guardrails/profiles", headers=AUTH)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["available"] is True
    assert body["profiles"][0]["profile"] == "house-policy"
    assert "policy" in {parameter["name"] for parameter in body["profiles"][0]["parameters"]}


def test_guardrail_profiles_reports_an_unconfigured_service(tmp_path: Path) -> None:
    """A deployment with no guardrails service gets a reason, not an error.

    This drives the page an operator configures guardrails on, so it has to
    render before the service they are configuring exists.
    """
    with _client(tmp_path) as client:
        resp = client.get(f"{API_ROOT}/tool-settings/guardrails/profiles", headers=AUTH)

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["profiles"] == []
    assert body["reason"]


def test_guardrail_profiles_never_returns_the_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader router serves a tenant, from whom the GET above withholds URLs."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _stub_guardrails_service(monkeypatch, handler)
    with _client(tmp_path, guardrails_url="https://guardrails.internal.example") as client:
        resp = client.get(f"{API_ROOT}/tool-settings/guardrails/profiles", headers=AUTH)

    assert resp.status_code == 200
    assert "guardrails.internal.example" not in resp.text


def test_guardrail_profiles_requires_master_key(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get(f"{API_ROOT}/tool-settings/guardrails/profiles").status_code == 401
        

def test_guardrail_profiles_refuses_an_oversized_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout bounds how long the answer takes, not how much of it is held."""

    def handler(_request: httpx.Request) -> httpx.Response:
        row = {"name": "x" * 200, "guardrail_name": "injec_guard"}
        return httpx.Response(200, json=[row] * 20_000)

    _stub_guardrails_service(monkeypatch, handler)
    with _client(tmp_path, guardrails_url="http://anyguardrails:8000") as client:
        resp = client.get(f"{API_ROOT}/tool-settings/guardrails/profiles", headers=AUTH)

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert body["profiles"] == []
