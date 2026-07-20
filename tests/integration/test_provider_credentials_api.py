"""Integration tests for the /v1/provider-credentials CRUD + test endpoints.

Covers the security-critical behavior: keys are write-only and never echoed,
storing a key requires OTARI_SECRET_KEY, updates are optimistic, and every route
is master-key gated.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from gateway.api.routes import providers as providers_route
from gateway.services.model_discovery_service import ProviderDiscovery
from gateway.services.provider_store_service import reset_provider_cache
from gateway.services.secret_box import generate_secret_key


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    reset_provider_cache()
    yield
    reset_provider_cache()


def _with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


def _create(client: TestClient, headers: dict[str, str], instance: str = "openai", key: str = "sk-1234") -> None:
    resp = client.post("/v1/provider-credentials", json={"instance": instance, "api_key": key}, headers=headers)
    assert resp.status_code == 201, resp.text


def test_create_requires_secret_key(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET_KEY", raising=False)
    resp = client.post(
        "/v1/provider-credentials",
        json={"instance": "openai", "api_key": "sk-live-1234"},
        headers=master_key_header,
    )
    assert resp.status_code == 400
    assert "OTARI_SECRET_KEY" in resp.json()["detail"]


def test_create_lists_and_never_returns_key(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    resp = client.post(
        "/v1/provider-credentials",
        json={"instance": "openai", "api_key": "sk-live-1234", "api_base": "https://api.openai.com/v1"},
        headers=master_key_header,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["instance"] == "openai"
    assert body["last4"] == "1234"
    assert "api_key" not in body
    assert "sk-live-1234" not in resp.text

    listed = client.get("/v1/provider-credentials", headers=master_key_header)
    assert listed.status_code == 200
    assert [p["instance"] for p in listed.json()] == ["openai"]
    assert "sk-live-1234" not in listed.text


def test_list_flags_undecryptable_key(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    _create(client, master_key_header, instance="openai", key="sk-orig")
    # Rotate the encryption key: the stored key can no longer be decrypted.
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    rows = client.get("/v1/provider-credentials", headers=master_key_header).json()
    assert rows[0]["instance"] == "openai"
    assert rows[0]["decryptable"] is False


def test_create_duplicate_conflicts(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    payload = {"instance": "openai", "api_key": "sk-1234"}
    assert client.post("/v1/provider-credentials", json=payload, headers=master_key_header).status_code == 201
    dup = client.post("/v1/provider-credentials", json=payload, headers=master_key_header)
    assert dup.status_code == 409


def test_patch_updates_base_keeps_key_then_rotates(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    client.post(
        "/v1/provider-credentials",
        json={"instance": "openai", "api_key": "sk-orig-1111"},
        headers=master_key_header,
    )
    # Update the base only; the stored key (last4) is unchanged.
    patched = client.patch(
        "/v1/provider-credentials/openai",
        json={"api_base": "https://proxy/v1"},
        headers=master_key_header,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["api_base"] == "https://proxy/v1"
    assert patched.json()["last4"] == "1111"
    # Rotate the key.
    rotated = client.patch(
        "/v1/provider-credentials/openai",
        json={"api_key": "sk-new-2222"},
        headers=master_key_header,
    )
    assert rotated.status_code == 200
    assert rotated.json()["last4"] == "2222"
    assert "sk-new-2222" not in rotated.text


def test_patch_optimistic_precondition(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    client.post(
        "/v1/provider-credentials",
        json={"instance": "openai", "api_key": "sk-1234"},
        headers=master_key_header,
    )
    stale = client.patch(
        "/v1/provider-credentials/openai",
        json={"api_base": "https://x/v1", "expected_updated_at": "1999-01-01T00:00:00+00:00"},
        headers=master_key_header,
    )
    assert stale.status_code == 412


def test_patch_and_delete_missing_are_404(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    assert client.patch(
        "/v1/provider-credentials/nope", json={"api_base": "x"}, headers=master_key_header
    ).status_code == 404
    assert client.delete("/v1/provider-credentials/nope", headers=master_key_header).status_code == 404


def test_delete_round_trip(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    _create(client, master_key_header)
    assert client.delete("/v1/provider-credentials/openai", headers=master_key_header).status_code == 204
    assert client.get("/v1/provider-credentials", headers=master_key_header).json() == []


def test_invalid_instance_and_provider_type(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    bad_name = client.post(
        "/v1/provider-credentials", json={"instance": "a:b", "api_key": "sk"}, headers=master_key_header
    )
    assert bad_name.status_code == 400
    bad_type = client.post(
        "/v1/provider-credentials",
        json={"instance": "x", "provider_type": "not-a-provider", "api_key": "sk"},
        headers=master_key_header,
    )
    assert bad_type.status_code == 400


def test_test_connection_before_save_maps_result(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _ok(impl: str, **_kwargs: object) -> ProviderDiscovery:
        return ProviderDiscovery(provider=impl, models=[], error=None)

    monkeypatch.setattr(providers_route, "test_provider_credentials", _ok)
    resp = client.post(
        "/v1/provider-credentials/test",
        json={"provider_type": "anthropic-compatible", "api_base": "http://x/v1", "api_key": "k"},
        headers=master_key_header,
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "model_count": 0, "error": None}


def test_test_connection_requires_a_target(client: TestClient, master_key_header: dict[str, str]) -> None:
    assert client.post("/v1/provider-credentials/test", json={}, headers=master_key_header).status_code == 400


def test_test_connection_requires_master_key(client: TestClient) -> None:
    assert client.post("/v1/provider-credentials/test", json={"instance": "openai"}).status_code in (401, 403)


def test_catalog_lists_known_providers(client: TestClient, master_key_header: dict[str, str]) -> None:
    resp = client.get("/v1/providers/catalog", headers=master_key_header)
    assert resp.status_code == 200
    by_id = {p["id"]: p for p in resp.json()}
    assert "openai" in by_id
    assert by_id["openai"]["requires_api_key"] is True
    assert by_id["openai"]["default_api_base"]  # openai has an explicit built-in base
    # Keyless local backends are reported as not requiring a key.
    assert by_id["ollama"]["requires_api_key"] is False


def test_catalog_requires_master_key(client: TestClient) -> None:
    assert client.get("/v1/providers/catalog").status_code in (401, 403)


def test_all_routes_require_master_key(client: TestClient) -> None:
    assert client.get("/v1/provider-credentials").status_code in (401, 403)
    assert client.post("/v1/provider-credentials", json={"instance": "x"}).status_code in (401, 403)
    assert client.patch("/v1/provider-credentials/x", json={}).status_code in (401, 403)
    assert client.delete("/v1/provider-credentials/x").status_code in (401, 403)
    assert client.post("/v1/provider-credentials/x/test").status_code in (401, 403)


def test_test_endpoint_maps_discovery_result(
    client: TestClient, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_key(monkeypatch)
    _create(client, master_key_header)

    # Unknown instance is a 404 before any provider is contacted.
    assert client.post("/v1/provider-credentials/ghost/test", headers=master_key_header).status_code == 404

    async def _ok(_config: object, instance: str) -> ProviderDiscovery:
        return ProviderDiscovery(provider=instance, models=[], error=None)

    monkeypatch.setattr(providers_route, "discover_provider_models", _ok)
    ok = client.post("/v1/provider-credentials/openai/test", headers=master_key_header)
    assert ok.status_code == 200
    assert ok.json() == {"ok": True, "model_count": 0, "error": None}

    async def _fail(_config: object, instance: str) -> ProviderDiscovery:
        return ProviderDiscovery(provider=instance, models=[], error="401 Unauthorized")

    monkeypatch.setattr(providers_route, "discover_provider_models", _fail)
    failed = client.post("/v1/provider-credentials/openai/test", headers=master_key_header)
    assert failed.status_code == 200
    assert failed.json() == {"ok": False, "model_count": 0, "error": "401 Unauthorized"}
