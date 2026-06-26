from pathlib import Path

from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app


def _client(tmp_path: Path, *, default_pricing: bool = False, require_pricing: bool = True) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'settings-test.db'}",
        master_key="sk-test-master",
        default_pricing=default_pricing,
        require_pricing=require_pricing,
    )
    return TestClient(create_app(config))


def test_settings_requires_auth(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        assert client.get("/v1/settings").status_code == 401


def test_settings_rejects_non_master_key(tmp_path: Path) -> None:
    # The settings route is admin-only: a token that is not the master key is rejected.
    with _client(tmp_path) as client:
        response = client.get("/v1/settings", headers={"Authorization": "Bearer not-the-master-key"})
    assert response.status_code == 401


def test_settings_reports_pricing_flags(tmp_path: Path) -> None:
    with _client(tmp_path, default_pricing=True, require_pricing=False) as client:
        response = client.get("/v1/settings", headers={"Authorization": "Bearer sk-test-master"})

    assert response.status_code == 200
    body = response.json()
    assert body["default_pricing"] is True
    assert body["require_pricing"] is False
    assert body["mode"] == "standalone"
    assert "version" in body


def test_settings_defaults(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/v1/settings", headers={"Authorization": "Bearer sk-test-master"})

    assert response.status_code == 200
    body = response.json()
    # default_pricing is off by default; require_pricing is fail-closed by default.
    assert body["default_pricing"] is False
    assert body["require_pricing"] is True
