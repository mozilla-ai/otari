from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from gateway.core.config import GatewayConfig
from gateway.main import create_app
from gateway.services.pricing_service import configure_default_pricing, default_pricing_enabled


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


def test_settings_patch_does_not_apply_when_commit_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A toggle that is persisted but never committed must not mutate this
    # worker's in-memory config or the process-wide pricing flag; otherwise a
    # failed write would leave the gateway metering against an unpersisted value.
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'settings-rollback.db'}",
        master_key="sk-test-master",
        default_pricing=False,
        require_pricing=False,
    )
    configure_default_pricing(False)  # establish a known baseline for the global flag

    async def _boom(self: object) -> None:
        raise OperationalError("commit failed", None, Exception("boom"))

    # raise_server_exceptions=False so we observe the 500 the operator would see,
    # rather than the exception being re-raised into the test.
    with TestClient(create_app(config), raise_server_exceptions=False) as client:
        monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession.commit", _boom)
        response = client.patch(
            "/v1/settings",
            headers={"Authorization": "Bearer sk-test-master"},
            json={"default_pricing": True},
        )

    assert response.status_code == 500
    # The in-memory config and the global pricing flag stay at their pre-request value.
    assert config.default_pricing is False
    assert default_pricing_enabled() is False
