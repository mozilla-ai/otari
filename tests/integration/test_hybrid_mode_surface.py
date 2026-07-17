import pytest
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app


def test_hybrid_mode_starts_without_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="hybrid",
        database_url="postgresql://127.0.0.1:1/does-not-exist",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["mode"] == "hybrid"
    assert payload["platform_reachable"] in {"yes", "no"}

    reset_config()
    reset_db()


def test_hybrid_mode_disables_local_management_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        users_response = client.post("/v1/users", json={"user_id": "u1"})
        keys_response = client.get("/v1/keys")
        budgets_response = client.get("/v1/budgets")
        usage_response = client.get("/v1/usage")

    expected = {"detail": "This endpoint is not available in hybrid mode. Manage this resource via the platform UI."}
    assert users_response.status_code == 404
    assert users_response.json() == expected
    assert keys_response.status_code == 404
    assert keys_response.json() == expected
    assert budgets_response.status_code == 404
    assert budgets_response.json() == expected
    assert usage_response.status_code == 404
    assert usage_response.json() == expected

    reset_config()
    reset_db()


def test_hybrid_mode_disables_dashboard_management_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    # The admin-dashboard management surface is standalone-only; in hybrid mode
    # it must be unavailable (owned by the platform), with the same helpful hint.
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    expected = {"detail": "This endpoint is not available in hybrid mode. Manage this resource via the platform UI."}
    with TestClient(app) as client:
        for path in ("/v1/settings", "/v1/aliases", "/v1/providers", "/v1/pricing"):
            response = client.get(path)
            assert response.status_code == 404, path
            assert response.json() == expected, path

        # State-changing verbs are stubbed too (api_route covers every method), so
        # a write cannot slip past the hybrid gate and reach a local handler.
        patch_settings = client.patch("/v1/settings", json={"model_discovery": False})
        assert patch_settings.status_code == 404
        assert patch_settings.json() == expected
        post_alias = client.post("/v1/aliases", json={"name": "x", "target": "anthropic:claude-opus-4"})
        assert post_alias.status_code == 404
        assert post_alias.json() == expected
        assert client.delete("/v1/aliases/x").status_code == 404

    reset_config()
    reset_db()


def test_hybrid_mode_omits_model_management_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    # models.router is standalone-only (register_routers returns early in hybrid),
    # so the dashboard's model-management reads have no route at all. Guards
    # against re-mounting models.router in hybrid, which would expose them.
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        for path in ("/v1/models/metadata", "/v1/models/discoverable"):
            assert client.get(path).status_code == 404, path

    reset_config()
    reset_db()


def test_hybrid_mode_root_serves_tutorial(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hybrid has no local management API, so the root serves the get-started
    # tutorial rather than the admin dashboard SPA.
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Your gateway is running." in response.text

    reset_config()
    reset_db()


def test_hybrid_mode_health_reports_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    async def _reachable(_: GatewayConfig) -> bool:
        return True

    monkeypatch.setattr("gateway.api.routes.health._check_platform_reachability", _reachable)

    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health")
        readiness_response = client.get("/health/readiness")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "mode": "hybrid", "platform_reachable": "yes"}
    assert readiness_response.status_code == 200
    assert readiness_response.json()["platform"] == "connected"

    reset_config()
    reset_db()


def test_hybrid_mode_readiness_fails_when_platform_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")

    async def _unreachable(_: GatewayConfig) -> bool:
        return False

    monkeypatch.setattr("gateway.api.routes.health._check_platform_reachability", _unreachable)

    config = GatewayConfig(
        mode="hybrid",
        platform={"base_url": "http://localhost:8100/api/v1"},
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/health/readiness")

    assert response.status_code == 503
    assert response.json()["detail"]["platform"] == "unavailable"

    reset_config()
    reset_db()
