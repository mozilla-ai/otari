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
        # The tenancy-scoped ceilings are the second budget surface and are
        # standalone-only for the same reason as the first: a hybrid gateway
        # holds no local budget rows, so the router is never registered.
        scoped_budgets_response = client.get("/v1/scoped-budgets")
        usage_response = client.get("/v1/usage")
        # Invitations are tenancy too: a hybrid gateway holds no membership
        # state to accept one into, and the router is never registered.
        validate_response = client.post(
            "/v1/invitations/validate", json={"token": "some-token"}
        )
        accept_response = client.post("/v1/invitations/accept", json={"token": "some-token"})

    expected = {"detail": "This endpoint is not available in hybrid mode. Manage this resource via the platform UI."}
    assert users_response.status_code == 404
    assert users_response.json() == expected
    assert keys_response.status_code == 404
    assert keys_response.json() == expected
    assert budgets_response.status_code == 404
    assert budgets_response.json() == expected
    assert scoped_budgets_response.status_code == 404
    assert validate_response.status_code == 404
    assert validate_response.json() == expected
    assert accept_response.status_code == 404
    assert accept_response.json() == expected
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
        for path in (
            "/v1/settings",
            # Covered by the /v1/settings/{path} stub: a hybrid gateway sends no
            # mail of its own, and the mail surface must 404 with the same hint
            # rather than reporting an unconfigured transport as if it were one
            # this deployment could configure.
            "/v1/settings/mail",
            "/v1/aliases",
            "/v1/providers",
            "/v1/pricing",
            "/v1/organizations/me",
            "/v1/workspaces",
            # A workspace's MCP servers are stored with a bearer token, and in
            # hybrid mode they live on the platform. Listed explicitly rather
            # than left to the `/v1/workspaces` entry above: this is a distinct
            # router, so re-mounting it would not show up in that check.
            "/v1/workspaces/11111111-1111-1111-1111-111111111111/mcp-servers",
            # A workspace's web-search configuration lives on the platform in
            # hybrid mode, where `prepare_gateway_tools` resolves it from
            # otari.ai rather than from a local row. Listed for the same reason
            # as the servers above: its own router, so re-mounting it would not
            # show up in the `/v1/workspaces` check.
            "/v1/workspaces/11111111-1111-1111-1111-111111111111/web-search",
        ):
            response = client.get(path)
            assert response.status_code == 404, path
            assert response.json() == expected, path

        # State-changing verbs are stubbed too (every method in the stubs'
        # shared ``_METHODS``), so a write cannot slip past the hybrid gate and
        # reach a local handler.
        patch_settings = client.patch("/v1/settings", json={"model_discovery": False})
        assert patch_settings.status_code == 404
        assert patch_settings.json() == expected
        # The send route specifically, not just the status GET: a regression that
        # left POST mounted while GET was stubbed would expose the one mail route
        # that does something.
        post_mail_test = client.post("/v1/settings/mail/test", json={"to": "ada@example.com"})
        assert post_mail_test.status_code == 404
        assert post_mail_test.json() == expected
        post_alias = client.post("/v1/aliases", json={"name": "x", "target": "anthropic:claude-opus-4"})
        assert post_alias.status_code == 404
        assert post_alias.json() == expected
        assert client.delete("/v1/aliases/x").status_code == 404
        # HEAD is in that list rather than derived from GET, which FastAPI does
        # only for a route that leaves its methods unspecified. Dropping it
        # would answer 405, saying the path is served here and only the verb was
        # wrong. No body to compare on a HEAD, so the status is the assertion.
        assert client.head("/v1/keys").status_code == 404

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


def test_hybrid_mode_root_falls_back_to_tutorial_without_a_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Hybrid serves the same dashboard bundle as standalone (it renders the
    # data-plane landing page there), so an unbuilt checkout degrades the same
    # way: the get-started tutorial at the root. Pinned with the bundle absent so
    # the assertion does not depend on whether this checkout happens to have one.
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    monkeypatch.setattr("gateway.main.get_dashboard_dir", lambda: None)

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
