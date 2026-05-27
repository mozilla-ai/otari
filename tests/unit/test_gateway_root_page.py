from pathlib import Path

from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.main import create_app


def test_root_tutorial_page_is_available(tmp_path: Path) -> None:
    database_path = tmp_path / "gateway-root-test.db"
    config = GatewayConfig(database_url=f"sqlite:///{database_path}")
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Gateway (Proxy Server)" in response.text
    assert "Gateway Quickstart" in response.text
    assert "bootstrap API key" in response.text
    assert "from openai import OpenAI" in response.text
    assert "YOUR_BOOTSTRAP_GATEWAY_KEY" in response.text
    assert "mozilla-ai.github.io/otari/gateway/quickstart" in response.text


def test_admin_dashboard_page_is_available(tmp_path: Path) -> None:
    database_path = tmp_path / "gateway-admin-test.db"
    config = GatewayConfig(
        database_url=f"sqlite:///{database_path}",
        master_key="test-master-key",
        bootstrap_api_key=False,
    )
    app = create_app(config)

    with TestClient(app) as client:
        response = client.get("/admin")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Otari Gateway Admin" in response.text
    assert 'data-view="policies"' in response.text
    assert "/v1/routing-policies" in response.text
    assert "/v1/route-traces/summary" in response.text
    assert "/v1/usage/summary" in response.text
    assert "/v1/budgets/alerts" in response.text
    assert "Otari-Key" in response.text
