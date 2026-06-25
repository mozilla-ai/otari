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
    assert "Otari (Proxy Server)" in response.text
    assert "Otari Quickstart" in response.text
    assert "bootstrap API key" in response.text
    assert "from openai import OpenAI" in response.text
    assert "YOUR_BOOTSTRAP_OTARI_KEY" in response.text
    assert "https://github.com/mozilla-ai/otari/blob/main/docs/quickstart.md" in response.text
    assert "mozilla-ai.github.io/otari/gateway/quickstart" not in response.text
