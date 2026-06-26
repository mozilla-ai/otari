import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gateway.main as gateway_main
from gateway.core.config import GatewayConfig
from gateway.dashboard import get_dashboard_dir
from gateway.main import create_app


def _config(tmp_path: Path, name: str) -> GatewayConfig:
    database_path = tmp_path / name
    return GatewayConfig(database_url=f"sqlite:///{database_path}")


def test_welcome_tutorial_page_is_available(tmp_path: Path) -> None:
    app = create_app(_config(tmp_path, "gateway-welcome-test.db"))

    with TestClient(app) as client:
        response = client.get("/welcome")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Otari" in response.text
    assert "Proxy Server" not in response.text
    assert "Otari Quickstart" in response.text
    assert "bootstrap API key" in response.text
    assert "from openai import OpenAI" in response.text
    assert "YOUR_BOOTSTRAP_OTARI_KEY" in response.text
    assert "https://github.com/mozilla-ai/otari/blob/main/docs/quickstart.md" in response.text
    assert "mozilla-ai.github.io/otari/gateway/quickstart" not in response.text
    assert '<link rel="icon" type="image/svg+xml" href="/favicon.svg" />' in response.text


def test_favicon_is_served(tmp_path: Path) -> None:
    app = create_app(_config(tmp_path, "gateway-favicon-test.db"))

    with TestClient(app) as client:
        response = client.get("/favicon.svg")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.text.lstrip().startswith("<svg")
    assert response.headers["cache-control"] == "public, max-age=86400"


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built (run: npm --prefix web run build)")
def test_dashboard_is_served_at_root(tmp_path: Path) -> None:
    app = create_app(_config(tmp_path, "gateway-dashboard-test.db"))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div id="root">' in response.text
    assert "Otari Dashboard" in response.text


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built (run: npm --prefix web run build)")
def test_dashboard_assets_are_mounted_and_cacheable(tmp_path: Path) -> None:
    app = create_app(_config(tmp_path, "gateway-dashboard-assets-test.db"))

    with TestClient(app) as client:
        index = client.get("/").text
        asset_match = re.search(r'/assets/[^"\']+\.js', index)
        assert asset_match is not None, "expected a hashed JS asset reference in index.html"
        asset_response = client.get(asset_match.group(0))

    assert asset_response.status_code == 200
    # Hashed bundles are immutable, so the security middleware must not force no-store.
    assert "no-store" not in asset_response.headers.get("cache-control", "")


def test_root_falls_back_to_tutorial_without_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a source checkout that has not built the frontend.
    monkeypatch.setattr(gateway_main, "get_dashboard_dir", lambda: None)
    app = create_app(_config(tmp_path, "gateway-no-dashboard-test.db"))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Otari Quickstart" in response.text
