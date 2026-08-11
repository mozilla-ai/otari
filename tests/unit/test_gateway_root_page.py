import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import gateway.main as gateway_main
from gateway.core.config import GatewayConfig
from gateway.dashboard import get_dashboard_dir
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app
from gateway.services.secret_box import SecretBoxUnavailableError, generate_secret_key


def _config(tmp_path: Path, name: str) -> GatewayConfig:
    database_path = tmp_path / name
    return GatewayConfig(database_url=f"sqlite:///{database_path}")


# What the security middleware gives the PWA assets: public, but a day rather
# than the immutable year the content-hashed /assets bundles get.
_PWA_CACHE_CONTROL = "public, max-age=86400"


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


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built (run: make dashboard)")
def test_dashboard_is_served_at_root(tmp_path: Path) -> None:
    app = create_app(_config(tmp_path, "gateway-dashboard-test.db"))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert '<div id="root">' in response.text
    assert "Otari Dashboard" in response.text


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built (run: make dashboard)")
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


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built (run: make dashboard)")
def test_pwa_manifest_and_icons_are_served(tmp_path: Path) -> None:
    """Installing the dashboard to a phone home screen needs these to be reachable."""
    app = create_app(_config(tmp_path, "gateway-pwa-test.db"))

    with TestClient(app) as client:
        index = client.get("/").text
        manifest = client.get("/pwa/manifest.webmanifest")
        assert manifest.status_code == 200
        assert manifest.headers["cache-control"] == _PWA_CACHE_CONTROL
        payload = manifest.json()
        # Every icon the manifest advertises must resolve, or the launcher falls
        # back to a screenshot of the page instead of the Otari mark.
        icon_responses = {entry["src"]: client.get(entry["src"]) for entry in payload["icons"]}
        apple_icon = client.get("/pwa/apple-touch-icon.png")

    assert '<link rel="manifest" href="/pwa/manifest.webmanifest" />' in index
    assert '<link rel="apple-touch-icon" href="/pwa/apple-touch-icon.png" />' in index

    assert payload["name"] == "Otari Dashboard"
    assert payload["short_name"] == "Otari"
    assert payload["display"] == "standalone"
    # Android offers an install only with both a 192 and a 512 icon; the maskable
    # one keeps the mark from being cropped by the launcher's icon shape.
    assert {(entry["sizes"], entry["purpose"]) for entry in payload["icons"]} == {
        ("192x192", "any"),
        ("512x512", "any"),
        ("512x512", "maskable"),
    }
    for src, response in icon_responses.items():
        assert response.status_code == 200, src
        assert response.headers["content-type"] == "image/png", src
        # Pin the policy, not just "cacheable": these filenames are not
        # content-hashed, so a drift to no-cache or a different max-age changes
        # what an installed app re-fetches.
        assert response.headers["cache-control"] == _PWA_CACHE_CONTROL, src

    assert apple_icon.status_code == 200
    assert apple_icon.headers["content-type"] == "image/png"
    assert apple_icon.headers["cache-control"] == _PWA_CACHE_CONTROL


def test_pwa_assets_are_absent_without_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the standalone dashboard is installable, so nothing mounts /pwa without it."""
    monkeypatch.setattr(gateway_main, "get_dashboard_dir", lambda: None)
    app = create_app(_config(tmp_path, "gateway-no-pwa-test.db"))

    with TestClient(app) as client:
        response = client.get("/pwa/manifest.webmanifest")

    assert response.status_code == 404


def test_dashboard_without_pwa_dir_still_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bundle built before the PWA assets existed must not break startup.

    The guard this covers is not the one above: there the dashboard is absent
    entirely, while here ``assets/`` is present and only ``pwa/`` is missing, as
    in an older wheel or a stale Docker layer. Mounting ``StaticFiles`` on a
    directory that does not exist raises at construction, so without the
    ``is_dir()`` check the gateway would fail to boot rather than degrade.
    """
    stale_bundle = tmp_path / "dashboard"
    (stale_bundle / "assets").mkdir(parents=True)
    (stale_bundle / "index.html").write_text('<html><link rel="manifest" href="/pwa/manifest.webmanifest" /></html>')
    monkeypatch.setattr(gateway_main, "get_dashboard_dir", lambda: stale_bundle)

    app = create_app(_config(tmp_path, "gateway-stale-bundle-test.db"))

    with TestClient(app) as client:
        index = client.get("/")
        manifest = client.get("/pwa/manifest.webmanifest")

    assert index.status_code == 200
    # The stale index.html still links a manifest; it 404s rather than taking
    # the gateway down with it, and the page itself keeps working.
    assert manifest.status_code == 404


def test_create_app_rejects_invalid_secret_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", "not-a-valid-fernet-key")
    with pytest.raises(SecretBoxUnavailableError) as excinfo:
        create_app(_config(tmp_path, "gateway-bad-secret-test.db"))
    assert "not-a-valid-fernet-key" not in str(excinfo.value)


def test_create_app_accepts_valid_secret_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    app = create_app(_config(tmp_path, "gateway-good-secret-test.db"))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_root_falls_back_to_tutorial_without_dashboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a source checkout that has not built the frontend.
    monkeypatch.setattr(gateway_main, "get_dashboard_dir", lambda: None)
    app = create_app(_config(tmp_path, "gateway-no-dashboard-test.db"))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Otari Quickstart" in response.text


def _capture_gateway_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Route the ``gateway`` logger (which does not propagate) into caplog."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")


def test_missing_dashboard_is_explained_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The bundle is built rather than committed, so an unbuilt checkout is ordinary.

    Serving the tutorial at "/" then looks like a broken dashboard unless startup
    says which build step was not run.
    """
    monkeypatch.setattr(gateway_main, "get_dashboard_dir", lambda: None)
    _capture_gateway_logs(caplog)
    try:
        create_app(_config(tmp_path, "gateway-no-dashboard-log-test.db"))
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert "make dashboard" in caplog.text
    assert "static/dashboard" in caplog.text


def test_hybrid_mode_does_not_report_a_missing_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Hybrid mode has no local management API, so the tutorial is the intended root.

    Telling that operator to run `make dashboard` would send them after a bundle
    this mode does not serve, so the notice is standalone-only.
    """
    monkeypatch.setattr(gateway_main, "get_dashboard_dir", lambda: None)
    # The platform token alone selects hybrid, which is how a hybrid deployment is
    # configured (see deploy/render/render.hybrid.yaml). It is resolved once at
    # config-load time, so it has to be set before the config is built.
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw-test-token")
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'gateway-hybrid-log-test.db'}",
        platform={"base_url": "http://platform.test"},
    )
    assert config.is_hybrid_mode

    _capture_gateway_logs(caplog)
    try:
        create_app(config)
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert "make dashboard" not in caplog.text
