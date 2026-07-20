"""Unit tests for the dashboard build id served to open tabs.

A tab that loaded the dashboard before a deploy keeps running the old bundle
until someone reloads it, and the usual folk remedy is a hard refresh or
clearing storage. /dashboard-build.json gives the tab a way to notice, so it can
offer an ordinary reload instead.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import GatewayConfig
from gateway.dashboard import get_dashboard_build_id, get_dashboard_dir
from gateway.main import create_app


def _config(tmp_path: Path) -> GatewayConfig:
    return GatewayConfig(database_url=f"sqlite:///{tmp_path / 'build-id.db'}")


def test_build_id_tracks_index_html(tmp_path: Path) -> None:
    index = tmp_path / "index.html"
    index.write_text('<script src="/assets/index-AAAA.js"></script>')
    first = get_dashboard_build_id(tmp_path)

    # Same bundle, asked twice: an unchanged app must not nag anyone to reload.
    assert get_dashboard_build_id(tmp_path) == first

    # Vite stamps the content-hashed asset name into the page, so a rebuilt app
    # always changes this file even when index.html's structure is identical.
    index.write_text('<script src="/assets/index-BBBB.js"></script>')
    assert get_dashboard_build_id(tmp_path) != first


def test_build_id_is_short_and_opaque(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<html></html>")
    build_id = get_dashboard_build_id(tmp_path)

    assert len(build_id) == 16
    assert build_id.isalnum()


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built")
def test_endpoint_reports_the_served_bundle(tmp_path: Path) -> None:
    app = create_app(_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/dashboard-build.json")

    assert response.status_code == 200
    body = response.json()
    dashboard_dir = get_dashboard_dir()
    assert dashboard_dir is not None
    assert body["build"] == get_dashboard_build_id(dashboard_dir)
    assert body["version"]


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built")
def test_endpoint_is_public_and_uncacheable(tmp_path: Path) -> None:
    # The poll only works if it is not served from cache, and it has to answer
    # the login screen too, which has no key to send.
    app = create_app(_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/dashboard-build.json")

    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]


@pytest.mark.skipif(get_dashboard_dir() is None, reason="dashboard bundle not built")
def test_index_html_is_uncacheable_so_a_reload_gets_the_new_bundle(tmp_path: Path) -> None:
    # The whole prompt rests on this: "Update now" is a plain reload, which only
    # helps if the browser refetches the page rather than reusing a cached copy
    # pointing at the old hashed asset.
    app = create_app(_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
