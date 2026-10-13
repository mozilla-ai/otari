"""The web provider client: one per app, built and closed by the lifespan (decision 55).

Every route that calls a web provider takes the app's client through ``get_web_provider_client``.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from gateway.api.deps import WebProviderClientDep, get_web_provider_client, reset_config
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.services.tools import build_web_provider_client


@pytest.fixture(autouse=True)
def _clean_process_globals() -> Iterator[None]:
    yield
    reset_config()
    reset_db()


def _app(tmp_path: Path) -> tuple[FastAPI, list[httpx.AsyncClient]]:
    """An app with a route that records the client each request gets."""
    app = create_app(GatewayConfig(database_url=f"sqlite:///{tmp_path / 'client.db'}", master_key="sk-test-master"))
    served: list[httpx.AsyncClient] = []

    async def probe(client: WebProviderClientDep) -> dict[str, bool]:
        served.append(client)
        return {"ok": True}

    app.add_api_route("/probe-web-provider-client", probe)
    return app, served


def test_the_lifespan_builds_one_client_that_every_request_gets_and_closes_it(tmp_path: Path) -> None:
    app, served = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/probe-web-provider-client").status_code == 200
        assert client.get("/probe-web-provider-client").status_code == 200
        built = app.state.web_provider_client

    first, second = served
    assert first is second is built
    assert built.is_closed
    # Removed as well as closed, so a route served after shutdown fails plainly.
    assert not hasattr(app.state, "web_provider_client")


def test_an_app_whose_lifespan_is_not_running_has_no_client() -> None:
    request = Request({"type": "http", "app": FastAPI()})
    with pytest.raises(RuntimeError, match="lifespan"):
        get_web_provider_client(request)


def _through(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[dict[str, Any]]:
    """Build the client with ``handler`` as its transport, recording what it was built with."""
    built: list[dict[str, Any]] = []
    real = httpx.AsyncClient

    def recording(**kwargs: Any) -> httpx.AsyncClient:
        built.append(kwargs)
        return real(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("gateway.services.tools._web_provider_client.httpx.AsyncClient", recording)
    return built


@pytest.mark.asyncio
async def test_a_cookie_a_provider_sets_never_rides_on_the_next_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """One client serves every tenant's calls, so a cookie set for one must not reach another's."""
    sent: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.headers.get("cookie"))
        return httpx.Response(200, headers={"set-cookie": "session=tenant-a; Path=/"})

    _through(monkeypatch, handler)
    async with build_web_provider_client() as client:
        await client.get("https://provider.example/search")
        await client.get("https://provider.example/search")
        assert not client.cookies

    assert sent == [None, None]


@pytest.mark.asyncio
async def test_the_client_caps_no_connections_and_keeps_20_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx's default of 100 would queue a burst of calls into pool timeouts."""
    built = _through(monkeypatch, lambda _request: httpx.Response(200))
    await build_web_provider_client().aclose()
    (kwargs,) = built
    assert kwargs["limits"] == httpx.Limits(max_connections=None, max_keepalive_connections=20)
    # httpx's defaults otherwise, so environment proxies still apply.
    assert "trust_env" not in kwargs
    assert "timeout" not in kwargs
