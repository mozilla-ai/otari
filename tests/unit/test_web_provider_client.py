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

from gateway.api.deps import (
    WebProviderClientDep,
    get_optional_web_provider_client,
    get_web_provider_client,
    reset_config,
)
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app
from gateway.services.providers import owned_endpoint_http_client
from gateway.services.tools import ResolvedFetch, build_web_provider_client
from gateway.services.tools._web_fetch import FetchFailure, fetch_page
from gateway.services.tools._web_provider_client import web_provider_client_for
from gateway.services.web_retrieval_policy import DomainPolicy


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


def test_the_tool_ports_read_the_client_without_depending_on_it(tmp_path: Path) -> None:
    """Every completion route takes the tool ports, so one that runs no web tool must not need the client."""
    app, _ = _app(tmp_path)
    with TestClient(app):
        assert get_optional_web_provider_client(Request({"type": "http", "app": app})) is app.state.web_provider_client
    assert get_optional_web_provider_client(Request({"type": "http", "app": app})) is None


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


@pytest.mark.asyncio
async def test_a_tenants_endpoint_is_dialed_as_an_owned_endpoint_and_others_on_the_shared_client() -> None:
    async with httpx.AsyncClient() as shared:
        assert web_provider_client_for(shared, tenant_endpoint=False) is shared
        assert web_provider_client_for(shared, tenant_endpoint=True) is owned_endpoint_http_client()


@pytest.mark.asyncio
async def test_the_owned_endpoint_client_keeps_no_cookie() -> None:
    """It serves every tenant's endpoints, so it keeps no cookie, as the shared client does not."""
    client = owned_endpoint_http_client()
    request = httpx.Request("GET", "https://provider.example/search")
    client.cookies.extract_cookies(httpx.Response(200, headers={"set-cookie": "session=tenant-a"}, request=request))
    assert not client.cookies


@pytest.mark.asyncio
@pytest.mark.parametrize("api_base", ["https://10.0.0.7/v1", "https://169.254.169.254/v1"])
async def test_a_fetch_on_a_tenants_internal_endpoint_is_refused_at_the_call(api_base: str) -> None:
    """A key stored before its endpoint was checked, or a name re-pointed since, still reaches no internal address."""
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    resolved = ResolvedFetch(
        name=None,
        provider="tabstack",
        runner="library",
        api_key="org-key",
        api_base=api_base,
        options={},
        timeout=5.0,
        label=None,
        rate_keys=(),
        max_chars=None,
        on_credential=True,
        tenant_endpoint=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as shared:
        outcome = await fetch_page(
            resolved,
            "https://example.com/page",
            client=shared,
            policy=DomainPolicy(),
            allow_private=False,
            max_chars=None,
        )

    assert isinstance(outcome, FetchFailure)
    assert sent == []
