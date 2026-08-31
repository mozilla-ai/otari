"""Endpoint tests for GET /v1/web-search/search.

The search backend a data-plane gateway calls when the deployment holding the
search credential is a different process. Covers what is mounted, who may call
it, and that no upstream detail reaches the caller.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.api.deps import reset_config
from gateway.api.routes.web_search_backend import _authorize
from gateway.core.config import GatewayConfig
from gateway.core.database import reset_db
from gateway.main import create_app

BACKEND_TOKEN = "gw-default-token"
AUTH = {"X-Gateway-Token": BACKEND_TOKEN}

TAVILY_OK = {
    "results": [
        {"url": "https://example.com/a", "title": "A", "content": "snippet a", "raw_content": "page a"},
    ]
}


@pytest.fixture(autouse=True)
def _clean_process_globals() -> Iterator[None]:
    yield
    reset_config()
    reset_db()


class _Recorder(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response | Exception) -> None:
        self._response = response
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Answer the provider call in-process, through the route's pooled client."""
    recorder = _Recorder(httpx.Response(200, json=TAVILY_OK))
    monkeypatch.setattr(
        "gateway.api.routes.web_search_backend.get_search_client",
        lambda: httpx.AsyncClient(transport=recorder),
    )
    return recorder


def _client(tmp_path: Path, **overrides: Any) -> TestClient:
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'web-search-backend.db'}",
        master_key="sk-test-master",
        **overrides,
    )
    return TestClient(create_app(config))


def _configured(tmp_path: Path, **overrides: Any) -> TestClient:
    return _client(
        tmp_path,
        web_search_provider="tavily",
        web_search_provider_api_key="tvly-x",
        web_search_backend_token=BACKEND_TOKEN,
        **overrides,
    )


def test_not_mounted_without_a_provider(tmp_path: Path) -> None:
    with _client(tmp_path, web_search_backend_token=BACKEND_TOKEN) as client:
        assert client.get("/v1/web-search/search", params={"q": "x"}, headers=AUTH).status_code == 404


def test_not_mounted_without_a_backend_token(tmp_path: Path) -> None:
    """A route that cannot recognize its own gateway is absent, not open.

    It spends this deployment's search quota, and a control plane is
    internet-reachable, so an unauthenticated version of it is not a lesser
    version of the same thing.
    """
    with _client(tmp_path, web_search_provider="tavily", web_search_provider_api_key="tvly-x") as client:
        assert client.get("/v1/web-search/search", params={"q": "x"}, headers=AUTH).status_code == 404


def test_serves_searxng_shaped_results(tmp_path: Path, upstream: _Recorder) -> None:
    with _configured(tmp_path) as client:
        response = client.get("/v1/web-search/search", params={"q": "claude code"}, headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "url": "https://example.com/a",
                "title": "A",
                "content": "snippet a",
                "extracted_content": "page a",
            }
        ]
    }
    assert upstream.requests[0].url.host == "api.tavily.com"


def test_refuses_a_missing_token(tmp_path: Path, upstream: _Recorder) -> None:
    with _configured(tmp_path) as client:
        assert client.get("/v1/web-search/search", params={"q": "x"}).status_code == 401
    assert upstream.requests == []


def test_refuses_a_wrong_token(tmp_path: Path, upstream: _Recorder) -> None:
    with _configured(tmp_path) as client:
        response = client.get(
            "/v1/web-search/search",
            params={"q": "x"},
            headers={"X-Gateway-Token": "gw-someone-elses"},
        )
    assert response.status_code == 401
    assert upstream.requests == []


def test_forwards_only_the_provider_options_it_declares(tmp_path: Path, upstream: _Recorder) -> None:
    with _configured(tmp_path) as client:
        response = client.get(
            "/v1/web-search/search",
            params={"q": "x", "search_depth": "advanced", "format": "json", "engines": "google"},
            headers=AUTH,
        )

    assert response.status_code == 200
    body = json.loads(upstream.requests[0].content)
    assert body["search_depth"] == "advanced"
    assert "engines" not in body
    assert "format" not in body


def test_upstream_failure_does_not_leak_the_provider_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder(httpx.Response(401, text="invalid api key tvly-secret"))
    monkeypatch.setattr(
        "gateway.api.routes.web_search_backend.get_search_client",
        lambda: httpx.AsyncClient(transport=recorder),
    )
    with _configured(tmp_path) as client:
        response = client.get("/v1/web-search/search", params={"q": "x"}, headers=AUTH)

    assert response.status_code == 502
    assert "tvly-secret" not in response.text


def test_a_non_ascii_token_is_a_refusal_not_a_crash(tmp_path: Path) -> None:
    """Starlette decodes a header as latin-1, so a byte above 0x7f reaches the
    comparison as a non-ASCII string, on which ``compare_digest`` raises.

    Asserted against ``_authorize`` rather than over HTTP because httpx refuses
    to *send* such a header, while curl or a raw socket sends it happily. Without
    the hash it is an unhandled 500 from an unauthenticated caller.
    """
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'x.db'}",
        master_key="sk-test-master",
        web_search_backend_token=BACKEND_TOKEN,
    )
    with pytest.raises(HTTPException) as raised:
        _authorize(config, "gw-\u00e9\u00e9\u00e9")
    assert raised.value.status_code == 401


def test_a_half_configured_provider_is_refused_at_startup(tmp_path: Path) -> None:
    """Either half alone runs no search, and without this the deployment that set
    the provider precisely so it would need no URL answers 400 to every search."""
    with pytest.raises(ValueError, match="web_search_provider_api_key"):
        GatewayConfig(
            database_url=f"sqlite:///{tmp_path / 'x.db'}",
            master_key="sk-test-master",
            web_search_provider="tavily",
        ).validate_web_search_provider()

    with pytest.raises(ValueError, match="names no provider"):
        GatewayConfig(
            database_url=f"sqlite:///{tmp_path / 'x.db'}",
            master_key="sk-test-master",
            web_search_provider_api_key="tvly-x",
        ).validate_web_search_provider()


def test_both_halves_together_pass_startup(tmp_path: Path) -> None:
    GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'x.db'}",
        master_key="sk-test-master",
        web_search_provider="tavily",
        web_search_provider_api_key="tvly-x",
    ).validate_web_search_provider()
