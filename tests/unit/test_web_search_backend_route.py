"""Endpoint tests for GET /v1/web-search/search.

The search backend a data-plane gateway calls when the deployment holding the
search credential is a different process. Covers what is mounted, who may call
it, and that no upstream detail reaches the caller.
"""

import json
import logging
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
from gateway.inflight import InFlightRegistry
from gateway.log_config import logger as gateway_logger
from gateway.main import create_app

BACKEND_TOKEN = "gw-default-token"
AUTH = {"X-Gateway-Token": BACKEND_TOKEN}

TAVILY_OK = {
    "results": [
        {"url": "https://example.com/a", "title": "A", "content": "snippet a", "raw_content": "page a"},
    ]
}

BRAVE_DATED = {
    "web": {
        "results": [
            {"url": "https://a.example", "title": "A", "description": "a", "page_age": "2026-08-30T00:00:00"},
        ]
    }
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


def test_the_recency_signal_survives_the_hop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Undeclared here, Pydantic would drop it, and a search over this hop would
    render without the date an in-process one renders with."""
    recorder = _Recorder(httpx.Response(200, json=BRAVE_DATED))
    monkeypatch.setattr(
        "gateway.api.routes.web_search_backend.get_search_client",
        lambda: httpx.AsyncClient(transport=recorder),
    )
    with _client(
        tmp_path,
        web_search_provider="brave",
        web_search_provider_api_key="brv-x",
        web_search_backend_token=BACKEND_TOKEN,
    ) as client:
        response = client.get("/v1/web-search/search", params={"q": "x"}, headers=AUTH)

    assert response.status_code == 200
    assert response.json()["results"][0]["published_date"] == "2026-08-30T00:00:00"


def test_max_results_is_bounded_by_the_server(tmp_path: Path, upstream: _Recorder) -> None:
    """The caller is another gateway forwarding an opaque ``provider_options``
    bag, so the ceiling on upstream work and response size is enforced here."""
    with _configured(tmp_path) as client:
        response = client.get("/v1/web-search/search", params={"q": "x", "max_results": 500}, headers=AUTH)

    assert response.status_code == 422
    assert upstream.requests == []


def test_a_search_in_progress_is_registered_in_flight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A paid call nobody can see running is a paid call nobody can see hanging.

    Read from inside the provider call, which is the only moment the entry
    exists: ``InFlightMiddleware`` drops it once the response is sent.
    """
    in_flight: list[tuple[str, str | None]] = []
    registry: list[InFlightRegistry] = []

    class _Watcher(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            in_flight.extend((entry.endpoint, entry.provider) for entry in registry[0].snapshot())
            return httpx.Response(200, json=TAVILY_OK)

    monkeypatch.setattr(
        "gateway.api.routes.web_search_backend.get_search_client",
        lambda: httpx.AsyncClient(transport=_Watcher()),
    )
    with _configured(tmp_path) as client:
        registry.append(client.app.state.inflight)  # type: ignore[attr-defined]
        assert client.get("/v1/web-search/search", params={"q": "x"}, headers=AUTH).status_code == 200

    assert in_flight == [("/v1/web-search/search", "tavily")]


def test_an_upstream_failure_is_logged_for_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing else on this path logs: the provider error only becomes an
    ``HTTPException``'s ``__cause__``, which FastAPI renders and discards."""
    recorder = _Recorder(httpx.Response(401, text="invalid api key tvly-secret"))
    monkeypatch.setattr(
        "gateway.api.routes.web_search_backend.get_search_client",
        lambda: httpx.AsyncClient(transport=recorder),
    )
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.ERROR, logger="gateway")
    try:
        with _configured(tmp_path) as client:
            assert client.get("/v1/web-search/search", params={"q": "x"}, headers=AUTH).status_code == 502
    finally:
        gateway_logger.removeHandler(caplog.handler)

    assert "tavily" in caplog.text
    assert "401" in caplog.text
    assert "tvly-secret" not in caplog.text


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


def _warnings_from(config: GatewayConfig, caplog: pytest.LogCaptureFixture) -> str:
    """What the (non-propagating) gateway logger said, per ``test_chat_output_cap``."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.WARNING, logger="gateway")
    try:
        config.warn_about_half_configured_web_search()
        return caplog.text
    finally:
        gateway_logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"web_search_provider": "tavily"}, "web_search_provider_api_key is not set"),
        ({"web_search_provider_api_key": "tvly-x"}, "web_search_provider is not set"),
    ],
)
def test_a_half_configured_provider_is_warned_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, overrides: dict[str, Any], expected: str
) -> None:
    """Warned rather than refused, like half-configured OAuth: web search is one
    optional tool, and a compose file can default the provider name without being
    able to default the secret, so refusing to boot would take the process down
    over a setting nobody typed.

    Silent is the alternative: neither half is read without the other, so a
    deployment that named a provider precisely so it would need no URL answers
    400 to every search and says nowhere why.
    """
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'x.db'}",
        master_key="sk-test-master",
        **overrides,
    )
    assert expected in _warnings_from(config, caplog)


def test_a_backend_token_with_no_provider_is_warned_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The token gates a route that is not mounted without a provider to serve,
    so on its own it silently does nothing."""
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'x.db'}",
        master_key="sk-test-master",
        web_search_backend_token=BACKEND_TOKEN,
    )
    assert "web_search_backend_token is set but no web-search provider" in _warnings_from(config, caplog)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"web_search_provider": "tavily", "web_search_provider_api_key": "tvly-x"}],
)
def test_neither_half_and_both_halves_are_both_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, overrides: dict[str, Any]
) -> None:
    """Configuring nothing is the ordinary state, not a mistake."""
    config = GatewayConfig(
        database_url=f"sqlite:///{tmp_path / 'x.db'}",
        master_key="sk-test-master",
        **overrides,
    )
    assert _warnings_from(config, caplog) == ""
