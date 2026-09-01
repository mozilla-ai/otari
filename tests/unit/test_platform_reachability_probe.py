"""The ``platform_reachable`` probe must hit the configured path and believe only a 2xx.

Two defects met here. The probe's path could only come from a config file, so an
env-only deployment was stuck on a route specific to otari.ai; and any status
below 500 counted as reachable, so the peer that does not serve that route
answered 404 and still reported healthy. ``PLATFORM_HEALTH_PATH`` fixes the
first, requiring a 2xx the second.
"""

from pathlib import Path
from typing import Any

import httpx
import pytest

from gateway.api.routes.health import _check_platform_reachability
from gateway.core.config import DEFAULT_PLATFORM_HEALTH_PATH, load_config


class _StubTransport(httpx.AsyncBaseTransport):
    """Answers every probe with one canned response, or raises one canned error."""

    def __init__(self, response: httpx.Response | Exception) -> None:
        self._response = response
        self.requested: list[httpx.URL] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(request.url)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a factory installing a stub transport and reporting what was probed."""

    def install(response: httpx.Response | Exception = httpx.Response(200)) -> _StubTransport:
        stub = _StubTransport(response)
        original_init = httpx.AsyncClient.__init__

        def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = stub
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
        return stub

    return install


@pytest.fixture(autouse=True)
def _isolated_platform_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give each test the env these tests set and nothing else.

    ``load_config`` reads a ``.env`` from the working directory, which would put
    a contributor's local ``PLATFORM_*`` values back after the deletes below.
    """
    for name in ("OTARI_AI_TOKEN", "PLATFORM_BASE_URL", "PLATFORM_HEALTH_PATH", "OTARI_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _hybrid_config(monkeypatch: pytest.MonkeyPatch, health_path: str | None = None) -> Any:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    monkeypatch.setenv("PLATFORM_BASE_URL", "http://platform.test/v1")
    if health_path is not None:
        monkeypatch.setenv("PLATFORM_HEALTH_PATH", health_path)
    return load_config()


@pytest.mark.asyncio
async def test_health_path_env_var_moves_the_probe(monkeypatch: pytest.MonkeyPatch, probe: Any) -> None:
    stub = probe()
    config = _hybrid_config(monkeypatch, "/healthz")

    assert await _check_platform_reachability(config) is True
    assert [str(url) for url in stub.requested] == ["http://platform.test/v1/healthz"]


@pytest.mark.asyncio
async def test_probe_falls_back_to_the_platform_route(monkeypatch: pytest.MonkeyPatch, probe: Any) -> None:
    stub = probe()
    config = _hybrid_config(monkeypatch)

    assert await _check_platform_reachability(config) is True
    assert [str(url) for url in stub.requested] == [
        f"http://platform.test/v1{DEFAULT_PLATFORM_HEALTH_PATH}"
    ]


@pytest.mark.parametrize("status", [200, 204])
@pytest.mark.asyncio
async def test_a_success_reports_reachable(monkeypatch: pytest.MonkeyPatch, probe: Any, status: int) -> None:
    probe(httpx.Response(status))
    config = _hybrid_config(monkeypatch)

    assert await _check_platform_reachability(config) is True


@pytest.mark.parametrize(
    ("status", "why"),
    [
        (404, "the peer does not serve the configured path"),
        (401, "the route exists but refused the probe"),
        (403, "the route exists but refused the probe"),
        (500, "the peer is serving errors"),
    ],
)
@pytest.mark.asyncio
async def test_a_non_success_reports_unreachable(
    monkeypatch: pytest.MonkeyPatch, probe: Any, status: int, why: str
) -> None:
    """A status below 500 used to count as reachable, which is what #877 reports."""
    probe(httpx.Response(status))
    config = _hybrid_config(monkeypatch)

    assert await _check_platform_reachability(config) is False, why


@pytest.mark.asyncio
async def test_a_redirect_is_not_followed_and_reports_unreachable(
    monkeypatch: pytest.MonkeyPatch, probe: Any
) -> None:
    """A peer route that bounces to a login page has not answered the probe."""
    stub = probe(httpx.Response(302, headers={"location": "https://platform.test/login"}))
    config = _hybrid_config(monkeypatch)

    assert await _check_platform_reachability(config) is False
    assert [str(url) for url in stub.requested] == [
        f"http://platform.test/v1{DEFAULT_PLATFORM_HEALTH_PATH}"
    ]


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectTimeout("timed out"),
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("malformed response"),
        httpx.ProxyError("bad proxy"),
    ],
)
@pytest.mark.asyncio
async def test_a_transport_error_reports_unreachable(
    monkeypatch: pytest.MonkeyPatch, probe: Any, error: Exception
) -> None:
    """Only timeouts and network errors were caught, so the rest reached the client as a 500."""
    probe(error)
    config = _hybrid_config(monkeypatch)

    assert await _check_platform_reachability(config) is False
