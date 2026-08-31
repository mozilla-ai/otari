"""The ``platform_reachable`` probe must hit the path the deployment configured.

The default path is a platform route. A peer that does not serve it answers
404, which the probe counts as reachable, so an unverified peer reports healthy.
``PLATFORM_HEALTH_PATH`` is what moves the probe off that default in an
env-only deployment.
"""

from typing import Any

import httpx
import pytest

from gateway.api.routes.health import _check_platform_reachability
from gateway.core.config import load_config


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requested: list[httpx.URL] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(request.url)
        return httpx.Response(200)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _RecordingTransport:
    recorder = _RecordingTransport()
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = recorder
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return recorder


@pytest.fixture(autouse=True)
def _isolated_platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OTARI_AI_TOKEN", "PLATFORM_BASE_URL", "PLATFORM_HEALTH_PATH", "OTARI_MODE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_health_path_env_var_moves_the_probe(
    monkeypatch: pytest.MonkeyPatch, transport: _RecordingTransport
) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    monkeypatch.setenv("PLATFORM_BASE_URL", "http://platform.test/v1")
    monkeypatch.setenv("PLATFORM_HEALTH_PATH", "/healthz")

    config = load_config()

    assert await _check_platform_reachability(config) is True
    assert [str(url) for url in transport.requested] == ["http://platform.test/v1/healthz"]


@pytest.mark.asyncio
async def test_probe_falls_back_to_the_platform_route(
    monkeypatch: pytest.MonkeyPatch, transport: _RecordingTransport
) -> None:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gw_test_token")
    monkeypatch.setenv("PLATFORM_BASE_URL", "http://platform.test/v1")

    config = load_config()

    assert await _check_platform_reachability(config) is True
    assert [str(url) for url in transport.requested] == ["http://platform.test/v1/utils/health-check/"]
