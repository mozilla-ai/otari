"""The pure parts of connecting to otari.ai: the address rule, the setting, and the client's reading of answers."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from gateway.core.config import API_ROOT, GatewayConfig, load_config
from gateway.core.settings.providers import https_or_loopback_problem
from gateway.services.providers._platform_connect import PlatformClient, PlatformProtocolError, PollDelay, _IssuedKey
from gateway.services.url_safety import UnsafeURLError, validate_credential_address


@pytest.mark.parametrize(
    "url",
    [
        "https://api.otari.ai",
        "https://api.otari.ai/api/v1",
        "http://localhost:8000",
        "http://127.0.0.1:8000/x",
        "http://[::1]:8000",
    ],
)
def test_https_and_local_http_addresses_are_accepted(url: str) -> None:
    assert https_or_loopback_problem(url) is None
    validate_credential_address(url, label="test")


@pytest.mark.parametrize(
    "url",
    [
        "http://api.otari.ai",
        "http://localhost.example.com",
        "http://10.0.0.5:8000",
        "ftp://localhost",
        "javascript:alert(1)",
        "api.otari.ai",
        "https://user:secret@api.otari.ai",
        "https://[::1",
    ],
)
def test_other_addresses_are_refused(url: str) -> None:
    assert https_or_loopback_problem(url) is not None
    with pytest.raises(UnsafeURLError):
        validate_credential_address(url, label="test")


def test_the_platform_url_defaults_to_otari_ai() -> None:
    config = GatewayConfig(database_url="sqlite://")
    assert config.platform_url == "https://api.otari.ai"
    assert config.platform_connect_available is True


@pytest.mark.parametrize("value", ["", "   ", None])
def test_an_empty_platform_url_switches_the_feature_off(value: str | None) -> None:
    config = GatewayConfig(database_url="sqlite://", platform_url=value)  # type: ignore[arg-type]
    assert config.platform_url == ""
    assert config.platform_connect_available is False


def test_the_platform_url_is_trimmed_of_its_trailing_slash() -> None:
    assert GatewayConfig(database_url="sqlite://", platform_url=" http://localhost:9000/ ").platform_url == (
        "http://localhost:9000"
    )


@pytest.mark.parametrize(
    "value", ["http://platform.example.com", "https://u:p@platform.example.com", "https://platform.example.com?x=1"]
)
def test_a_platform_url_a_device_code_may_not_travel_to_fails_the_load(value: str) -> None:
    with pytest.raises(ValidationError, match="platform_url"):
        GatewayConfig(database_url="sqlite://", platform_url=value)


def test_an_empty_environment_variable_switches_the_feature_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other ``OTARI_<FIELD>`` reads an empty variable as unset; this one is how the feature is removed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OTARI_DATABASE_URL", "sqlite://")
    monkeypatch.setenv("OTARI_PLATFORM_URL", "")
    assert load_config().platform_connect_available is False

    monkeypatch.setenv("OTARI_PLATFORM_URL", "http://localhost:9000")
    assert load_config().platform_url == "http://localhost:9000"


def _client(status: int, body: Any, seen: list[httpx.Request] | None = None) -> PlatformClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if isinstance(body, bytes):
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=body)

    return PlatformClient("https://platform.test", transport=httpx.MockTransport(handler))


_GRANT = {
    "device_code": "dc",
    "user_code": "ABCD",
    "verification_uri": "https://otari.test/connect",
    "verification_uri_complete": "https://otari.test/connect?code=ABCD",
    "expires_in": 600,
    "interval": 5,
}


@pytest.mark.asyncio
async def test_request_code_posts_the_gateway_name_without_credentials() -> None:
    seen: list[httpx.Request] = []

    grant = await _client(200, _GRANT, seen).request_code("gateway.example.com")

    assert (grant.device_code, grant.user_code, grant.interval) == ("dc", "ABCD", 5)
    [request] = seen
    assert str(request.url) == f"https://platform.test{API_ROOT}/connect/device"
    assert json.loads(request.content) == {"gateway_name": "gateway.example.com"}
    assert "authorization" not in request.headers
    assert "cookie" not in request.headers


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, {}),
        (302, {}),
        (200, b"not json"),
        (200, {**_GRANT, "device_code": ""}),
        (200, {key: value for key, value in _GRANT.items() if key != "user_code"}),
        (200, {**_GRANT, "expires_in": 0}),
    ],
)
@pytest.mark.asyncio
async def test_request_code_refuses_an_answer_outside_the_contract(status: int, body: Any) -> None:
    with pytest.raises(PlatformProtocolError):
        await _client(status, body).request_code("gateway")


@pytest.mark.parametrize("error", ["authorization_pending", "expired_token", "access_denied"])
@pytest.mark.asyncio
async def test_exchange_reads_each_refusal(error: str) -> None:
    assert await _client(400, {"error": error}).exchange("dc") == error


@pytest.mark.asyncio
async def test_exchange_reads_slow_down_and_a_rate_limit_as_a_delay() -> None:
    assert await _client(400, {"error": "slow_down"}).exchange("dc") == PollDelay()

    def limited(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Too many requests"}, headers={"Retry-After": "30"})

    client = PlatformClient("https://platform.test", transport=httpx.MockTransport(limited))
    assert await client.exchange("dc") == PollDelay(30)


@pytest.mark.parametrize("status", [422, 429, 503])
@pytest.mark.asyncio
async def test_request_code_treats_a_refusal_to_start_as_unavailable(status: int) -> None:
    with pytest.raises(PlatformProtocolError):
        await _client(status, {"detail": "otari.ai's own words"}).request_code("gateway")


@pytest.mark.asyncio
async def test_exchange_reads_the_issued_key_and_keeps_it_out_of_its_repr() -> None:
    issued = await _client(200, {"api_key": "tk-secret", "api_base": "https://api.otari.ai/api/v1"}).exchange("dc")

    assert isinstance(issued, _IssuedKey)
    assert (issued.api_key, issued.api_base) == ("tk-secret", "https://api.otari.ai/api/v1")
    assert "tk-secret" not in repr(issued)


@pytest.mark.parametrize(
    ("status", "body"), [(400, {"error": "invalid_grant"}), (400, {}), (401, {}), (200, {"api_key": "k"}), (503, {})]
)
@pytest.mark.asyncio
async def test_exchange_refuses_an_answer_outside_the_contract(status: int, body: Any) -> None:
    with pytest.raises(PlatformProtocolError):
        await _client(status, body).exchange("dc")


@pytest.mark.asyncio
async def test_an_unreachable_platform_is_a_protocol_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = PlatformClient("https://platform.test", transport=httpx.MockTransport(handler))
    with pytest.raises(PlatformProtocolError):
        await client.exchange("dc")
