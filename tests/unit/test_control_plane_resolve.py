"""What the resolve client tells a caller when the control plane will not answer.

The errors here carry their own status and are rendered by the API layer rather
than by the registered family handler, which would give a 5xx a generic detail
instead of the one the caller has always seen.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from conftest import InstallControlPlane
from gateway.exceptions.control_plane_exceptions import (
    ControlPlaneNotConfiguredError,
    ControlPlaneRefusedError,
    ControlPlaneUnavailableError,
)
from gateway.services.control_plane import NOT_CONFIGURED_DETAIL, UNAVAILABLE_DETAIL, ResolveEndpoint, resolve


def _config(base_url: str | None = "https://control-plane.local") -> Any:
    platform: dict[str, Any] = {"resolve_timeout_ms": 5000}
    if base_url is not None:
        platform["base_url"] = base_url
    return SimpleNamespace(platform=platform, platform_token="gw_test_token")


def _answers(response: httpx.Response) -> Any:
    async def handler(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> Any:
        return response

    return handler


def _raises(error: Exception) -> Any:
    async def handler(*, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float) -> Any:
        raise error

    return handler


async def _ask(endpoint: ResolveEndpoint = ResolveEndpoint.PROVIDER_KEYS, config: Any | None = None) -> Any:
    return await resolve(config or _config(), user_token="tk_user", endpoint=endpoint, body={})


@pytest.mark.asyncio
async def test_an_address_the_deployment_does_not_have_is_its_own_fault(
    control_plane_transport: InstallControlPlane,
) -> None:
    control_plane_transport(_answers(httpx.Response(200, json={})))

    with pytest.raises(ControlPlaneNotConfiguredError) as raised:
        await _ask(config=_config(base_url=None))

    assert raised.value.status_code == 500
    assert raised.value.message == NOT_CONFIGURED_DETAIL


@pytest.mark.asyncio
async def test_a_refusal_keeps_the_status_and_the_wording_the_peer_chose(
    control_plane_transport: InstallControlPlane,
) -> None:
    control_plane_transport(_answers(httpx.Response(402, json={"detail": "Wallet empty"})))

    with pytest.raises(ControlPlaneRefusedError) as raised:
        await _ask()

    assert raised.value.status_code == 402
    assert raised.value.message == "Wallet empty"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "detail"),
    [
        (ResolveEndpoint.CODE_EXECUTION, "Code execution resolution failed"),
        (ResolveEndpoint.MCP_SERVERS, "MCP server resolution failed"),
        (ResolveEndpoint.PROVIDER_KEYS, "Authorization request rejected"),
        (ResolveEndpoint.WEB_SEARCH, "Web search resolution failed"),
    ],
)
async def test_a_refusal_with_nothing_usable_falls_back_to_the_endpoints_own_wording(
    control_plane_transport: InstallControlPlane,
    endpoint: ResolveEndpoint,
    detail: str,
) -> None:
    control_plane_transport(_answers(httpx.Response(403, json={"detail": {"nested": "object"}})))

    with pytest.raises(ControlPlaneRefusedError) as raised:
        await _ask(endpoint)

    assert raised.value.message == detail


@pytest.mark.asyncio
async def test_only_a_rate_limit_carries_the_peers_retry_hint(
    control_plane_transport: InstallControlPlane,
) -> None:
    control_plane_transport(_answers(httpx.Response(429, json={"detail": "Slow down"}, headers={"Retry-After": "30"})))

    with pytest.raises(ControlPlaneRefusedError) as raised:
        await _ask()

    assert raised.value.retry_after == "30"

    control_plane_transport(_answers(httpx.Response(404, json={"detail": "No"}, headers={"Retry-After": "30"})))

    with pytest.raises(ControlPlaneRefusedError) as raised:
        await _ask()

    assert raised.value.retry_after is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"detail": "peer exploded"}),
        httpx.Response(422, json={"detail": "field required"}),
        httpx.Response(200, content=b"not json"),
    ],
)
async def test_nothing_about_the_peers_own_failure_reaches_the_caller(
    control_plane_transport: InstallControlPlane,
    response: httpx.Response,
) -> None:
    """A 5xx, a validation error and an unreadable body are one answer to a caller."""
    control_plane_transport(_answers(response))

    with pytest.raises(ControlPlaneUnavailableError) as raised:
        await _ask()

    assert raised.value.status_code == 502
    assert raised.value.message == UNAVAILABLE_DETAIL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.ConnectTimeout("timed out"), httpx.ConnectError("refused")],
)
async def test_a_peer_that_cannot_be_reached_is_the_same_answer(
    control_plane_transport: InstallControlPlane,
    error: Exception,
) -> None:
    control_plane_transport(_raises(error))

    with pytest.raises(ControlPlaneUnavailableError):
        await _ask()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code", "detail", "retry_after"),
    [
        (ControlPlaneUnavailableError(UNAVAILABLE_DETAIL), 502, UNAVAILABLE_DETAIL, None),
        (ControlPlaneNotConfiguredError(NOT_CONFIGURED_DETAIL), 500, NOT_CONFIGURED_DETAIL, None),
        (ControlPlaneRefusedError("Slow down", status_code=429, retry_after="30"), 429, "Slow down", "30"),
    ],
)
async def test_an_error_that_reaches_the_app_is_rendered_whole(
    error: Exception,
    status_code: int,
    detail: str,
    retry_after: str | None,
) -> None:
    """The tenancy family handler would give a 5xx a generic detail and drop the retry hint."""
    from gateway.main import _control_plane_error_handler

    response = await _control_plane_error_handler(cast(Any, None), error)

    assert response.status_code == status_code
    assert json.loads(bytes(response.body)) == {"detail": detail}
    assert response.headers.get("Retry-After") == retry_after
