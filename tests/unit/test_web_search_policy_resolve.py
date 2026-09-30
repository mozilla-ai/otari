"""Resolving a workspace's web search policy against a peer control plane."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from conftest import InstallControlPlane
from gateway.adapters.web_search_policy_adapter import LocalWebSearchPolicy, RemoteWebSearchPolicy
from gateway.exceptions.control_plane_exceptions import ControlPlaneError, ControlPlaneRefusedError
from gateway.exceptions.tools_exceptions import WebSearchPolicyResolutionFailedError, WebSearchPolicyResolutionFailure
from gateway.ports.web_search_policy_port import WebSearchPolicyScope


def _scope(user_token: str | None = "tk_user") -> WebSearchPolicyScope:
    """The scope a hybrid caller supplies, which carries a token and no workspace."""
    return WebSearchPolicyScope(workspace_id=None, user_token=user_token)


def _config(*, base_url: str | None = "https://platform.local") -> Any:
    cfg = MagicMock()
    cfg.platform = {"base_url": base_url, "resolve_timeout_ms": 5000} if base_url else {}
    cfg.platform_token = "gw_test_token"
    return cfg


def _answer(response: httpx.Response, install: InstallControlPlane) -> dict[str, Any]:
    """Answer the next control plane call with ``response``, and record the request."""
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured.update(url=url, headers=headers, body=body)
        return response

    install(fake_post)
    return captured


@pytest.mark.asyncio
async def test_resolve_reads_the_policy(control_plane_transport: InstallControlPlane) -> None:
    captured = _answer(
        httpx.Response(
            200,
            json={
                "enabled": True,
                "provider": "tavily",
                "max_results": 7,
                "purpose_hint": "search the docs",
                "allowed_domains": ["docs.python.org"],
                "blocked_domains": None,
                "provider_options": {"search_depth": "advanced"},
            },
        ),
        control_plane_transport,
    )

    policy = await RemoteWebSearchPolicy(_config()).resolve(_scope(), ["web_search"])

    assert policy is not None
    assert policy.enabled is True
    assert policy.max_results == 7
    assert policy.allowed_domains == ("docs.python.org",)
    assert policy.provider_options == {"search_depth": "advanced"}
    assert policy.authorized_tools == frozenset({"web_search"})
    assert captured["url"].endswith("/gateway/web-search/resolve")
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
    assert captured["headers"]["X-User-Token"] == "tk_user"


@pytest.mark.asyncio
async def test_resolve_sends_the_requested_web_tools(control_plane_transport: InstallControlPlane) -> None:
    captured = _answer(
        httpx.Response(200, json={"enabled": True, "authorized_tools": ["web_search", "web_fetch"]}),
        control_plane_transport,
    )

    policy = await RemoteWebSearchPolicy(_config()).resolve(_scope(), ["web_search", "web_fetch"])

    assert captured["body"] == {"requested_tools": ["web_search", "web_fetch"]}
    assert policy is not None
    assert policy.authorized_tools == frozenset({"web_search", "web_fetch"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=["enabled"]),
        httpx.Response(200, json={"enabled": "yes"}),
        httpx.Response(200, json={"enabled": True, "authorized_tools": None}),
    ],
    ids=["not-an-object", "enabled-not-bool", "authorized-tools-null"],
)
async def test_an_unreadable_answer_is_a_resolution_failure(
    control_plane_transport: InstallControlPlane, response: httpx.Response
) -> None:
    _answer(response, control_plane_transport)

    with pytest.raises(WebSearchPolicyResolutionFailedError) as ei:
        await RemoteWebSearchPolicy(_config()).resolve(_scope(), ["web_search"])

    assert ei.value.reason is WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE


@pytest.mark.asyncio
async def test_a_scope_without_a_caller_token_is_a_resolution_failure() -> None:
    with pytest.raises(WebSearchPolicyResolutionFailedError) as ei:
        await RemoteWebSearchPolicy(_config()).resolve(_scope(user_token=None), ["web_search"])

    assert ei.value.reason is WebSearchPolicyResolutionFailure.NO_CALLER_CREDENTIAL


@pytest.mark.asyncio
async def test_resolve_403_passes_through(control_plane_transport: InstallControlPlane) -> None:
    _answer(httpx.Response(403, json={"detail": "web search disabled"}), control_plane_transport)

    with pytest.raises(ControlPlaneRefusedError) as ei:
        await RemoteWebSearchPolicy(_config()).resolve(_scope(), ["web_search"])

    assert ei.value.status_code == 403
    assert ei.value.message == "web search disabled"


@pytest.mark.asyncio
async def test_resolve_429_passthrough_with_retry_after(control_plane_transport: InstallControlPlane) -> None:
    _answer(httpx.Response(429, json={"detail": "slow down"}, headers={"Retry-After": "30"}), control_plane_transport)

    with pytest.raises(ControlPlaneRefusedError) as ei:
        await RemoteWebSearchPolicy(_config()).resolve(_scope(), ["web_search"])

    assert ei.value.status_code == 429
    assert ei.value.retry_after == "30"
    assert ei.value.message == "slow down"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="busy"),
        httpx.Response(422, json={"detail": "schema mismatch"}),
    ],
    ids=["5xx", "422"],
)
async def test_an_unusable_status_maps_to_502(
    control_plane_transport: InstallControlPlane, response: httpx.Response
) -> None:
    _answer(response, control_plane_transport)

    with pytest.raises(ControlPlaneError) as ei:
        await RemoteWebSearchPolicy(_config()).resolve(_scope(), ["web_search"])

    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_network_error_maps_to_502(control_plane_transport: InstallControlPlane) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.NetworkError("connection refused")

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneError) as ei:
        await RemoteWebSearchPolicy(_config()).resolve(_scope(), ["web_search"])

    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_misconfigured_platform_500() -> None:
    with pytest.raises(ControlPlaneError) as ei:
        await RemoteWebSearchPolicy(_config(base_url=None)).resolve(_scope(), ["web_search"])

    assert ei.value.status_code == 500


@pytest.mark.asyncio
async def test_a_local_scope_without_a_workspace_is_a_resolution_failure() -> None:
    with pytest.raises(WebSearchPolicyResolutionFailedError) as ei:
        await LocalWebSearchPolicy(MagicMock()).resolve(_scope(), ["web_search"])

    assert ei.value.reason is WebSearchPolicyResolutionFailure.NO_WORKSPACE
