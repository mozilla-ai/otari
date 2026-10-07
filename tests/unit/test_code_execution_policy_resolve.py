"""Resolving a workspace's code execution policy from its own rows or from a peer control plane."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from conftest import InstallControlPlane
from gateway.adapters.code_execution_policy_adapter import LocalCodeExecutionPolicy, RemoteCodeExecutionPolicy
from gateway.exceptions.control_plane_exceptions import (
    ControlPlaneNotConfiguredError,
    ControlPlaneRefusedError,
    ControlPlaneUnavailableError,
)
from gateway.exceptions.tools_exceptions import (
    CodeExecutionPolicyResolutionFailedError,
    CodeExecutionPolicyResolutionFailure,
)
from gateway.models.tools import CodeExecutor, ResolvedCodeExecutionPolicy
from gateway.ports.code_execution_policy_port import CodeExecutionPolicyScope


def _token_scope(user_token: str | None = "tk_user") -> CodeExecutionPolicyScope:
    """A scope that carries a caller token and no workspace."""
    return CodeExecutionPolicyScope(workspace_id=None, user_token=user_token)


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
                "tools": ["code_execution"],
                "default_purpose_hint": "prefer running code",
                "max_iterations": 5,
                "exec_timeout_s": 30,
                "executor": "otari",
            },
        ),
        control_plane_transport,
    )

    policy = await RemoteCodeExecutionPolicy(_config()).resolve(_token_scope())

    assert policy == ResolvedCodeExecutionPolicy(
        enabled=True,
        default_purpose_hint="prefer running code",
        max_iterations=5,
        exec_timeout_s=30,
        image=None,
        tools=frozenset({"code_execution"}),
        executor=CodeExecutor.OTARI,
    )
    assert captured["url"].endswith("/gateway/code-execution/resolve")
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
    assert captured["headers"]["X-User-Token"] == "tk_user"
    assert captured["body"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(200, json={"enabled": "yes"}),
        httpx.Response(200, json={"enabled": True, "max_iterations": "4"}),
    ],
    ids=["not-an-object", "enabled-not-bool", "ceiling-not-int"],
)
async def test_an_unreadable_answer_is_a_resolution_failure(
    control_plane_transport: InstallControlPlane, response: httpx.Response
) -> None:
    _answer(response, control_plane_transport)

    with pytest.raises(CodeExecutionPolicyResolutionFailedError) as ei:
        await RemoteCodeExecutionPolicy(_config()).resolve(_token_scope())

    assert ei.value.reason is CodeExecutionPolicyResolutionFailure.ANSWER_UNREADABLE


@pytest.mark.asyncio
async def test_a_scope_without_a_caller_token_is_a_resolution_failure() -> None:
    with pytest.raises(CodeExecutionPolicyResolutionFailedError) as ei:
        await RemoteCodeExecutionPolicy(_config()).resolve(_token_scope(user_token=None))

    assert ei.value.reason is CodeExecutionPolicyResolutionFailure.NO_CALLER_CREDENTIAL


@pytest.mark.asyncio
async def test_resolve_403_passes_through(control_plane_transport: InstallControlPlane) -> None:
    _answer(httpx.Response(403, json={"detail": "caller not accepted"}), control_plane_transport)

    with pytest.raises(ControlPlaneRefusedError) as ei:
        await RemoteCodeExecutionPolicy(_config()).resolve(_token_scope())

    assert ei.value.status_code == 403
    assert ei.value.message == "caller not accepted"


@pytest.mark.asyncio
async def test_resolve_429_passthrough_with_retry_after(control_plane_transport: InstallControlPlane) -> None:
    _answer(httpx.Response(429, json={"detail": "slow down"}, headers={"Retry-After": "30"}), control_plane_transport)

    with pytest.raises(ControlPlaneRefusedError) as ei:
        await RemoteCodeExecutionPolicy(_config()).resolve(_token_scope())

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

    with pytest.raises(ControlPlaneUnavailableError) as ei:
        await RemoteCodeExecutionPolicy(_config()).resolve(_token_scope())

    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_network_error_maps_to_502(control_plane_transport: InstallControlPlane) -> None:
    async def fake_post(**kwargs: Any) -> httpx.Response:
        raise httpx.NetworkError("connection refused")

    control_plane_transport(fake_post)

    with pytest.raises(ControlPlaneUnavailableError) as ei:
        await RemoteCodeExecutionPolicy(_config()).resolve(_token_scope())

    assert ei.value.status_code == 502


@pytest.mark.asyncio
async def test_resolve_misconfigured_platform_500() -> None:
    with pytest.raises(ControlPlaneNotConfiguredError) as ei:
        await RemoteCodeExecutionPolicy(_config(base_url=None)).resolve(_token_scope())

    assert ei.value.status_code == 500


@pytest.mark.asyncio
async def test_a_local_scope_without_a_workspace_is_a_resolution_failure() -> None:
    with pytest.raises(CodeExecutionPolicyResolutionFailedError) as ei:
        await LocalCodeExecutionPolicy(MagicMock()).resolve(_token_scope())

    assert ei.value.reason is CodeExecutionPolicyResolutionFailure.NO_WORKSPACE


@pytest.mark.asyncio
async def test_a_local_scope_reads_the_workspace_row() -> None:
    workspace_id = uuid.uuid4()
    stored = ResolvedCodeExecutionPolicy(
        enabled=False, default_purpose_hint=None, max_iterations=None, exec_timeout_s=None, image=None, tools=None
    )
    policies = MagicMock()
    policies.resolve = AsyncMock(return_value=stored)

    policy = await LocalCodeExecutionPolicy(policies).resolve(
        CodeExecutionPolicyScope(workspace_id=workspace_id, user_token="tk_ignored")
    )

    assert policy is stored
    policies.resolve.assert_awaited_once_with(workspace_id)
