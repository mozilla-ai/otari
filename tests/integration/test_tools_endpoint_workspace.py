"""What an API key's workspace policy does to ``GET /api/v1/tools``.

The deployment-only answer is covered in ``tests/unit/test_tools_endpoint.py``;
this is the workspace half: a tool the key's workspace does not allow is listed
as unavailable, so discovery agrees with what the request path would refuse.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT, GatewayConfig

_SEARCH_URL = "http://127.0.0.1:9998/search"
_SANDBOX_URL = "http://127.0.0.1:9999/sandbox"


@pytest.fixture(autouse=True)
def _tools_configured(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that runs all three tools, so only the workspace decides."""
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", _SEARCH_URL)
    monkeypatch.setenv("OTARI_SANDBOX_URL", _SANDBOX_URL)


def _tools(client: TestClient, headers: dict[str, str]) -> dict[str, dict[str, Any]]:
    response = client.get(f"{API_ROOT}/tools", headers=headers)
    assert response.status_code == 200, response.text
    return {tool["id"]: tool for tool in response.json()["data"]}


def _status(tools: dict[str, dict[str, Any]]) -> dict[str, tuple[bool, str | None]]:
    return {tool_id: (tool["available"], tool["reason"]) for tool_id, tool in tools.items()}


def _default_workspace_id(client: TestClient, master_key_header: dict[str, str]) -> str:
    listed = client.get(f"{API_ROOT}/workspaces", headers=master_key_header)
    assert listed.status_code == 200
    workspace_id: str = listed.json()["data"][0]["id"]
    return workspace_id


def _put(client: TestClient, master_key_header: dict[str, str], path: str, body: dict[str, Any]) -> None:
    workspace_id = _default_workspace_id(client, master_key_header)
    response = client.put(f"{API_ROOT}/workspaces/{workspace_id}/{path}", json=body, headers=master_key_header)
    assert response.status_code == 200, response.text


_ALL_AVAILABLE = {
    "otari_web_fetch": (True, None),
    "otari_web_search": (True, None),
    "otari_code_execution": (True, None),
}
_ALL_WORKSPACE_DISABLED = {
    "otari_web_fetch": (False, "workspace_disabled"),
    "otari_web_search": (False, "workspace_disabled"),
    "otari_code_execution": (False, "workspace_disabled"),
}


def test_a_workspace_with_no_policy_gets_the_deployment_default(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    assert _status(_tools(client, api_key_header)) == _ALL_AVAILABLE


def test_no_policy_reads_as_off_where_the_deployment_makes_the_tools_opt_in(
    client: TestClient,
    test_config: GatewayConfig,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "web_search_workspace_default", False)
    monkeypatch.setattr(test_config, "code_execution_workspace_default", False)

    assert _status(_tools(client, api_key_header)) == _ALL_WORKSPACE_DISABLED


def test_a_workspace_that_turned_the_tools_off_sees_them_unavailable(
    client: TestClient, api_key_header: dict[str, str], master_key_header: dict[str, str]
) -> None:
    _put(client, master_key_header, "web-search", {"enabled": False})
    _put(client, master_key_header, "code-execution-policy", {"enabled": False})

    assert _status(_tools(client, api_key_header)) == _ALL_WORKSPACE_DISABLED


def test_turning_the_tools_on_overrides_an_opt_in_default(
    client: TestClient,
    test_config: GatewayConfig,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_config, "web_search_workspace_default", False)
    monkeypatch.setattr(test_config, "code_execution_workspace_default", False)
    _put(client, master_key_header, "web-search", {"enabled": True})
    _put(client, master_key_header, "code-execution-policy", {"enabled": True})

    assert _status(_tools(client, api_key_header)) == _ALL_AVAILABLE


def test_the_master_key_gets_the_deployment_answer(
    client: TestClient, test_config: GatewayConfig, master_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The master key belongs to no single workspace, so no workspace narrows its answer."""
    monkeypatch.setattr(test_config, "web_search_workspace_default", False)
    monkeypatch.setattr(test_config, "code_execution_workspace_default", False)

    assert _status(_tools(client, master_key_header)) == _ALL_AVAILABLE


def test_an_unconfigured_tool_says_so_whatever_the_workspace_allows(
    client: TestClient, api_key_header: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OTARI_SANDBOX_URL")

    assert _status(_tools(client, api_key_header))["otari_code_execution"] == (False, "not_configured")
