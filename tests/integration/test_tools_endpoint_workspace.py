"""What an API key's workspace policy does to ``GET /api/v1/tools``.

The deployment-only answer is covered in ``tests/unit/test_tools_endpoint.py``;
this is the workspace half: a tool the key's workspace does not allow is listed
as unavailable, so discovery agrees with what the request path would refuse.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.tools import OrgWebSearchKey, WorkspaceWebSearchConfig
from gateway.services.secret_box import generate_secret_key

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


def _add_org_search_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Store a search key as the organization's default, which every workspace without a pin then uses."""
    keys = f"{API_ROOT}/organizations/me/web-search-keys"
    created = client.post(
        keys, json={"provider": "tavily", "name": "ours", "api_key": "tvly-org-secret"}, headers=master_key_header
    )
    assert created.status_code == 201, created.text
    made_default = client.post(f"{keys}/{created.json()['id']}/default", headers=master_key_header)
    assert made_default.status_code == 200, made_default.text


def test_an_organization_search_key_serves_where_the_deployment_has_no_search(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request path searches on the organization's key, so the key's answer counts it; the master key's does not."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    assert _status(_tools(client, api_key_header))["otari_web_search"] == (False, "not_configured")

    _add_org_search_key(client, master_key_header)

    assert _status(_tools(client, api_key_header))["otari_web_search"] == (True, None)
    assert _status(_tools(client, master_key_header))["otari_web_search"] == (False, "not_configured")


def test_an_organization_search_key_does_not_override_the_workspace_switch(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _add_org_search_key(client, master_key_header)
    _put(client, master_key_header, "web-search", {"enabled": False})

    assert _status(_tools(client, api_key_header))["otari_web_search"] == (False, "workspace_disabled")


def test_a_policy_whose_tool_list_leaves_nothing_to_run_reads_as_disabled(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission refuses a policy whose stored ``tools`` the sandbox no longer serves, so the catalog does too."""
    _put(client, master_key_header, "code-execution-policy", {"enabled": True, "tools": ["code_execution"]})
    monkeypatch.setattr("gateway.services.tools._code_execution_policy.SERVED_TOOL_NAMES", ("something_else",))

    assert _status(_tools(client, api_key_header))["otari_code_execution"] == (False, "workspace_disabled")


def test_an_invalid_stored_row_reads_as_the_workspace_refusing_not_as_nothing_to_search_with(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _add_org_search_key(client, master_key_header)
    _put(client, master_key_header, "web-search", {"enabled": True, "allowed_domains": ["example.com"]})
    with db_session_factory() as db:
        row = db.get(WorkspaceWebSearchConfig, uuid.UUID(_default_workspace_id(client, master_key_header)))
        assert row is not None
        row.allowed_domains = ["https://example.com/private"]
        db.commit()

    assert _status(_tools(client, api_key_header))["otari_web_search"] == (False, "workspace_disabled")


def test_a_key_whose_provider_this_gateway_cannot_call_serves_no_search(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The request path skips such a key, so with no search of the deployment's there is nothing to run on."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _add_org_search_key(client, master_key_header)
    with db_session_factory() as db:
        db.execute(update(OrgWebSearchKey).values(provider="retired-provider"))
        db.commit()

    assert _status(_tools(client, api_key_header))["otari_web_search"] == (False, "not_configured")


def test_a_pinned_image_the_operator_withdrew_reads_as_disabled(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission refuses every sandbox request on a withdrawn pin, so the catalog does too."""
    monkeypatch.setenv("OTARI_SANDBOX_ALLOWED_SESSION_IMAGES", "ghcr.io/acme/sandbox:2")
    _put(client, master_key_header, "code-execution-policy", {"enabled": True, "image": "ghcr.io/acme/sandbox:2"})
    assert _status(_tools(client, api_key_header))["otari_code_execution"] == (True, None)

    monkeypatch.delenv("OTARI_SANDBOX_ALLOWED_SESSION_IMAGES")

    assert _status(_tools(client, api_key_header))["otari_code_execution"] == (False, "workspace_disabled")


def test_a_key_served_search_advertises_only_the_canonical_type(
    client: TestClient,
    test_config: GatewayConfig,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interception claims a provider's own keywords only for a deployment that searches itself."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL")
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(test_config, "web_search_intercept", True)
    _add_org_search_key(client, master_key_header)

    web_search = _tools(client, api_key_header)["otari_web_search"]

    assert (web_search["available"], web_search["accepted_types"]) == (True, ["otari_web_search"])


def _add_default_key(client: TestClient, master_key_header: dict[str, str], **key: Any) -> str:
    keys = f"{API_ROOT}/organizations/me/web-search-keys"
    created = client.post(keys, json=key, headers=master_key_header)
    assert created.status_code == 201, created.text
    key_id: str = created.json()["id"]
    assert client.post(f"{keys}/{key_id}/default", headers=master_key_header).status_code == 200
    return key_id


def test_a_fetch_key_admission_refuses_takes_both_web_tools_with_it(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetch key with no secret refuses the fetch tool and the enrichment of a search on the libraries."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    _add_default_key(client, master_key_header, provider="exa", name="search", api_key="exa-1")
    fetch_key = _add_default_key(
        client, master_key_header, provider="tabstack", name="pages", api_key="tbs-1", kind="fetch"
    )
    assert _status(_tools(client, api_key_header)) == _ALL_AVAILABLE
    # A row stored before the write checks required a secret.
    with db_session_factory() as db:
        db.execute(
            update(OrgWebSearchKey).where(OrgWebSearchKey.id == uuid.UUID(fetch_key)).values(encrypted_api_key=None)
        )
        db.commit()

    status = _status(_tools(client, api_key_header))

    assert status["otari_web_fetch"] == (False, "not_configured")
    assert status["otari_web_search"] == (False, "not_configured")
