"""A stored MCP server's access list: everyone_tools, grants for keys and users, and everywhere they bind.

The stateless MCP routes, the access list routes, test access, a key's summary and the workspace's
request-server switch run against the real database. URLs are public IP literals, so no resolver is reached.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import patch

import anyio
import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, ToolAnnotations
from mcp.types import Tool as MCPTool
from sqlalchemy import event, func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, CONVERSATION_HEADER
from gateway.models.api_keys import APIKey
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tenancy import OrganizationMember, Workspace, WorkspaceMember
from gateway.models.tenancy import User as Identity
from gateway.models.tools import WorkspaceMcpServer, WorkspaceMcpServerGrant
from gateway.models.traces import Trace, TraceSpan
from gateway.models.users import User as BillingUser
from gateway.repositories.tools import WorkspaceMcpServerGrantRepository
from gateway.repositories.users_repository import lock_live_users, provision_attribution_users
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import CODE_SERVER_NOT_FOUND, CODE_TOOL_NOT_ALLOWED
from gateway.services.model_access import resolve_request_mcp_access, revoke_user_mcp_grants
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tenancy import workspace_mcp_server_service
from gateway.services.tools import OTARI_TOOL_CALLS_FIELD

from .conftest import MODEL_NAME, _to_async_url

PUBLIC_URL = "https://93.184.216.34/mcp"
CLIENT_EXECUTION_ID = "6e51b3bc-6f48-4c68-a61e-0786bc80cd67"
TOOL_NAMES = ("list_issues", "create_issue", "delete_issue")


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    yield


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self, *, params: PaginatedRequestParams | None = None) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                MCPTool(
                    name=name,
                    description=name,
                    input_schema={"type": "object"},
                    # The server claims both list and delete are read-only; only one of them is.
                    annotations=ToolAnnotations(read_only_hint=True) if name != "create_issue" else None,
                )
                for name in TOOL_NAMES
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="done")])


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    fake = _FakeSession()

    @asynccontextmanager
    async def open_session(*args: Any, **kwargs: Any) -> Any:
        yield fake

    monkeypatch.setattr(mcp_stateless, "open_session", open_session)
    return fake


def _workspace_id(test_db: Session, key: dict[str, Any]) -> uuid.UUID:
    row = test_db.get(APIKey, key["id"])
    assert row is not None
    return row.workspace_id


def _store(
    test_db: Session,
    workspace_id: uuid.UUID,
    name: str,
    *,
    everyone: list[str] | None = None,
    enabled: bool = True,
    allowed_tools: list[str] | None = None,
    read_tools: list[str] | None = None,
) -> WorkspaceMcpServer:
    row = WorkspaceMcpServer(
        workspace_id=workspace_id,
        name=name,
        url=PUBLIC_URL,
        encrypted_token=encrypt_secret("ghp_token"),
        enabled=enabled,
        everyone_tools=["*"] if everyone is None else everyone,
        allowed_tools=allowed_tools,
        read_tools=read_tools or [],
        purpose_hint=f"{name} hint",
    )
    test_db.add(row)
    test_db.commit()
    test_db.refresh(row)
    return row


def _bearer(key: dict[str, Any]) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {key['key']}"}


def _new_key(client: TestClient, master: dict[str, str], **body: Any) -> dict[str, Any]:
    response = client.post(f"{API_ROOT}/keys", json={"key_name": "k", **body}, headers=master)
    assert response.status_code == 200, response.text
    created: dict[str, Any] = response.json()
    return created


def _servers(workspace: uuid.UUID) -> str:
    return f"{API_ROOT}/workspaces/{workspace}/mcp-servers"


def _grant(
    client: TestClient,
    master: dict[str, str],
    row: WorkspaceMcpServer,
    tools: list[str],
    *,
    keys: Sequence[dict[str, Any]] = (),
    users: Sequence[str] = (),
) -> dict[str, Any]:
    response = client.post(
        f"{_servers(row.workspace_id)}/{row.id}/access/grants",
        json={"api_key_ids": [key["id"] for key in keys], "user_ids": list(users), "tools": tools},
        headers=master,
    )
    assert response.status_code == 200, response.text
    access: dict[str, Any] = response.json()
    return access


def _execute_body(row: WorkspaceMcpServer, tool: str) -> dict[str, Any]:
    return {
        "mcp_server_id": str(row.id),
        "tool_name": tool,
        "arguments": {},
        "server_revision": ResolvedMcpServer(
            id=row.id,
            name=row.name,
            url=row.url,
            authorization_token="ghp_token",
            enabled=row.enabled,
            allowed_tools=row.allowed_tools,
        ).revision,
        "client_execution_id": CLIENT_EXECUTION_ID,
    }


def _listed(client: TestClient, key: dict[str, Any]) -> list[str]:
    response = client.get(f"{API_ROOT}/mcp/servers", headers=_bearer(key))
    assert response.status_code == 200, response.text
    return [server["name"] for server in response.json()["servers"]]


# --------------------------------------------------------------------------- #
# GET /mcp/servers
# --------------------------------------------------------------------------- #


def test_a_key_lists_the_enabled_servers_whose_everyone_rule_lets_it_in(
    client: TestClient, test_db: Session, api_key_obj: dict[str, Any], api_key_header: dict[str, str]
) -> None:
    workspace = _workspace_id(test_db, api_key_obj)
    visible = _store(test_db, workspace, "github")
    _store(test_db, workspace, "locked", everyone=[])
    _store(test_db, workspace, "off", enabled=False)

    response = client.get(f"{API_ROOT}/mcp/servers", headers=api_key_header)

    assert response.status_code == 200, response.text
    assert response.json() == {"servers": [{"id": str(visible.id), "name": "github", "purpose_hint": "github hint"}]}
    for secret in ("ghp_token", "93.184.216.34"):
        assert secret not in response.text


def test_a_users_grant_opens_a_locked_server_to_every_key_they_own_and_a_keys_own_grant_wins(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    laptop = _new_key(client, master_key_header, user_id="ana", key_name="laptop")
    ci = _new_key(client, master_key_header, user_id="ana", key_name="ci")
    stranger = _new_key(client, master_key_header, user_id="bo")
    locked = _store(test_db, _workspace_id(test_db, laptop), "locked", everyone=[])

    assert _listed(client, laptop) == []
    _grant(client, master_key_header, locked, ["*"], users=["ana"])
    assert _listed(client, laptop) == _listed(client, ci) == ["locked"]
    assert _listed(client, stranger) == []

    _grant(client, master_key_header, locked, [], keys=[ci])
    assert _listed(client, ci) == [], "a key's own grant decides over its owner's"
    assert _listed(client, laptop) == ["locked"]


def test_the_master_key_has_no_workspace_to_list(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.get(f"{API_ROOT}/mcp/servers", headers=master_key_header)

    assert response.status_code == 404
    assert response.json()["code"] == CODE_SERVER_NOT_FOUND


def test_the_listing_needs_a_key(client: TestClient) -> None:
    assert client.get(f"{API_ROOT}/mcp/servers").status_code == 401


# --------------------------------------------------------------------------- #
# Discovery and execution
# --------------------------------------------------------------------------- #


def test_discovery_lists_only_the_tools_the_deciding_rule_allows(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, session: _FakeSession
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=["list_*"])

    def names() -> list[str]:
        response = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=_bearer(key))
        assert response.status_code == 200, response.text
        return [tool["name"] for tool in response.json()["tools"]]

    assert names() == ["list_issues"]
    _grant(client, master_key_header, row, ["list_*", "create_issue"], keys=[key])
    assert names() == ["list_issues", "create_issue"]


def test_the_server_ceiling_still_applies_beneath_a_grant(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, session: _FakeSession
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", allowed_tools=["list_issues"])
    _grant(client, master_key_header, row, ["*"], keys=[key])

    discovered = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=_bearer(key))
    executed = client.post(f"{API_ROOT}/mcp/execute", headers=_bearer(key), json=_execute_body(row, "create_issue"))

    assert [tool["name"] for tool in discovered.json()["tools"]] == ["list_issues"]
    assert executed.status_code == 403
    assert executed.json()["code"] == CODE_TOOL_NOT_ALLOWED


def test_a_tool_outside_the_rule_is_refused_with_the_ceilings_code(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, session: _FakeSession
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=["list_*"])

    refused = client.post(f"{API_ROOT}/mcp/execute", headers=_bearer(key), json=_execute_body(row, "delete_issue"))
    allowed = client.post(f"{API_ROOT}/mcp/execute", headers=_bearer(key), json=_execute_body(row, "list_issues"))

    assert refused.status_code == 403
    assert refused.json()["code"] == CODE_TOOL_NOT_ALLOWED
    assert refused.json()["execution_state"] == "not_started"
    assert allowed.status_code == 200, allowed.text
    assert session.calls == ["list_issues"]


def test_a_read_rule_reaches_only_the_tools_a_manager_labeled_read(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, session: _FakeSession
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])
    _grant(client, master_key_header, row, ["@read"], keys=[key])

    assert _listed(client, key) == []

    labeled = client.patch(
        f"{_servers(row.workspace_id)}/{row.id}", json={"read_tools": ["list_issues"]}, headers=master_key_header
    )
    assert labeled.status_code == 200, labeled.text
    assert labeled.json()["read_tools"] == ["list_issues"]

    discovered = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=_bearer(key))
    hinted = client.post(f"{API_ROOT}/mcp/execute", headers=_bearer(key), json=_execute_body(row, "delete_issue"))
    allowed = client.post(f"{API_ROOT}/mcp/execute", headers=_bearer(key), json=_execute_body(row, "list_issues"))

    assert _listed(client, key) == ["github"]
    assert [tool["name"] for tool in discovered.json()["tools"]] == ["list_issues"]
    assert hinted.status_code == 403
    assert hinted.json()["code"] == CODE_TOOL_NOT_ALLOWED
    assert allowed.status_code == 200, allowed.text
    assert session.calls == ["list_issues"]


def test_the_connection_check_suggests_read_tools_without_labeling_them(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, session: _FakeSession
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")

    checked = client.post(f"{_servers(row.workspace_id)}/{row.id}/check", headers=master_key_header)

    assert checked.status_code == 200, checked.text
    stored = client.get(f"{_servers(row.workspace_id)}/{row.id}", headers=master_key_header).json()
    assert stored["health"]["read_only_hinted_tools"] == ["list_issues", "delete_issue"]
    assert stored["read_tools"] == []


def test_the_connection_check_leaves_out_a_tool_named_like_a_pattern(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool the server names ``*`` or ``search*`` would be stored as a wildcard once a manager ticked it."""
    monkeypatch.setattr(sys.modules[__name__], "TOOL_NAMES", ("list_issues", "*", "search*", "@read", " @read", ","))
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")

    checked = client.post(f"{_servers(row.workspace_id)}/{row.id}/check", headers=master_key_header)

    assert checked.status_code == 200, checked.text
    assert checked.json()["health"]["tools"] == ["list_issues"]


def test_discovery_applies_the_caller_s_rule_before_the_returned_tool_limit(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    session: _FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server offering more tools than a response may carry still answers a key allowed one of them."""
    names = tuple(f"tool_{index}" for index in range(mcp_stateless.DISCOVERY_MAX_TOOLS + 1))
    monkeypatch.setattr(sys.modules[__name__], "TOOL_NAMES", names)
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])
    _grant(client, master_key_header, row, ["tool_7"], keys=[key])

    response = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=_bearer(key))

    assert response.status_code == 200, response.text
    assert [tool["name"] for tool in response.json()["tools"]] == ["tool_7"]


def test_a_catalog_stored_before_the_screen_is_read_back_without_pattern_names(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")
    row.health = {
        "status": "healthy",
        "checked_at": "2026-01-01T00:00:00+00:00",
        "tools": ["list_issues", "*", " @read", "get_issue,delete_issue"],
        "read_only_hinted_tools": ["list_issues", " @read"],
    }
    test_db.commit()

    stored = client.get(f"{_servers(row.workspace_id)}/{row.id}", headers=master_key_header).json()

    assert stored["health"]["tools"] == ["list_issues"]
    assert stored["health"]["read_only_hinted_tools"] == ["list_issues"]


@pytest.mark.parametrize("read_tools", [["list_*"], ["@read"], [""], None, ["x" * 257]])
def test_read_tools_name_tools_exactly(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, read_tools: Any
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")

    response = client.patch(
        f"{_servers(row.workspace_id)}/{row.id}", json={"read_tools": read_tools}, headers=master_key_header
    )

    assert response.status_code == 422


@pytest.mark.parametrize("tools", [["@write"], ["x" * 257]])
def test_an_unknown_preset_or_an_overlong_pattern_is_refused(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, tools: list[str]
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")

    response = client.post(
        f"{_servers(row.workspace_id)}/{row.id}/access/grants",
        json={"api_key_ids": [key["id"]], "user_ids": [], "tools": tools},
        headers=master_key_header,
    )

    assert response.status_code in (400, 422)


def test_a_server_that_denies_the_key_is_as_missing_as_a_disabled_one(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, session: _FakeSession
) -> None:
    key = _new_key(client, master_key_header)
    workspace = _workspace_id(test_db, key)
    github = _store(test_db, workspace, "github")
    denied = _store(test_db, workspace, "denied")
    disabled = _store(test_db, workspace, "off", enabled=False)
    _grant(client, master_key_header, denied, [], keys=[key])

    def status_of(row: WorkspaceMcpServer) -> tuple[int, str]:
        response = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=_bearer(key))
        return response.status_code, response.json().get("code", "")

    assert status_of(github)[0] == 200
    assert status_of(denied) == status_of(disabled) == (404, CODE_SERVER_NOT_FOUND)
    executed = client.post(f"{API_ROOT}/mcp/execute", headers=_bearer(key), json=_execute_body(denied, "list_issues"))
    assert executed.status_code == 404
    assert session.calls == []


# --------------------------------------------------------------------------- #
# The access list routes
# --------------------------------------------------------------------------- #


def test_the_access_list_names_each_grant_and_everyone_can_be_changed(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, session: _FakeSession
) -> None:
    key = _new_key(client, master_key_header, user_id="ana", key_name="Ana's laptop")
    _new_key(client, master_key_header, user_id="bo", key_name="Bo's laptop")
    workspace = _workspace_id(test_db, key)
    created = client.post(_servers(workspace), json={"name": "github", "url": PUBLIC_URL}, headers=master_key_header)
    assert created.status_code == 201, created.text
    assert created.json()["everyone_tools"] == ["*"]
    server_id = created.json()["id"]
    row = test_db.get(WorkspaceMcpServer, uuid.UUID(server_id))
    assert row is not None

    _grant(client, master_key_header, row, ["get_issue"], keys=[key])
    access = _grant(client, master_key_header, row, ["*"], users=["bo"])

    assert access["everyone_tools"] == ["*"]
    by_subject = {grant["subject"]: grant for grant in access["grants"]}
    assert set(by_subject["key"]) == {
        "id",
        "subject",
        "api_key_id",
        "key_name",
        "key_active",
        "user_id",
        "tools",
        "created_at",
        "updated_at",
    }
    assert (by_subject["key"]["api_key_id"], by_subject["key"]["key_name"], by_subject["key"]["user_id"]) == (
        key["id"],
        "Ana's laptop",
        "ana",
    )
    assert by_subject["key"]["key_active"] is True
    assert (by_subject["user"]["api_key_id"], by_subject["user"]["user_id"]) == (None, "bo")

    patched = client.patch(f"{_servers(workspace)}/{server_id}", json={"everyone_tools": []}, headers=master_key_header)
    assert patched.status_code == 200, patched.text
    assert patched.json()["everyone_tools"] == []
    read = client.get(f"{_servers(workspace)}/{server_id}/access", headers=master_key_header).json()
    assert read["everyone_tools"] == []
    assert len(read["grants"]) == 2
    one = client.get(f"{_servers(workspace)}/{server_id}", headers=master_key_header)
    assert one.status_code == 200
    assert one.json()["name"] == "github"


def test_granting_again_replaces_the_tools_and_grants_can_be_changed_and_removed(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])
    base = f"{_servers(row.workspace_id)}/{row.id}/access/grants"

    first = _grant(client, master_key_header, row, ["get_issue"], keys=[key])
    again = _grant(client, master_key_header, row, ["list_*"], keys=[key])
    assert len(again["grants"]) == 1
    grant_id = again["grants"][0]["id"]
    assert grant_id == first["grants"][0]["id"]
    assert again["grants"][0]["tools"] == ["list_*"]

    changed = client.patch(f"{base}/{grant_id}", json={"tools": ["*"]}, headers=master_key_header)
    assert changed.status_code == 200, changed.text
    assert changed.json()["grants"][0]["tools"] == ["*"]
    assert _listed(client, key) == ["github"]

    removed = client.delete(f"{base}/{grant_id}", headers=master_key_header)
    assert removed.status_code == 200, removed.text
    assert removed.json()["grants"] == []
    assert _listed(client, key) == [], "without its grant the key falls back to everyone, which is denied"
    assert client.delete(f"{base}/{grant_id}", headers=master_key_header).status_code == 404


def test_a_grant_that_loses_an_insert_race_updates_the_winner_s_row(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a lock on the server (SQLite), another add can insert the grant after this one read none."""
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])
    first = _grant(client, master_key_header, row, ["get_issue"], keys=[key])
    existing = WorkspaceMcpServerGrantRepository._existing
    reads: list[int] = []

    async def stale_once(
        self: WorkspaceMcpServerGrantRepository, *args: Any
    ) -> dict[tuple[str | None, str | None], WorkspaceMcpServerGrant]:
        reads.append(1)
        return {} if len(reads) == 1 else await existing(self, *args)

    monkeypatch.setattr(WorkspaceMcpServerGrantRepository, "_existing", stale_once)
    again = _grant(client, master_key_header, row, ["list_*"], keys=[key])

    assert len(reads) == 2, "the write ran again after the unique constraint refused it"
    assert [(grant["id"], grant["tools"]) for grant in again["grants"]] == [(first["grants"][0]["id"], ["list_*"])]


def test_a_grant_must_name_keys_and_people_of_the_servers_workspace(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _new_key(client, master_key_header, user_id="ana")
    row = _store(test_db, _workspace_id(test_db, key), "github")
    base = f"{_servers(row.workspace_id)}/{row.id}/access/grants"

    def post(body: dict[str, Any]) -> int:
        return client.post(base, json=body, headers=master_key_header).status_code

    assert post({"api_key_ids": ["no-such-key"], "tools": ["*"]}) == 400
    assert post({"user_ids": ["nobody"], "tools": ["*"]}) == 400
    assert post({"tools": ["*"]}) == 422
    assert post({"user_ids": ["ana"], "tools": ["*_issue"]}) == 422
    assert post({"user_ids": ["ana"], "tools": ["*"]}) == 200
    other = client.get(f"{_servers(row.workspace_id)}/{uuid.uuid4()}/access", headers=master_key_header)
    assert other.status_code == 404


def test_a_server_holds_a_bounded_number_of_grants(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workspace_mcp_server_service, "MAX_GRANTS_PER_SERVER", 2)
    keys = [_new_key(client, master_key_header, key_name=f"k{index}") for index in range(3)]
    row = _store(test_db, _workspace_id(test_db, keys[0]), "github")

    _grant(client, master_key_header, row, ["*"], keys=keys[:2])
    refused = client.post(
        f"{_servers(row.workspace_id)}/{row.id}/access/grants",
        json={"api_key_ids": [keys[2]["id"]], "tools": ["*"]},
        headers=master_key_header,
    )
    regrant = _grant(client, master_key_header, row, [], keys=keys[:2])

    assert refused.status_code == 400
    assert len(regrant["grants"]) == 2, "changing the grants a server already holds is never capped"


def test_test_access_says_which_rule_decides(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    laptop = _new_key(client, master_key_header, user_id="ana")
    ci = _new_key(client, master_key_header, user_id="ana")
    other = _new_key(client, master_key_header, user_id="bo")
    row = _store(test_db, _workspace_id(test_db, laptop), "github", everyone=["list_*"], allowed_tools=None)
    user_rule = _grant(client, master_key_header, row, ["get_issue"], users=["ana"])["grants"][0]["id"]
    own = _grant(client, master_key_header, row, ["*"], keys=[ci])["grants"][1]["id"]

    def check(key: dict[str, Any], tool: str) -> dict[str, Any]:
        response = client.get(
            f"{_servers(row.workspace_id)}/{row.id}/access/check",
            params={"api_key_id": key["id"], "tool": tool},
            headers=master_key_header,
        )
        assert response.status_code == 200, response.text
        verdict: dict[str, Any] = response.json()
        return verdict

    assert check(laptop, "get_issue") == {
        "allowed": True,
        "decided_by": "user",
        "grant_id": user_rule,
        "tools": ["get_issue"],
    }
    assert check(laptop, "list_issues")["allowed"] is False
    assert check(ci, "delete_repo") == {"allowed": True, "decided_by": "key", "grant_id": own, "tools": ["*"]}
    assert check(other, "list_issues") == {
        "allowed": True,
        "decided_by": "everyone",
        "grant_id": None,
        "tools": ["list_*"],
    }

    client.patch(
        f"{_servers(row.workspace_id)}/{row.id}", json={"allowed_tools": ["list_issues"]}, headers=master_key_header
    )
    hidden = check(ci, "delete_repo")
    assert (hidden["allowed"], hidden["decided_by"]) == (False, "server")

    unknown = client.get(
        f"{_servers(row.workspace_id)}/{row.id}/access/check",
        params={"api_key_id": "no-such-key", "tool": "x"},
        headers=master_key_header,
    )
    assert unknown.status_code == 404


def test_a_keys_summary_covers_every_server_of_its_workspace(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _new_key(client, master_key_header, user_id="ana")
    workspace = _workspace_id(test_db, key)
    github = _store(test_db, workspace, "github", read_tools=["list_issues"])
    jira = _store(test_db, workspace, "jira", everyone=[], allowed_tools=["search"])
    _grant(client, master_key_header, jira, ["search"], users=["ana"])

    response = client.get(f"{_servers(workspace)}/keys/{key['id']}/access", headers=master_key_header)

    assert response.status_code == 200, response.text
    summary = {entry["name"]: entry for entry in response.json()}
    assert summary["github"]["decided_by"] == "everyone"
    assert summary["github"]["tools"] == ["*"]
    assert summary["github"]["server_id"] == str(github.id)
    # What '@read' in a rule would stand for, so the key editor can say which tools that reaches.
    assert summary["github"]["read_tools"] == ["list_issues"]
    assert (summary["jira"]["decided_by"], summary["jira"]["tools"], summary["jira"]["allowed_tools"]) == (
        "user",
        ["search"],
        ["search"],
    )
    assert client.get(f"{_servers(workspace)}/keys/no-such-key/access", headers=master_key_header).status_code == 404


def test_a_member_without_a_key_can_be_given_a_rule_and_an_outsider_cannot(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    """The Playground binds a member as their user before they hold any key, so a manager can restrict them first."""
    key = _new_key(client, master_key_header)
    workspace = _workspace_id(test_db, key)
    row = _store(test_db, workspace, "github")
    home = test_db.get(Workspace, workspace)
    assert home is not None
    member = Identity(email="keyless@example.test", full_name="Keyless", active_organization_id=home.organization_id)
    outsider = Identity(
        email="outsider@example.test", full_name="Outsider", active_organization_id=home.organization_id
    )
    test_db.add_all([member, outsider])
    test_db.commit()
    test_db.add(WorkspaceMember(workspace_id=workspace, user_id=member.id, role="member", status="active"))
    test_db.commit()

    granted = _grant(client, master_key_header, row, [], users=[str(member.id)])
    refused = client.post(
        f"{_servers(workspace)}/{row.id}/access/grants",
        json={"user_ids": [str(outsider.id)], "tools": []},
        headers=master_key_header,
    )

    assert [grant["user_id"] for grant in granted["grants"]] == [str(member.id)]
    assert test_db.get(BillingUser, str(member.id)) is not None
    assert refused.status_code == 400, refused.text


def test_a_member_s_grants_end_when_they_leave_the_workspace(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    """A grant left behind would come back unseen if the person were invited again."""
    key = _new_key(client, master_key_header)
    workspace = _workspace_id(test_db, key)
    row = _store(test_db, workspace, "github", everyone=[])
    home = test_db.get(Workspace, workspace)
    assert home is not None
    member = Identity(email="leaver@example.test", full_name="Leaver", active_organization_id=home.organization_id)
    test_db.add(member)
    test_db.commit()
    test_db.add(WorkspaceMember(workspace_id=workspace, user_id=member.id, role="member", status="active"))
    test_db.commit()
    _grant(client, master_key_header, row, ["*"], users=[str(member.id)])

    removed = client.delete(f"{API_ROOT}/workspaces/{workspace}/members/{member.id}", headers=master_key_header)

    assert removed.status_code == 200, removed.text
    test_db.expire_all()
    remaining = test_db.execute(
        select(WorkspaceMcpServerGrant).where(WorkspaceMcpServerGrant.user_id == str(member.id))
    ).all()
    assert remaining == []


def test_a_member_named_by_another_spelling_of_their_id_is_refused(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    """Their Playground binds the canonical spelling, so a rule under another one would restrict nobody."""
    key = _new_key(client, master_key_header)
    workspace = _workspace_id(test_db, key)
    row = _store(test_db, workspace, "github")
    home = test_db.get(Workspace, workspace)
    assert home is not None
    member = Identity(email="upper@example.test", full_name="Upper", active_organization_id=home.organization_id)
    test_db.add(member)
    test_db.commit()
    test_db.add(WorkspaceMember(workspace_id=workspace, user_id=member.id, role="member", status="active"))
    test_db.commit()

    refused = client.post(
        f"{_servers(workspace)}/{row.id}/access/grants",
        json={"user_ids": [str(member.id).upper()], "tools": []},
        headers=master_key_header,
    )

    assert refused.status_code == 400, refused.text


def test_an_organization_admin_with_no_workspace_row_can_be_given_a_rule(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    """The Playground lets an organization owner or admin into every workspace, so a rule can name them too."""
    key = _new_key(client, master_key_header)
    workspace = _workspace_id(test_db, key)
    row = _store(test_db, workspace, "github")
    home = test_db.get(Workspace, workspace)
    assert home is not None
    admin = Identity(email="admin@example.test", full_name="Admin", active_organization_id=home.organization_id)
    plain = Identity(email="plain@example.test", full_name="Plain", active_organization_id=home.organization_id)
    test_db.add_all([admin, plain])
    test_db.commit()
    test_db.add_all(
        [
            OrganizationMember(organization_id=home.organization_id, user_id=admin.id, role="admin", status="active"),
            OrganizationMember(organization_id=home.organization_id, user_id=plain.id, role="member", status="active"),
        ]
    )
    test_db.commit()

    granted = _grant(client, master_key_header, row, [], users=[str(admin.id)])
    refused = client.post(
        f"{_servers(workspace)}/{row.id}/access/grants",
        json={"user_ids": [str(plain.id)], "tools": []},
        headers=master_key_header,
    )

    assert [grant["user_id"] for grant in granted["grants"]] == [str(admin.id)]
    assert refused.status_code == 400, refused.text


def test_naming_many_outsiders_costs_no_more_organization_lookups_than_naming_one(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")

    def lookups_naming(count: int) -> int:
        seen: list[str] = []

        def record(conn: Any, cursor: Any, statement: str, *args: Any) -> None:
            if "FROM organization_member" in statement:
                seen.append(statement)

        event.listen(Engine, "before_cursor_execute", record)
        try:
            refused = client.post(
                f"{_servers(row.workspace_id)}/{row.id}/access/grants",
                json={"user_ids": [str(uuid.uuid4()) for _ in range(count)], "tools": []},
                headers=master_key_header,
            )
        finally:
            event.remove(Engine, "before_cursor_execute", record)
        assert refused.status_code == 400, refused.text
        return len(seen)

    assert lookups_naming(50) == lookups_naming(1)


def test_a_grant_sent_while_the_user_is_deleted_waits_and_is_refused(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real endpoints: the deletion pauses after removing the user's grants, and a grant arrives."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": "victim"}, headers=master_key_header)
    assert created.status_code in (200, 201), created.text
    key = _new_key(client, master_key_header, user_id="victim")
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])
    reached, release = threading.Event(), threading.Event()
    revoke = revoke_user_mcp_grants

    async def paused_revoke(db: Any, user_id: str) -> None:
        await revoke(db, user_id)
        reached.set()
        await anyio.to_thread.run_sync(release.wait)

    monkeypatch.setattr("gateway.api.routes.users.revoke_user_mcp_grants", paused_revoke)
    answers: dict[str, Any] = {}
    deleting = threading.Thread(
        target=lambda: answers.update(deleted=client.delete(f"{API_ROOT}/users/victim", headers=master_key_header))
    )
    granting = threading.Thread(
        target=lambda: answers.update(
            granted=client.post(
                f"{_servers(row.workspace_id)}/{row.id}/access/grants",
                json={"user_ids": ["victim"], "tools": ["*"]},
                headers=master_key_header,
            )
        )
    )
    deleting.start()
    assert reached.wait(10)
    granting.start()
    granting.join(0.5)
    still_waiting = granting.is_alive()
    release.set()
    deleting.join(10)
    granting.join(10)

    assert still_waiting
    assert answers["deleted"].status_code in (200, 204), answers["deleted"].text
    assert answers["granted"].status_code == 400, answers["granted"].text
    recreated = client.post(f"{API_ROOT}/users", json={"user_id": "victim"}, headers=master_key_header)
    assert recreated.status_code in (200, 201), recreated.text
    assert _listed(client, _new_key(client, master_key_header, user_id="victim")) == []


def test_concurrent_grants_naming_the_same_people_do_not_deadlock(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each request reaches its member check, where each used to lock one of the two, before either goes on."""
    first_key = _new_key(client, master_key_header)
    here = _workspace_id(test_db, first_key)
    home = test_db.get(Workspace, here)
    assert home is not None
    there = Workspace(name="There", organization_id=home.organization_id)
    alice = Identity(email="alice@example.test", full_name="Alice", active_organization_id=home.organization_id)
    bob = Identity(email="bob@example.test", full_name="Bob", active_organization_id=home.organization_id)
    test_db.add_all([there, alice, bob])
    test_db.commit()
    test_db.add_all(
        [
            WorkspaceMember(workspace_id=workspace, user_id=person.id, role="member", status="active")
            for workspace in (here, there.id)
            for person in (alice, bob)
        ]
    )
    test_db.commit()
    for person, workspace in ((alice, here), (bob, there.id)):
        made = client.post(f"{API_ROOT}/users", json={"user_id": str(person.id)}, headers=master_key_header)
        assert made.status_code in (200, 201), made.text
        _new_key(client, master_key_header, user_id=str(person.id), workspace_id=str(workspace))
    servers = [_store(test_db, here, "github", everyone=[]), _store(test_db, there.id, "jira", everyone=[])]
    both_checked = threading.Barrier(2)
    provision = provision_attribution_users

    async def provision_then_wait(db: Any, user_ids: Any) -> set[str]:
        live = await provision(db, user_ids)
        await anyio.to_thread.run_sync(both_checked.wait, 10)
        return live

    monkeypatch.setattr(
        "gateway.services.tenancy.workspace_mcp_server_service.provision_attribution_users", provision_then_wait
    )
    answers: dict[str, Any] = {}

    def grant(row: WorkspaceMcpServer) -> None:
        answers[row.name] = client.post(
            f"{_servers(row.workspace_id)}/{row.id}/access/grants",
            json={"user_ids": [str(alice.id), str(bob.id)], "tools": ["*"]},
            headers=master_key_header,
        )

    threads = [threading.Thread(target=grant, args=(row,)) for row in servers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert {name: answer.status_code for name, answer in answers.items()} == {"github": 200, "jira": 200}


@pytest.mark.asyncio
async def test_minting_a_member_s_row_never_revives_one_deleted_meanwhile(postgres_url: str) -> None:
    """The row is created and deleted by another transaction while the insert waits; the insert leaves it so."""
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as other, sessions() as granting:
            other.add(BillingUser(user_id="raced", alias=None, deleted_at=datetime.now(UTC)))
            await other.flush()
            minting = asyncio.ensure_future(provision_attribution_users(granting, ["raced"]))
            await asyncio.sleep(0.3)
            await other.commit()

            assert await asyncio.wait_for(minting, timeout=10) == set()
            await granting.rollback()
        async with sessions() as check:
            stored = await check.get(BillingUser, "raced")
            assert stored is not None and stored.deleted_at is not None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_grant_and_the_user_s_deletion_cannot_interleave(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session], postgres_url: str
) -> None:
    """Both lock the user's row first, so a grant waiting on a deletion sees the user deleted once it commits."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": "racer"}, headers=master_key_header)
    assert created.status_code in (200, 201), created.text
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as deleting, sessions() as granting:
            assert await lock_live_users(deleting, ["racer"]) == {"racer"}
            await deleting.execute(
                update(BillingUser).where(BillingUser.user_id == "racer").values(deleted_at=func.now())
            )

            waiting = asyncio.ensure_future(lock_live_users(granting, ["racer"]))
            await asyncio.sleep(0.3)
            assert not waiting.done()
            await deleting.commit()

            assert await asyncio.wait_for(waiting, timeout=10) == set()
    finally:
        await engine.dispose()


def test_a_deleted_user_s_grants_do_not_come_back_with_a_user_recreated_under_its_id(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    created = client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=master_key_header)
    assert created.status_code in (200, 201), created.text
    key = _new_key(client, master_key_header, user_id="alice")
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])
    _grant(client, master_key_header, row, ["*"], users=["alice"])
    assert _listed(client, key) == ["github"]

    deleted = client.delete(f"{API_ROOT}/users/alice", headers=master_key_header)
    recreated = client.post(f"{API_ROOT}/users", json={"user_id": "alice"}, headers=master_key_header)
    fresh = _new_key(client, master_key_header, user_id="alice")

    assert deleted.status_code in (200, 204), deleted.text
    assert recreated.status_code in (200, 201), recreated.text
    assert _listed(client, fresh) == []


@pytest.mark.asyncio
async def test_a_request_reads_only_its_own_workspace_s_grants(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, async_db: AsyncSession
) -> None:
    """A user shared across workspaces, as ``default`` is, costs a request in one only that one's grants."""
    created = client.post(f"{API_ROOT}/users", json={"user_id": "shared"}, headers=master_key_header)
    assert created.status_code in (200, 201), created.text
    key = _new_key(client, master_key_header, user_id="shared")
    here = _workspace_id(test_db, key)
    home = test_db.get(Workspace, here)
    assert home is not None
    elsewhere = Workspace(name="Elsewhere", organization_id=home.organization_id)
    test_db.add(elsewhere)
    test_db.commit()
    mine = _store(test_db, here, "github", everyone=[])
    theirs = _store(test_db, elsewhere.id, "jira", everyone=[])
    test_db.add_all(
        [WorkspaceMcpServerGrant(server_id=server.id, user_id="shared", tools=["*"]) for server in (mine, theirs)]
    )
    test_db.commit()

    access = await resolve_request_mcp_access(async_db, await async_db.get(APIKey, key["id"]), here)
    grants = await WorkspaceMcpServerGrantRepository(async_db).for_caller(key["id"], "shared", [mine.id])

    assert [grant.server_id for grant in grants] == [mine.id]
    assert set(access.user_grants) == {mine.id}


def test_renaming_a_server_keeps_its_grants(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])
    _grant(client, master_key_header, row, ["*"], keys=[key])

    renamed = client.patch(f"{_servers(row.workspace_id)}/{row.id}", json={"name": "gh"}, headers=master_key_header)

    assert renamed.status_code == 200, renamed.text
    assert _listed(client, key) == ["gh"]


def test_a_workspace_can_refuse_servers_a_request_names_itself(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
) -> None:
    workspace = _workspace_id(test_db, api_key_obj)
    body = {
        "model": "openai:gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "mcp_servers": [{"name": "mine", "url": PUBLIC_URL}],
    }

    switched = client.patch(
        f"{API_ROOT}/workspaces/{workspace}", json={"request_mcp_servers_allowed": False}, headers=master_key_header
    )
    refused = client.post(f"{API_ROOT}/chat/completions", json=body, headers=api_key_header)

    assert switched.status_code == 200, switched.text
    assert switched.json()["request_mcp_servers_allowed"] is False
    assert refused.status_code == 400, refused.text
    assert "does not let a request name MCP servers" in refused.text


def test_a_new_workspace_refuses_servers_a_request_names_until_turned_on(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, api_key_obj: dict[str, Any]
) -> None:
    """The migration kept the workspaces it found on; one created after it starts off."""
    created = client.post(f"{API_ROOT}/workspaces", json={"name": "Locked"}, headers=master_key_header)
    existing = client.get(f"{API_ROOT}/workspaces/{_workspace_id(test_db, api_key_obj)}", headers=master_key_header)

    assert created.status_code in (200, 201), created.text
    assert created.json()["request_mcp_servers_allowed"] is False
    assert existing.json()["request_mcp_servers_allowed"] is True


def test_the_switch_also_refuses_an_mcp_server_the_provider_would_dial(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
) -> None:
    """A Responses ``type: "mcp"`` tool names a server by URL as an ``mcp_servers`` entry does."""
    workspace = _workspace_id(test_db, api_key_obj)
    switched = client.patch(
        f"{API_ROOT}/workspaces/{workspace}", json={"request_mcp_servers_allowed": False}, headers=master_key_header
    )
    assert switched.status_code == 200, switched.text
    body = {
        "model": "openai:gpt-4o-mini",
        "input": "hi",
        "tools": [{"type": "mcp", "server_label": "mine", "server_url": PUBLIC_URL, "require_approval": "never"}],
    }

    with patch("gateway.api.routes.responses.aresponses") as provider:
        refused = client.post(f"{API_ROOT}/responses", json=body, headers=api_key_header)

    assert refused.status_code == 400, refused.text
    assert "does not let a request name MCP servers" in refused.text
    provider.assert_not_called()


@pytest.mark.parametrize("field", ["mcp_servers", "mcpServers"])
def test_the_switch_also_refuses_mcp_servers_handed_to_gemini(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    field: str,
) -> None:
    """Gemini takes MCP servers on a tool entry, in either spelling its SDK accepts, and dials them itself."""
    workspace = _workspace_id(test_db, api_key_obj)
    switched = client.patch(
        f"{API_ROOT}/workspaces/{workspace}", json={"request_mcp_servers_allowed": False}, headers=master_key_header
    )
    assert switched.status_code == 200, switched.text
    body = {
        "model": "gemini:gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{field: [{"name": "external", "streamable_http_transport": {"url": PUBLIC_URL}}]}],
    }

    with patch("gateway.api.routes.chat.acompletion") as provider:
        refused = client.post(f"{API_ROOT}/chat/completions", json=body, headers=api_key_header)

    assert refused.status_code == 400, refused.text
    assert "does not let a request name MCP servers" in refused.text
    provider.assert_not_called()


@pytest.mark.parametrize("empty", [[], None])
def test_an_empty_gemini_mcp_field_is_not_refused(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    api_key_obj: dict[str, Any],
    api_key_header: dict[str, str],
    empty: list[Any] | None,
) -> None:
    """A Gemini tool whose MCP field names no server asks the provider to dial nothing."""
    workspace = _workspace_id(test_db, api_key_obj)
    switched = client.patch(
        f"{API_ROOT}/workspaces/{workspace}", json={"request_mcp_servers_allowed": False}, headers=master_key_header
    )
    assert switched.status_code == 200, switched.text
    body = {
        "model": "gemini:gemini-2.5-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"google_search": {}, "mcp_servers": empty}],
    }

    with patch("gateway.api.routes.chat.acompletion") as provider:
        answered = client.post(f"{API_ROOT}/chat/completions", json=body, headers=api_key_header)

    assert "does not let a request name MCP servers" not in answered.text
    assert provider.called


# --------------------------------------------------------------------------- #
# Traces of POST /mcp/execute
# --------------------------------------------------------------------------- #


def _tool_spans(db_session_factory: Callable[[], Session], key: dict[str, Any], count: int) -> list[TraceSpan]:
    """The key's tool spans, waiting for the writer, which stores traces off the request path."""
    deadline = time.monotonic() + 10
    while True:
        with db_session_factory() as db:
            spans = list(
                db.scalars(
                    select(TraceSpan)
                    .join(Trace, Trace.trace_id == TraceSpan.trace_id)
                    .where(Trace.api_key_id == key["id"], TraceSpan.kind == "tool")
                    .order_by(TraceSpan.start_time)
                )
            )
        if len(spans) >= count or time.monotonic() > deadline:
            return spans
        time.sleep(0.1)


def test_a_direct_call_is_traced_with_its_server_and_the_rule_that_allowed_it(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    session: _FakeSession,
    db_session_factory: Callable[[], Session],
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")

    response = client.post(
        f"{API_ROOT}/mcp/execute",
        json={**_execute_body(row, "list_issues"), "arguments": {"q": "secret"}},
        headers=_bearer(key),
    )

    assert response.status_code == 200, response.text
    [span] = _tool_spans(db_session_factory, key, 1)
    assert (span.tool_name, span.tool_type, span.outcome) == ("list_issues", "mcp", "ok")
    assert span.attributes == {
        "otari.mcp.server": "github",
        "otari.mcp.server_id": str(row.id),
        "otari.tool.decision": "accept",
        "otari.tool.decision_source": "mcp_everyone",
    }
    # Names and outcomes only: the arguments are not kept.
    assert span.content_sealed is None


def test_a_refused_call_is_traced_as_blocked_naming_the_rule_that_refused_it(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    session: _FakeSession,
    db_session_factory: Callable[[], Session],
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", read_tools=["list_issues"])
    _grant(client, master_key_header, row, ["@read"], keys=[key])

    response = client.post(f"{API_ROOT}/mcp/execute", json=_execute_body(row, "create_issue"), headers=_bearer(key))

    assert response.status_code == 403, response.text
    assert session.calls == []
    [span] = _tool_spans(db_session_factory, key, 1)
    assert (span.tool_name, span.outcome, span.error_class) == ("create_issue", "blocked", None)
    assert span.attributes is not None
    assert span.attributes["otari.tool.decision"] == "reject"
    assert span.attributes["otari.tool.decision_source"] == "mcp_key"


def test_a_call_outside_the_servers_tool_list_is_traced_as_refused_by_the_server(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    session: _FakeSession,
    db_session_factory: Callable[[], Session],
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", allowed_tools=["list_issues"])

    response = client.post(f"{API_ROOT}/mcp/execute", json=_execute_body(row, "create_issue"), headers=_bearer(key))

    assert response.status_code == 403, response.text
    [span] = _tool_spans(db_session_factory, key, 1)
    assert span.outcome == "blocked"
    assert span.attributes is not None
    assert span.attributes["otari.tool.decision_source"] == "mcp_server"


def test_a_call_to_a_server_the_key_cannot_reach_is_a_404_traced_as_blocked(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    session: _FakeSession,
    db_session_factory: Callable[[], Session],
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=[])

    response = client.post(f"{API_ROOT}/mcp/execute", json=_execute_body(row, "create_issue"), headers=_bearer(key))

    assert response.status_code == 404, response.text
    assert response.json()["code"] == CODE_SERVER_NOT_FOUND
    [span] = _tool_spans(db_session_factory, key, 1)
    assert (span.tool_name, span.outcome) == ("create_issue", "blocked")
    assert span.attributes is not None
    assert span.attributes["otari.mcp.server"] == "github"
    assert span.attributes["otari.tool.decision_source"] == "mcp_everyone"


def test_an_unreadable_token_does_not_tell_a_key_without_access_that_the_server_exists(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    allowed = _new_key(client, master_key_header)
    denied = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, allowed), "github", everyone=[])
    _grant(client, master_key_header, row, ["*"], keys=[allowed])
    row.encrypted_token = "not-a-sealed-token"
    test_db.commit()

    seen_by_denied = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=_bearer(denied))
    unknown = client.get(f"{API_ROOT}/mcp/servers/{uuid.uuid4()}/tools", headers=_bearer(denied))
    seen_by_allowed = client.get(f"{API_ROOT}/mcp/servers/{row.id}/tools", headers=_bearer(allowed))

    assert (seen_by_denied.status_code, seen_by_denied.json()["code"]) == (404, CODE_SERVER_NOT_FOUND)
    assert seen_by_denied.json()["detail"] == unknown.json()["detail"]
    assert seen_by_allowed.status_code == 500


def test_direct_calls_naming_a_conversation_are_steps_of_one_session(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    session: _FakeSession,
    db_session_factory: Callable[[], Session],
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github")
    headers = {**_bearer(key), CONVERSATION_HEADER: "agent-run-7"}

    for tool in ("list_issues", "delete_issue"):
        response = client.post(f"{API_ROOT}/mcp/execute", json=_execute_body(row, tool), headers=headers)
        assert response.status_code == 200, response.text

    spans = _tool_spans(db_session_factory, key, 2)
    assert [span.tool_name for span in spans] == ["list_issues", "delete_issue"]
    assert len({span.trace_id for span in spans}) == 1
    with db_session_factory() as db:
        trace = db.scalars(select(Trace).where(Trace.trace_id == spans[0].trace_id)).one()
        assert trace.name == "agent-run-7"


# --------------------------------------------------------------------------- #
# A completion's tool loop over a stored server
# --------------------------------------------------------------------------- #


def _round(tool: str | None) -> ChatCompletion:
    calls = (
        [
            ChatCompletionMessageFunctionToolCall(
                id="call_0", type="function", function=Function(name=tool, arguments="{}")
            )
        ]
        if tool
        else None
    )
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None if tool else "no access",
                    tool_calls=cast(Any, calls if tool else None),
                ),
                finish_reason="tool_calls" if tool else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_the_loop_refuses_a_withheld_tool_and_tells_the_model_what_it_holds(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    monkeypatch: pytest.MonkeyPatch,
    db_session_factory: Callable[[], Session],
) -> None:
    key = _new_key(client, master_key_header)
    row = _store(test_db, _workspace_id(test_db, key), "github", everyone=["list_*"])
    fake = _FakeSession()

    class _Session(_FakeSession):
        async def __aenter__(self) -> _FakeSession:
            return fake

        async def __aexit__(self, *exc: object) -> None:
            return None

    @asynccontextmanager
    async def transport(*args: Any, **kwargs: Any) -> Any:
        yield object(), object()

    async def initialize() -> None:
        return None

    monkeypatch.setattr(fake, "initialize", initialize, raising=False)
    monkeypatch.setattr("gateway.services.mcp_client.streamable_http_client", transport)
    monkeypatch.setattr("gateway.services.mcp_client.ClientSession", lambda *args: _Session())
    sent: list[dict[str, Any]] = []
    rounds = iter([_round("delete_issue"), _round(None)])

    async def provider(**kwargs: Any) -> ChatCompletion:
        sent.append(kwargs)
        return next(rounds)

    with patch("gateway.services.mcp_loop.acompletion", new=provider):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "delete issue 7"}],
                "mcp_server_ids": [str(row.id)],
            },
            headers=_bearer(key),
        )

    assert response.status_code == 200, response.text
    # Never handed to the client as a call of its own, and never sent to the server.
    assert response.json()["choices"][0]["message"].get("tool_calls") is None
    assert fake.calls == []
    system = sent[0]["messages"][0]
    assert system["role"] == "system"
    assert (
        "The user may use list_issues here. create_issue, delete_issue exist but are not granted" in system["content"]
    )
    [refusal] = [m["content"] for m in sent[1]["messages"] if m.get("role") == "tool"]
    assert refusal == (
        f"[tool error] {CODE_TOOL_NOT_ALLOWED}: delete_issue on github is not granted to you; ask a workspace admin"
    )
    [entry] = response.json()[OTARI_TOOL_CALLS_FIELD]
    assert (entry["server"], entry["name"], entry["status"]) == ("github", "delete_issue", "failed")
    [span] = _tool_spans(db_session_factory, key, 1)
    assert (span.tool_name, span.outcome) == ("delete_issue", "blocked")
    assert span.attributes is not None
    assert span.attributes["otari.tool.decision_source"] == "mcp_everyone"
    assert span.attributes["otari.mcp.server_id"] == str(row.id)
