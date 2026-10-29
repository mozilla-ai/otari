"""A stored MCP server's access list: everyone_tools, grants for keys and users, and everywhere they bind.

The stateless MCP routes, the access list routes, test access, a key's summary and the workspace's
request-server switch run against the real database. URLs are public IP literals, so no resolver is reached.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent
from mcp.types import Tool as MCPTool
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.models.api_keys import APIKey
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tools import WorkspaceMcpServer
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import CODE_SERVER_NOT_FOUND, CODE_TOOL_NOT_ALLOWED
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tenancy import workspace_mcp_server_service

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
            tools=[MCPTool(name=name, description=name, input_schema={"type": "object"}) for name in TOOL_NAMES]
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
) -> WorkspaceMcpServer:
    row = WorkspaceMcpServer(
        workspace_id=workspace_id,
        name=name,
        url=PUBLIC_URL,
        encrypted_token=encrypt_secret("ghp_token"),
        enabled=enabled,
        everyone_tools=["*"] if everyone is None else everyone,
        allowed_tools=allowed_tools,
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
    assert unknown.status_code == 400


def test_a_keys_summary_covers_every_server_of_its_workspace(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _new_key(client, master_key_header, user_id="ana")
    workspace = _workspace_id(test_db, key)
    github = _store(test_db, workspace, "github")
    jira = _store(test_db, workspace, "jira", everyone=[], allowed_tools=["search"])
    _grant(client, master_key_header, jira, ["search"], users=["ana"])

    response = client.get(f"{_servers(workspace)}/keys/{key['id']}/access", headers=master_key_header)

    assert response.status_code == 200, response.text
    summary = {entry["name"]: entry for entry in response.json()}
    assert summary["github"]["decided_by"] == "everyone"
    assert summary["github"]["tools"] == ["*"]
    assert summary["github"]["server_id"] == str(github.id)
    assert (summary["jira"]["decided_by"], summary["jira"]["tools"], summary["jira"]["allowed_tools"]) == (
        "user",
        ["search"],
        ["search"],
    )
    assert client.get(f"{_servers(workspace)}/keys/no-such-key/access", headers=master_key_header).status_code == 400


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
