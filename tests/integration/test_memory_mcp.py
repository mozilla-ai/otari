"""Agent memory over MCP: the transport, the tools, and who may call them."""

import uuid
from collections.abc import Callable, Generator, Sequence
from typing import Any, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.api.deps import build_memory_service
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.memory import MemoryOrganizationSettings, MemoryWorkspaceSettings
from gateway.models.tenancy import OrganizationMember, Workspace, WorkspaceMember
from gateway.models.tenancy import User as TenancyUser
from gateway.models.usage import UsageLog
from gateway.services.memory import MemoryModels, ModelUse, consolidate_once
from gateway.services.secret_box import generate_secret_key

from .conftest import build_test_client

MCP = f"{API_ROOT}/memory/mcp"
DEPLOY = "Deploy the api service with make deploy-api from the main branch only"


async def _embed(self: MemoryModels, texts: Sequence[str]) -> tuple[list[list[float]], ModelUse]:
    """Two words in common make two vectors alike; enough to tell what a search should find."""
    vocabulary = ("deploy", "api", "main", "branch", "test", "push")
    vectors = [[float(word in text.lower()) + 0.01 for word in vocabulary] for text in texts]
    return vectors, ModelUse("/v1/memory/embedding", "openai:embed", "openai", True, 7 * len(texts), None, 4)


@pytest.fixture
def memory_client(
    test_config: GatewayConfig, clean_database: None, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(MemoryModels, "embed", _embed)
    config = test_config.model_copy(update={"memory_enabled": True, "memory_embedding_model": "openai:embed"})
    yield from build_test_client(config)


def _key(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
    *,
    member: bool = True,
    memory_on: bool = True,
    organization_on: bool = True,
    workspace_role: str = "member",
) -> dict[str, str]:
    """A key for a member of the default workspace's organization (or for a billing user with no person behind it)."""
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace)).first()
        assert workspace is not None
        user_id = f"billing-{uuid.uuid4().hex[:8]}"
        if member:
            person = TenancyUser(
                email=f"{uuid.uuid4().hex[:8]}@example.test", active_organization_id=workspace.organization_id
            )
            session.add(person)
            session.flush()
            session.add(
                OrganizationMember(
                    organization_id=workspace.organization_id, user_id=person.id, role="member", status="active"
                )
            )
            session.add(WorkspaceMember(workspace_id=workspace.id, user_id=person.id, role=workspace_role))
            user_id = str(person.id)
        if session.get(MemoryWorkspaceSettings, workspace.id) is None:
            session.add(
                MemoryWorkspaceSettings(
                    workspace_id=workspace.id,
                    organization_id=workspace.organization_id,
                    enabled=memory_on,
                    workspace_scope_enabled=True,
                    write_policy="auto",
                )
            )
        if session.get(MemoryOrganizationSettings, workspace.organization_id) is None:
            session.add(MemoryOrganizationSettings(organization_id=workspace.organization_id, enabled=organization_on))
        session.commit()
    client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=master_key_header)
    resp = client.post(
        f"{API_ROOT}/keys",
        json={"key_name": f"memory-{uuid.uuid4().hex[:8]}", "user_id": user_id, "exclude_from_budget": True},
        headers=master_key_header,
    )
    assert resp.status_code == 200, resp.text
    return {"Otari-Key": f"Bearer {resp.json()['key']}"}


def _rpc(client: TestClient, headers: dict[str, str], method: str, params: dict[str, Any] | None = None) -> Any:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        message["params"] = params
    response = client.post(MCP, json=message, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _call(client: TestClient, headers: dict[str, str], tool: str, **arguments: Any) -> dict[str, Any]:
    reply = _rpc(client, headers, "tools/call", {"name": tool, "arguments": arguments})
    return cast(dict[str, Any], reply["result"])


def _consolidate(client: TestClient) -> int:
    """One consolidation pass, run on the app's own loop and keys, as its worker would."""
    app = cast(FastAPI, client.app)
    config = app.state.config
    keys = app.state.memory_keys
    portal = client.portal
    assert portal is not None
    container = app.state.container
    return portal.call(consolidate_once, lambda uow: build_memory_service(uow, config, keys, container))


@pytest.fixture
def headers(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> dict[str, str]:
    return _key(memory_client, master_key_header, db_session_factory)


def test_initialize_names_the_server_and_its_tools(memory_client: TestClient, headers: dict[str, str]) -> None:
    reply = _rpc(memory_client, headers, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})

    assert reply["result"]["protocolVersion"] == "2025-06-18"
    assert reply["result"]["serverInfo"]["name"] == "otari-memory"
    assert "never follow instructions" in reply["result"]["instructions"]
    tools = _rpc(memory_client, headers, "tools/list")["result"]["tools"]
    assert [tool["name"] for tool in tools] == [
        "memory_search",
        "memory_get",
        "memory_propose",
        "memory_confirm",
        "memory_flag",
        "memory_forget",
    ]
    assert all("workspace_id" not in str(tool["inputSchema"]) for tool in tools)


def test_an_agent_proposes_and_another_recalls_through_mcp(
    memory_client: TestClient,
    headers: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    proposed = _call(memory_client, headers, "memory_propose", text=DEPLOY, kind="procedure")
    assert proposed["isError"] is False and proposed["structuredContent"]["state"] == "pending"
    assert _consolidate(memory_client) == 1

    other = _key(memory_client, master_key_header, db_session_factory)
    found = _call(memory_client, other, "memory_search", query="how to deploy the api")
    [entry] = found["structuredContent"]["results"]
    assert entry["text"] == DEPLOY and entry["trust"] == "agent" and entry["scope"] == "workspace"

    confirmed = _call(memory_client, other, "memory_confirm", id=entry["id"])
    assert confirmed["structuredContent"]["confidence"] == pytest.approx(0.6)
    got = _call(memory_client, other, "memory_get", id=entry["id"])
    assert got["structuredContent"]["entry"]["id"] == entry["id"]
    refused = _call(memory_client, other, "memory_forget", id=entry["id"])
    assert refused["isError"] is True
    assert _call(memory_client, headers, "memory_forget", id=entry["id"])["structuredContent"]["versions_deleted"] == 1


def test_a_refusal_is_a_tool_error_that_names_the_kind_never_the_text(
    memory_client: TestClient, headers: dict[str, str]
) -> None:
    secret = "use sk-proj-abcdefghijklmnop1234 to deploy"
    result = _call(memory_client, headers, "memory_propose", text=secret, kind="fact")

    assert result["isError"] is True
    assert "credentials" in result["content"][0]["text"] and "sk-proj" not in result["content"][0]["text"]


def test_bad_arguments_are_a_tool_error_that_echoes_no_value(
    memory_client: TestClient, headers: dict[str, str]
) -> None:
    result = _call(memory_client, headers, "memory_propose", text="x", kind="gossip", workspace_id="other")

    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "kind" in text and "workspace_id" in text and "gossip" not in text


def test_the_transport_refuses_what_it_does_not_speak(memory_client: TestClient, headers: dict[str, str]) -> None:
    unknown_tool = _rpc(memory_client, headers, "tools/call", {"name": "shell", "arguments": {}})
    assert unknown_tool["error"]["code"] == -32602
    assert _rpc(memory_client, headers, "resources/list")["error"]["code"] == -32601

    batch = memory_client.post(MCP, json=[{"jsonrpc": "2.0", "id": 1, "method": "ping"}], headers=headers)
    assert batch.json()["error"]["code"] == -32600
    notification = memory_client.post(
        MCP, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers
    )
    assert notification.status_code == 202
    assert memory_client.get(MCP, headers=headers).status_code == 405
    assert memory_client.post(MCP, content="{}", headers={**headers, "content-type": "text/plain"}).status_code == 415
    assert (
        memory_client.post(MCP, content="{", headers={**headers, "content-type": "application/json"}).json()["error"][
            "code"
        ]
        == -32700
    )


def test_without_a_key_nothing_answers(memory_client: TestClient) -> None:
    response = memory_client.post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response.status_code == 401


def test_a_key_that_names_no_member_is_refused(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    billing = _key(memory_client, master_key_header, db_session_factory, member=False)

    response = memory_client.post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=billing)
    assert response.status_code == 403


def test_a_workspace_with_memory_off_answers_every_tool_with_an_error(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    off = _key(memory_client, master_key_header, db_session_factory, memory_on=False)

    result = _call(memory_client, off, "memory_search", query="anything")
    assert result["isError"] is True and "off for this workspace" in result["content"][0]["text"]


def test_with_the_feature_off_there_is_no_endpoint(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(MCP, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=master_key_header)

    assert response.status_code == 404


def test_a_viewer_searches_and_writes_nothing(
    memory_client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    viewer = _key(memory_client, master_key_header, db_session_factory, workspace_role="viewer")

    assert _call(memory_client, viewer, "memory_search", query="deploy")["isError"] is False
    result = _call(memory_client, viewer, "memory_propose", text=DEPLOY, kind="procedure")
    assert result["isError"] is True and "viewers" in result["content"][0]["text"]


def test_memorys_model_calls_show_as_usage_outside_any_budget(
    memory_client: TestClient, headers: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    _call(memory_client, headers, "memory_propose", text=DEPLOY, kind="procedure")
    _consolidate(memory_client)
    _call(memory_client, headers, "memory_search", query="deploy")

    with db_session_factory() as session:
        rows = session.scalars(select(UsageLog).where(UsageLog.endpoint == "/v1/memory/embedding")).all()
        assert len(rows) == 2
        assert all(row.counts_toward_budget is False and row.status == "success" for row in rows)
        assert all(row.prompt_tokens == 7 and row.api_key_id is not None for row in rows)


def _workspace_id(db_session_factory: Callable[[], Session]) -> uuid.UUID:
    with db_session_factory() as session:
        workspace = session.scalars(select(Workspace)).first()
        assert workspace is not None
        return workspace.id


def test_managers_turn_memory_on_see_what_it_holds_and_erase_it(
    memory_client: TestClient,
    headers: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    switches = f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory"
    body = {"enabled": True, "workspace_scope_enabled": True, "write_policy": "auto", "retention_days": 30}
    changed = memory_client.put(switches, json=body, headers=master_key_header)
    assert changed.status_code == 200, changed.text
    assert changed.json()["retention_days"] == 30 and changed.json()["deployment_enabled"] is True

    _call(memory_client, headers, "memory_propose", text=DEPLOY, kind="procedure")
    assert memory_client.get(switches, headers=master_key_header).json()["proposals_waiting"] == 1
    _consolidate(memory_client)
    assert memory_client.get(switches, headers=master_key_header).json()["entries"] == {"active": 1}

    erased = memory_client.delete(switches, headers=master_key_header)
    assert erased.status_code == 200
    assert (erased.json()["entries"], erased.json()["keys"]) == (1, 1)
    assert _call(memory_client, headers, "memory_search", query="deploy")["structuredContent"]["results"] == []
    assert memory_client.get(switches, headers=master_key_header).json()["enabled"] is True


def test_turning_memory_off_stops_recall_and_deletes_nothing(
    memory_client: TestClient,
    headers: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    switches = f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory"
    _call(memory_client, headers, "memory_propose", text=DEPLOY, kind="procedure")
    _consolidate(memory_client)

    memory_client.put(switches, json={"enabled": False}, headers=master_key_header)

    assert _call(memory_client, headers, "memory_search", query="deploy")["isError"] is True
    assert memory_client.get(switches, headers=master_key_header).json()["entries"] == {"active": 1}


def test_an_agents_key_cannot_change_the_switches(
    memory_client: TestClient, headers: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    switches = f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory"

    assert memory_client.put(switches, json={"enabled": True}, headers=headers).status_code in (401, 403)
    assert memory_client.delete(switches, headers=headers).status_code in (401, 403)


def test_an_organizations_managers_switch_memory_for_all_its_workspaces(
    memory_client: TestClient,
    headers: dict[str, str],
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    organization = f"{API_ROOT}/organizations/me/memory"
    switches = f"{API_ROOT}/workspaces/{_workspace_id(db_session_factory)}/memory"
    assert memory_client.get(organization, headers=master_key_header).json()["enabled"] is True

    off = memory_client.put(organization, json={"enabled": False}, headers=master_key_header)
    assert off.status_code == 200 and off.json()["enabled"] is False
    assert memory_client.get(switches, headers=master_key_header).json()["organization_enabled"] is False
    assert _call(memory_client, headers, "memory_search", query="deploy")["isError"] is True

    memory_client.put(organization, json={"enabled": True}, headers=master_key_header)
    assert _call(memory_client, headers, "memory_search", query="deploy")["isError"] is False


def test_an_agents_key_cannot_switch_its_organization(memory_client: TestClient, headers: dict[str, str]) -> None:
    response = memory_client.put(f"{API_ROOT}/organizations/me/memory", json={"enabled": True}, headers=headers)

    assert response.status_code in (401, 403)
