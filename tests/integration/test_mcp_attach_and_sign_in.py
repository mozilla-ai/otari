"""Servers a workspace attaches to every request, and the sign-in link a conversation is handed instead of a refusal.

Driven over HTTP with the provider and the MCP transport stood in for, on the three API shapes and
the streamed chat path. `test_mcp_user_connections.py` covers the resolution rules at the service layer.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, cast
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function,
)
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock, ToolUseBlock
from any_llm.types.responses import Response
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent
from mcp.types import Tool as MCPTool
from openai.types.responses import ResponseFunctionToolCall, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.error_codes import MCP_SIGN_IN_REQUIRED
from gateway.models.api_keys import APIKey
from gateway.models.mcp import ResolvedMcpServer
from gateway.models.tools import McpOAuthFlow, McpUserConnection
from gateway.services import mcp_stateless
from gateway.services.mcp_stateless import ServerCheck
from gateway.services.secret_box import encrypt_secret, generate_secret_key
from gateway.services.tools import OTARI_TOOL_CALLS_FIELD, _mcp_oauth
from gateway.services.tools._mcp_oauth import AUTH_METHOD_NONE, McpOAuthClient, McpOAuthMetadata, McpOAuthTokens

from .conftest import MODEL_NAME

PUBLIC_URL = "https://93.184.216.34/mcp"
ISSUER = "https://93.184.216.36"
ORIGIN = "https://otari.example.com"
LINK_PREFIX = f"{ORIGIN}{API_ROOT}/mcp-connections/sign-in?ticket="
USER_TOKEN = "user-access-token"
CLIENT_TOOL = {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}


@pytest.fixture(autouse=True)
def _deployment(test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    monkeypatch.setattr(test_config, "public_base_url", ORIGIN)
    yield


@pytest.fixture(autouse=True)
def _wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for the authorization server and for the connection check."""

    async def discover(server_url: str) -> McpOAuthMetadata:
        return McpOAuthMetadata(
            authorization_endpoint=f"{ISSUER}/authorize",
            token_endpoint=f"{ISSUER}/token",
            registration_endpoint=f"{ISSUER}/register",
        )

    async def register_client(registration_endpoint: str, redirect_uri: str) -> McpOAuthClient:
        return McpOAuthClient(client_id="dynamic", auth_method=AUTH_METHOD_NONE)

    async def exchange_code(*args: Any, **kwargs: Any) -> McpOAuthTokens:
        return McpOAuthTokens(access_token=USER_TOKEN, refresh_token="refresh", expires_in=3600)

    async def check(server: ResolvedMcpServer) -> ServerCheck:
        return ServerCheck(tool_names=[])

    monkeypatch.setattr(_mcp_oauth, "discover", discover)
    monkeypatch.setattr(_mcp_oauth, "register_client", register_client)
    monkeypatch.setattr(_mcp_oauth, "exchange_code", exchange_code)
    monkeypatch.setattr(mcp_stateless, "check_stored_server", check)


class _Mcp:
    """The MCP servers the tool loop connects to, as one fake that records what it was sent."""

    def __init__(self) -> None:
        self.headers: list[dict[str, str] | None] = []
        self.calls: list[str] = []

    async def list_tools(self, *, params: PaginatedRequestParams | None = None) -> ListToolsResult:
        return ListToolsResult(tools=[MCPTool(name="list_issues", description="list", input_schema={"type": "object"})])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="3 open issues")])

    async def initialize(self) -> None:
        return None


@pytest.fixture
def mcp(monkeypatch: pytest.MonkeyPatch) -> _Mcp:
    fake = _Mcp()

    @asynccontextmanager
    async def http_client(url: str, headers: dict[str, str] | None) -> Any:
        fake.headers.append(headers)
        yield object()

    @asynccontextmanager
    async def transport(*args: Any, **kwargs: Any) -> Any:
        yield object(), object()

    class _Session:
        async def __aenter__(self) -> _Mcp:
            return fake

        async def __aexit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr("gateway.services.mcp_client._origin_bound_http_client", http_client)
    monkeypatch.setattr("gateway.services.mcp_client.streamable_http_client", transport)
    monkeypatch.setattr("gateway.services.mcp_client.ClientSession", lambda *args: _Session())
    return fake


def _bearer(key: dict[str, Any]) -> dict[str, str]:
    return {API_KEY_HEADER: f"Bearer {key['key']}"}


def _key(client: TestClient, master: dict[str, str], user_id: str | None = "alice") -> dict[str, Any]:
    body: dict[str, Any] = {"key_name": f"k-{uuid.uuid4().hex[:6]}"}
    if user_id is not None:
        client.post(f"{API_ROOT}/users", json={"user_id": user_id}, headers=master)
        body["user_id"] = user_id
    response = client.post(f"{API_ROOT}/keys", json=body, headers=master)
    assert response.status_code == 200, response.text
    key: dict[str, Any] = response.json()
    return key


def _workspace(test_db: Session, key: dict[str, Any]) -> uuid.UUID:
    row = test_db.get(APIKey, key["id"])
    assert row is not None
    return row.workspace_id


def _server(client: TestClient, master: dict[str, str], workspace: uuid.UUID, name: str, **fields: Any) -> str:
    body: dict[str, Any] = {"name": name, "url": PUBLIC_URL, "purpose_hint": f"{name} hint", **fields}
    if body.get("credential_mode") != "per_user":
        body.setdefault("authorization_token", "shared-token")
    response = client.post(f"{API_ROOT}/workspaces/{workspace}/mcp-servers", json=body, headers=master)
    assert response.status_code == 201, response.text
    assert response.json()["attach_by_default"] is fields.get("attach_by_default", False)
    server_id: str = response.json()["id"]
    return server_id


def _round(tool: str | None, text: str = "done") -> ChatCompletion:
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
                    role="assistant", content=None if tool else text, tool_calls=cast(Any, calls)
                ),
                finish_reason="tool_calls" if tool else "stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@contextmanager
def _scripted(target: str, responses: list[Any]) -> Iterator[list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def fake(**kwargs: Any) -> Any:
        sent.append(kwargs)
        return next(remaining)

    with patch(target, new=fake):
        yield sent


def _tool_names(kwargs: dict[str, Any]) -> list[str]:
    return [tool["function"]["name"] if "function" in tool else tool["name"] for tool in kwargs.get("tools") or []]


def _tool_results(kwargs: dict[str, Any]) -> list[str]:
    return [m["content"] for m in kwargs["messages"] if m.get("role") == "tool"]


def _chat(client: TestClient, key: dict[str, Any], **extra: Any) -> Any:
    return client.post(
        f"{API_ROOT}/chat/completions",
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "check my open issues"}], **extra},
        headers=_bearer(key),
    )


def _link(text: str) -> str:
    start = text.index(LINK_PREFIX)
    return text[start:].split()[0]


# --------------------------------------------------------------------------- #
# Attached by default
# --------------------------------------------------------------------------- #


def test_an_attached_server_joins_a_request_that_named_nothing_beside_the_clients_own_tools(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, mcp: _Mcp
) -> None:
    key = _key(client, master_key_header)
    workspace = _workspace(test_db, key)
    _server(client, master_key_header, workspace, "jira", attach_by_default=True)
    _server(client, master_key_header, workspace, "quiet")

    with _scripted("gateway.services.mcp_loop.acompletion", [_round("list_issues"), _round(None)]) as sent:
        response = _chat(client, key, tools=[CLIENT_TOOL])

    assert response.status_code == 200, response.text
    assert _tool_names(sent[0]) == ["read_file", "list_issues"], "the client's tools stay, one server is attached"
    assert mcp.calls == ["list_issues"]
    assert mcp.headers == [{"Authorization": "Bearer shared-token"}]
    assert _tool_results(sent[1]) == ["3 open issues"]
    assert response.json()["choices"][0]["message"]["content"] == "done"
    [entry] = response.json()[OTARI_TOOL_CALLS_FIELD]
    assert (entry["server"], entry["name"], entry["status"]) == ("jira", "list_issues", "completed")


def test_a_client_call_still_goes_back_to_the_client(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, mcp: _Mcp
) -> None:
    key = _key(client, master_key_header)
    _server(client, master_key_header, _workspace(test_db, key), "jira", attach_by_default=True)

    with _scripted("gateway.services.mcp_loop.acompletion", [_round("read_file")]):
        response = _chat(client, key, tools=[CLIENT_TOOL])

    assert response.status_code == 200, response.text
    [call] = response.json()["choices"][0]["message"]["tool_calls"]
    assert call["function"]["name"] == "read_file"
    assert mcp.calls == []


def test_a_key_the_access_rules_deny_is_left_without_the_server(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, mcp: _Mcp
) -> None:
    key = _key(client, master_key_header)
    workspace = _workspace(test_db, key)
    server_id = _server(client, master_key_header, workspace, "jira", attach_by_default=True)
    granted = client.post(
        f"{API_ROOT}/workspaces/{workspace}/mcp-servers/{server_id}/access/grants",
        json={"api_key_ids": [key["id"]], "user_ids": [], "tools": []},
        headers=master_key_header,
    )
    assert granted.status_code == 200, granted.text

    with patch("gateway.api.routes.chat.acompletion", return_value=_round(None)) as direct:
        response = _chat(client, key)

    assert response.status_code == 200, response.text
    assert mcp.headers == [], "a server the key may not reach is never connected"
    assert direct.await_count == 1, "with nothing attached the request skips the tool loop"


def test_turning_attach_off_takes_the_server_back_off(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, mcp: _Mcp
) -> None:
    key = _key(client, master_key_header)
    workspace = _workspace(test_db, key)
    server_id = _server(client, master_key_header, workspace, "jira", attach_by_default=True)
    edited = client.patch(
        f"{API_ROOT}/workspaces/{workspace}/mcp-servers/{server_id}",
        json={"attach_by_default": False},
        headers=master_key_header,
    )
    assert edited.status_code == 200 and edited.json()["attach_by_default"] is False
    nulled = client.patch(
        f"{API_ROOT}/workspaces/{workspace}/mcp-servers/{server_id}",
        json={"attach_by_default": None},
        headers=master_key_header,
    )
    assert nulled.status_code == 422

    with patch("gateway.api.routes.chat.acompletion", return_value=_round(None)):
        response = _chat(client, key)

    assert response.status_code == 200, response.text
    assert mcp.headers == []


# --------------------------------------------------------------------------- #
# The sign-in link in the conversation
# --------------------------------------------------------------------------- #


def test_an_unsigned_in_user_gets_a_link_in_the_conversation_and_the_next_ask_works(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, mcp: _Mcp
) -> None:
    key = _key(client, master_key_header)
    workspace = _workspace(test_db, key)
    server_id = _server(
        client, master_key_header, workspace, "jira", credential_mode="per_user", attach_by_default=True
    )

    with _scripted("gateway.services.mcp_loop.acompletion", [_round("sign_in_to_jira"), _round(None)]) as sent:
        response = _chat(client, key, tools=[CLIENT_TOOL])

    assert response.status_code == 200, response.text
    assert _tool_names(sent[0]) == ["read_file", "sign_in_to_jira"]
    assert "jira hint" in sent[0]["messages"][0]["content"]
    [told] = _tool_results(sent[1])
    link = _link(told)
    assert mcp.headers == [], "nothing connects to a server the user has not signed in to"
    assert test_db.execute(select(McpOAuthFlow)).first() is None, "minting a link stores nothing"

    opened = client.get(link.removeprefix(ORIGIN), follow_redirects=False)
    assert opened.status_code == 303, opened.text
    assert opened.headers["location"].startswith(f"{ISSUER}/authorize?")
    again = client.get(link.removeprefix(ORIGIN), follow_redirects=False)
    test_db.expire_all()
    [flow] = test_db.execute(select(McpOAuthFlow)).scalars().all()
    assert (flow.user_id, flow.identity_id) == ("alice", None), "opening it twice leaves one pending sign-in"
    state = parse_qs(urlsplit(again.headers["location"]).query)["state"][0]

    landed = client.get(f"{API_ROOT}/mcp-connections/callback", params={"state": state, "code": "c"})
    assert landed.status_code == 200
    assert "You're signed in" in landed.text and "jira is now connected" in landed.text
    assert "Go back to your conversation" in landed.text
    assert USER_TOKEN not in landed.text and state not in landed.text
    assert test_db.execute(select(McpUserConnection)).scalar_one().server_id == uuid.UUID(server_id)

    with _scripted("gateway.services.mcp_loop.acompletion", [_round("list_issues"), _round(None)]) as sent:
        response = _chat(client, key)

    assert response.status_code == 200, response.text
    assert _tool_names(sent[0]) == ["list_issues"]
    assert mcp.headers == [{"Authorization": f"Bearer {USER_TOKEN}"}], "the next ask carries the user's own token"


def test_a_link_that_is_forged_or_altered_starts_nothing(client: TestClient) -> None:
    for ticket in ("", "nonsense", encrypt_secret('{"p":"other","s":"x","u":"alice"}').rstrip("=")):
        opened = client.get(f"{API_ROOT}/mcp-connections/sign-in", params={"ticket": ticket}, follow_redirects=False)
        assert opened.status_code == 400
        assert "This link has expired" in opened.text
        assert opened.headers["content-security-policy"].startswith("default-src 'none'")


def test_a_failed_sign_in_from_a_link_ends_on_the_plain_page(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _key(client, master_key_header)
    _server(
        client, master_key_header, _workspace(test_db, key), "jira", credential_mode="per_user", attach_by_default=True
    )
    listed = client.get(f"{API_ROOT}/mcp/servers", headers=_bearer(key)).json()["servers"]
    opened = client.get(listed[0]["sign_in_url"].removeprefix(ORIGIN), follow_redirects=False)
    state = parse_qs(urlsplit(opened.headers["location"]).query)["state"][0]

    refused = client.get(f"{API_ROOT}/mcp-connections/callback", params={"state": state, "error": "access_denied"})

    assert refused.status_code == 400
    assert "The sign-in did not finish" in refused.text


def test_the_streamed_chat_answer_carries_the_link_and_no_gateway_call(
    client: TestClient, master_key_header: dict[str, str], test_db: Session, mcp: _Mcp
) -> None:
    key = _key(client, master_key_header)
    _server(
        client, master_key_header, _workspace(test_db, key), "jira", credential_mode="per_user", attach_by_default=True
    )

    def chunk(delta: dict[str, Any], finish: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk.model_validate(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4} if finish else None,
            }
        )

    async def events(*items: Any) -> AsyncIterator[Any]:
        for item in items:
            yield item

    call = {"index": 0, "id": "call_0", "type": "function", "function": {"name": "sign_in_to_jira", "arguments": "{}"}}
    rounds = [
        events(chunk({"tool_calls": [call]}), chunk({}, "tool_calls")),
        events(chunk({"content": "ok"}), chunk({}, "stop")),
    ]
    with _scripted("gateway.services.mcp_loop.acompletion", rounds) as sent:
        response = _chat(client, key, stream=True)

    assert response.status_code == 200, response.text
    payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    deltas = [p["choices"][0]["delta"] for p in payloads if p.get("choices")]
    assert not any(delta.get("tool_calls") for delta in deltas), "the stand-in call leaked to the client"
    assert "".join(delta.get("content") or "" for delta in deltas) == "ok"
    assert _link(_tool_results(sent[1])[0]).startswith(LINK_PREFIX)


def _message(stop: str, content: list[Any]) -> MessageResponse:
    return MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="fake",
        content=content,
        stop_reason=cast(Any, stop),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def test_the_messages_shape_offers_the_stand_in_and_returns_the_link(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _key(client, master_key_header)
    _server(
        client, master_key_header, _workspace(test_db, key), "jira", credential_mode="per_user", attach_by_default=True
    )
    rounds = [
        _message("tool_use", [ToolUseBlock(type="tool_use", id="tu_0", name="sign_in_to_jira", input={})]),
        _message("end_turn", [TextBlock(type="text", text="done", citations=None)]),
    ]

    with _scripted("gateway.services.mcp_loop_messages.amessages", rounds) as sent:
        response = client.post(
            f"{API_ROOT}/messages",
            json={"model": MODEL_NAME, "max_tokens": 100, "messages": [{"role": "user", "content": "my issues"}]},
            headers=_bearer(key),
        )

    assert response.status_code == 200, response.text
    assert [tool["name"] for tool in sent[0]["tools"]] == ["sign_in_to_jira"]
    results = [
        block["content"]
        for message in sent[1]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert LINK_PREFIX in json.dumps(results)


def test_the_responses_shape_offers_the_stand_in_and_returns_the_link(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _key(client, master_key_header)
    _server(
        client, master_key_header, _workspace(test_db, key), "jira", credential_mode="per_user", attach_by_default=True
    )

    def response_with(output: list[Any]) -> Response:
        return Response(
            id="resp_1",
            created_at=0.0,
            model=MODEL_NAME,
            object="response",
            status="completed",
            output=output,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
            usage=ResponseUsage(
                input_tokens=5,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=2,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=7,
            ),
        )

    call = ResponseFunctionToolCall(type="function_call", call_id="c0", name="sign_in_to_jira", arguments="{}")
    with _scripted(
        "gateway.services.mcp_loop_responses.aresponses", [response_with([call]), response_with([])]
    ) as sent:
        response = client.post(
            f"{API_ROOT}/responses", json={"model": MODEL_NAME, "input": "my issues"}, headers=_bearer(key)
        )

    assert response.status_code == 200, response.text
    assert [tool["name"] for tool in sent[0]["tools"]] == ["sign_in_to_jira"]
    [output] = [item for item in sent[1]["input_data"] if item.get("type") == "function_call_output"]
    assert LINK_PREFIX in output["output"]


def test_a_key_with_no_user_is_told_why_and_gets_no_link(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _key(client, master_key_header, user_id=None)
    workspace = _workspace(test_db, key)
    named = _server(client, master_key_header, workspace, "jira", credential_mode="per_user")
    _server(client, master_key_header, workspace, "drive", credential_mode="per_user", attach_by_default=True)

    with _scripted("gateway.services.mcp_loop.acompletion", [_round("sign_in_to_jira"), _round(None)]) as sent:
        response = _chat(client, key, mcp_server_ids=[named])

    assert response.status_code == 200, response.text
    assert _tool_names(sent[0]) == ["sign_in_to_jira"], "an attached one is left out for a key with no user"
    [told] = _tool_results(sent[1])
    assert "belongs to no person" in told
    assert LINK_PREFIX not in told


def test_the_caller_run_routes_carry_the_link_beside_their_refusal(
    client: TestClient, master_key_header: dict[str, str], test_db: Session
) -> None:
    key = _key(client, master_key_header)
    server_id = _server(client, master_key_header, _workspace(test_db, key), "jira", credential_mode="per_user")

    [listed] = client.get(f"{API_ROOT}/mcp/servers", headers=_bearer(key)).json()["servers"]
    tools = client.get(f"{API_ROOT}/mcp/servers/{server_id}/tools", headers=_bearer(key))

    assert listed["sign_in_required"] is True and listed["sign_in_url"].startswith(LINK_PREFIX)
    assert tools.status_code == 403
    assert tools.json()["code"] == MCP_SIGN_IN_REQUIRED
    assert tools.json()["sign_in_url"].startswith(LINK_PREFIX)
    assert test_db.execute(select(McpOAuthFlow)).first() is None, "listing and refusing mint no sign-in"


def test_an_attached_server_that_is_down_is_left_out_and_a_named_one_still_fails(
    client: TestClient,
    master_key_header: dict[str, str],
    test_db: Session,
    mcp: _Mcp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _key(client, master_key_header)
    workspace = _workspace(test_db, key)
    _server(client, master_key_header, workspace, "jira", attach_by_default=True)
    named = _server(client, master_key_header, workspace, "named")

    async def unreachable() -> None:
        raise ConnectionError("down")

    monkeypatch.setattr(mcp, "initialize", unreachable)
    with patch("gateway.api.routes.chat.acompletion", return_value=_round(None)):
        with _scripted("gateway.services.mcp_loop.acompletion", [_round(None)]) as sent:
            response = _chat(client, key, tools=[CLIENT_TOOL])
        refused = _chat(client, key, mcp_server_ids=[named])

    assert response.status_code == 200, response.text
    assert _tool_names(sent[0]) == ["read_file"], "the request goes on without the server that is down"
    assert refused.status_code >= 500, refused.text
