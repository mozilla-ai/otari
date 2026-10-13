"""Who runs a provider's web search or fetch keyword when a request's candidates disagree about it.

Under ``auto``, each attempt is decided on its own, as for code execution
(``test_code_execution_per_attempt.py``): a candidate that runs the keyword natively
is sent it, and one that cannot runs Otari's search or fetch and answers in the same
blocks. A fallback that cannot be served that way is skipped rather than refusing the
request. The pure decision is in ``tests/unit/test_candidate_tools.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.messages import (
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
    ToolUseBlock,
)
from fastapi.testclient import TestClient
from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog
from gateway.services.sandbox_backend import CODE_EXECUTION_TOOL_NAME
from gateway.services.tools import (
    WEB_FETCH_HEADER,
    WEB_FETCH_TOOL_NAME,
    WEB_SEARCH_HEADER,
    WEB_SEARCH_TOOL_NAME,
    ToolUseBudget,
)

_CLAUDE = "claude-haiku-4-5"
_SEARCH = {"type": "web_search_20250305", "name": "web_search"}
_FETCH = {"type": "web_fetch_20250910", "name": "web_fetch"}
_CODE = {"type": "code_execution_20250825", "name": "code_execution"}
_CODE_HEADER = "Otari-Code-Execution"
_HEADERS = {"search": WEB_SEARCH_HEADER, "fetch": WEB_FETCH_HEADER}
_DECLARATIONS = {"search": _SEARCH, "fetch": _FETCH}
_NAMES = {"search": WEB_SEARCH_TOOL_NAME, "fetch": WEB_FETCH_TOOL_NAME}
_RESULT_BLOCKS = {"search": "web_search_tool_result", "fetch": "web_fetch_tool_result"}


@pytest.fixture(scope="module")
def test_config(postgres_url: str) -> GatewayConfig:
    return GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "claude-then-vertex": {
                        "select": [{"default": f"anthropic:{_CLAUDE}"}],
                        "on_failure": [f"vertexai:{_CLAUDE}"],
                    },
                    "claude-then-bedrock": {
                        "select": [{"default": f"anthropic:{_CLAUDE}"}],
                        "on_failure": [f"bedrock:{_CLAUDE}"],
                    },
                    "claude-only": {
                        "select": [{"default": f"anthropic:{_CLAUDE}"}],
                        "on_failure": ["anthropic:claude-sonnet-4-5"],
                    },
                    "openai-then-mistral": {
                        "select": [{"default": "openai:gpt-4o-mini"}],
                        "on_failure": ["mistral:mistral-small-latest"],
                    },
                }
            }
        ),
    )


@pytest.fixture(autouse=True)
def _web_tools(monkeypatch: pytest.MonkeyPatch, test_config: GatewayConfig) -> None:
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    monkeypatch.setattr(test_config, "web_fetch_enabled", True)


def _unavailable() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))


def _reply(text: str, content: list[Any] | None = None, stop_reason: str = "end_turn") -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model=_CLAUDE,
        content=content if content is not None else [TextBlock(type="text", text=text, citations=None)],
        stop_reason=cast(Any, stop_reason),
        stop_sequence=None,
        usage=MessageUsage(input_tokens=5, output_tokens=2),
    )


def _responses_reply() -> Response:
    return Response(
        id="resp_test",
        created_at=0.0,
        model="fake",
        object="response",
        status=cast(Any, "completed"),
        output=[],
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
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
    )


async def _events(text: str) -> AsyncIterator[MessageStreamEvent]:
    yield MessageStartEvent(type="message_start", message=cast(Any, _reply(text)))
    yield MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=cast(Any, "end_turn"), stop_sequence=None),
        usage=MessageDeltaUsage(
            input_tokens=None,
            output_tokens=1,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
            server_tool_use=None,
        ),
    )
    yield MessageStopEvent(type="message_stop")


class _FakeWebBackend:
    """The retrieval backend's surface, answering every search and fetch from memory."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> _FakeWebBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}
            for name in (WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME)
        ]

    def owns_tool(self, name: str) -> bool:
        return name in (WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME)

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "page text"

    def take_last_results(self) -> list[dict[str, Any]]:
        return [{"url": "https://example.com", "title": "Example"}]

    def take_last_fetch(self) -> Any:
        return SimpleNamespace(text="page text", final_url=SimpleNamespace(display_url="https://example.com"))


@dataclass
class _Seen:
    """Every provider call, tool loop and backend one request made, in order."""

    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    loop_calls: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    web_backends: list[_FakeWebBackend] = field(default_factory=list)
    sandboxes: list[dict[str, Any]] = field(default_factory=list)

    def tool_types(self, kwargs: dict[str, Any]) -> set[str]:
        return {tool["type"] for tool in kwargs.get("tools") or []}


def _fake_sandbox(seen: _Seen) -> Any:
    def _build(**kwargs: Any) -> Any:
        seen.sandboxes.append(kwargs)
        backend = AsyncMock()
        backend.purpose_hints = lambda: []
        backend.__aenter__ = AsyncMock(return_value=backend)
        backend.__aexit__ = AsyncMock(return_value=None)
        return backend

    return _build


def _fake_web_backend(seen: _Seen) -> Any:
    def _build(**_kwargs: Any) -> _FakeWebBackend:
        backend = _FakeWebBackend()
        seen.web_backends.append(backend)
        return backend

    return _build


def _post_messages(
    client: TestClient,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    provider_fails: bool = False,
    loop_fails: bool = False,
    run_loop_for: str | None = None,
) -> tuple[Any, _Seen]:
    """POST ``body`` to Messages with the providers, loops and backends faked.

    ``run_loop_for`` runs the real tool loop, whose model asks for that tool once.
    """
    seen = _Seen()

    async def fake_amessages(**kwargs: Any) -> Any:
        seen.provider_calls.append(kwargs)
        if provider_fails and len(seen.provider_calls) == 1:
            raise _unavailable()
        if kwargs.get("stream"):
            return _events("via-provider")
        return _reply("via-provider")

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> MessageResponse:
        seen.loop_calls.append((completion_kwargs, extra))
        if loop_fails and len(seen.loop_calls) == 1:
            raise _unavailable()
        return _reply("via-otari")

    def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any
    ) -> AsyncIterator[MessageStreamEvent]:
        seen.loop_calls.append((completion_kwargs, extra))
        return _events("via-otari")

    rounds: list[dict[str, Any]] = []

    async def fake_loop_amessages(**kwargs: Any) -> MessageResponse:
        rounds.append(kwargs)
        if len(rounds) == 1:
            arguments: dict[str, object] = (
                {"query": "otari"} if run_loop_for == "search" else {"url": "https://example.com"}
            )
            use = ToolUseBlock(type="tool_use", id="toolu_1", name=_NAMES[run_loop_for or "search"], input=arguments)
            return _reply("", content=[use], stop_reason="tool_use")
        return _reply("via-otari")

    loops = (
        [patch("gateway.services.mcp_loop_messages.amessages", new=fake_loop_amessages)]
        if run_loop_for is not None
        else [
            patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
            patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        ]
    )
    with ExitStack() as stack:
        stack.enter_context(patch("gateway.api.routes.messages.amessages", new=fake_amessages))
        stack.enter_context(
            patch("gateway.api.routes._pipeline._build_web_retrieval_backend", new=_fake_web_backend(seen))
        )
        stack.enter_context(patch("gateway.api.routes._pipeline.SandboxBackend", new=_fake_sandbox(seen)))
        for loop in loops:
            stack.enter_context(loop)
        response = client.post(f"{API_ROOT}/messages", json=body, headers=headers)
    return response, seen


def _messages_body(policy: str, *tools: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "model": policy,
        "messages": [{"role": "user", "content": "look it up"}],
        "max_tokens": 100,
        "tools": [dict(tool) for tool in tools],
        **extra,
    }


def _rows(db_session: Session) -> list[UsageLog]:
    return list(db_session.scalars(select(UsageLog)).all())


def _auto(api_key_header: dict[str, str], *tools: str) -> dict[str, str]:
    return {**api_key_header, **{_HEADERS[tool]: "auto" for tool in tools}}


def _default_workspace_id(client: TestClient, master_key_header: dict[str, str]) -> str:
    workspace_id: str = client.get(f"{API_ROOT}/workspaces", headers=master_key_header).json()["data"][0]["id"]
    return workspace_id


# --- Messages ---------------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_a_native_head_runs_the_tool_itself_and_opens_no_backend(
    client: TestClient, api_key_header: dict[str, str], db_session: Session, tool: str
) -> None:
    response, seen = _post_messages(
        client, _auto(api_key_header, tool), _messages_body("claude-then-vertex", _DECLARATIONS[tool])
    )

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}]
    assert seen.loop_calls == []
    assert seen.web_backends == [], "a gateway backend was opened for a request the provider answered"
    served = [row for row in _rows(db_session) if row.status == "success"]
    assert [row.provider for row in served] == ["anthropic"]
    assert not (served[0].billing_meters or {}).get("tools"), "a gateway tool was charged"


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_a_fallback_that_cannot_run_the_tool_runs_otaris_with_anthropic_blocks(
    client: TestClient, api_key_header: dict[str, str], db_session: Session, tool: str
) -> None:
    response, seen = _post_messages(
        client,
        _auto(api_key_header, tool),
        _messages_body("claude-then-vertex", _DECLARATIONS[tool]),
        provider_fails=True,
        run_loop_for=tool,
    )

    assert response.status_code == 200, response.text
    content_types = [block["type"] for block in response.json()["content"]]
    assert content_types == ["server_tool_use", _RESULT_BLOCKS[tool], "text"]
    assert response.json()["content"][0]["id"].startswith("otari_srvtoolu_")
    # The head was sent Anthropic's tool, and only the fallback ran Otari's.
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}]
    assert len(seen.web_backends) == 1
    assert [name for name, _ in seen.web_backends[0].calls] == [_NAMES[tool]]
    assert {(row.provider, row.status) for row in _rows(db_session)} >= {
        ("anthropic", "absorbed"),
        ("vertexai", "success"),
    }


def test_the_fallbacks_request_carries_no_keyword_for_the_tool_otari_runs(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_messages(
        client, _auto(api_key_header, "search"), _messages_body("claude-then-bedrock", _SEARCH), provider_fails=True
    )

    assert response.status_code == 200, response.text
    loop_kwargs, loop_extra = seen.loop_calls[0]
    assert seen.tool_types(loop_kwargs) == set(), "the claimed declaration was forwarded to the fallback"
    assert loop_extra.get("native_tools") == frozenset({WEB_SEARCH_TOOL_NAME})


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_a_streamed_fallback_that_cannot_run_the_tool_runs_otaris(
    client: TestClient, api_key_header: dict[str, str], tool: str
) -> None:
    response, seen = _post_messages(
        client,
        _auto(api_key_header, tool),
        _messages_body("claude-then-vertex", _DECLARATIONS[tool], stream=True),
        provider_fails=True,
    )

    assert response.status_code == 200, response.text
    assert "via-otari" in response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}]
    loop_kwargs, loop_extra = seen.loop_calls[0]
    assert seen.tool_types(loop_kwargs) == set()
    assert loop_extra.get("native_tools") == frozenset({_NAMES[tool]})
    assert len(seen.web_backends) == 1


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_a_streamed_native_head_opens_no_backend(client: TestClient, api_key_header: dict[str, str], tool: str) -> None:
    response, seen = _post_messages(
        client, _auto(api_key_header, tool), _messages_body("claude-then-vertex", _DECLARATIONS[tool], stream=True)
    )

    assert response.status_code == 200, response.text
    assert "via-provider" in response.text
    assert seen.loop_calls == []
    assert seen.web_backends == []


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_a_web_access_policy_refusal_skips_only_the_fallback_that_needs_it(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    db_session: Session,
    tool: str,
) -> None:
    """A workspace with web access off still has Anthropic run its own."""
    workspace_id = _default_workspace_id(client, master_key_header)
    stored = client.put(
        f"{API_ROOT}/workspaces/{workspace_id}/web-search", json={"enabled": False}, headers=master_key_header
    )
    assert stored.status_code == 200, stored.text
    body = _messages_body("claude-then-vertex", _DECLARATIONS[tool])

    served, served_seen = _post_messages(client, _auto(api_key_header, tool), body)
    assert served.status_code == 200, served.text
    assert served_seen.web_backends == []

    failed, seen = _post_messages(client, _auto(api_key_header, tool), body, provider_fails=True)

    assert failed.status_code >= 500, failed.text
    assert len(seen.provider_calls) == 1
    assert seen.loop_calls == [], "the fallback used the web where the workspace switched it off"
    skipped = [row for row in _rows(db_session) if row.status == "absorbed" and row.provider == "vertexai"]
    assert len(skipped) == 1
    assert (skipped[0].error_message or "").startswith("Skipped: ")


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_an_unpriced_tool_skips_only_the_fallback_that_needs_it(
    client: TestClient, api_key_header: dict[str, str], db_session: Session, tool: str
) -> None:
    async def unpriced(_ctx: Any, tools: Any) -> str | None:
        return _NAMES[tool] if _NAMES[tool] in list(tools) else None

    body = _messages_body("claude-then-vertex", _DECLARATIONS[tool])
    with patch("gateway.api.routes._pipeline._unpriced_tool", new=unpriced):
        served, _ = _post_messages(client, _auto(api_key_header, tool), body)
        failed, seen = _post_messages(client, _auto(api_key_header, tool), body, provider_fails=True)

    assert served.status_code == 200, served.text
    assert failed.status_code >= 500, failed.text
    assert seen.loop_calls == []
    skipped = [row for row in _rows(db_session) if row.status == "absorbed" and row.provider == "vertexai"]
    assert len(skipped) == 1
    assert f"otari:{_NAMES[tool]}" in (skipped[0].error_message or "")
    rejections = [row for row in _rows(db_session) if row.provider == "otari" and row.status != "success"]
    assert rejections == [], "a refusal that answered no request was logged as one"


def test_a_plan_whose_candidates_all_run_the_tool_never_uses_otaris(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_messages(
        client,
        _auto(api_key_header, "search", "fetch"),
        _messages_body("claude-only", _SEARCH, _FETCH),
        provider_fails=True,
    )

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [
        {"web_search_20250305", "web_fetch_20250910"}
    ] * 2
    assert seen.loop_calls == []
    assert seen.web_backends == []


@pytest.mark.parametrize("tool", ["search", "fetch"])
def test_otari_runs_otaris_tool_on_every_candidate(
    client: TestClient, api_key_header: dict[str, str], tool: str
) -> None:
    response, seen = _post_messages(
        client, {**api_key_header, _HEADERS[tool]: "otari"}, _messages_body("claude-then-vertex", _DECLARATIONS[tool])
    )

    assert response.status_code == 200, response.text
    assert seen.provider_calls == []
    loop_kwargs, loop_extra = seen.loop_calls[0]
    assert seen.tool_types(loop_kwargs) == set()
    assert loop_extra.get("native_tools") == frozenset({_NAMES[tool]})


@pytest.mark.parametrize("tool", ["search", "fetch"])
@pytest.mark.parametrize("header", [None, "provider"])
@pytest.mark.parametrize("stream", [False, True])
def test_without_a_header_or_under_provider_the_keyword_is_forwarded_to_every_candidate(
    client: TestClient, api_key_header: dict[str, str], tool: str, header: str | None, stream: bool
) -> None:
    headers = {**api_key_header, _HEADERS[tool]: header} if header is not None else api_key_header
    body = _messages_body("claude-then-vertex", _DECLARATIONS[tool], **({"stream": True} if stream else {}))

    response, seen = _post_messages(client, headers, body, provider_fails=True)

    assert response.status_code == 200, response.text
    assert "via-provider" in response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{_DECLARATIONS[tool]["type"]}] * 2
    assert seen.loop_calls == []
    assert seen.web_backends == []


@pytest.mark.parametrize("stream", [False, True])
def test_code_and_search_on_a_fallback_that_runs_neither_run_together_on_otari(
    client: TestClient, api_key_header: dict[str, str], db_session: Session, stream: bool
) -> None:
    """The head runs Anthropic's own two tools, and the fallback runs Otari's sandbox and search in one loop."""
    headers = {**_auto(api_key_header, "search"), _CODE_HEADER: "auto"}
    body = _messages_body("claude-then-vertex", _CODE, _SEARCH, **({"stream": True} if stream else {}))

    served, served_seen = _post_messages(client, headers, body)
    assert served.status_code == 200, served.text
    assert [served_seen.tool_types(call) for call in served_seen.provider_calls] == [
        {"code_execution_20250825", "web_search_20250305"}
    ]
    assert served_seen.sandboxes == []
    assert served_seen.web_backends == []

    response, seen = _post_messages(client, headers, body, provider_fails=True)

    assert response.status_code == 200, response.text
    assert "via-otari" in response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [
        {"code_execution_20250825", "web_search_20250305"}
    ]
    [(loop_kwargs, loop_extra)] = seen.loop_calls
    assert seen.tool_types(loop_kwargs) == set()
    assert loop_extra.get("native_tools") == frozenset({CODE_EXECUTION_TOOL_NAME, WEB_SEARCH_TOOL_NAME})
    assert len(seen.sandboxes) == len(seen.web_backends) == 1
    if not stream:
        assert {(row.provider, row.status) for row in _rows(db_session)} >= {
            ("anthropic", "absorbed"),
            ("vertexai", "success"),
        }


def test_otaris_search_beside_the_providers_own_code_falls_back_to_running_both_on_otari(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """The head runs Anthropic's code beside Otari's search; the fallback runs both on Otari."""
    headers = {**api_key_header, WEB_SEARCH_HEADER: "otari", _CODE_HEADER: "auto"}
    body = _messages_body("claude-then-vertex", _CODE, _SEARCH)

    served, served_seen = _post_messages(client, headers, body)

    assert served.status_code == 200, served.text
    loop_kwargs, loop_extra = served_seen.loop_calls[0]
    assert served_seen.tool_types(loop_kwargs) == {"code_execution_20250825"}
    assert loop_extra.get("native_tools") == frozenset({WEB_SEARCH_TOOL_NAME})
    assert served_seen.sandboxes == []

    response, seen = _post_messages(client, headers, body, loop_fails=True)

    assert response.status_code == 200, response.text
    (head_kwargs, _), (fallback_kwargs, fallback_extra) = seen.loop_calls
    assert seen.tool_types(head_kwargs) == {"code_execution_20250825"}
    assert seen.tool_types(fallback_kwargs) == set()
    assert fallback_extra.get("native_tools") == frozenset({CODE_EXECUTION_TOOL_NAME, WEB_SEARCH_TOOL_NAME})
    assert len(seen.sandboxes) == 1, "only the fallback runs the code on the sandbox"
    assert len(seen.web_backends) == 2


class _ShadowingMcpPool:
    """An MCP pool whose one server exposes a tool named ``web_search``."""

    def __init__(self, configs: Any, *, tally: Any = None, trace: Any = None) -> None:
        return None

    async def __aenter__(self) -> _ShadowingMcpPool:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": WEB_SEARCH_TOOL_NAME, "description": "", "parameters": {}}}]

    def owns_tool(self, name: str) -> bool:
        return name == WEB_SEARCH_TOOL_NAME

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []


@pytest.mark.parametrize("stream", [False, True])
def test_an_mcp_tool_shadowing_a_builtin_only_the_fallback_runs_is_refused_before_the_head_is_called(
    client: TestClient, api_key_header: dict[str, str], stream: bool
) -> None:
    """Whether the request is refused does not wait on its head failing."""
    body = _messages_body(
        "claude-then-vertex",
        _SEARCH,
        mcp_servers=[{"name": "docs", "url": "http://127.0.0.1:9999/mcp"}],
        **({"stream": True} if stream else {}),
    )

    with patch("gateway.api.routes._pipeline.MCPClientPool", new=_ShadowingMcpPool):
        response, seen = _post_messages(client, _auto(api_key_header, "search"), body)

    assert response.status_code == 400, response.text
    assert f"'{WEB_SEARCH_TOOL_NAME}'" in response.text
    assert seen.provider_calls == []
    assert seen.loop_calls == []
    assert seen.web_backends == []


def test_caps_are_shared_by_attempts_that_run_different_tools(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    """Every attempt draws on one budget, which holds every cap the request declared."""
    headers = {**api_key_header, WEB_SEARCH_HEADER: "otari", WEB_FETCH_HEADER: "auto"}
    body = _messages_body("claude-then-vertex", {**_SEARCH, "max_uses": 3}, {**_FETCH, "max_uses": 0})

    response, seen = _post_messages(client, headers, body, loop_fails=True)

    assert response.status_code == 200, response.text
    assert len(seen.loop_calls) == 2
    (head_kwargs, head_extra), (fallback_kwargs, fallback_extra) = seen.loop_calls
    # The head runs Otari's search and Anthropic's fetch; the fallback runs both here.
    assert seen.tool_types(head_kwargs) == {"web_fetch_20250910"}
    assert seen.tool_types(fallback_kwargs) == set()
    budget = head_extra["use_budget"]
    assert isinstance(budget, ToolUseBudget)
    assert fallback_extra["use_budget"] is budget
    assert budget.exhausted(WEB_FETCH_TOOL_NAME), "the fetch cap only the fallback declares is not on the budget"
    assert not budget.exhausted(WEB_SEARCH_TOOL_NAME)


def test_gateway_minted_blocks_are_dropped_and_native_ones_forwarded_to_each_attempt(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    echoed = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "otari_srvtoolu_1", "name": "web_search", "input": {"query": "a"}},
                {"type": "web_search_tool_result", "tool_use_id": "otari_srvtoolu_1", "content": []},
                {"type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search", "input": {"query": "b"}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_2",
                    "content": [
                        {"type": "web_search_result", "url": "https://b", "title": "B", "encrypted_content": "x"}
                    ],
                },
                {"type": "text", "text": "answer"},
            ],
        },
        {"role": "user", "content": "again"},
    ]
    body = {**_messages_body("claude-then-vertex", _SEARCH), "messages": echoed}

    response, seen = _post_messages(client, _auto(api_key_header, "search"), body, provider_fails=True)

    assert response.status_code == 200, response.text
    head_ids = [
        block.get("id") or block.get("tool_use_id") for block in seen.provider_calls[0]["messages"][1]["content"]
    ]
    loop_kwargs, _ = seen.loop_calls[0]
    fallback_ids = [block.get("id") or block.get("tool_use_id") for block in loop_kwargs["messages"][1]["content"]]
    assert head_ids == fallback_ids == ["srvtoolu_2", "srvtoolu_2", None]


# --- Responses --------------------------------------------------------------------------------


def _post_responses(
    client: TestClient, headers: dict[str, str], body: dict[str, Any], *, provider_fails: bool = False
) -> tuple[Any, _Seen]:
    seen = _Seen()

    async def fake_aresponses(**kwargs: Any) -> Response:
        seen.provider_calls.append(kwargs)
        if provider_fails and len(seen.provider_calls) == 1:
            raise _unavailable()
        return _responses_reply()

    async def fake_loop(*, completion_kwargs: Any, pool: Any, max_iterations: int, **extra: Any) -> Response:
        seen.loop_calls.append((completion_kwargs, extra))
        return _responses_reply()

    with (
        patch("gateway.api.routes.responses.aresponses", new=fake_aresponses),
        patch("gateway.api.routes.responses.responses_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline._build_web_retrieval_backend", new=_fake_web_backend(seen)),
    ):
        response = client.post(f"{API_ROOT}/responses", json=body, headers=headers)
    return response, seen


def _responses_body() -> dict[str, Any]:
    return {"model": "openai-then-mistral", "input": "look it up", "tools": [{"type": "web_search"}]}


def test_responses_leaves_the_search_with_an_openai_head(client: TestClient, api_key_header: dict[str, str]) -> None:
    response, seen = _post_responses(client, _auto(api_key_header, "search"), _responses_body())

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"web_search"}]
    assert seen.loop_calls == []
    assert seen.web_backends == []


def test_responses_runs_otaris_search_for_a_fallback_without_one(
    client: TestClient, api_key_header: dict[str, str]
) -> None:
    response, seen = _post_responses(client, _auto(api_key_header, "search"), _responses_body(), provider_fails=True)

    assert response.status_code == 200, response.text
    assert [seen.tool_types(call) for call in seen.provider_calls] == [{"web_search"}]
    loop_kwargs, loop_extra = seen.loop_calls[0]
    assert seen.tool_types(loop_kwargs) == set()
    # A ``web_search_call`` item back, as OpenAI would have answered.
    assert loop_extra.get("native_tools") == frozenset({WEB_SEARCH_TOOL_NAME})
    assert len(seen.web_backends) == 1
