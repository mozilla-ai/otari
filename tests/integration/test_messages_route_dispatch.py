"""Route-level tests for the /v1/messages endpoint wiring.

These complement :mod:`tests.unit.test_mcp_loop_messages` (which tests the
Anthropic tool loop in isolation) by exercising the FastAPI route handler:
tool extraction, mutual-exclusivity validation, error-body mapping to the
Anthropic shape, the per-tool dispatch into the right backend, and the
prompt-cache contract (``cache_control`` markers reaching the provider
unchanged, cache usage reaching the client).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    TextBlock,
    TextDelta,
)
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.services.mcp_client import MCPToolCallOutcome
from gateway.services.mcp_loop_messages import MCP_ACTIVITY_ID_PREFIX, MCP_CLIENT_BETA

_CONTEXT_MANAGEMENT = {"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}}]}
_BETAS = ["compact-2026-01-12"]
_AUTOMATIC_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}


def _text_response(
    text: str = "ok",
    *,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=MessageUsage(
            input_tokens=5,
            output_tokens=2,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation=None,
            server_tool_use=None,
            service_tier=None,
        ),
        container=None,
    )


def _compaction_response() -> MessageResponse:
    return MessageResponse.model_validate(
        {
            "id": "msg_compaction",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5",
            "content": [{"type": "compaction", "content": "Conversation summary"}],
            "stop_reason": "compaction",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "iterations": [
                    {
                        "type": "compaction",
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                    {
                        "type": "message",
                        "model": "claude-opus-5",
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                ],
            },
            "context_management": {
                "applied_edits": [
                    {
                        "type": "clear_tool_uses_20250919",
                        "cleared_input_tokens": 42,
                        "cleared_tool_uses": 2,
                    }
                ]
            },
        }
    )


# ---------- plain amessages (no gateway tools) ----------


def test_no_tools_falls_through_to_plain_amessages(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A vanilla request with no gateway-managed tools hits ``amessages`` directly,
    bypassing the tool loop entirely.
    """
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("hi")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["content"][0]["text"] == "hi"
    # Direct amessages call: tool-loop-only fields shouldn't appear.
    assert "tools" not in captured


def test_container_reaches_plain_amessages_unchanged(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Top-level container continuity reaches the upstream provider call unchanged."""
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Continue"}],
                "max_tokens": 100,
                "container": "container_01ABC",
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["container"] == "container_01ABC"


def test_container_is_dropped_when_the_gateway_runs_code_execution(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``otari_code_execution`` means the sandbox runs the code, so Anthropic is
    never asked to stand up a container this request could reach."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    forwarded: dict[str, Any] = {}

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        forwarded.update(completion_kwargs)
        return _text_response()

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes._pipeline.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_code_execution"}],
                "container": "container_01ABC",
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert "container" not in forwarded


def test_container_survives_provider_native_code_execution(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """The provider-named keyword leaves execution with the provider, which is
    the case the container id is for."""
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "tools": [{"type": "code_execution_20250825", "name": "code_execution"}],
                "betas": ["code-execution-2025-08-25"],
                "container": "container_01ABC",
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["container"] == "container_01ABC"


def test_cache_control_and_non_stream_usage_round_trip(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Block and automatic cache controls reach Anthropic unchanged, and cache
    usage reaches the client.
    """
    system_block = {
        "type": "text",
        "text": "Stable system instructions",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    message_block = {
        "type": "text",
        "text": "Stable reference material",
        "cache_control": {"type": "ephemeral"},
    }
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response(cache_creation_input_tokens=13, cache_read_input_tokens=8)

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "system": [system_block],
                "messages": [{"role": "user", "content": [message_block]}],
                "max_tokens": 100,
                "cache_control": _AUTOMATIC_CACHE_CONTROL,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["system"] == [system_block]
    assert captured["messages"] == [{"role": "user", "content": [message_block]}]
    assert captured["cache_control"] == _AUTOMATIC_CACHE_CONTROL
    usage = resp.json()["usage"]
    assert usage["cache_creation_input_tokens"] == 13
    assert usage["cache_read_input_tokens"] == 8


def test_context_management_non_stream_contract(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Context management reaches any-llm and beta response data reaches the client."""
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _compaction_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-opus-5",
                "messages": [{"role": "user", "content": "Summarize when needed"}],
                "max_tokens": 100,
                "context_management": _CONTEXT_MANAGEMENT,
                "betas": _BETAS,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["context_management"] == _CONTEXT_MANAGEMENT
    assert captured["betas"] == _BETAS
    body = resp.json()
    assert body["content"] == [{"type": "compaction", "content": "Conversation summary"}]
    assert body["context_management"]["applied_edits"][0]["cleared_input_tokens"] == 42
    assert body["usage"]["iterations"][0]["type"] == "compaction"


def test_mcp_client_beta_is_not_forwarded_to_provider(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """The MCP client capability is consumed while other betas reach any-llm."""
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages?beta=true",
            json={
                "model": "anthropic:claude-opus-5",
                "messages": [{"role": "user", "content": "Use the beta"}],
                "max_tokens": 100,
                "betas": _BETAS,
            },
            headers={
                **api_key_header,
                "anthropic-beta": f"{MCP_CLIENT_BETA},files-api-2025-04-14,{MCP_CLIENT_BETA}",
            },
        )

    assert resp.status_code == 200, resp.text
    assert captured["betas"] == [*_BETAS, "files-api-2025-04-14"]


def test_mcp_client_beta_is_removed_for_translated_provider(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages?beta=true",
            json={
                "model": "openai:gpt-4o",
                "messages": [{"role": "user", "content": "Use the MCP beta"}],
                "max_tokens": 100,
            },
            headers={**api_key_header, "anthropic-beta": MCP_CLIENT_BETA},
        )

    assert resp.status_code == 200, resp.text
    assert "betas" not in captured


def test_gateway_internal_fields_are_stripped_from_upstream_kwargs(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``mcp_servers`` / ``tools_header`` / ``max_tool_iterations`` are gateway-
    only knobs; Anthropic rejects unknown kwargs with a 400. Stripping must
    happen at the boundary so they never reach ``amessages``.
    """
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "tools_header": "Tools available:",
                "max_tool_iterations": 3,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    for field in ("mcp_servers", "mcp_server_ids", "tools_header", "max_tool_iterations", "user"):
        assert field not in captured, f"gateway-internal field {field!r} leaked to upstream"


def test_user_supplied_openai_shape_tools_get_converted_to_anthropic(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A caller can mix gateway-managed tools and their own function tools; the
    OpenAI-shape ones must be converted to ``{name, description, input_schema}``
    before the call reaches Anthropic.
    """
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "do it"}],
                "max_tokens": 100,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "look stuff up",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    forwarded_tools = captured.get("tools")
    assert forwarded_tools is not None
    assert forwarded_tools[0]["name"] == "lookup"
    assert forwarded_tools[0]["input_schema"] == {"type": "object", "properties": {}}
    # The OpenAI wrapper keys must be gone.
    assert "function" not in forwarded_tools[0]
    assert forwarded_tools[0].get("type") != "function"


# ---------- gateway tool dispatch ----------


def test_cache_control_survives_route_level_purpose_hint_injection(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Purpose hints add a block without altering the caller's cached block."""
    system_block = {
        "type": "text",
        "text": "Stable system instructions",
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    captured: dict[str, Any] = {}

    class FakePool:
        @property
        def openai_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up information",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]

        def purpose_hints(self) -> list[tuple[str, str]]:
            return [("reference", "Use for authoritative lookups")]

        def owns_tool(self, name: str) -> bool:
            return name == "lookup"

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            raise AssertionError(f"Unexpected tool call: {name}({arguments})")

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response()

    fake_pool = FakePool()
    with (
        patch("gateway.services.mcp_loop_messages.amessages", new=fake_amessages),
        patch("gateway.services.mcp_client.MCPClientPool.__aenter__", new=AsyncMock(return_value=fake_pool)),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "system": [system_block],
                "messages": [{"role": "user", "content": "Look this up"}],
                "max_tokens": 100,
                "mcp_servers": [
                    {
                        "name": "reference",
                        "url": "http://127.0.0.1:9999/mcp",
                        "purpose_hint": "Use for authoritative lookups",
                    }
                ],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["system"][0]["type"] == "text"
    assert "Use for authoritative lookups" in captured["system"][0]["text"]
    assert captured["system"][1] == system_block


def test_mcp_servers_dispatches_through_anthropic_tool_loop(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """A request with ``mcp_servers`` routes through ``anthropic_tool_loop`` rather
    than calling ``amessages`` directly. Catches regressions where the dispatch
    if/elif chain silently falls through to the plain path.
    """
    seen: dict[str, Any] = {}

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        seen["completion_kwargs"] = completion_kwargs
        seen["pool"] = pool
        seen["max_iterations"] = max_iterations
        return _text_response("via-mcp-loop")

    plain_amessages_called = False

    async def fake_amessages(**_kwargs: Any) -> MessageResponse:
        nonlocal plain_amessages_called
        plain_amessages_called = True
        return _text_response()

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "mcp_servers": [
                    {"name": "test", "url": "http://127.0.0.1:9999/mcp"},
                ],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "via-mcp-loop"
    assert "completion_kwargs" in seen, "anthropic_tool_loop was not invoked"
    assert plain_amessages_called is False


def test_web_search_replay_is_stripped_when_interception_is_off(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway provenance, not the current interception setting, controls replay stripping."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    replayed_messages = [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_echoed", "name": "web_search", "input": {}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_echoed",
                    "content": [],
                },
                {"type": "text", "text": "answer"},
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    captured: dict[str, Any] = {}

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        captured.update(completion_kwargs)
        return _text_response("ok")

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": replayed_messages,
                "max_tokens": 100,
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:9999/mcp"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["messages"] == [
        replayed_messages[0],
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        replayed_messages[2],
    ]


def test_code_execution_dispatches_through_sandbox_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tools: [{"type": "otari_code_execution"}]`` routes through ``SandboxBackend``."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    pool_seen: list[Any] = []

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        pool_seen.append(pool)
        return _text_response("via-sandbox-loop")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes._pipeline.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "via-sandbox-loop"
    assert pool_seen == [fake_backend], "loop didn't receive the SandboxBackend"


def test_web_search_dispatches_through_web_search_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tools: [{"type": "otari_web_search"}]`` routes through ``WebSearchBackend``."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")

    pool_seen: list[Any] = []

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        pool_seen.append(pool)
        return _text_response("via-web-search-loop")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    fake_builder_result = AsyncMock(
        __aenter__=AsyncMock(return_value=fake_backend),
        __aexit__=AsyncMock(return_value=None),
    )

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline._build_web_search_backend", return_value=fake_builder_result),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_web_search"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "via-web-search-loop"
    assert pool_seen == [fake_backend], "loop didn't receive the WebSearchBackend"


# ---------- provider-named keyword passthrough ----------


@pytest.mark.parametrize("tool_type", ["code_execution", "code_interpreter", "code_execution_20250825"])
def test_provider_code_execution_passes_through_to_upstream(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tool_type: str,
) -> None:
    """Provider-named code-execution keywords are NOT intercepted. They stay in
    ``tools[]`` and reach ``amessages`` so Anthropic runs the code in its own
    native sandbox — even with no gateway sandbox configured.
    """
    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("ok")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "tools": [{"type": tool_type}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    forwarded = captured.get("tools")
    assert forwarded is not None, "provider-named tool was dropped instead of forwarded"
    assert {t["type"] for t in forwarded} == {tool_type}


@pytest.mark.parametrize("tool_type", ["web_search", "web_search_20250305"])
def test_provider_web_search_passes_through_to_upstream(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tool_type: str,
) -> None:
    """Provider-named web_search keywords pass through to Anthropic even when
    no gateway web_search backend is configured."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("ok")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"type": tool_type}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    forwarded = captured.get("tools")
    assert forwarded is not None, "provider-named tool was dropped instead of forwarded"
    assert {t["type"] for t in forwarded} == {tool_type}


# ---------- validation errors (Anthropic-shaped 400) ----------


def _assert_anthropic_error(body: dict[str, Any], *, error_type: str, message_substr: str) -> None:
    assert "detail" in body, body
    detail = body["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == error_type
    assert message_substr in detail["error"]["message"]


def test_code_execution_without_sandbox_env_returns_400_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTARI_SANDBOX_URL", raising=False)
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"type": "otari_code_execution"}],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    _assert_anthropic_error(resp.json(), error_type="invalid_request_error", message_substr="OTARI_SANDBOX_URL")


def test_code_execution_combined_with_mcp_servers_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"type": "otari_code_execution"}],
            "mcp_servers": [{"name": "x", "url": "http://127.0.0.1:9999/mcp"}],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    _assert_anthropic_error(
        resp.json(),
        error_type="invalid_request_error",
        message_substr="otari_code_execution and mcp_servers cannot be combined",
    )


@pytest.mark.parametrize("native_type", ["code_execution", "code_interpreter", "code_execution_20250825"])
def test_code_execution_combined_with_a_provider_native_tool_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    native_type: str,
) -> None:
    """Two sandboxes in one request have no single home for the caller's state."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"type": "otari_code_execution"}, {"type": native_type}],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    _assert_anthropic_error(
        resp.json(),
        error_type="invalid_request_error",
        message_substr="cannot be combined with a provider-native code-execution tool",
    )


def test_code_execution_allows_an_unrelated_caller_function(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conflict keys on the tool ``type``. A caller's own function named
    ``code_execution`` is theirs to dispatch and never claimed by the gateway."""
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        return _text_response()

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes._pipeline.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "tools": [
                    {"type": "otari_code_execution"},
                    {"type": "function", "function": {"name": "code_execution"}},
                ],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text


def test_web_search_combined_with_sandbox_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [
                {"type": "otari_code_execution"},
                {"type": "otari_web_search"},
            ],
        },
        headers=api_key_header,
    )
    assert resp.status_code == 400
    _assert_anthropic_error(
        resp.json(),
        error_type="invalid_request_error",
        message_substr="otari_web_search cannot be combined",
    )


# ---------- gateway-side runtime errors ----------


def test_max_tool_iterations_exceeded_returns_422_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gateway's own iteration cap is distinct from a provider outage —
    422 with the Anthropic error envelope lets callers tell them apart.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.mcp_loop_messages import MaxToolIterationsExceeded

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        raise MaxToolIterationsExceeded(f"Exceeded max_tool_iterations={max_iterations}")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch(
            "gateway.api.routes._pipeline.SandboxBackend",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=fake_backend),
                __aexit__=AsyncMock(return_value=None),
            ),
        ),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_code_execution"}],
                "max_tool_iterations": 1,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 422
    _assert_anthropic_error(
        resp.json(),
        error_type="invalid_request_error",
        message_substr="max_tool_iterations",
    )


def test_sandbox_unreachable_returns_502_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.sandbox_backend import SandboxNotReachableError

    with patch(
        "gateway.api.routes._pipeline.SandboxBackend",
        return_value=AsyncMock(__aenter__=AsyncMock(side_effect=SandboxNotReachableError("boom"))),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 502
    _assert_anthropic_error(resp.json(), error_type="api_error", message_substr="sandbox unreachable")


# ---------- streaming dispatch ----------


def _stream_message_start(
    *,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> MessageStartEvent:
    return MessageStartEvent(
        type="message_start",
        message=cast(
            Any,
            _text_response(
                "",
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            ),
        ),
    )


def _stream_text_delta(text: str) -> ContentBlockDeltaEvent:
    return ContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=cast(Any, TextDelta(type="text_delta", text=text)),
    )


def _stream_message_delta() -> MessageDeltaEvent:
    return MessageDeltaEvent(
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


def _stream_message_stop() -> MessageStopEvent:
    return MessageStopEvent(type="message_stop")


async def _stream_iter(*events: MessageStreamEvent) -> AsyncIterator[MessageStreamEvent]:
    for event in events:
        yield event


@pytest.mark.parametrize(
    "activity_blocks",
    [
        [
            {
                "type": "mcp_tool_use",
                "id": f"{MCP_ACTIVITY_ID_PREFIX}echoed",
                "name": "lookup",
                "server_name": "fixture",
                "input": {"id": 755},
            },
            {
                "type": "mcp_tool_result",
                "tool_use_id": f"{MCP_ACTIVITY_ID_PREFIX}echoed",
                "content": "large-result" * 10_000,
                "is_error": False,
            },
        ],
        [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_echoed",
                "name": "web_search",
                "input": {"query": "otari"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_echoed",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.test",
                        "title": "large-result" * 10_000,
                        "encrypted_content": "",
                    }
                ],
            },
        ],
    ],
    ids=["mcp", "web-search"],
)
def test_echoed_gateway_activity_is_removed_before_prompt_estimation(
    client: TestClient,
    api_key_header: dict[str, str],
    activity_blocks: list[dict[str, Any]],
) -> None:
    """Gateway-owned result content must not consume budget headroom before being stripped."""
    messages = [
        {"role": "user", "content": "first turn"},
        {
            "role": "assistant",
            "content": [*activity_blocks, {"type": "text", "text": "prior answer"}],
        },
        {"role": "user", "content": "continue"},
    ]
    sanitized = [
        messages[0],
        {"role": "assistant", "content": [{"type": "text", "text": "prior answer"}]},
        messages[2],
    ]
    captured: dict[str, Any] = {}

    async def fake_normalize_messages(input_messages: Any, **kwargs: Any) -> Any:
        captured["normalized_messages"] = input_messages
        return input_messages, SimpleNamespace(vision_usage=lambda: None)

    async def fake_resolve_request_context(**kwargs: Any) -> Any:
        captured.update(kwargs)
        await kwargs["normalize_messages"]("user", None, "model", None, None)
        raise HTTPException(status_code=418, detail="stop after admission inputs")

    with (
        patch(
            "gateway.api.routes.messages.resolve_request_context",
            new=fake_resolve_request_context,
        ),
        patch(
            "gateway.api.routes.messages.normalize_request_messages",
            new=fake_normalize_messages,
        ),
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": messages,
                "max_tokens": 100,
            },
            headers=api_key_header,
        )

    assert response.status_code == 418
    assert captured["estimate_prompt_chars"] == len(str(sanitized))
    assert captured["normalized_messages"] == sanitized


def test_stream_no_tools_returns_sse_response(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``stream: true`` with no gateway tools wraps the upstream stream in an
    SSE response. The route should NOT call the tool loop.
    """

    async def fake_amessages(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        return _stream_iter(_stream_message_start(), _stream_message_stop())

    tool_loop_called = False

    async def fake_loop_stream(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        nonlocal tool_loop_called
        tool_loop_called = True
        # Async-generator shape to match anthropic_tool_loop_stream.
        return
        yield  # noqa: F811 — needed to classify this as an async generator

    with (
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert tool_loop_called is False


def test_stream_cache_control_and_usage_round_trip(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Automatic cache control reaches Anthropic and usage remains in SSE."""
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        captured.update(kwargs)
        return _stream_iter(
            _stream_message_start(cache_creation_input_tokens=13, cache_read_input_tokens=8),
            _stream_message_stop(),
        )

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
                "cache_control": _AUTOMATIC_CACHE_CONTROL,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["cache_control"] == _AUTOMATIC_CACHE_CONTROL
    payloads = [json.loads(line.removeprefix("data: ")) for line in resp.text.splitlines() if line.startswith("data: ")]
    message_start = next(payload for payload in payloads if payload["type"] == "message_start")
    usage = message_start["message"]["usage"]
    assert usage["cache_creation_input_tokens"] == 13
    assert usage["cache_read_input_tokens"] == 8


def test_context_management_stream_contract(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """Compaction events and telemetry survive the streaming route unchanged."""
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        captured.update(kwargs)
        return _stream_iter(
            MessageStartEvent.model_validate(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_compaction",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-opus-5",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                }
            ),
            ContentBlockStartEvent.model_validate(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "compaction", "content": None},
                }
            ),
            ContentBlockDeltaEvent.model_validate(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "compaction_delta", "content": "Conversation summary"},
                }
            ),
            MessageDeltaEvent.model_validate(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "compaction", "stop_sequence": None},
                    "usage": {
                        "output_tokens": 5,
                        "iterations": [
                            {
                                "type": "compaction",
                                "input_tokens": 100,
                                "output_tokens": 20,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                            },
                            {
                                "type": "message",
                                "model": "claude-opus-5",
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0,
                            },
                        ],
                    },
                    "context_management": {
                        "applied_edits": [
                            {
                                "type": "clear_thinking_20251015",
                                "cleared_input_tokens": 21,
                                "cleared_thinking_turns": 1,
                            }
                        ]
                    },
                }
            ),
            _stream_message_stop(),
        )

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-opus-5",
                "messages": [{"role": "user", "content": "Summarize when needed"}],
                "max_tokens": 100,
                "stream": True,
                "context_management": _CONTEXT_MANAGEMENT,
                "betas": _BETAS,
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert captured["context_management"] == _CONTEXT_MANAGEMENT
    assert captured["betas"] == _BETAS
    payloads = [json.loads(line.removeprefix("data: ")) for line in resp.text.splitlines() if line.startswith("data: ")]
    compaction_start = next(payload for payload in payloads if payload["type"] == "content_block_start")
    assert compaction_start["content_block"] == {"type": "compaction"}
    compaction_delta = next(payload for payload in payloads if payload["type"] == "content_block_delta")
    assert compaction_delta["delta"] == {"type": "compaction_delta", "content": "Conversation summary"}
    message_delta = next(payload for payload in payloads if payload["type"] == "message_delta")
    assert message_delta["context_management"]["applied_edits"][0]["cleared_input_tokens"] == 21
    assert message_delta["usage"]["iterations"][0]["type"] == "compaction"


def test_stream_mcp_servers_dispatches_through_tool_loop_stream(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    """``stream: true`` with ``mcp_servers`` routes through
    ``anthropic_tool_loop_stream`` rather than calling ``amessages`` directly.
    """
    seen: dict[str, Any] = {}

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> AsyncIterator[MessageStreamEvent]:
        seen["pool"] = pool
        seen["max_iterations"] = max_iterations
        yield _stream_message_stop()

    plain_amessages_called = False

    async def fake_amessages(**_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        nonlocal plain_amessages_called
        plain_amessages_called = True
        return _stream_iter()

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes.messages.amessages", new=fake_amessages),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=AsyncMock(purpose_hints=lambda: [])),
        ),
        patch("gateway.services.mcp_client.MCPClientPool.__aexit__", new=AsyncMock(return_value=None)),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "stream": True,
                "mcp_servers": [{"name": "test", "url": "http://127.0.0.1:9999/mcp"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert seen.get("pool") is not None, "anthropic_tool_loop_stream was not invoked"
    assert plain_amessages_called is False


@pytest.mark.parametrize(
    ("beta_header", "expected_block_types"),
    [
        (None, ["text"]),
        (MCP_CLIENT_BETA, ["mcp_tool_use", "mcp_tool_result", "text"]),
    ],
)
def test_stream_mcp_activity_requires_beta(
    client: TestClient,
    api_key_header: dict[str, str],
    beta_header: str | None,
    expected_block_types: list[str],
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    provider_calls: list[dict[str, Any]] = []
    streams = iter(
        [
            _stream_iter(
                _stream_message_start(),
                ContentBlockStartEvent.model_validate(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_internal",
                            "name": "lookup",
                            "input": {},
                        },
                    }
                ),
                ContentBlockDeltaEvent.model_validate(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"issue": 755}',
                        },
                    }
                ),
                ContentBlockStopEvent(type="content_block_stop", index=0),
                MessageDeltaEvent.model_validate(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    }
                ),
                _stream_message_stop(),
            ),
            _stream_iter(
                _stream_message_start(),
                ContentBlockStartEvent.model_validate(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                ),
                _stream_text_delta("done"),
                ContentBlockStopEvent(type="content_block_stop", index=0),
                _stream_message_delta(),
                _stream_message_stop(),
            ),
        ]
    )

    async def fake_amessages(**kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        provider_calls.append(kwargs)
        return next(streams)

    class ActivityPool:
        @property
        def openai_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "function": {"name": "lookup", "description": "", "parameters": {}},
                }
            ]

        def purpose_hints(self) -> list[tuple[str, str]]:
            return []

        def owns_tool(self, name: str) -> bool:
            return name == "lookup"

        def server_name_for_tool(self, name: str) -> str | None:
            return "fixture" if self.owns_tool(name) else None

        async def call_tool_outcome(self, name: str, arguments: dict[str, Any]) -> MCPToolCallOutcome:
            assert name == "lookup"
            assert arguments == {"issue": 755}
            calls.append((name, arguments))
            return MCPToolCallOutcome(
                content="issue result",
                activity_content="issue result",
                is_error=False,
            )

    with (
        patch("gateway.services.mcp_loop_messages.amessages", new=fake_amessages),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aenter__",
            new=AsyncMock(return_value=ActivityPool()),
        ),
        patch(
            "gateway.services.mcp_client.MCPClientPool.__aexit__",
            new=AsyncMock(return_value=None),
        ),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "look it up"}],
                "max_tokens": 100,
                "stream": True,
                "mcp_servers": [{"name": "fixture", "url": "http://127.0.0.1:9999/mcp"}],
            },
            headers={
                **api_key_header,
                **({"anthropic-beta": beta_header} if beta_header is not None else {}),
            },
        )

    assert resp.status_code == 200, resp.text
    payloads = [json.loads(line.removeprefix("data: ")) for line in resp.text.splitlines() if line.startswith("data: ")]
    blocks = [payload["content_block"] for payload in payloads if payload.get("type") == "content_block_start"]
    assert [block["type"] for block in blocks] == expected_block_types
    assert calls == [("lookup", {"issue": 755})]
    assert all(MCP_CLIENT_BETA not in call.get("betas", []) for call in provider_calls)
    if beta_header is not None:
        assert blocks[0]["name"] == "lookup"
        assert blocks[0]["server_name"] == "fixture"
        assert blocks[0]["input"] == {"issue": 755}
        assert blocks[1] == {
            "type": "mcp_tool_result",
            "tool_use_id": blocks[0]["id"],
            "content": "issue result",
            "is_error": False,
        }


def test_stream_code_execution_dispatches_through_sandbox(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stream: true`` with ``otari_code_execution`` opens the sandbox backend
    and feeds it to ``anthropic_tool_loop_stream``.
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    pool_seen: list[Any] = []

    async def fake_loop_stream(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> AsyncIterator[MessageStreamEvent]:
        pool_seen.append(pool)
        yield _stream_message_stop()

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []
    fake_backend.__aenter__ = AsyncMock(return_value=fake_backend)
    fake_backend.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop_stream", new=fake_loop_stream),
        patch("gateway.api.routes._pipeline.SandboxBackend", return_value=fake_backend),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "compute"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert pool_seen == [fake_backend], "tool loop didn't receive the SandboxBackend"


def test_stream_sandbox_unreachable_returns_502_anthropic_body(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the eager-open error mapping bug: when the
    streaming sandbox eager-open fails, the route must return a 502 with the
    Anthropic error envelope rather than a 500 (which is what would happen
    if the streaming dispatch wasn't wrapped in the error-mapping
    try/except).
    """
    monkeypatch.setenv("OTARI_SANDBOX_URL", "http://127.0.0.1:9999/sandbox")

    from gateway.services.sandbox_backend import SandboxNotReachableError

    with patch(
        "gateway.api.routes._pipeline.SandboxBackend",
        return_value=AsyncMock(__aenter__=AsyncMock(side_effect=SandboxNotReachableError("boom"))),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "go"}],
                "max_tokens": 100,
                "stream": True,
                "tools": [{"type": "otari_code_execution"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 502
    _assert_anthropic_error(resp.json(), error_type="api_error", message_substr="sandbox unreachable")


# ---------- web-search interception (opt-in) ----------


@pytest.mark.parametrize(
    "tool_entry",
    [
        {"type": "web_search"},
        {"type": "web_search_20250305"},
        # The exact shape Claude Code sends.
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
    ],
)
def test_intercept_routes_provider_keywords_to_the_gateway_backend(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tool_entry: dict[str, Any],
) -> None:
    """With interception on, a provider-named declaration reaches WebSearchBackend
    instead of being forwarded, so a client that cannot say ``otari_web_search``
    still gets the gateway's search."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")

    pool_seen: list[Any] = []

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        pool_seen.append(pool)
        return _text_response("via-web-search-loop")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []
    fake_builder_result = AsyncMock(
        __aenter__=AsyncMock(return_value=fake_backend),
        __aexit__=AsyncMock(return_value=None),
    )

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline._build_web_search_backend", return_value=fake_builder_result),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [tool_entry],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert pool_seen == [fake_backend], "declaration was forwarded instead of intercepted"


def test_intercept_emits_native_blocks_only_for_a_native_declaration(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dated keyword opts into native server-tool blocks; the bare short form and
    the canonical otari type do not."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")

    seen: list[bool] = []

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        seen.append(emit_native_web_search)
        return _text_response("ok")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []
    fake_builder_result = AsyncMock(
        __aenter__=AsyncMock(return_value=fake_backend),
        __aexit__=AsyncMock(return_value=None),
    )

    def post(tool_entry: dict[str, Any]) -> None:
        with (
            patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
            patch("gateway.api.routes._pipeline._build_web_search_backend", return_value=fake_builder_result),
        ):
            resp = client.post(
                "/v1/messages",
                json={
                    "model": "anthropic:claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "search"}],
                    "max_tokens": 100,
                    "tools": [tool_entry],
                },
                headers=api_key_header,
            )
        assert resp.status_code == 200, resp.text

    post({"type": "web_search_20250305"})
    post({"type": "web_search"})
    post({"type": "otari_web_search"})

    assert seen == [True, False, False]


def test_intercept_off_still_forwards_provider_keywords(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default: a configured backend alone does not change who runs the search."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.delenv("OTARI_WEB_SEARCH_INTERCEPT", raising=False)
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("ok")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"type": "web_search_20250305"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert [tool["type"] for tool in captured.get("tools") or []] == ["web_search_20250305"]


def test_intercept_without_a_backend_forwards_rather_than_400s(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interception with nothing to intercept *to* must not turn a request the
    provider would have served into a 400."""
    monkeypatch.delenv("OTARI_WEB_SEARCH_URL", raising=False)
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("ok")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"type": "web_search_20250305"}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert [tool["type"] for tool in captured.get("tools") or []] == ["web_search_20250305"]


def test_intercept_never_claims_a_caller_function_named_web_search(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller's own tool stays theirs to dispatch, so it must reach the provider."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")
    captured: dict[str, Any] = {}

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        captured.update(kwargs)
        return _text_response("ok")

    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"name": "web_search", "input_schema": {"type": "object", "properties": {}}}],
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert [tool.get("name") for tool in captured.get("tools") or []] == ["web_search"]


def test_intercept_retargets_a_forced_tool_choice(
    client: TestClient,
    api_key_header: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool_choice forcing the caller's own name must be repointed at the
    backend's tool, or the provider is asked to force a tool it never received."""
    monkeypatch.setenv("OTARI_WEB_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setenv("OTARI_WEB_SEARCH_INTERCEPT", "true")

    seen: list[Any] = []

    async def fake_loop(
        *, completion_kwargs: Any, pool: Any, max_iterations: int, emit_native_web_search: bool = False
    ) -> MessageResponse:
        seen.append(completion_kwargs.get("tool_choice"))
        return _text_response("ok")

    fake_backend = AsyncMock()
    fake_backend.purpose_hints = lambda: []
    fake_builder_result = AsyncMock(
        __aenter__=AsyncMock(return_value=fake_backend),
        __aexit__=AsyncMock(return_value=None),
    )

    with (
        patch("gateway.api.routes.messages.anthropic_tool_loop", new=fake_loop),
        patch("gateway.api.routes._pipeline._build_web_search_backend", return_value=fake_builder_result),
    ):
        resp = client.post(
            "/v1/messages",
            json={
                "model": "anthropic:claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "search"}],
                "max_tokens": 100,
                "tools": [{"type": "web_search_20250305", "name": "search_the_web"}],
                "tool_choice": {"type": "tool", "name": "search_the_web"},
            },
            headers=api_key_header,
        )

    assert resp.status_code == 200, resp.text
    assert seen == [{"type": "tool", "name": "web_search"}]
