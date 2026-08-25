"""Tests for the /v1/messages gateway endpoint."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.messages import (
    MessageResponse,
    MessageUsage,
    TextBlock,
    ToolUseBlock,
)
from fastapi.testclient import TestClient

from gateway.api.routes.messages import CountTokensRequest
from gateway.core.config import API_KEY_HEADER


def _make_message_response(**overrides: Any) -> MessageResponse:
    defaults: dict[str, Any] = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [TextBlock(type="text", text="Hello!")],
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "usage": MessageUsage(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return MessageResponse(**defaults)


def test_messages_endpoint_basic_completion(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Test basic non-streaming message completion."""
    mock_response = _make_message_response()
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hello!"
    assert data["stop_reason"] == "end_turn"


def test_messages_endpoint_requires_auth(
    client: TestClient,
    messages_request_body: dict[str, Any],
) -> None:
    """Test that the endpoint requires authentication."""
    response = client.post(
        "/v1/messages",
        json=messages_request_body,
    )
    assert response.status_code == 401


def test_messages_endpoint_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
    messages_request_body: dict[str, Any],
) -> None:
    """Test that master key auth requires user_id in metadata."""
    mock_response = _make_message_response()

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"]["type"] == "invalid_request_error"


def test_messages_endpoint_validation_error(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that validation errors are returned for missing fields."""
    response = client.post(
        "/v1/messages",
        json={"model": "anthropic:claude-3-5-sonnet"},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_messages_endpoint_with_tools(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test message completion with tools."""
    tool_use_response = _make_message_response(
        content=[
            ToolUseBlock(
                type="tool_use",
                id="toolu_123",
                name="get_weather",
                input={"city": "London"},
            )
        ],
        stop_reason="tool_use",
    )

    request_body = {
        "model": "anthropic:claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "metadata": {"user_id": "test-user"},
    }

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=tool_use_response):
        response = client.post(
            "/v1/messages",
            json=request_body,
            headers=master_key_header,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["stop_reason"] == "tool_use"
    assert data["content"][0]["type"] == "tool_use"
    assert data["content"][0]["name"] == "get_weather"


def test_messages_endpoint_provider_error_format(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Test that provider errors are returned in Anthropic error format."""
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch(
        "gateway.api.routes.messages.amessages",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Provider unavailable"),
    ):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == "api_error"


def test_messages_endpoint_rejected_param_is_a_400_naming_it(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """A param the resolved provider's SDK has no keyword for is the caller's to
    fix, so it is a 400 rather than the generic 500 this format falls back to.

    Pinned here as well as on the chat route because the classification is shared
    (``classify_provider_error``) while the envelope is not: this format renders
    the mapped status as an Anthropic ``invalid_request_error`` body, which the
    chat tests cannot show.
    """
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch(
        "gateway.api.routes.messages.amessages",
        new_callable=AsyncMock,
        side_effect=TypeError("AsyncMessages.create() got an unexpected keyword argument 'seed'"),
    ):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"]["type"] == "invalid_request_error"
    assert "'seed'" in detail["error"]["message"]
    assert "AsyncMessages" not in detail["error"]["message"]


def test_messages_endpoint_provider_error_streaming(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Provider errors raised before the stream starts return an HTTP error
    in Anthropic format, not an uncaught 500 leaking into the SSE channel.
    """
    messages_request_body["metadata"] = {"user_id": "test-user"}
    messages_request_body["stream"] = True

    with patch(
        "gateway.api.routes.messages.amessages",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Provider unavailable"),
    ):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == "api_error"


def test_messages_endpoint_bearer_auth(
    client: TestClient,
    api_key_obj: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Test authentication via standard Bearer token."""
    mock_response = _make_message_response()

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers={API_KEY_HEADER: f"Bearer {api_key_obj['key']}"},
        )

    assert response.status_code == 200


def _claude_code_request_body() -> dict[str, Any]:
    """A request shaped the way Claude Code sends it: a structured ``system``
    block with cache_control, tool definitions, and ``metadata.user_id``.
    """
    return {
        "model": "anthropic:claude-3-5-sonnet",
        "max_tokens": 1024,
        "system": [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": "List the files here."}],
        "tools": [
            {
                "name": "Bash",
                "description": "Run a shell command",
                "input_schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ],
        "metadata": {"user_id": "test-user"},
    }


def test_messages_endpoint_claude_code_shape(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A realistic Claude Code request (system blocks + tools) completes."""
    mock_response = _make_message_response()

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=_claude_code_request_body(),
            headers=master_key_header,
        )

    assert response.status_code == 200
    assert response.json()["type"] == "message"


# What Claude Code actually sends: client telemetry, not a user id, so it can
# never equal an Otari user id and no provisioned user makes it match.
_CLAUDE_CODE_TELEMETRY_USER_ID = '{"device_id":"9f2c","account_uuid":"","session_id":"7b1e-4a"}'


def test_messages_telemetry_user_id_rejected_by_default(
    client: TestClient,
    api_key_obj: dict[str, Any],
) -> None:
    """Strict default: a key naming a different 'user' is rejected (403)."""
    body = _claude_code_request_body()
    body["metadata"] = {"user_id": _CLAUDE_CODE_TELEMETRY_USER_ID}

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=_make_message_response()):
        response = client.post(
            "/v1/messages",
            json=body,
            headers={API_KEY_HEADER: f"Bearer {api_key_obj['key']}"},
        )

    assert response.status_code == 403


def test_messages_telemetry_user_id_allowed_by_per_key_override(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A key overriding reject_user_mismatch to false completes while the
    deployment stays strict.

    The deployment-wide reject_user_mismatch is left at its default, so this is the
    per-key exception rather than a deployment-wide relaxation (issue #493).
    """
    created = client.post(
        "/v1/keys",
        json={"key_name": "claude-code", "user_id": "cc-user", "reject_user_mismatch": False},
        headers=master_key_header,
    )
    assert created.status_code == 200
    lenient_key = created.json()["key"]

    body = _claude_code_request_body()
    body["metadata"] = {"user_id": _CLAUDE_CODE_TELEMETRY_USER_ID}

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=_make_message_response()):
        response = client.post(
            "/v1/messages",
            json=body,
            headers={API_KEY_HEADER: f"Bearer {lenient_key}"},
        )

    assert response.status_code == 200, response.text

    # A second, unflagged key on the same deployment is still rejected.
    strict = client.post("/v1/keys", json={"key_name": "strict"}, headers=master_key_header)
    assert strict.status_code == 200, strict.text
    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=_make_message_response()):
        strict_response = client.post(
            "/v1/messages",
            json=body,
            headers={API_KEY_HEADER: f"Bearer {strict.json()['key']}"},
        )
    assert strict_response.status_code == 403


def test_messages_per_key_override_re_tightens_a_lenient_deployment(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override works in both directions: a key pinned strict is still rejected
    on a deployment that relaxed the check globally."""
    monkeypatch.setattr(test_config, "reject_user_mismatch", False)
    created = client.post(
        "/v1/keys",
        json={"key_name": "pinned-strict", "reject_user_mismatch": True},
        headers=master_key_header,
    )
    assert created.status_code == 200, created.text

    body = _claude_code_request_body()
    body["metadata"] = {"user_id": _CLAUDE_CODE_TELEMETRY_USER_ID}

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=_make_message_response()):
        response = client.post(
            "/v1/messages",
            json=body,
            headers={API_KEY_HEADER: f"Bearer {created.json()['key']}"},
        )

    assert response.status_code == 403


def test_count_tokens_basic(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """count_tokens returns a positive integer input-token estimate."""
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello, world!"}],
        },
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0


def test_count_tokens_accepts_context_management_and_betas(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Token-count requests accept the same context-management controls as messages."""
    assert "context_management" in CountTokensRequest.model_fields
    assert "betas" in CountTokensRequest.model_fields

    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-opus-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "context_management": {
                "edits": [
                    {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}}
                ]
            },
            "betas": ["compact-2026-01-12"],
        },
        headers=master_key_header,
    )

    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


def test_count_tokens_requires_auth(client: TestClient) -> None:
    """count_tokens rejects unauthenticated callers."""
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert response.status_code == 401


def test_count_tokens_validation_error(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Missing required fields produce a 422 before any counting."""
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "anthropic:claude-3-5-sonnet"},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_count_tokens_scales_with_input(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A longer prompt yields a strictly larger token estimate."""
    short = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hi"}],
        },
        headers=master_key_header,
    ).json()["input_tokens"]

    long = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello " * 200}],
        },
        headers=master_key_header,
    ).json()["input_tokens"]

    with_extras = client.post(
        "/v1/messages/count_tokens",
        json=_claude_code_request_body(),
        headers=master_key_header,
    ).json()["input_tokens"]

    assert long > short
    assert with_extras > 0


def test_count_tokens_bearer_auth(
    client: TestClient,
    api_key_obj: dict[str, Any],
) -> None:
    """count_tokens authenticates via the standard ``Authorization: Bearer``
    header, which is what Claude Code sends when ANTHROPIC_AUTH_TOKEN is set.
    """
    response = client.post(
        "/v1/messages/count_tokens",
        json={
            "model": "anthropic:claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers={"Authorization": f"Bearer {api_key_obj['key']}"},
    )
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0
