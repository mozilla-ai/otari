"""Integration tests: classified provider failures surface per format.

Each adapter (chat, responses, messages) turns a status-carrying upstream
exception into a specific HTTP error in its own wire envelope. A rejection of
the caller's request carries the provider's own message; a failure that is the
gateway's own fault keeps a fixed detail and never echoes upstream text.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.exceptions import UnsupportedParameterError
from fastapi.testclient import TestClient

from gateway.api.routes._pipeline import (
    PROVIDER_CREDENTIALS_DETAIL,
    PROVIDER_ERROR_DETAIL,
    PROVIDER_RATE_LIMITED_DETAIL,
)

_RAW = "raw upstream message SECRET-9f3a"

# The upstream OpenAI rejection for function tools + a non-'none' reasoning_effort.
_REASONING_TOOLS_MSG = (
    "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
    "/v1/chat/completions. To use function tools, use /v1/responses or set "
    "reasoning_effort to 'none'."
)


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(_RAW)
        self.status_code = status_code


class _ParamError(Exception):
    """Raw OpenAI-style bad-request error carrying ``param`` + ``message``."""

    def __init__(self, status_code: int, param: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.param = param
        self.message = message


# (upstream status, mapped HTTP status, mapped detail). 500 and the bare case
# fall through to the generic provider error each format already returned.
# 400/422/404 are the caller's request to fix, so they carry the upstream
# message; the rest are the gateway's own fault and keep a fixed string.
_CASES = [
    (400, 400, _RAW),
    (422, 400, _RAW),
    (404, 404, _RAW),
    (401, 502, PROVIDER_CREDENTIALS_DETAIL),
    (403, 502, PROVIDER_CREDENTIALS_DETAIL),
    (429, 429, PROVIDER_RATE_LIMITED_DETAIL),
]


@pytest.mark.parametrize(("upstream", "expected_status", "expected_detail"), _CASES)
def test_chat_classifies_provider_error(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    upstream: int,
    expected_status: int,
    expected_detail: str,
) -> None:
    with patch(
        "gateway.api.routes.chat.acompletion",
        new_callable=AsyncMock,
        side_effect=_StatusError(upstream),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "openai:nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    if expected_detail != _RAW:
        assert "SECRET" not in response.text


def test_chat_surfaces_unsupported_prompt_cache_key_as_client_error(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A final provider that cannot honor the key returns an actionable 400."""
    with patch(
        "gateway.api.routes.chat.acompletion",
        new_callable=AsyncMock,
        side_effect=UnsupportedParameterError("prompt_cache_key", "anthropic"),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "anthropic:claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
                "prompt_cache_key": "tenant-session-123",
            },
            headers=api_key_header,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "'prompt_cache_key' is not supported for anthropic"


def test_chat_unknown_status_stays_generic_502(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    with patch(
        "gateway.api.routes.chat.acompletion",
        new_callable=AsyncMock,
        side_effect=_StatusError(500),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "openai:nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == 502
    assert response.json()["detail"] == PROVIDER_ERROR_DETAIL
    assert "SECRET" not in response.text


def test_chat_surfaces_reasoning_effort_tools_conflict(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """#331's case, with no probe behind it. OpenAI's message already names the
    remedy, so it reaches the caller intact through the chat adapter."""
    with patch(
        "gateway.api.routes.chat.acompletion",
        new_callable=AsyncMock,
        side_effect=_ParamError(400, "reasoning_effort", _REASONING_TOOLS_MSG),
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "openai:gpt-5.6-sol", "messages": [{"role": "user", "content": "Hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == _REASONING_TOOLS_MSG
    # The model name is signal, not a leak: it is what tells the caller which
    # model in their routing policy rejected the request.
    assert "gpt-5.6-sol" in response.text


def test_responses_surfaces_reasoning_effort_tools_conflict(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    """The shared classifier passes the same message through the responses
    adapter."""
    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        side_effect=_ParamError(400, "reasoning_effort", _REASONING_TOOLS_MSG),
    ):
        response = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert response.status_code == 400
    assert response.json()["detail"] == _REASONING_TOOLS_MSG
    # The model name is signal, not a leak: it is what tells the caller which
    # model in their routing policy rejected the request.
    assert "gpt-5.6-sol" in response.text


def test_messages_surfaces_reasoning_effort_tools_conflict(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """The messages (Anthropic) envelope carries the upstream message as an
    invalid_request_error."""
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch(
        "gateway.api.routes.messages.amessages",
        new_callable=AsyncMock,
        side_effect=_ParamError(400, "reasoning_effort", _REASONING_TOOLS_MSG),
    ):
        response = client.post("/v1/messages", json=messages_request_body, headers=master_key_header)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == "invalid_request_error"
    assert detail["error"]["message"] == _REASONING_TOOLS_MSG
    assert "gpt-5.6-sol" in response.text


@pytest.mark.parametrize(("upstream", "expected_status", "expected_detail"), _CASES)
def test_responses_classifies_provider_error(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
    upstream: int,
    expected_status: int,
    expected_detail: str,
) -> None:
    with patch(
        "gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        side_effect=_StatusError(upstream),
    ):
        response = client.post("/v1/responses", json=responses_request_body, headers=master_key_header)

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    if expected_detail != _RAW:
        assert "SECRET" not in response.text


# (upstream status, mapped HTTP status, mapped detail, anthropic error.type)
_MESSAGES_CASES = [
    (400, 400, _RAW, "invalid_request_error"),
    (404, 404, _RAW, "not_found_error"),
    (401, 502, PROVIDER_CREDENTIALS_DETAIL, "api_error"),
    (429, 429, PROVIDER_RATE_LIMITED_DETAIL, "rate_limit_error"),
]


@pytest.mark.parametrize(("upstream", "expected_status", "expected_detail", "expected_type"), _MESSAGES_CASES)
def test_messages_classifies_provider_error(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
    upstream: int,
    expected_status: int,
    expected_detail: str,
    expected_type: str,
) -> None:
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch(
        "gateway.api.routes.messages.amessages",
        new_callable=AsyncMock,
        side_effect=_StatusError(upstream),
    ):
        response = client.post("/v1/messages", json=messages_request_body, headers=master_key_header)

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == expected_type
    assert detail["error"]["message"] == expected_detail
    if expected_detail != _RAW:
        assert "SECRET" not in response.text
