"""Integration tests: classified provider failures surface per format.

Each adapter (chat, responses, messages) turns a status-carrying upstream
exception into a specific HTTP error in its own wire envelope. A rejection of
the caller's request carries the provider's own message; a failure that is the
gateway's own fault keeps a fixed detail and never echoes upstream text.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import openai
import pytest
from any_llm.exceptions import UnsupportedParameterError
from cohere.errors import BadRequestError as CohereBadRequestError
from fastapi.testclient import TestClient

from gateway.api.routes._pipeline import (
    PROVIDER_CREDENTIALS_DETAIL,
    PROVIDER_RATE_LIMITED_DETAIL,
)
from gateway.core.config import API_ROOT

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
# 400/422/404 are the caller's request to fix and 429 is theirs to act on, so
# those carry the upstream message; the rest are the gateway's own fault and
# keep a fixed string.
_CASES = [
    (400, 400, _RAW),
    (422, 400, _RAW),
    (404, 404, _RAW),
    (429, 429, _RAW),
    (401, 424, PROVIDER_CREDENTIALS_DETAIL),
    (403, 424, PROVIDER_CREDENTIALS_DETAIL),
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
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    if expected_detail != _RAW:
        assert "SECRET" not in response.text


def test_chat_rate_limit_falls_back_when_the_provider_said_nothing(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A 429 whose body carried no explanation still gets a usable detail."""

    class _Silent(Exception):
        def __init__(self) -> None:
            super().__init__("")
            self.status_code = 429

    with patch(
        "gateway.api.routes.chat.acompletion",
        new_callable=AsyncMock,
        side_effect=_Silent(),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == 429
    assert response.json()["detail"] == PROVIDER_RATE_LIMITED_DETAIL


def test_chat_forwards_the_upstream_retry_after(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """The one upstream header a rate-limited caller can act on reaches the
    wire, so a client can honor the provider's own backoff window instead of
    guessing one."""

    class _RateLimited(Exception):
        def __init__(self) -> None:
            super().__init__("Quota exceeded.")
            self.status_code = 429
            self.response = httpx.Response(429, headers={"Retry-After": "34"})

    with patch(
        "gateway.api.routes.chat.acompletion",
        new_callable=AsyncMock,
        side_effect=_RateLimited(),
    ):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "34"


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
            f"{API_ROOT}/chat/completions",
            json={
                "model": "anthropic:claude-3-5-sonnet",
                "messages": [{"role": "user", "content": "Hi"}],
                "prompt_cache_key": "tenant-session-123",
            },
            headers=api_key_header,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "'prompt_cache_key' is not supported for anthropic"


def test_chat_unknown_status_is_a_502_naming_only_the_upstream_status(
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
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:nonexistent-model-xyz", "messages": [{"role": "user", "content": "Hi"}]},
            headers=api_key_header,
        )

    assert response.status_code == 502
    assert response.json()["detail"] == "The provider failed the request with HTTP 500"
    assert "SECRET" not in response.text
    assert "x-should-retry" not in response.headers


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
            f"{API_ROOT}/chat/completions",
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
        response = client.post(f"{API_ROOT}/responses", json=responses_request_body, headers=master_key_header)

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
        response = client.post(f"{API_ROOT}/messages", json=messages_request_body, headers=master_key_header)

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
        response = client.post(f"{API_ROOT}/responses", json=responses_request_body, headers=master_key_header)

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    if expected_detail != _RAW:
        assert "SECRET" not in response.text


# (upstream status, mapped HTTP status, mapped detail, anthropic error.type)
_MESSAGES_CASES = [
    (400, 400, _RAW, "invalid_request_error"),
    (404, 404, _RAW, "not_found_error"),
    (429, 429, _RAW, "rate_limit_error"),
    (401, 424, PROVIDER_CREDENTIALS_DETAIL, "api_error"),
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
        response = client.post(f"{API_ROOT}/messages", json=messages_request_body, headers=master_key_header)

    assert response.status_code == expected_status
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == expected_type
    assert detail["error"]["message"] == expected_detail
    if expected_detail != _RAW:
        assert "SECRET" not in response.text


_TEMPERATURE_MSG = (
    "Unsupported value: 'temperature' does not support 0.5 with this model. Only the default (1) value is supported."
)


def _openai_bad_request() -> openai.BadRequestError:
    """OpenAI's 400 as its SDK raises it: the message is ``Error code: 400 - `` and a dict repr."""
    error = {
        "message": _TEMPERATURE_MSG,
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {error}}}",
        response=httpx.Response(400, request=request, json={"error": error}),
        body=error,
    )


def _cohere_bad_request() -> CohereBadRequestError:
    """Cohere's 400, whose string form leads with the response headers."""
    return CohereBadRequestError(
        body={"id": "c0ffee", "message": "invalid validation: invalid 'json_schema'"},
        headers={"x-debug-trace-id": "trace-SECRET-9f3a", "server": "envoy"},
    )


@pytest.mark.parametrize(
    ("upstream", "expected"),
    [(_openai_bad_request(), _TEMPERATURE_MSG), (_cohere_bad_request(), "invalid validation: invalid 'json_schema'")],
    ids=["openai", "cohere"],
)
def test_chat_answers_a_provider_400_with_the_providers_sentence(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    upstream: Exception,
    expected: str,
) -> None:
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, side_effect=upstream):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={"model": "openai:gpt-5.6-sol", "messages": [{"role": "user", "content": "Hi"}], "temperature": 0.5},
            headers=api_key_header,
        )

    assert response.status_code == 400
    assert response.json() == {"detail": expected}
    assert "SECRET" not in response.text


def test_responses_answers_a_provider_400_with_the_providers_sentence(
    client: TestClient,
    master_key_header: dict[str, str],
    responses_request_body: dict[str, Any],
) -> None:
    with patch("gateway.api.routes.responses.aresponses", new_callable=AsyncMock, side_effect=_openai_bad_request()):
        response = client.post(f"{API_ROOT}/responses", json=responses_request_body, headers=master_key_header)

    assert response.status_code == 400
    assert response.json() == {"detail": _TEMPERATURE_MSG}


def test_messages_answers_a_provider_400_with_the_providers_sentence(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: Field required"}}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    upstream = anthropic.BadRequestError(
        f"Error code: 400 - {body}", response=httpx.Response(400, request=request, json=body), body=body
    )
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch("gateway.api.routes.messages.amessages", new_callable=AsyncMock, side_effect=upstream):
        response = client.post(f"{API_ROOT}/messages", json=messages_request_body, headers=master_key_header)

    assert response.status_code == 400
    assert response.json()["detail"]["error"] == {
        "type": "invalid_request_error",
        "message": "max_tokens: Field required",
    }


def _dropped_connection() -> openai.APIConnectionError:
    request = httpx.Request("POST", "https://internal-proxy.example/v1/chat/completions")
    try:
        raise openai.APIConnectionError(request=request) from httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        )
    except openai.APIConnectionError as exc:
        return exc


@pytest.mark.parametrize(
    ("path", "target", "body"),
    [
        (
            "chat/completions",
            "gateway.api.routes.chat.acompletion",
            {"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "Hi"}]},
        ),
        ("embeddings", "gateway.api.routes.embeddings.aembedding", {"model": "openai:gpt-5-mini", "input": "Hi"}),
    ],
)
def test_a_dropped_connection_is_a_502_saying_the_provider_closed_it(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
    path: str,
    target: str,
    body: dict[str, Any],
) -> None:
    with patch(target, new_callable=AsyncMock, side_effect=_dropped_connection()):
        response = client.post(f"{API_ROOT}/{path}", json=body, headers=api_key_header)

    assert response.status_code == 502
    assert response.json()["detail"] == "The provider closed the connection before answering"
    assert "internal-proxy" not in response.text
    assert "x-should-retry" not in response.headers


_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_IMAGE_DETAIL_TAIL = (
    "does not accept image content. Remove the image parts, or send the request to a model that accepts images."
)


def _fireworks_image_rejection() -> openai.BadRequestError:
    """Fireworks' 400, through its OpenAI-compatible SDK, for an image sent to a text-only model."""
    error = {"message": "Unsupported ChatMessageContent type: image_url", "type": "invalid_request_error"}
    request = httpx.Request("POST", "https://api.fireworks.ai/inference/v1/chat/completions")
    return openai.BadRequestError(
        f"Error code: 400 - {{'error': {error}}}",
        response=httpx.Response(400, request=request, json={"error": error}),
        body=error,
    )


def _cohere_image_rejection() -> CohereBadRequestError:
    """Cohere's 400 for the same request, whose string form redacts to nothing."""
    return CohereBadRequestError(
        body={"id": "c0ffee", "message": "invalid request: image content is not supported for this model"},
        headers={"x-debug-trace-id": "trace-SECRET-9f3a", "server": "envoy"},
    )


@pytest.mark.parametrize("stream", [False, True], ids=["unstreamed", "streamed"])
@pytest.mark.parametrize(
    "upstream", [_fireworks_image_rejection(), _cohere_image_rejection()], ids=["fireworks", "cohere"]
)
def test_chat_answers_an_image_the_model_cannot_take_in_otaris_words(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    upstream: Exception,
    stream: bool,
) -> None:
    content = [
        {"type": "text", "text": "What is in this picture?"},
        {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
    ]
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, side_effect=upstream):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": "openai:text-only-model",
                "stream": stream,
                "messages": [{"role": "user", "content": content}],
            },
            headers=api_key_header,
        )

    detail = f"The model 'openai:text-only-model' {_IMAGE_DETAIL_TAIL}"
    assert response.status_code == 400
    assert response.json() == {"detail": detail, "code": "unsupported_content"}
    assert response.headers["Otari-Error-Code"] == "unsupported_content"
    assert "SECRET" not in response.text

    usage = client.get(f"{API_ROOT}/users/{api_key_obj['user_id']}/usage", headers=master_key_header).json()
    errors = [row for row in usage if row["status"] == "error"]
    assert len(errors) == 1
    assert errors[0]["error_message"] == f"The model serving this request {_IMAGE_DETAIL_TAIL}"
