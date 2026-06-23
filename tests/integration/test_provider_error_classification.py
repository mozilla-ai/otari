"""Integration tests: classified provider failures surface per format.

Each adapter (chat, responses, messages) turns a status-carrying upstream
exception into a safe, specific HTTP error in its own wire envelope, without
leaking the raw provider message.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.api.routes._pipeline import (
    PROVIDER_BAD_REQUEST_DETAIL,
    PROVIDER_CREDENTIALS_DETAIL,
    PROVIDER_ERROR_DETAIL,
    PROVIDER_MODEL_NOT_FOUND_DETAIL,
    PROVIDER_RATE_LIMITED_DETAIL,
)

_RAW = "raw upstream message SECRET-9f3a"


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(_RAW)
        self.status_code = status_code


# (upstream status, mapped HTTP status, mapped detail). 500 and the bare case
# fall through to the generic provider error each format already returned.
_CASES = [
    (400, 400, PROVIDER_BAD_REQUEST_DETAIL),
    (422, 400, PROVIDER_BAD_REQUEST_DETAIL),
    (404, 404, PROVIDER_MODEL_NOT_FOUND_DETAIL),
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
    assert "SECRET" not in response.text


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
    assert "SECRET" not in response.text


# (upstream status, mapped HTTP status, mapped detail, anthropic error.type)
_MESSAGES_CASES = [
    (400, 400, PROVIDER_BAD_REQUEST_DETAIL, "invalid_request_error"),
    (404, 404, PROVIDER_MODEL_NOT_FOUND_DETAIL, "not_found_error"),
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
    assert "SECRET" not in response.text
