"""Tests for error message sanitization.

The gateway does not sanitize provider errors uniformly. A rejection of the
caller's own request carries the provider's message, because otari cannot
describe what it did not diagnose. A failure that is the gateway's own keeps a
fixed detail: it names the operator's credentials or topology, and the caller
has no remedy to apply either way.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


def test_provider_error_does_not_leak_details(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that provider errors return a generic message without internal details."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers=api_key_header,
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail == "LLM provider error"


class _UpstreamError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _post_chat(client: TestClient, headers: dict[str, str], exc: Exception) -> Any:
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, side_effect=exc):
        return client.post(
            "/v1/chat/completions",
            json={"model": "openai:gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers=headers,
        )


def test_caller_fault_error_carries_the_provider_message(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """The case the fixed strings used to bury: the caller can only fix a bad
    request if they are told what was wrong with it."""
    response = _post_chat(client, api_key_header, _UpstreamError(400, "max_tokens must be <= 8192, got 200000"))
    assert response.status_code == 400
    assert response.json()["detail"] == "max_tokens must be <= 8192, got 200000"


def test_caller_fault_error_still_redacts_credentials(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Passing the provider's message through is not a licence to pass its
    credentials through: the gateway calls providers with the operator's key."""
    response = _post_chat(
        client,
        api_key_header,
        _UpstreamError(400, "Bad request for key sk-proj-AAAAAAAAAAAAAAAAAAAA via http://10.0.0.4:8000/v1"),
    )
    assert response.status_code == 400
    assert "sk-proj-AAAAAAAAAAAAAAAAAAAA" not in response.text
    assert "10.0.0.4" not in response.text


def test_gateway_fault_error_keeps_a_fixed_detail(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A provider rejecting the gateway's credentials is the operator's problem.
    It surfaces as a 502 with a fixed detail and never echoes the upstream text,
    which is where a provider names the account it rejected."""
    response = _post_chat(client, api_key_header, _UpstreamError(401, "Incorrect API key sk-proj-BBBBBBBBBBBB"))
    assert response.status_code == 502
    assert response.json()["detail"] == "The provider rejected the gateway's credentials"
    assert "sk-proj-BBBBBBBBBBBB" not in response.text


def test_health_readiness_does_not_leak_db_details(client: TestClient) -> None:
    """Test that readiness endpoint doesn't leak database details on success."""
    response = client.get("/health/readiness")
    assert response.status_code == 200
    data = response.json()
    # Should not contain "error" key
    assert "error" not in data
