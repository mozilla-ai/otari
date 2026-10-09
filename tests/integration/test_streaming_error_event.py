"""Tests for error handling on streaming requests."""

from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from gateway.core.config import API_ROOT


def test_streaming_creation_error_returns_http_error(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that a streaming request whose stream cannot be opened returns an HTTP error.

    When the stream cannot be created, the gateway returns a proper HTTP error
    response rather than starting a stream and emitting an SSE error event.
    """
    with patch("gateway.api.routes.chat.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
        response = client.post(
            f"{API_ROOT}/chat/completions",
            json={
                "model": "openai:totally-invalid-model-xyz",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
            headers=api_key_header,
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider error"}
