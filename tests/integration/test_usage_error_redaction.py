"""The error text a usage row stores gets the redaction the HTTP detail gets.

``UsageLog.error_message`` is read back by every member of a workspace through
``/api/v1/organizations/me/usage`` (a viewer included), while the HTTP detail
for the same failure was the only copy being redacted. These pin the persisted
copy: key material, URLs and account identifiers are masked, an echoed request
payload is replaced by the fixed detail, and the length is capped, so the row
carries what went wrong and not what served the request.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.routes._pipeline import PROVIDER_ERROR_DETAIL, log_usage
from gateway.services.upstream_redaction import MAX_EXPOSED_DETAIL_CHARS

from .test_log_usage_commit_scope import StubLogWriter
from .test_usage_status_code import _chat, _error_rows, _upstream_fails

_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"
_URL = "https://llm.internal.example:8443/v1/chat/completions"
_ORG = "org-abc123def456"


class _LeakyError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("stream", [False, True])
def test_stored_error_masks_credentials_topology_and_account(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    stream: bool,
) -> None:
    message = f"Incorrect API key {_KEY} for {_ORG} at {_URL}; check the model name"
    with _upstream_fails(_LeakyError(message)):
        _chat(client, api_key_header, stream=stream)

    rows = _error_rows(client, master_key_header)
    assert len(rows) == 1
    stored = rows[0]["error_message"]
    assert _KEY not in stored
    assert _URL not in stored
    assert _ORG not in stored
    assert "[redacted]" in stored
    assert "check the model name" in stored


def test_stored_error_drops_an_echoed_request_payload(
    client: TestClient,
    api_key_header: dict[str, str],
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """A provider that reflects the request back must not put another member's
    prompt on a row every member can read."""
    prompt = "the quarterly numbers nobody else should see"
    with _upstream_fails(_LeakyError(f'messages.0.content: Input should be a string, input_value="{prompt}"')):
        _chat(client, api_key_header)

    rows = _error_rows(client, master_key_header)
    assert len(rows) == 1
    assert rows[0]["error_message"] == PROVIDER_ERROR_DETAIL
    assert prompt not in rows[0]["error_message"]


@pytest.mark.asyncio
async def test_log_usage_caps_the_stored_error_and_keeps_gateway_reasons(async_db: AsyncSession) -> None:
    writer = StubLogWriter()
    common: dict[str, Any] = {
        "db": async_db,
        "log_writer": writer,
        "api_key_id": None,
        "model": "gpt-4o",
        "provider": "openai",
        "endpoint": "/v1/chat/completions",
    }

    await log_usage(**common, error="x " * 2000)
    await log_usage(**common, error="Provider timeout")
    await log_usage(**common, error=None)

    capped, gateway_reason, success = writer.logs
    assert len(capped.error_message or "") <= MAX_EXPOSED_DETAIL_CHARS
    assert capped.status == "error"
    assert gateway_reason.error_message == "Provider timeout"
    assert success.error_message is None
    assert success.status == "success"
