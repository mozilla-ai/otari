"""Integration tests for the copy an attached file gets at the provider that runs the code.

Anthropic's ``container_upload`` block names a file from Anthropic's own Files
API, so a request that asks Anthropic to run code over a stored upload has a
short-lived copy made there first. Anthropic's Files API is stubbed at the HTTP
level.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
import pytest
from any_llm.types.messages import MessageResponse, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.core.config import API_ROOT
from gateway.models.files import FileObject, FileProviderCopy

_CODE_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
_MODEL = "anthropic:claude-sonnet-4-5"
_PROVIDER_FILE_ID = "file_011CqStubUpload"


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


class _StubAnthropicFiles:
    """Anthropic's Files API, as far as uploading a copy needs it."""

    def __init__(self) -> None:
        self.uploads: list[bytes] = []
        self.accepting = True

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith("/v1/files"):
            return httpx.Response(404)
        if not self.accepting:
            return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "nope"}})
        self.uploads.append(request.content)
        now = datetime.now(UTC)
        return httpx.Response(
            200,
            json={
                "id": _PROVIDER_FILE_ID,
                "type": "file",
                "filename": "data.csv",
                "mime_type": "text/csv",
                "size_bytes": 12,
                "created_at": now.isoformat(),
                "expires_at": (now + timedelta(hours=1)).isoformat(),
                "downloadable": False,
            },
        )


@pytest.fixture
def anthropic_files(monkeypatch: pytest.MonkeyPatch) -> _StubAnthropicFiles:
    """A deployment credentialed for Anthropic by the SDK's own variable, whose Files API is stubbed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    stub = _StubAnthropicFiles()
    original_init = httpx.AsyncClient.__init__

    def patched_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(stub.handle)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)
    return stub


def _reply() -> MessageResponse:
    return MessageResponse(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4-5",
        content=[TextBlock(type="text", text="ok", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=cast(Any, {"input_tokens": 3, "output_tokens": 2}),
    )


def _upload_file(client: TestClient, headers: dict[str, str]) -> str:
    stored = client.post(
        f"{API_ROOT}/files",
        headers=headers,
        files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert stored.status_code == 200, stored.text
    return str(stored.json()["id"])


def _run(client: TestClient, headers: dict[str, str], file_id: str) -> tuple[Any, list[Any]]:
    """Post a request whose code Anthropic runs, and report what reached the provider."""
    forwarded: list[Any] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        forwarded.append(kwargs.get("messages"))
        return _reply()

    body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": [{"type": "container_upload", "file_id": file_id}]}],
        "max_tokens": 100,
        "tools": [_CODE_TOOL],
    }
    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        response = client.post(f"{API_ROOT}/messages", json=body, headers=headers)
    return response, forwarded


def test_an_attached_file_reaches_the_providers_container(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 200, response.text
    assert forwarded[0][0]["content"][0] == {"type": "container_upload", "file_id": _PROVIDER_FILE_ID}
    assert len(anthropic_files.uploads) == 1
    stored = db_session.get(FileObject, file_id)
    assert stored is not None
    row = db_session.get(FileProviderCopy, (file_id, "anthropic", "anthropic", stored.workspace_id))
    assert row is not None, "the copy was not recorded under the workspace whose credential made it"
    assert row.provider_file_id == _PROVIDER_FILE_ID
    assert row.expires_at > datetime.now(UTC)


def test_a_second_request_reuses_the_copy(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)

    first, _ = _run(client, api_key_header, file_id)
    assert first.status_code == 200, first.text
    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 200, response.text
    assert forwarded[0][0]["content"][0]["file_id"] == _PROVIDER_FILE_ID
    assert len(anthropic_files.uploads) == 1, "the same file was uploaded twice"


def test_a_file_the_deployment_does_not_hold_is_refused(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    """A provider file ID of the caller's choosing must never reach the provider."""
    response, forwarded = _run(client, api_key_header, "file_011CqSomeoneElses")

    assert response.status_code == 400, response.text
    assert forwarded == []
    assert anthropic_files.uploads == []


def test_a_deployment_that_makes_no_copies_refuses(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_id = _upload_file(client, api_key_header)
    monkeypatch.setattr(
        cast(Any, client.app).state.config, "files_provider_upload_enabled", False, raising=True
    )

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 400, response.text
    assert forwarded == []
    assert anthropic_files.uploads == []


def test_a_provider_that_will_not_take_the_copy_refuses(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)
    anthropic_files.accepting = False

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 502, response.text
    assert forwarded == []


def test_file_understanding_off_refuses_rather_than_forwarding_the_block(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With normalization off nothing reads the block, so it must not reach the provider."""
    file_id = _upload_file(client, api_key_header)
    monkeypatch.setattr(cast(Any, client.app).state.config, "file_understanding_enabled", False, raising=True)

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 400, response.text
    assert forwarded == []
    assert anthropic_files.uploads == []
