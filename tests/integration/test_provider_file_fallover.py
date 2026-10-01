"""Integration tests for the copy each routing candidate is sent of an attached file.

A provider file ID exists only in the account of the key that uploaded it, so a
candidate that falls over to another key must be sent a copy in that key's
account, and a candidate that cannot hold a copy must not be sent the request.
Anthropic's Files API is stubbed at the HTTP level, and mints IDs per key.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.messages import MessageResponse, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.models.files import FileProviderCopy
from gateway.models.routing import RoutingConfig
from gateway.models.usage import UsageLog

_CODE_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
_MODEL = "claude-sonnet-4-5"


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
        providers={
            "claude-a": {"provider_type": "anthropic", "api_key": "sk-ant-a"},
            "claude-b": {"provider_type": "anthropic", "api_key": "sk-ant-b"},
            "openai": {"api_key": "sk-openai"},
        },
        routing=RoutingConfig.model_validate(
            {
                "policies": {
                    "two-keys": {"select": [{"default": f"claude-a:{_MODEL}"}], "on_failure": [f"claude-b:{_MODEL}"]},
                    "then-openai": {"select": [{"default": f"claude-a:{_MODEL}"}], "on_failure": ["openai:gpt-5"]},
                }
            }
        ),
    )


@pytest.fixture
def file_store(client: TestClient, tmp_path: Path) -> Generator[None]:
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))
    yield


class _AccountFiles:
    """Anthropic's Files API, minting each copy's ID from the key that uploaded it."""

    def __init__(self) -> None:
        self.uploads: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "/v1/files/" in request.url.path:
            return httpx.Response(200, json=self._metadata(request.url.path.rsplit("/", 1)[-1]))
        if request.method != "POST" or not request.url.path.endswith("/v1/files"):
            return httpx.Response(404)
        key = request.headers.get("x-api-key", "")
        self.uploads.append(key)
        return httpx.Response(200, json=self._metadata(f"file_copy_for_{key}"))

    @staticmethod
    def _metadata(file_id: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "id": file_id,
            "type": "file",
            "filename": "data.csv",
            "mime_type": "text/csv",
            "size_bytes": 8,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "downloadable": False,
        }


@pytest.fixture
def account_files(monkeypatch: pytest.MonkeyPatch) -> _AccountFiles:
    stub = _AccountFiles()
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
        model=_MODEL,
        content=[TextBlock(type="text", text="ok", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=cast(Any, {"input_tokens": 3, "output_tokens": 2}),
    )


def _upload_file(client: TestClient, headers: dict[str, str]) -> str:
    stored = client.post(f"{API_ROOT}/files", headers=headers, files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")})
    assert stored.status_code == 200, stored.text
    return str(stored.json()["id"])


def _run(client: TestClient, headers: dict[str, str], policy: str, file_id: str, provider: AsyncMock) -> Any:
    body = {
        "model": policy,
        "messages": [{"role": "user", "content": [{"type": "container_upload", "file_id": file_id}]}],
        "max_tokens": 100,
        "tools": [_CODE_TOOL],
    }
    with patch("gateway.api.routes.messages.amessages", new=provider):
        return client.post(f"{API_ROOT}/messages", json=body, headers=headers)


def _unavailable() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))


def test_a_fallover_to_another_key_is_sent_a_copy_in_that_keys_account(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    file_store: None,
    account_files: _AccountFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)
    provider = AsyncMock(side_effect=[_unavailable(), _reply()])

    response = _run(client, api_key_header, "two-keys", file_id, provider)

    assert response.status_code == 200, response.text
    sent = [call.kwargs["messages"][0]["content"][0]["file_id"] for call in provider.await_args_list]
    assert sent == ["file_copy_for_sk-ant-a", "file_copy_for_sk-ant-b"]
    rows = db_session.scalars(select(FileProviderCopy).where(FileProviderCopy.file_id == file_id)).all()
    assert {row.provider_instance for row in rows} == {"claude-a", "claude-b"}
    assert len({row.account_identity for row in rows}) == 2


def test_a_fallover_to_a_provider_that_cannot_hold_a_copy_is_not_sent_the_request(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    file_store: None,
    account_files: _AccountFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)
    provider = AsyncMock(side_effect=[_unavailable(), _reply()])

    response = _run(client, api_key_header, "then-openai", file_id, provider)

    assert response.status_code == 502, response.text
    assert provider.await_count == 1, "the request reached a candidate that cannot open the file"
    rows = db_session.scalars(select(UsageLog)).all()
    assert [(row.provider, row.status) for row in rows] == [("claude-a", "error")], (
        "the failure was not recorded against the only provider called"
    )
