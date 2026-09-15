"""Public Files routing is scoped, GA-only, and never exposes unfinalized IDs."""

import uuid
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from gateway.api.deps import get_config
from gateway.api.routes import hybrid_files
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import (
    FileAccount,
    FileMetadata,
    FilePage,
    FilesError,
    Operation,
    ResolvedFile,
    WireModel,
)


@pytest.fixture
def file_client(monkeypatch: pytest.MonkeyPatch) -> Generator[tuple[TestClient, list[str]]]:
    monkeypatch.setenv("OTARI_AI_TOKEN", "gateway-token")
    config = GatewayConfig(
        mode="hybrid",
        files_provider_native_enabled=True,
        platform={"base_url": "https://authority.test"},
        files_max_bytes=1024,
    )
    app = FastAPI()
    app.dependency_overrides[get_config] = lambda: config
    app.include_router(hybrid_files.router, prefix=API_ROOT)
    events: list[str] = []
    data = FileMetadata(
        id="file_provider",
        filename="example.csv",
        mime_type="text/csv",
        size_bytes=4,
        created_at=datetime.now(UTC),
        downloadable=True,
    )
    account = FileAccount(generation_id=uuid.uuid4(), api_key=SecretStr("provider-secret"))
    operation = Operation(
        id=uuid.uuid4(),
        cleanup_token=SecretStr("operation-token"),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        account=account,
        max_bytes=1024,
        expires_in_seconds=604800,
    )

    async def post(self: object, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        events.append(path)
        if path == "uploads/prepare":
            return operation
        if path.endswith("/finalize"):
            return data
        if path == "list":
            return FilePage(data=[data], next_page=None)
        if path.endswith("/resolve"):
            return ResolvedFile(
                metadata=data, account=account, operation_id=operation.id, cleanup_token=SecretStr("operation-token")
            )
        return WireModel()

    class Provider:
        async def aupload_file(self, file: Any, **kwargs: Any) -> FileMetadata:
            events.append("provider-upload")
            assert kwargs["max_retries"] == 0
            assert file.read() == b"data"
            assert "x-api-key" not in kwargs["extra_headers"]
            assert "anthropic-workspace-id" not in kwargs["extra_headers"]
            return data

        async def adelete_file(self, file_id: str, **kwargs: Any) -> None:
            events.append("provider-delete")

    @asynccontextmanager
    async def provider(*args: Any, **kwargs: Any) -> AsyncIterator[Provider]:
        yield Provider()

    monkeypatch.setattr(PlatformFilesClient, "post", post)
    monkeypatch.setattr(hybrid_files, "provider_client", provider)
    with TestClient(app) as client:
        yield client, events


HEADERS = {"anthropic-version": "2023-06-01", "Authorization": "Bearer caller-token"}


@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/files"),
        ("GET", "/files"),
        ("GET", "/files/file_x"),
        ("GET", "/files/file_x/content"),
        ("DELETE", "/files/file_x"),
    ],
)
def test_legacy_beta_rejected_on_every_verb(file_client: tuple[TestClient, list[str]], method: str, path: str) -> None:
    client, events = file_client
    response = client.request(
        method, API_ROOT + path, headers={**HEADERS, "AnThRoPiC-BeTa": "other, files-api-2025-04-14 "}
    )
    assert response.status_code == 400
    assert events == []


def test_upload_finalized_before_id_returned(file_client: tuple[TestClient, list[str]]) -> None:
    client, events = file_client
    response = client.post(
        API_ROOT + "/files",
        headers={**HEADERS, "anthropic-workspace-id": "foreign"},
        files={"file": ("example.csv", b"data", "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["id"] == "file_provider"
    assert events[0] == "uploads/prepare"
    assert events[1] == "provider-upload"
    assert events[2].endswith("/finalize")
    assert response.headers["cache-control"] == "private, no-store"
    assert "purpose" not in response.json()


def test_legacy_pagination_rejected_before_authority(file_client: tuple[TestClient, list[str]]) -> None:
    client, events = file_client
    response = client.get(API_ROOT + "/files?after_id=file_x", headers=HEADERS)
    assert response.status_code == 400 and not events


def test_finalize_failure_compensates_without_exposing_id(
    file_client: tuple[TestClient, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, events = file_client
    original = PlatformFilesClient.post

    async def fail(self: Any, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        if path.endswith("/finalize"):
            events.append("finalize-failed")
            raise FilesError(502, "Authorization service unavailable")
        return await original(self, path, body, result_type)

    monkeypatch.setattr(PlatformFilesClient, "post", fail)
    response = client.post(API_ROOT + "/files", headers=HEADERS, files={"file": ("example.csv", b"data")})
    assert response.status_code == 502
    assert "file_provider" not in response.text
    assert events.count("provider-upload") == 1
    assert events.count("finalize-failed") == 3
    assert "provider-delete" in events
    assert events[-1].endswith("/abandon")
