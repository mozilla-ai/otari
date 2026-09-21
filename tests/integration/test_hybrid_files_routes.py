"""Public Files routing is scoped, GA-only, and never exposes unfinalized IDs."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from any_llm.types.files import AsyncFileDownload
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from gateway.api.deps import get_config
from gateway.api.routes import _file_formats, hybrid_files
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
from gateway.services.provider_files.transfers import receive_upload


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


@pytest.mark.parametrize("failure", ["conversion", "context-exit"])
def test_failure_after_finalize_does_not_compensate(
    file_client: tuple[TestClient, list[str]], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Finalize commits the binding, so a later failure must not delete the upload."""
    client, events = file_client
    if failure == "conversion":

        def convert(self: Any, value: Any) -> Any:
            raise FilesError(502, "Provider returned invalid file metadata")

        monkeypatch.setattr(_file_formats.AnthropicFilesFormat, "metadata", convert)
    else:
        original = receive_upload

        @asynccontextmanager
        async def leaky(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            async with original(*args, **kwargs) as value:
                yield value
            raise RuntimeError("upload context failed on exit")

        monkeypatch.setattr(hybrid_files, "receive_upload", leaky)
    response = client.post(API_ROOT + "/files", headers=HEADERS, files={"file": ("example.csv", b"data")})
    assert response.status_code == 502, response.text
    assert events[-1].endswith("/finalize")
    assert "provider-delete" not in events


@pytest.mark.parametrize("oversized", ["metadata", "content-length", None])
def test_oversized_download_is_refused_before_the_body(
    file_client: tuple[TestClient, list[str]], monkeypatch: pytest.MonkeyPatch, oversized: str | None
) -> None:
    """The 413 must precede the 200, since a streaming abort cannot take one back."""
    client, events = file_client
    payload = b"x" * (2048 if oversized else 4)
    data = FileMetadata(
        id="file_provider",
        filename="example.csv",
        mime_type="text/csv",
        size_bytes=len(payload) if oversized == "metadata" else None,
        created_at=datetime.now(UTC),
        downloadable=True,
    )
    account = FileAccount(generation_id=uuid.uuid4(), api_key=SecretStr("provider-secret"))

    async def post(self: object, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        events.append(path)
        return ResolvedFile(metadata=data, account=account)

    class Provider:
        @asynccontextmanager
        async def adownload_file(self, file_id: str, **kwargs: Any) -> AsyncIterator[AsyncFileDownload]:
            events.append("provider-download")

            async def chunks() -> AsyncIterator[bytes]:
                events.append("provider-chunk")
                yield payload

            yield AsyncFileDownload(
                status_code=200,
                headers={"Content-Type": "text/csv", "Content-Length": str(len(payload))},
                chunks=chunks(),
            )

    @asynccontextmanager
    async def provider(*args: Any, **kwargs: Any) -> AsyncIterator[Provider]:
        yield Provider()

    monkeypatch.setattr(PlatformFilesClient, "post", post)
    monkeypatch.setattr(hybrid_files, "provider_client", provider)
    response = client.get(API_ROOT + "/files/file_provider/content", headers=HEADERS)
    if oversized is None:
        assert response.status_code == 200, response.text
        assert response.content == payload
        assert response.headers["content-length"] == "4"
        return
    assert response.status_code == 413, response.text
    assert "provider-chunk" not in events
    assert ("provider-download" in events) == (oversized == "content-length")


@pytest.mark.parametrize("path", ["/files", "/files/file_provider"])
def test_invalid_openai_metadata_returns_fixed_error(
    file_client: tuple[TestClient, list[str]], monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    client, _ = file_client
    data = FileMetadata.model_validate({"id": "file_provider", "bytes": "private-invalid-value"})

    async def post(self: object, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        if path == "list":
            return FilePage(data=[data])
        return ResolvedFile(metadata=data)

    monkeypatch.setattr(PlatformFilesClient, "post", post)
    response = client.get(
        API_ROOT + path, headers={"Authorization": "Bearer caller-token", "X-Otari-Files-Provider": "openai"}
    )
    assert response.status_code == 502
    assert response.json() == {"detail": "Provider returned invalid file metadata"}
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-otari-files-protocol"] == "2"


@pytest.mark.parametrize("authorization", [None, "Basic invalid", "Bearer "])
@pytest.mark.parametrize(
    "method,path",
    [
        ("POST", "/files"),
        ("GET", "/files"),
        ("GET", "/files/file_provider"),
        ("GET", "/files/file_provider/content"),
        ("DELETE", "/files/file_provider"),
    ],
)
def test_authentication_errors_include_files_headers(
    file_client: tuple[TestClient, list[str]], method: str, path: str, authorization: str | None
) -> None:
    client, events = file_client
    headers = {"anthropic-version": "2023-06-01"}
    if authorization is not None:
        headers["Authorization"] = authorization
    response = client.request(method, API_ROOT + path, headers=headers)
    assert response.status_code == 401
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-otari-files-protocol"] == "2"
    assert not events


@pytest.mark.parametrize("standalone_enabled", [False, True])
@pytest.mark.parametrize("native_enabled", [False, True])
def test_hybrid_files_only_uses_native_feature_flag(
    file_client: tuple[TestClient, list[str]], standalone_enabled: bool, native_enabled: bool
) -> None:
    client, events = file_client
    app: Any = client.app
    config = app.dependency_overrides[get_config]()
    app.dependency_overrides[get_config] = lambda: config.model_copy(
        update={"files_enabled": standalone_enabled, "files_provider_native_enabled": native_enabled}
    )
    response = client.get(API_ROOT + "/files", headers=HEADERS)
    assert response.status_code == (200 if native_enabled else 404)
    assert events == (["list"] if native_enabled else [])


@pytest.mark.parametrize("stage,budget", [("setup", "transfer"), ("download", "transfer"), ("download", "idle")])
def test_download_setup_timeout_returns_504_and_closes_resources(
    file_client: tuple[TestClient, list[str]], monkeypatch: pytest.MonkeyPatch, stage: str, budget: str
) -> None:
    client, events = file_client
    app: Any = client.app
    config = app.dependency_overrides[get_config]()
    app.dependency_overrides[get_config] = lambda: config.model_copy(
        update={
            "files_transfer_timeout_seconds": 0.01 if budget == "transfer" else 1,
            "files_idle_timeout_seconds": 0.01 if budget == "idle" else 1,
        }
    )

    async def stall() -> None:
        await asyncio.sleep(0.1)
        raise RuntimeError("Transfer deadline did not interrupt provider setup")

    class Provider:
        @asynccontextmanager
        async def adownload_file(self, file_id: str, **kwargs: Any) -> AsyncIterator[AsyncFileDownload]:
            try:
                await stall()
                yield AsyncFileDownload(status_code=200, headers={}, chunks=aiter_bytes())
            finally:
                events.append("download-closed")

    async def aiter_bytes() -> AsyncIterator[bytes]:
        yield b"data"

    @asynccontextmanager
    async def provider(*args: Any, **kwargs: Any) -> AsyncIterator[Provider]:
        try:
            if stage == "setup":
                await stall()
            yield Provider()
        finally:
            events.append("provider-closed")

    monkeypatch.setattr(hybrid_files, "provider_client", provider)
    response = client.get(API_ROOT + "/files/file_provider/content", headers=HEADERS)
    assert response.status_code == 504, response.text
    assert response.json() == {"detail": "File transfer timed out"}
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-otari-files-protocol"] == "2"
    assert events[-1] == "provider-closed"
    assert ("download-closed" in events) == (stage == "download")


@pytest.mark.parametrize("stage", ["setup", "delete"])
def test_delete_timeout_reports_failed_cleanup_before_504(
    file_client: tuple[TestClient, list[str]], monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    client, events = file_client
    app: Any = client.app
    config = app.dependency_overrides[get_config]()
    app.dependency_overrides[get_config] = lambda: config.model_copy(update={"files_transfer_timeout_seconds": 0.01})
    original = PlatformFilesClient.post

    async def post(self: Any, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        if path.endswith("/cleanup-result"):
            assert body == {"cleanup_token": "operation-token", "deleted": False}
        return await original(self, path, body, result_type)

    async def stall() -> None:
        await asyncio.sleep(0.1)
        raise RuntimeError("Transfer deadline did not interrupt provider deletion")

    class Provider:
        async def adelete_file(self, file_id: str, **kwargs: Any) -> None:
            await stall()

    @asynccontextmanager
    async def provider(*args: Any, **kwargs: Any) -> AsyncIterator[Provider]:
        try:
            if stage == "setup":
                await stall()
            yield Provider()
        finally:
            events.append("provider-closed")

    monkeypatch.setattr(PlatformFilesClient, "post", post)
    monkeypatch.setattr(hybrid_files, "provider_client", provider)
    response = client.delete(API_ROOT + "/files/file_provider", headers=HEADERS)
    assert response.status_code == 504, response.text
    assert response.json() == {"detail": "File transfer timed out"}
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-otari-files-protocol"] == "2"
    assert events[-2] == "provider-closed"
    assert events[-1].endswith("/cleanup-result")
