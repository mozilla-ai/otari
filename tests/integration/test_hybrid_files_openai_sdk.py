"""The official OpenAI SDK traverses Otari and any-llm, with only network peers mocked."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import SecretStr

from gateway.api.deps import get_config
from gateway.api.routes import hybrid_files
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.services.provider_files import transport
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import (
    FileAccount,
    FileMetadata,
    FilePage,
    Operation,
    ResolvedFile,
    WireModel,
)

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("retention", [604800, 7776000])
async def test_openai_sdk_files_round_trip(monkeypatch: pytest.MonkeyPatch, retention: int) -> None:
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
    account = FileAccount(generation_id=uuid.uuid4(), provider="openai", api_key=SecretStr("provider-secret"))
    operation = Operation(
        id=uuid.uuid4(),
        cleanup_token=SecretStr("cleanup"),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        account=account,
        max_bytes=1024,
        expires_in_seconds=retention,
    )
    data = FileMetadata(
        id="file_test",
        filename="input.csv",
        size_bytes=4,
        created_at=datetime.now(UTC),
        purpose="user_data",
        status="processed",
    ).model_copy(update={"object": "file"})
    assert data.created_at is not None
    created_at = int(data.created_at.timestamp())
    calls: list[str] = []

    async def authority(self: Any, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        calls.append(path)
        if path in {"uploads/prepare", "list"} or path.endswith("/resolve"):
            assert body["provider"] == "openai"
        if path == "uploads/prepare":
            return operation
        if path.endswith("/finalize"):
            assert body["metadata"]["purpose"] == "user_data"
            assert body["metadata"]["status"] == "processed"
            return data
        if path == "list":
            assert body["purpose"] == "user_data"
            return FilePage(data=[data])
        if path.endswith("/resolve"):
            return ResolvedFile(
                metadata=data, account=account, operation_id=operation.id, cleanup_token=SecretStr("cleanup")
            )
        return WireModel()

    async def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(f"provider:{request.method}:{request.url.path}")
        assert request.headers["authorization"] == "Bearer provider-secret"
        assert "anthropic-version" not in request.headers
        assert "x-otari-files-provider" not in request.headers
        if request.method == "POST":
            body = await request.aread()
            assert b"user_data" in body and str(min(retention, 2592000)).encode() in body
            return httpx.Response(
                200,
                json={
                    "id": data.id,
                    "filename": "input.csv",
                    "bytes": 4,
                    "created_at": created_at,
                    "purpose": "user_data",
                    "status": "processed",
                    "object": "file",
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": data.id, "object": "file", "deleted": True})
        return httpx.Response(200, content=b"data", headers={"content-type": "text/csv"})

    monkeypatch.setattr(PlatformFilesClient, "post", authority)
    original = httpx.AsyncClient
    gateway_http = original(transport=httpx.ASGITransport(app=app))
    monkeypatch.setattr(transport, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(upstream), **kw))
    async with AsyncOpenAI(
        api_key="user-token",
        base_url=f"http://gateway.test{API_ROOT}/",
        http_client=gateway_http,
        max_retries=0,
        default_headers={"X-Otari-Files-Provider": "openai"},
    ) as sdk:
        uploaded = await sdk.files.create(file=("input.csv", b"data", "text/csv"), purpose="user_data")
        assert uploaded.id == data.id and uploaded.bytes == 4
        assert uploaded.purpose == "user_data"
        retrieved = await sdk.files.retrieve(data.id)
        assert retrieved.created_at == created_at
        page = await sdk.files.list(purpose="user_data", limit=1)
        assert [item.id for item in page.data] == [data.id]
        assert page.has_more is False
        assert page.model_dump(exclude_unset=True)["object"] == "list"
        downloaded = await sdk.files.content(data.id)
        assert downloaded.read() == b"data"
        deleted = await sdk.files.delete(data.id)
        assert deleted.deleted is True
    assert calls.index("provider:POST:/v1/files") < next(
        i for i, path in enumerate(calls) if path.endswith("/finalize")
    )
