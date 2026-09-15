"""Official GA Anthropic client through Otari and the merged any-llm Files transport."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import anthropic
import httpx
import pytest
from any_llm import AnyLLM
from fastapi import FastAPI
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

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not hasattr(AnyLLM, "aupload_file"), reason="Requires any-llm Files interface (#1395, planned 1.28)"
    ),
]


async def test_official_sdk_upload_list_download_delete(monkeypatch: pytest.MonkeyPatch) -> None:
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
    account = FileAccount(generation_id=uuid.uuid4(), api_key=SecretStr("provider-secret"))
    operation = Operation(
        id=uuid.uuid4(),
        cleanup_token=SecretStr("cleanup"),
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        account=account,
        max_bytes=1024,
        expires_in_seconds=604800,
    )
    metadata = FileMetadata(
        id="file_contract",
        type="file",
        filename="input.csv",
        mime_type="text/csv",
        size_bytes=4,
        created_at=datetime.now(UTC),
        downloadable=True,
    )
    calls: list[str] = []

    async def authority(self: Any, path: str, body: dict[str, Any], result_type: type[Any]) -> Any:
        calls.append(path)
        if path == "uploads/prepare":
            return operation
        if path.endswith("/finalize"):
            assert body["metadata"]["id"] == "file_contract"
            return metadata
        if path == "list":
            return FilePage(data=[metadata], next_page=None)
        if path.endswith("/resolve"):
            return ResolvedFile(
                metadata=metadata, account=account, operation_id=operation.id, cleanup_token=SecretStr("cleanup")
            )
        return WireModel()

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "provider-secret"
        assert "anthropic-workspace-id" not in request.headers
        calls.append(f"provider:{request.method}:{request.url.path}")
        if request.method == "POST":
            body = await request.aread()
            assert b"data" in body and b"604800" in body
            return httpx.Response(200, json=metadata.model_dump(mode="json", exclude_unset=True))
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "file_contract", "type": "file_deleted"})
        return httpx.Response(
            200,
            content=b"data",
            headers={"content-type": "text/csv", "content-disposition": 'attachment; filename="output.csv"'},
        )

    monkeypatch.setattr(PlatformFilesClient, "post", authority)
    original_client = httpx.AsyncClient
    gateway_http = original_client(transport=httpx.ASGITransport(app=app))
    monkeypatch.setattr(
        transport,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(upstream), **kwargs),
    )
    async with anthropic.AsyncAnthropic(
        auth_token="user-token", base_url="http://gateway.test/api/", http_client=gateway_http, max_retries=0
    ) as sdk:
        uploaded = await sdk.files.upload(file=("input.csv", b"data", "text/csv"))
        assert uploaded.id == "file_contract"
        assert calls.index("provider:POST:/v1/files") < next(
            i for i, call in enumerate(calls) if call.endswith("/finalize")
        )
        page = await sdk.files.list()
        assert [item.id for item in page.data] == [uploaded.id]
        downloaded = await sdk.files.download(uploaded.id)
        assert await downloaded.read() == b"data"
        deleted = await sdk.files.delete(uploaded.id)
        assert deleted.id == uploaded.id and deleted.type == "file_deleted"
