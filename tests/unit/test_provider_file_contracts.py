"""Provider-neutral Files contracts preserve the SDK's unknown values."""

import uuid
from datetime import UTC, datetime

import httpx
import pytest
from any_llm.types.files import FileMetadata as SDKFileMetadata
from pydantic import SecretStr, ValidationError

from gateway.services.provider_files import transport
from gateway.services.provider_files.contracts import FileAccount, FileMetadata, FilesError


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_account_preserves_provider(provider: str) -> None:
    account = FileAccount(generation_id=uuid.uuid4(), provider=provider, api_key=SecretStr("test-only"))
    assert account.provider == provider


def test_normalized_metadata_preserves_unknowns_and_provider_fields() -> None:
    native = SDKFileMetadata(id="file_test", size_bytes=4, purpose="user_data", status="processed")
    metadata = FileMetadata.model_validate(native.model_dump(exclude_unset=True))
    assert metadata.model_dump(exclude_unset=True) == native.model_dump(exclude_unset=True)
    assert metadata.downloadable is None
    assert metadata.mime_type is None


def test_anthropic_metadata_keeps_native_extras() -> None:
    metadata = FileMetadata(
        id="file_test",
        filename="data.csv",
        mime_type="text/csv",
        size_bytes=4,
        created_at=datetime.now(UTC),
        downloadable=False,
    ).model_copy(update={"type": "file"})
    assert metadata.model_dump()["type"] == "file"


def test_metadata_remains_bounded() -> None:
    with pytest.raises(ValidationError):
        FileMetadata.model_validate({"id": "file_test", "unexpected": "x" * 16384})


@pytest.mark.asyncio
async def test_transport_dispatches_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    async def safe(base: str) -> str:
        assert base == "https://api.openai.com/v1"
        return base

    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/file_test"
        assert request.headers["authorization"] == "Bearer test-only"
        assert "anthropic-workspace-id" not in request.headers
        return httpx.Response(200, json={"id": "file_test", "object": "file", "deleted": True})

    monkeypatch.setattr(transport, "validate_provider_api_base", safe)
    original = httpx.AsyncClient
    monkeypatch.setattr(transport, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(upstream), **kw))
    account = FileAccount(generation_id=uuid.uuid4(), provider="openai", api_key=SecretStr("test-only"))
    async with transport.provider_client(account) as client:
        result = await client.adelete_file("file_test")
    assert result.deleted is True


@pytest.mark.asyncio
async def test_transport_rejects_private_endpoint() -> None:
    account = FileAccount(
        generation_id=uuid.uuid4(), provider="openai", api_key=SecretStr("test-only"), api_base="http://127.0.0.1"
    )
    with pytest.raises(FilesError, match="Invalid provider file endpoint"):
        async with transport.provider_client(account):
            pytest.fail("Unsafe endpoint must not produce a client")
