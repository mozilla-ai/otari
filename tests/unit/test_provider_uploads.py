"""Moving an attachment into Gemini's Files API, with the provider client stubbed."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from any_llm import AnyLLM, LLMProvider
from any_llm.types.files import FileMetadata

from gateway.core.config import GatewayConfig
from gateway.services.files import provider_uploads
from gateway.services.files.provider_uploads import ProviderUploadError, upload_attachment, uploads_attachments

_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc"


class _Gemini:
    def __init__(self, *states: str | None, uri: str | None = _URI) -> None:
        self.states = list(states)
        self.uri = uri
        self.uploads: list[tuple[bytes, str | None, str | None]] = []

    def _metadata(self) -> FileMetadata:
        fields: dict[str, Any] = {"id": "files/abc", "status": self.states.pop(0)}
        if self.uri:
            fields["uri"] = self.uri
        return FileMetadata.model_validate(fields)

    async def aupload_file(self, data: bytes, *, filename: str | None, mime_type: str | None) -> FileMetadata:
        self.uploads.append((data, filename, mime_type))
        return self._metadata()

    async def aretrieve_file(self, file_id: str) -> FileMetadata:
        return self._metadata()


@pytest.fixture(autouse=True)
def _fresh_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_uploads, "_remembered", type(provider_uploads._remembered)())
    monkeypatch.setattr(provider_uploads, "_POLL_SECONDS", 0)


def _stub(monkeypatch: pytest.MonkeyPatch, gemini: _Gemini) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []

    def create(provider: str, **kwargs: Any) -> _Gemini:
        created.append({"provider": provider, **kwargs})
        return gemini

    monkeypatch.setattr(AnyLLM, "create", create)
    return created


async def _upload(data: bytes = b"%PDF", workspace_id: uuid.UUID | None = None) -> str:
    return await upload_attachment(
        GatewayConfig(providers={"gemini": {"api_key": "g-key"}}),
        provider=LLMProvider.GEMINI,
        instance="gemini",
        workspace_id=workspace_id,
        data=data,
        mime="application/pdf",
        filename="report.pdf",
    )


def test_only_gemini_uploads_attachments() -> None:
    assert uploads_attachments(LLMProvider.GEMINI)
    assert not uploads_attachments(LLMProvider.VERTEXAI)
    assert not uploads_attachments("not-a-provider")
    assert not uploads_attachments(None)


@pytest.mark.asyncio
async def test_an_active_upload_returns_its_uri_with_the_instance_key(monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = _Gemini("ACTIVE")
    created = _stub(monkeypatch, gemini)

    assert await _upload() == _URI
    assert created[0]["provider"] == "gemini"
    assert created[0]["api_key"] == "g-key"
    assert gemini.uploads == [(b"%PDF", "report.pdf", "application/pdf")]


@pytest.mark.asyncio
async def test_an_upload_still_processing_is_polled_until_active(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, _Gemini("PROCESSING", "PROCESSING", "ACTIVE"))

    assert await _upload() == _URI


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gemini", "reason"), [(_Gemini("FAILED"), "state FAILED"), (_Gemini("ACTIVE", uri=None), "no URI")]
)
async def test_an_unusable_upload_raises(monkeypatch: pytest.MonkeyPatch, gemini: _Gemini, reason: str) -> None:
    _stub(monkeypatch, gemini)

    with pytest.raises(ProviderUploadError, match=reason):
        await _upload()


@pytest.mark.asyncio
async def test_an_upload_that_never_finishes_processing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, _Gemini(*["PROCESSING"] * 5))
    monkeypatch.setattr(provider_uploads, "_PROCESSING_SECONDS", 0)

    with pytest.raises(ProviderUploadError, match="still processing"):
        await _upload()


@pytest.mark.asyncio
async def test_the_same_bytes_are_uploaded_once_per_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = _Gemini("ACTIVE", "ACTIVE")
    _stub(monkeypatch, gemini)
    workspace = uuid.uuid4()

    await _upload(workspace_id=workspace)
    await _upload(workspace_id=workspace)
    await _upload(workspace_id=uuid.uuid4())

    assert len(gemini.uploads) == 2
