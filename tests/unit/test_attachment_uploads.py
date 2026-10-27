"""Moving an attachment into Gemini's Files API, with the provider client stubbed."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, cast

import pytest
from any_llm import AnyLLM, LLMProvider
from any_llm.types.files import FileMetadata

from gateway.adapters import provider_file_adapter
from gateway.adapters.provider_file_adapter import AnyLlmProviderFiles
from gateway.core.config import GatewayConfig
from gateway.exceptions.files_exceptions import ProviderUploadFailedError
from gateway.ports.provider_file_port import ProviderFilePort
from gateway.services.content_normalizer import (
    InlineLimit,
    NormalizationStats,
    _Source,
    _within_inline_limit,
    inline_limit_for,
)
from gateway.services.files import FileService, _attachment_uploads, uploads_attachments
from gateway.services.files._attachment_uploads import upload_attachment
from gateway.services.providers import CountryGate, CountryNotServedError
from gateway.types.provider_account import ResolvedCredential

_URI = "https://generativelanguage.googleapis.com/v1beta/files/abc"
_KEY = ResolvedCredential(api_key="g-key")


class _Gemini:
    def __init__(self, *states: str | None, uri: str | None = _URI) -> None:
        self.states = list(states)
        self.uri = uri
        self.uploads: list[tuple[bytes, str | None, str | None]] = []

    def _metadata(self) -> FileMetadata:
        fields: dict[str, Any] = {"id": "files/abc", "status": self.states.pop(0) if self.states else "PROCESSING"}
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
    monkeypatch.setattr(_attachment_uploads, "_remembered", type(_attachment_uploads._remembered)())
    monkeypatch.setattr(provider_file_adapter, "_POLL_SECONDS", 0)


def _stub(monkeypatch: pytest.MonkeyPatch, gemini: _Gemini) -> list[dict[str, Any]]:
    created: list[dict[str, Any]] = []

    def create(provider: str, **kwargs: Any) -> _Gemini:
        created.append({"provider": provider, **kwargs})
        return gemini

    monkeypatch.setattr(AnyLLM, "create", create)
    return created


async def _adapter_upload(provider: LLMProvider = LLMProvider.GEMINI) -> str:
    return await AnyLlmProviderFiles().upload_attachment(
        provider=provider, credential=_KEY, data=b"%PDF", filename="report.pdf", mime_type="application/pdf"
    )


async def _upload(
    files: ProviderFilePort | None = None,
    *,
    credential: ResolvedCredential = _KEY,
    workspace_id: uuid.UUID | None = None,
) -> str:
    return await upload_attachment(
        files or AnyLlmProviderFiles(),
        provider=LLMProvider.GEMINI,
        credential=credential,
        workspace_id=workspace_id,
        data=b"%PDF",
        mime_type="application/pdf",
        filename="report.pdf",
    )


def test_only_gemini_uploads_attachments() -> None:
    assert uploads_attachments(LLMProvider.GEMINI)
    assert not uploads_attachments(LLMProvider.VERTEXAI)
    assert not uploads_attachments("not-a-provider")
    assert not uploads_attachments(None)


@pytest.mark.asyncio
async def test_an_active_upload_returns_its_uri_with_the_dispatch_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = _Gemini("ACTIVE")
    created = _stub(monkeypatch, gemini)

    assert await _adapter_upload() == _URI
    assert created[0]["provider"] == "gemini"
    assert created[0]["api_key"] == "g-key"
    assert gemini.uploads == [(b"%PDF", "report.pdf", "application/pdf")]


@pytest.mark.asyncio
async def test_an_upload_still_processing_is_polled_until_active(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, _Gemini("PROCESSING", "PROCESSING", "ACTIVE"))

    assert await _adapter_upload() == _URI


@pytest.mark.asyncio
@pytest.mark.parametrize("gemini", [_Gemini("FAILED"), _Gemini("ACTIVE", uri=None)])
async def test_an_unusable_upload_raises(monkeypatch: pytest.MonkeyPatch, gemini: _Gemini) -> None:
    _stub(monkeypatch, gemini)

    with pytest.raises(ProviderUploadFailedError):
        await _adapter_upload()


@pytest.mark.asyncio
async def test_an_upload_that_never_finishes_processing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, _Gemini())
    monkeypatch.setattr(provider_file_adapter, "_PROCESSING_SECONDS", 0)

    with pytest.raises(ProviderUploadFailedError):
        await _adapter_upload()


@pytest.mark.asyncio
async def test_a_refused_upload_raises_the_ports_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Refusing(_Gemini):
        async def aupload_file(self, data: bytes, *, filename: str | None, mime_type: str | None) -> FileMetadata:
            raise RuntimeError("quota")

    _stub(monkeypatch, _Refusing())

    with pytest.raises(ProviderUploadFailedError):
        await _adapter_upload()


@pytest.mark.asyncio
async def test_a_provider_that_reads_no_attachment_by_uri_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _stub(monkeypatch, _Gemini("ACTIVE"))

    with pytest.raises(ProviderUploadFailedError):
        await _adapter_upload(LLMProvider.OPENAI)
    assert created == []


@pytest.mark.asyncio
async def test_the_same_bytes_are_uploaded_once_per_account_and_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = _Gemini("ACTIVE", "ACTIVE", "ACTIVE")
    _stub(monkeypatch, gemini)
    workspace = uuid.uuid4()

    await _upload(workspace_id=workspace)
    await _upload(workspace_id=workspace)
    await _upload(workspace_id=uuid.uuid4())
    # A rotated key may reach another project, which cannot read the first URI.
    await _upload(workspace_id=workspace, credential=ResolvedCredential(api_key="rotated"))

    assert len(gemini.uploads) == 3


@pytest.mark.asyncio
async def test_the_same_bytes_under_another_type_are_uploaded_again(monkeypatch: pytest.MonkeyPatch) -> None:
    gemini = _Gemini("ACTIVE", "ACTIVE")
    _stub(monkeypatch, gemini)

    for mime_type in ("application/pdf", "text/plain"):
        await upload_attachment(
            AnyLlmProviderFiles(),
            provider=LLMProvider.GEMINI,
            credential=_KEY,
            workspace_id=None,
            data=b"%PDF",
            mime_type=mime_type,
            filename="report",
        )

    assert [mime for _, _, mime in gemini.uploads] == ["application/pdf", "text/plain"]


@pytest.mark.asyncio
async def test_an_upload_is_bounded_as_a_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stalled:
        async def upload_attachment(self, **_: Any) -> str:
            await asyncio.sleep(10)
            return _URI

    monkeypatch.setattr(_attachment_uploads, "_UPLOAD_SECONDS", 0.01)

    with pytest.raises(ProviderUploadFailedError):
        await _upload(cast(ProviderFilePort, _Stalled()))


def _limit(*candidates: LLMProvider | None, fmt: str = "openai", files: bool = True) -> Any:
    return inline_limit_for(
        GatewayConfig(files_gemini_inline_max_bytes=100),
        LLMProvider.GEMINI if candidates[0] is None else candidates[0],
        cast(Any, fmt),
        files=cast(FileService, object()) if files else None,
        candidates=candidates,
        instance=None,
        workspace_id=None,
    )


def test_a_gemini_request_gets_the_limit_and_an_upload() -> None:
    limit = _limit(LLMProvider.GEMINI, LLMProvider.GEMINI)

    assert limit is not None
    assert limit.max_bytes == 100
    assert limit.upload is not None


def test_with_no_files_service_the_limit_has_nowhere_to_upload() -> None:
    limit = _limit(LLMProvider.GEMINI, files=False)

    assert limit is not None
    assert limit.upload is None


def test_vertex_gets_no_limit() -> None:
    """Its provider serves Claude and Mistral models as well, under limits of their own."""
    assert _limit(LLMProvider.VERTEXAI) is None
    assert _limit(LLMProvider.GEMINI, LLMProvider.VERTEXAI) is None


def test_a_request_that_may_fall_back_beyond_gemini_is_sent_as_it_is() -> None:
    """A Gemini URI is no use to the fallback, and the fallback may well take the bytes inline."""
    assert _limit(LLMProvider.GEMINI, LLMProvider.OPENAI) is None
    assert _limit(LLMProvider.GEMINI, None) is None
    assert _limit(LLMProvider.OPENAI) is None


def test_responses_gets_no_limit() -> None:
    assert _limit(LLMProvider.GEMINI, fmt="responses") is None


class _Files:
    def __init__(self) -> None:
        self.credentials: list[ResolvedCredential] = []

    async def attachment_uri(self, data: bytes, *, credential: ResolvedCredential, **_: Any) -> str:
        self.credentials.append(credential)
        return _URI


def _hosted_limit(files: _Files, hosted: ResolvedCredential | None) -> Any:
    async def _hosted() -> ResolvedCredential | None:
        return hosted

    return inline_limit_for(
        GatewayConfig(files_gemini_inline_max_bytes=100),
        LLMProvider.GEMINI,
        "openai",
        files=cast(FileService, files),
        candidates=(LLMProvider.GEMINI,),
        instance=None,
        workspace_id=None,
        hosted_credential=_hosted,
    )


@pytest.mark.asyncio
async def test_with_no_stored_key_the_upload_uses_the_hosted_provider() -> None:
    files = _Files()
    hosted = ResolvedCredential(api_key="hosted-key")

    assert await _hosted_limit(files, hosted).upload(b"%PDF", "application/pdf", "report.pdf") == _URI
    assert files.credentials == [hosted]


@pytest.mark.asyncio
async def test_with_no_key_anywhere_the_upload_is_refused() -> None:
    files = _Files()

    with pytest.raises(LookupError):
        await _hosted_limit(files, None).upload(b"%PDF", "application/pdf", "report.pdf")
    assert files.credentials == []


@pytest.mark.asyncio
async def test_a_country_the_hosted_provider_does_not_serve_is_no_oversized_attachment() -> None:
    """The request is refused for its country, rather than for an attachment too large to send."""

    async def _refused(data: bytes, mime: str, filename: str) -> str:
        raise CountryNotServedError(CountryGate("FR"))

    stats = NormalizationStats()
    source = _Source(kind="image", data=b"0123456789", mime="image/png", filename=None, data_url=None)

    with pytest.raises(CountryNotServedError):
        await _within_inline_limit({}, source, InlineLimit(max_bytes=1, upload=_refused), stats)
    assert stats.oversized == 0
