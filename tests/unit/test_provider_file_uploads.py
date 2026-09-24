"""The copy an attached file gets at the provider that runs a request's code.

Covers the policy around the upload, with the provider's client stubbed: a copy
with time left is reused, one about to expire is replaced, the expiry the
provider reports is what the row keeps, and both refusals stop the request.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from any_llm.types.files import FileMetadata

from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import (
    AttachedFileExpiresTooSoonError,
    ProviderUploadDisabledError,
    ProviderUploadFailedError,
)
from gateway.models.files import FileProviderCopy
from gateway.ports.file_storage_port import FileStoragePort
from gateway.repositories.files import FileProviderCopyRepository, FileRepositories, FileRepository
from gateway.services.files import ProviderFileUploader, StagedFile, _provider_uploads

_STAGED = StagedFile(file_id="file-1", filename="report.csv", mime_type="text/csv", storage_ref="ref-1")
_PROVIDER = "anthropic"
_INSTANCE = "anthropic"
_WORKSPACE = uuid.uuid4()


class _Copies:
    """One row, in memory: what the uploader reads before it uploads and writes after."""

    def __init__(self, existing: FileProviderCopy | None = None) -> None:
        self.row = existing
        self.recorded: list[FileProviderCopy] = []
        self.asked: list[tuple[str, str, str, uuid.UUID]] = []

    async def in_account(
        self, file_id: str, *, provider: str, provider_instance: str, credential_workspace_id: uuid.UUID
    ) -> FileProviderCopy | None:
        self.asked.append((file_id, provider, provider_instance, credential_workspace_id))
        return self.row

    async def record(self, copy: FileProviderCopy) -> None:
        self.row = copy
        self.recorded.append(copy)


class _Uow:
    """A Unit of Work whose blocks open and close and do nothing else."""

    async def __aenter__(self) -> _Uow:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class _Store:
    def __init__(self) -> None:
        self.reads = 0

    async def get(self, storage_ref: str) -> bytes:
        self.reads += 1
        return b"id,value\n1,2\n"


class _Client:
    """The provider's files client, answering one upload."""

    def __init__(self, metadata: FileMetadata | None = None, error: Exception | None = None) -> None:
        self._metadata = metadata
        self._error = error
        self.uploads: list[dict[str, Any]] = []
        self.discarded: list[str] = []
        self.closed = False

    async def upload(self, data: bytes, *, filename: str, mime_type: str, expires_in: int) -> FileMetadata:
        self.uploads.append({"data": data, "filename": filename, "mime_type": mime_type, "expires_in": expires_in})
        if self._error is not None:
            raise self._error
        assert self._metadata is not None
        return self._metadata

    async def discard(self, provider_file_id: str) -> bool:
        self.discarded.append(provider_file_id)
        return True

    async def aclose(self) -> None:
        self.closed = True


def _copy(expires_at: datetime, provider_file_id: str = "file_old") -> FileProviderCopy:
    return FileProviderCopy(
        file_id=_STAGED.file_id,
        provider=_PROVIDER,
        provider_instance=_INSTANCE,
        credential_workspace_id=_WORKSPACE,
        provider_file_id=provider_file_id,
        expires_at=expires_at,
        created_at=datetime.now(UTC),
    )


def _uploader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    copies: _Copies,
    store: _Store,
    client: _Client | None = None,
    config: GatewayConfig | None = None,
) -> ProviderFileUploader:
    if client is not None:
        ready = client

        class _Factory:
            @staticmethod
            def for_run(_config: GatewayConfig, **_kwargs: object) -> _Client:
                return ready

        monkeypatch.setattr(_provider_uploads, "ProviderFileClient", _Factory)
    return ProviderFileUploader(
        cast(UnitOfWork, _Uow()),
        FileRepositories(
            files=cast(FileRepository, None),
            provider_copies=cast(FileProviderCopyRepository, copies),
        ),
        cast(FileStoragePort, store),
        config or GatewayConfig(),
        provider=_PROVIDER,
        provider_instance=_INSTANCE,
        workspace_id=_WORKSPACE,
    )


@pytest.mark.asyncio
async def test_a_copy_with_time_left_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies(_copy(datetime.now(UTC) + timedelta(hours=1)))
    store = _Store()

    file_id = await _uploader(monkeypatch, copies=copies, store=store).file_id_for(_STAGED)

    assert file_id == "file_old"
    assert store.reads == 0
    assert copies.recorded == []


@pytest.mark.asyncio
async def test_a_copy_about_to_expire_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies(_copy(datetime.now(UTC) + timedelta(minutes=1)))
    client = _Client(FileMetadata(id="file_new"))

    file_id = await _uploader(monkeypatch, copies=copies, store=_Store(), client=client).file_id_for(_STAGED)

    assert file_id == "file_new"
    assert copies.recorded[0].provider_file_id == "file_new"


@pytest.mark.asyncio
async def test_an_upload_asks_for_the_configured_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies()
    client = _Client(FileMetadata(id="file_new"))
    config = GatewayConfig(files_provider_upload_ttl_hours=6)

    await _uploader(monkeypatch, copies=copies, store=_Store(), client=client, config=config).file_id_for(_STAGED)

    assert client.uploads[0]["expires_in"] == 6 * 3600
    assert client.uploads[0]["filename"] == "report.csv"
    assert client.closed


@pytest.mark.asyncio
async def test_the_row_keeps_an_earlier_expiry_the_provider_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider is free to hold a copy for less time, and the row follows what it said."""
    copies = _Copies()
    reported = datetime.now(UTC) + timedelta(minutes=30)
    client = _Client(FileMetadata(id="file_new", expires_at=reported))

    await _uploader(monkeypatch, copies=copies, store=_Store(), client=client).file_id_for(_STAGED)

    assert copies.recorded[0].expires_at == reported
    assert client.discarded == []


@pytest.mark.asyncio
async def test_a_deployment_that_makes_no_copies_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store()
    config = GatewayConfig(files_provider_upload_enabled=False)

    with pytest.raises(ProviderUploadDisabledError):
        await _uploader(monkeypatch, copies=_Copies(), store=store, config=config).file_id_for(_STAGED)

    assert store.reads == 0


@pytest.mark.asyncio
async def test_a_provider_that_will_not_take_the_copy_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies()
    client = _Client(error=ProviderUploadFailedError())

    with pytest.raises(ProviderUploadFailedError):
        await _uploader(monkeypatch, copies=copies, store=_Store(), client=client).file_id_for(_STAGED)

    assert copies.recorded == []
    assert client.closed


@pytest.mark.asyncio
async def test_a_copy_never_outlives_the_files_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies()
    client = _Client(FileMetadata(id="file_new"))
    staged = replace(_STAGED, expires_at=datetime.now(UTC) + timedelta(hours=2))
    config = GatewayConfig(files_provider_upload_ttl_hours=48)

    await _uploader(monkeypatch, copies=copies, store=_Store(), client=client, config=config).file_id_for(staged)

    asked = client.uploads[0]["expires_in"]
    assert 0 < asked <= 2 * 3600, "the copy was asked for more life than the file has"


@pytest.mark.asyncio
async def test_a_file_expiring_sooner_than_the_provider_will_hold_a_copy_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic's shortest storable life is an hour, so a copy of this file would outlive it."""
    store = _Store()
    staged = replace(_STAGED, expires_at=datetime.now(UTC) + timedelta(minutes=10))

    with pytest.raises(AttachedFileExpiresTooSoonError):
        await _uploader(monkeypatch, copies=_Copies(), store=store, client=_Client()).file_id_for(staged)

    assert store.reads == 0


@pytest.mark.asyncio
async def test_the_copy_is_looked_up_under_the_workspace_that_supplies_the_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A copy belongs to one provider account, and the workspace is half of what selects it."""
    copies = _Copies()
    client = _Client(FileMetadata(id="file_new"))

    await _uploader(monkeypatch, copies=copies, store=_Store(), client=client).file_id_for(_STAGED)

    assert copies.asked == [(_STAGED.file_id, _PROVIDER, _INSTANCE, _WORKSPACE)]
    assert copies.recorded[0].credential_workspace_id == _WORKSPACE


@pytest.mark.asyncio
async def test_a_provider_holding_the_copy_too_long_has_it_taken_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otari cannot make a provider honor an expiry, so a copy that outlives the file is removed."""
    copies = _Copies()
    client = _Client(FileMetadata(id="file_new", expires_at=datetime.now(UTC) + timedelta(days=30)))
    staged = replace(_STAGED, expires_at=datetime.now(UTC) + timedelta(hours=2))

    with pytest.raises(ProviderUploadFailedError):
        await _uploader(monkeypatch, copies=copies, store=_Store(), client=client).file_id_for(staged)

    assert client.discarded == ["file_new"], "the copy was left at the provider"
    assert copies.recorded == []
