"""The copy an attached file gets in the provider account that runs a request's code.

Covers the policy around the upload, with the provider's client stubbed: a copy
with time left in the same account is reused, one about to expire or in another
account is not, every copy is reserved before it is uploaded and confirmed
after, and each refusal stops the request with nothing left behind.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from any_llm import LLMProvider
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
from gateway.repositories.files import FileProviderCopyRepository
from gateway.services.files import StagedFile, _provider_uploads
from gateway.services.files._provider_uploads import ProviderCopies
from gateway.types.provider_account import ProviderAccount, ResolvedCredential

_STAGED = StagedFile(file_id="file-1", filename="report.csv", mime_type="text/csv", storage_ref="ref-1")
_WORKSPACE = uuid.uuid4()
_ACCOUNT = ProviderAccount(provider=LLMProvider.ANTHROPIC, instance="anthropic", workspace_id=_WORKSPACE, identity="a1")
_CREDENTIAL = ResolvedCredential(api_key="sk-test")


class _Copies:
    """The copy table, in memory, recording what was done to it and in what order."""

    def __init__(self, *rows: FileProviderCopy, confirms: bool = True) -> None:
        self.rows = list(rows)
        self.events: list[str] = []
        self.lookups: list[tuple[list[str], str]] = []
        self._confirms = confirms

    async def usable(
        self, file_ids: list[str], account_identity: str, *, expiring_after: datetime
    ) -> dict[str, FileProviderCopy]:
        self.lookups.append((list(file_ids), account_identity))
        return {
            row.file_id: row
            for row in self.rows
            if row.file_id in file_ids
            and row.account_identity == account_identity
            and row.pending_since is None
            and row.expires_at is not None
            and row.expires_at > expiring_after
        }

    async def reserve(self, copy: FileProviderCopy) -> None:
        self.events.append("reserve")
        self.rows.append(copy)

    async def confirm(self, copy_id: uuid.UUID, *, provider_file_id: str, expires_at: datetime) -> bool:
        self.events.append("confirm")
        if not self._confirms:
            return False
        row = self._row(copy_id)
        row.provider_file_id, row.expires_at, row.pending_since = provider_file_id, expires_at, None
        return True

    async def cancel(self, copy_id: uuid.UUID) -> None:
        self.events.append("cancel")
        self.rows = [row for row in self.rows if row.id != copy_id]

    def _row(self, copy_id: uuid.UUID) -> FileProviderCopy:
        return next(row for row in self.rows if row.id == copy_id)

    @property
    def confirmed(self) -> list[FileProviderCopy]:
        return [row for row in self.rows if row.pending_since is None]


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
    """The provider's files client, answering each upload."""

    def __init__(
        self,
        metadata: FileMetadata | None = None,
        error: Exception | None = None,
        accept_delay: timedelta | None = None,
    ) -> None:
        self._metadata = metadata
        self._error = error
        self._accept_delay = accept_delay
        self.uploads: list[dict[str, Any]] = []
        self.discarded: list[str] = []
        self.closed = False
        self.provider = "anthropic"

    async def upload(self, data: bytes, *, filename: str, mime_type: str, expires_in: int) -> FileMetadata:
        self.uploads.append({"data": data, "filename": filename, "mime_type": mime_type, "expires_in": expires_in})
        if self._error is not None:
            raise self._error
        if self._accept_delay is not None:
            accepted = datetime.now(UTC) + self._accept_delay
            return FileMetadata(id="file_new", expires_at=accepted + timedelta(seconds=expires_in))
        if self._metadata is not None:
            return self._metadata
        return FileMetadata(id=f"file_new_{len(self.uploads)}")

    async def discard(self, provider_file_id: str) -> bool:
        self.discarded.append(provider_file_id)
        return True

    async def aclose(self) -> None:
        self.closed = True


def _confirmed(expires_at: datetime, *, identity: str = "a1", provider_file_id: str = "file_old") -> FileProviderCopy:
    return FileProviderCopy(
        id=uuid.uuid4(),
        file_id=_STAGED.file_id,
        account_identity=identity,
        provider="anthropic",
        provider_instance="anthropic",
        credential_workspace_id=_WORKSPACE,
        provider_file_id=provider_file_id,
        expires_at=expires_at,
    )


def _copies_service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    copies: _Copies,
    store: _Store | None = None,
    client: _Client | None = None,
    config: GatewayConfig | None = None,
) -> ProviderCopies:
    ready = client or _Client()
    monkeypatch.setattr(_provider_uploads, "ProviderFileClient", lambda **_kwargs: ready)
    return ProviderCopies(
        cast(UnitOfWork, _Uow()),
        cast(FileProviderCopyRepository, copies),
        cast(FileStoragePort, store or _Store()),
        config or GatewayConfig(),
    )


async def _file_ids(service: ProviderCopies, *files: StagedFile, account: ProviderAccount = _ACCOUNT) -> dict[str, str]:
    return await service.file_ids_for(list(files) or [_STAGED], account, _CREDENTIAL)


@pytest.mark.asyncio
async def test_a_copy_with_time_left_in_the_same_account_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies(_confirmed(datetime.now(UTC) + timedelta(hours=1)))
    store = _Store()

    ids = await _file_ids(_copies_service(monkeypatch, copies=copies, store=store))

    assert ids == {"file-1": "file_old"}
    assert store.reads == 0
    assert copies.events == []


@pytest.mark.asyncio
async def test_a_copy_in_another_account_is_not_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed credential names another account, so its copy is made again there."""
    copies = _Copies(_confirmed(datetime.now(UTC) + timedelta(hours=1), identity="a0"))
    client = _Client(FileMetadata(id="file_new"))

    ids = await _file_ids(_copies_service(monkeypatch, copies=copies, client=client))

    assert ids == {"file-1": "file_new"}
    assert copies.lookups == [(["file-1"], "a1")]


@pytest.mark.asyncio
async def test_a_copy_about_to_expire_is_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies(_confirmed(datetime.now(UTC) + timedelta(minutes=1)))
    client = _Client(FileMetadata(id="file_new"))

    ids = await _file_ids(_copies_service(monkeypatch, copies=copies, client=client))

    assert ids == {"file-1": "file_new"}


@pytest.mark.asyncio
async def test_a_copy_is_reserved_before_it_is_uploaded_and_confirmed_after(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies()
    client = _Client(FileMetadata(id="file_new"))

    await _file_ids(_copies_service(monkeypatch, copies=copies, client=client))

    assert copies.events == ["reserve", "confirm"]
    [row] = copies.confirmed
    assert (row.account_identity, row.provider_instance, row.credential_workspace_id) == ("a1", "anthropic", _WORKSPACE)
    assert row.provider_file_id == "file_new"
    assert client.closed


@pytest.mark.asyncio
async def test_several_files_are_looked_up_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies(_confirmed(datetime.now(UTC) + timedelta(hours=1)))
    second = replace(_STAGED, file_id="file-2", storage_ref="ref-2")

    ids = await _file_ids(_copies_service(monkeypatch, copies=copies), _STAGED, second)

    assert copies.lookups == [(["file-1", "file-2"], "a1")]
    assert ids["file-1"] == "file_old"
    assert ids["file-2"].startswith("file_new")


@pytest.mark.asyncio
async def test_an_upload_asks_for_the_configured_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(FileMetadata(id="file_new"))
    config = GatewayConfig(files_provider_upload_ttl_hours=6)

    await _file_ids(_copies_service(monkeypatch, copies=_Copies(), client=client, config=config))

    assert client.uploads[0]["expires_in"] == 6 * 3600
    assert client.uploads[0]["filename"] == "report.csv"


@pytest.mark.asyncio
async def test_the_row_keeps_an_earlier_expiry_the_provider_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider is free to hold a copy for less time, and the row follows what it said."""
    copies = _Copies()
    reported = datetime.now(UTC) + timedelta(minutes=30)
    client = _Client(FileMetadata(id="file_new", expires_at=reported))

    await _file_ids(_copies_service(monkeypatch, copies=copies, client=client))

    assert copies.confirmed[0].expires_at == reported
    assert client.discarded == []


@pytest.mark.asyncio
async def test_a_deployment_that_makes_no_copies_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store()
    copies = _Copies()
    config = GatewayConfig(files_provider_upload_enabled=False)

    with pytest.raises(ProviderUploadDisabledError):
        await _file_ids(_copies_service(monkeypatch, copies=copies, store=store, config=config))

    assert store.reads == 0
    assert copies.events == []


@pytest.mark.asyncio
async def test_a_provider_that_will_not_take_the_copy_leaves_no_reservation(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies()
    client = _Client(error=ProviderUploadFailedError())

    with pytest.raises(ProviderUploadFailedError):
        await _file_ids(_copies_service(monkeypatch, copies=copies, client=client))

    assert copies.events == ["reserve", "cancel"]
    assert copies.rows == []
    assert client.closed


@pytest.mark.asyncio
async def test_a_reservation_reclaimed_before_it_is_confirmed_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies(confirms=False)
    client = _Client(FileMetadata(id="file_new"))

    with pytest.raises(ProviderUploadFailedError):
        await _file_ids(_copies_service(monkeypatch, copies=copies, client=client))

    assert client.discarded == ["file_new"], "the provider kept a copy nothing names"


@pytest.mark.asyncio
async def test_a_copy_never_outlives_the_files_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client(FileMetadata(id="file_new"))
    staged = replace(_STAGED, expires_at=datetime.now(UTC) + timedelta(hours=2))
    config = GatewayConfig(files_provider_upload_ttl_hours=48)

    await _file_ids(_copies_service(monkeypatch, copies=_Copies(), client=client, config=config), staged)

    asked = client.uploads[0]["expires_in"]
    assert 0 < asked <= 2 * 3600, "the copy was asked for more life than the file has"


@pytest.mark.asyncio
async def test_a_copy_capped_by_the_files_expiry_survives_a_slow_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider counts the life it was asked for from when it accepts the upload, not from when it was asked."""
    copies = _Copies()
    client = _Client(accept_delay=timedelta(seconds=5))
    staged = replace(_STAGED, expires_at=datetime.now(UTC) + timedelta(hours=2))
    config = GatewayConfig(files_provider_upload_ttl_hours=48)

    ids = await _file_ids(_copies_service(monkeypatch, copies=copies, client=client, config=config), staged)

    assert ids == {"file-1": "file_new"}
    assert client.discarded == []
    recorded = copies.confirmed[0].expires_at
    assert staged.expires_at is not None
    assert recorded is not None
    assert recorded <= staged.expires_at


@pytest.mark.asyncio
async def test_a_file_expiring_sooner_than_the_provider_will_hold_a_copy_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic's shortest storable life is an hour, so a copy of this file would outlive it."""
    store = _Store()
    copies = _Copies()
    staged = replace(_STAGED, expires_at=datetime.now(UTC) + timedelta(minutes=10))

    with pytest.raises(AttachedFileExpiresTooSoonError):
        await _file_ids(_copies_service(monkeypatch, copies=copies, store=store), staged)

    assert store.reads == 0
    assert copies.events == []


@pytest.mark.asyncio
async def test_one_file_expiring_too_soon_leaves_no_copy_of_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    copies = _Copies()
    soon = replace(_STAGED, file_id="file-2", expires_at=datetime.now(UTC) + timedelta(minutes=10))

    with pytest.raises(AttachedFileExpiresTooSoonError):
        await _file_ids(_copies_service(monkeypatch, copies=copies), _STAGED, soon)

    assert copies.events == []


@pytest.mark.asyncio
async def test_a_provider_holding_the_copy_too_long_has_it_taken_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otari cannot make a provider honor an expiry, so a copy that outlives the file is removed."""
    copies = _Copies()
    client = _Client(FileMetadata(id="file_new", expires_at=datetime.now(UTC) + timedelta(days=30)))
    staged = replace(_STAGED, expires_at=datetime.now(UTC) + timedelta(hours=2))

    with pytest.raises(ProviderUploadFailedError):
        await _file_ids(_copies_service(monkeypatch, copies=copies, client=client), staged)

    assert client.discarded == ["file_new"], "the copy was left at the provider"
    assert copies.events == ["reserve", "cancel"]
