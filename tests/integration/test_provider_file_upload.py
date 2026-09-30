"""Integration tests for the copy an attached file gets at the provider that runs the code.

Anthropic's ``container_upload`` block names a file from Anthropic's own Files
API, so a request that asks Anthropic to run code over a stored upload has a
short-lived copy made there first. Anthropic's Files API is stubbed at the HTTP
level.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from any_llm.types.messages import MessageResponse, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from gateway.adapters.file_storage_adapter import LocalDirFileStore
from gateway.core.config import API_KEY_HEADER, API_ROOT
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.files_exceptions import ProviderCopyNotRecordedError
from gateway.models.files import FileObject, FileProviderCopy
from gateway.models.users import User
from gateway.repositories.files import FileProviderCopyRepository

_CODE_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
_MODEL = "anthropic:claude-sonnet-4-5"
_PROVIDER_FILE_ID = "file_011CqStubUpload"


@pytest.fixture
def tmp_file_store(client: TestClient, tmp_path: Path) -> None:
    cast(Any, client.app).state.file_store = LocalDirFileStore(str(tmp_path))


class _StubAnthropicFiles:
    """Anthropic's Files API, as far as uploading a copy needs it."""

    def __init__(self) -> None:
        self.uploads: list[bytes] = []
        self.accepting = True
        self.dropped: set[str] = set()

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "/v1/files/" in request.url.path:
            file_id = request.url.path.rsplit("/", 1)[-1]
            if file_id in self.dropped:
                return httpx.Response(
                    404, json={"type": "error", "error": {"type": "not_found_error", "message": "File not found"}}
                )
            return httpx.Response(200, json=self._metadata(file_id))
        if request.method != "POST" or not request.url.path.endswith("/v1/files"):
            return httpx.Response(404)
        if not self.accepting:
            return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "nope"}})
        self.uploads.append(request.content)
        return httpx.Response(200, json=self._metadata(_PROVIDER_FILE_ID))

    @staticmethod
    def _metadata(file_id: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "id": file_id,
            "type": "file",
            "filename": "data.csv",
            "mime_type": "text/csv",
            "size_bytes": 12,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(),
            "downloadable": False,
        }


@pytest.fixture
def anthropic_files(monkeypatch: pytest.MonkeyPatch) -> _StubAnthropicFiles:
    """A deployment credentialed for Anthropic by the SDK's own variable, whose Files API is stubbed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    stub = _StubAnthropicFiles()
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
        model="claude-sonnet-4-5",
        content=[TextBlock(type="text", text="ok", citations=None)],
        stop_reason=cast(Any, "end_turn"),
        stop_sequence=None,
        usage=cast(Any, {"input_tokens": 3, "output_tokens": 2}),
    )


def _upload_file(client: TestClient, headers: dict[str, str]) -> str:
    stored = client.post(
        f"{API_ROOT}/files",
        headers=headers,
        files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
    )
    assert stored.status_code == 200, stored.text
    return str(stored.json()["id"])


def _run(client: TestClient, headers: dict[str, str], file_id: str | None) -> tuple[Any, list[Any]]:
    """Post a request whose code Anthropic runs, and report what reached the provider."""
    forwarded: list[Any] = []

    async def fake_amessages(**kwargs: Any) -> MessageResponse:
        forwarded.append(kwargs.get("messages"))
        return _reply()

    body = {
        "model": _MODEL,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "container_upload", "file_id": file_id}]
                if file_id is not None
                else "Compute 2 + 2.",
            }
        ],
        "max_tokens": 100,
        "tools": [_CODE_TOOL],
    }
    with patch("gateway.api.routes.messages.amessages", new=fake_amessages):
        response = client.post(f"{API_ROOT}/messages", json=body, headers=headers)
    return response, forwarded


def test_an_attached_file_reaches_the_providers_container(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 200, response.text
    assert forwarded[0][0]["content"][0] == {"type": "container_upload", "file_id": _PROVIDER_FILE_ID}
    assert len(anthropic_files.uploads) == 1
    stored = db_session.get(FileObject, file_id)
    assert stored is not None
    [row] = db_session.scalars(select(FileProviderCopy).where(FileProviderCopy.file_id == file_id)).all()
    assert row.pending_since is None, "the copy was never confirmed"
    assert row.credential_workspace_id == stored.workspace_id
    assert row.provider_file_id == _PROVIDER_FILE_ID
    assert row.expires_at is not None
    assert row.expires_at > datetime.now(UTC)


def test_a_second_request_reuses_the_copy(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)

    first, _ = _run(client, api_key_header, file_id)
    assert first.status_code == 200, first.text
    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 200, response.text
    assert forwarded[0][0]["content"][0]["file_id"] == _PROVIDER_FILE_ID
    assert len(anthropic_files.uploads) == 1, "the same file was uploaded twice"


def test_a_copy_the_provider_dropped_is_made_again(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)
    first, _ = _run(client, api_key_header, file_id)
    assert first.status_code == 200, first.text
    anthropic_files.dropped.add(_PROVIDER_FILE_ID)

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 200, response.text
    assert forwarded[0][0]["content"][0]["file_id"] == _PROVIDER_FILE_ID
    assert len(anthropic_files.uploads) == 2, "the dropped copy was not replaced"
    rows = db_session.scalars(select(FileProviderCopy).where(FileProviderCopy.file_id == file_id)).all()
    assert len(rows) == 1, "the dropped copy's row was kept beside its replacement"


def test_a_file_the_deployment_does_not_hold_is_refused(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    """A provider file ID of the caller's choosing must never reach the provider."""
    response, forwarded = _run(client, api_key_header, "file_011CqSomeoneElses")

    assert response.status_code == 400, response.text
    assert forwarded == []
    assert anthropic_files.uploads == []


def test_a_deployment_that_makes_no_copies_refuses(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_id = _upload_file(client, api_key_header)
    monkeypatch.setattr(
        cast(Any, client.app).state.config, "files_provider_upload_enabled", False, raising=True
    )

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 400, response.text
    assert forwarded == []
    assert anthropic_files.uploads == []


def test_a_provider_that_will_not_take_the_copy_refuses(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
) -> None:
    file_id = _upload_file(client, api_key_header)
    anthropic_files.accepting = False

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 502, response.text
    assert forwarded == []


def test_file_understanding_off_refuses_rather_than_forwarding_the_block(
    client: TestClient,
    api_key_header: dict[str, str],
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With normalization off nothing reads the block, so it must not reach the provider."""
    file_id = _upload_file(client, api_key_header)
    monkeypatch.setattr(cast(Any, client.app).state.config, "file_understanding_enabled", False, raising=True)

    response, forwarded = _run(client, api_key_header, file_id)

    assert response.status_code == 400, response.text
    assert forwarded == []
    assert anthropic_files.uploads == []


def test_file_understanding_off_still_runs_code_that_attaches_nothing(
    client: TestClient,
    api_key_header: dict[str, str],
    anthropic_files: _StubAnthropicFiles,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cast(Any, client.app).state.config, "file_understanding_enabled", False, raising=True)

    response, forwarded = _run(client, api_key_header, None)

    assert response.status_code == 200, response.text
    assert forwarded[0][0]["content"] == "Compute 2 + 2."
    assert anthropic_files.uploads == []


@pytest.fixture
def priced_model(client: TestClient, master_key_header: dict[str, str]) -> None:
    """A price for the model, so a request reserves a nonzero estimate against the budget."""
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": _MODEL, "input_price_per_million": 3.0, "output_price_per_million": 15.0},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def budgeted_key_header(client: TestClient, master_key_header: dict[str, str]) -> dict[str, str]:
    """A key for a user with a budget, so a request holds a reservation against it."""
    budget = client.post(
        f"{API_ROOT}/budgets", json={"max_budget": 100.0, "budget_duration_sec": 86400}, headers=master_key_header
    )
    assert budget.status_code == 200, budget.text
    user = client.post(
        f"{API_ROOT}/users",
        json={"user_id": "budgeted-user", "budget_id": budget.json()["budget_id"]},
        headers=master_key_header,
    )
    assert user.status_code == 200, user.text
    key = client.post(
        f"{API_ROOT}/keys", json={"key_name": "budgeted", "user_id": "budgeted-user"}, headers=master_key_header
    )
    assert key.status_code == 200, key.text
    return {API_KEY_HEADER: f"Bearer {key.json()['key']}"}


def _reserved(db_session: Session) -> Decimal:
    db_session.expire_all()
    return sum((user.reserved for user in db_session.scalars(select(User)).all()), Decimal("0"))


@pytest.mark.parametrize("stream", [False, True])
def test_a_refusal_releases_the_reservation_and_answers_in_the_anthropic_envelope(
    client: TestClient,
    budgeted_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    anthropic_files: _StubAnthropicFiles,
    priced_model: None,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    file_id = _upload_file(client, budgeted_key_header)
    monkeypatch.setattr(cast(Any, client.app).state.config, "files_provider_upload_enabled", False, raising=True)
    body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": [{"type": "container_upload", "file_id": file_id}]}],
        "max_tokens": 100,
        "tools": [_CODE_TOOL],
        "stream": stream,
    }

    provider = AsyncMock()
    with patch("gateway.api.routes.messages.amessages", new=provider):
        response = client.post(f"{API_ROOT}/messages", json=body, headers=budgeted_key_header)

    assert response.status_code == 400, response.text
    envelope = response.json()["detail"]
    assert envelope["type"] == "error"
    assert envelope["error"]["message"] == "This deployment does not upload attached files to a provider"
    provider.assert_not_awaited()
    assert _reserved(db_session) == Decimal("0")


def _pending(file_id: str, workspace_id: uuid.UUID, *, since: datetime | None = None) -> FileProviderCopy:
    return FileProviderCopy(
        id=uuid.uuid4(),
        file_id=file_id,
        account_identity="account-1",
        provider="anthropic",
        provider_instance="anthropic",
        credential_workspace_id=workspace_id,
        pending_since=since or datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_a_copy_of_a_file_that_does_not_exist_is_not_reserved(async_db: AsyncSession) -> None:
    uow = UnitOfWork(async_db)

    with pytest.raises(ProviderCopyNotRecordedError):
        async with uow:
            await FileProviderCopyRepository(uow).reserve(_pending("file-that-does-not-exist", uuid.uuid4()))


def _stored_file(db_session: Session, client: TestClient, headers: dict[str, str]) -> FileObject:
    stored = db_session.get(FileObject, _upload_file(client, headers))
    assert stored is not None
    return stored


@pytest.mark.asyncio
async def test_only_a_confirmed_copy_in_the_same_account_is_usable(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    async_db: AsyncSession,
) -> None:
    stored = _stored_file(db_session, client, api_key_header)
    uow = UnitOfWork(async_db)
    repository = FileProviderCopyRepository(uow)
    later = datetime.now(UTC) + timedelta(hours=2)
    confirmed, pending = _pending(stored.id, stored.workspace_id), _pending(stored.id, stored.workspace_id)

    async with uow:
        await repository.reserve(confirmed)
        await repository.reserve(pending)
    async with uow:
        assert await repository.confirm(confirmed.id, provider_file_id="file_a", expires_at=later)
    async with uow:
        found = await repository.usable([stored.id], "account-1", expiring_after=datetime.now(UTC))
        elsewhere = await repository.usable([stored.id], "account-2", expiring_after=datetime.now(UTC))

    assert found[stored.id].provider_file_id == "file_a"
    assert elsewhere == {}


@pytest.mark.asyncio
async def test_a_copy_is_confirmed_once(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    async_db: AsyncSession,
) -> None:
    stored = _stored_file(db_session, client, api_key_header)
    uow = UnitOfWork(async_db)
    repository = FileProviderCopyRepository(uow)
    copy = _pending(stored.id, stored.workspace_id)
    later = datetime.now(UTC) + timedelta(hours=1)

    async with uow:
        await repository.reserve(copy)
    async with uow:
        first = await repository.confirm(copy.id, provider_file_id="file_a", expires_at=later)
    async with uow:
        second = await repository.confirm(copy.id, provider_file_id="file_b", expires_at=later)

    assert (first, second) == (True, False)


@pytest.mark.asyncio
async def test_the_sweep_removes_only_stale_reservations(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    async_db: AsyncSession,
) -> None:
    stored = _stored_file(db_session, client, api_key_header)
    uow = UnitOfWork(async_db)
    repository = FileProviderCopyRepository(uow)
    now = datetime.now(UTC)
    stale, fresh = (
        _pending(stored.id, stored.workspace_id, since=now - timedelta(hours=3)),
        _pending(stored.id, stored.workspace_id, since=now),
    )

    async with uow:
        await repository.reserve(stale)
        await repository.reserve(fresh)
    async with uow:
        removed = await repository.remove_stale_pending(pending_before=now - timedelta(hours=1), limit=10)

    remaining = db_session.scalars(select(FileProviderCopy.id).where(FileProviderCopy.file_id == stored.id)).all()
    assert removed == 1
    assert remaining == [fresh.id]


@pytest.mark.asyncio
async def test_the_sweep_removes_only_expired_copies(
    client: TestClient,
    api_key_header: dict[str, str],
    db_session: Session,
    tmp_file_store: None,
    async_db: AsyncSession,
) -> None:
    stored = _stored_file(db_session, client, api_key_header)
    uow = UnitOfWork(async_db)
    repository = FileProviderCopyRepository(uow)
    now = datetime.now(UTC)
    expired, live, pending = (
        _pending(stored.id, stored.workspace_id),
        _pending(stored.id, stored.workspace_id),
        _pending(stored.id, stored.workspace_id),
    )

    async with uow:
        for copy in (expired, live, pending):
            await repository.reserve(copy)
    async with uow:
        await repository.confirm(expired.id, provider_file_id="file_old", expires_at=now - timedelta(minutes=1))
        await repository.confirm(live.id, provider_file_id="file_live", expires_at=now + timedelta(hours=1))
    async with uow:
        removed = await repository.remove_expired(expired_before=now, limit=10)

    remaining = set(db_session.scalars(select(FileProviderCopy.id).where(FileProviderCopy.file_id == stored.id)).all())
    assert removed == 1
    assert remaining == {live.id, pending.id}
