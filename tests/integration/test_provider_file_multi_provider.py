"""Provider identity scopes lifecycle operations independently of native file IDs."""

import uuid
from typing import Literal

import pytest
from pydantic import SecretStr

from gateway.models.provider_files import ProviderAccountGeneration
from gateway.models.provider_keys import OrgProviderKey
from gateway.services.provider_files.accounts import FileAccountResolver
from gateway.services.provider_files.contracts import (
    FileAccount,
    FileListRequest,
    FileMetadata,
    FileScope,
    FilesError,
    PrepareUpload,
)
from gateway.services.provider_files.lifecycle import ProviderFileService
from gateway.services.secret_box import encrypt_secret

from .test_provider_file_lifecycle import files_setup as files_setup
from .test_provider_file_lifecycle import metadata

pytestmark = pytest.mark.asyncio


async def second_account(service: ProviderFileService, scope: FileScope) -> FileAccount:
    row = ProviderAccountGeneration(
        organization_id=scope.organization_id,
        provider="openai",
        credential_source="organization_key",
        credential_ref=str(uuid.uuid4()),
    )
    async with service.uow:
        await service.repo.save(row)
    return FileAccount(generation_id=row.id, provider="openai", api_key=SecretStr("test-only"))


async def put(service: ProviderFileService, scope: FileScope, account: FileAccount, data: FileMetadata) -> uuid.UUID:
    operation = await service.prepare(
        scope,
        account,
        PrepareUpload(
            operation_id=uuid.uuid4(),
            provider=account.provider,
            size_bytes=20,
        ),
    )
    await service.finalize(scope, operation.id, data)
    return operation.id


async def test_same_file_id_isolated_by_provider(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, anthropic = files_setup
    openai = await second_account(service, scope)
    await put(service, scope, anthropic, metadata("file_same"))
    await put(service, scope, openai, metadata("file_same").model_copy(update={"purpose": "user_data"}))
    page = await service.list_files(scope, FileListRequest(provider="openai"))
    assert len(page.data) == 1
    assert page.data[0].purpose == "user_data"
    assert await service.references(scope, ["file_same"], provider="openai") == openai.generation_id
    resolved = await service.resolve(scope, "file_same", "delete", openai, provider="openai")
    assert resolved.account == openai
    assert (await service.resolve(scope, "file_same", "metadata")).metadata.id == "file_same"


async def test_provider_is_bound_into_page_cursor(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, account = files_setup
    for index in range(2):
        await put(service, scope, account, metadata(f"file_{index}"))
    first = await service.list_files(scope, FileListRequest(limit=1))
    assert first.next_page is not None
    with pytest.raises(FilesError, match="Invalid file page"):
        await service.list_files(scope, FileListRequest(provider="openai", limit=1, page=first.next_page))


async def test_unknown_metadata_keeps_reserved_capacity(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, _ = files_setup
    account = await second_account(service, scope)
    data = FileMetadata(id="file_unknown", purpose="user_data")
    binding = await put(service, scope, account, data)
    async with service.uow:
        row = await service.repo.get(binding)
    assert row is not None and row.size_bytes == 20
    fetched = await service.resolve(scope, data.id, "download", account, provider="openai")
    assert fetched.metadata.downloadable is None
    assert fetched.metadata.size_bytes is None


async def test_prepare_rejects_provider_account_mismatch(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, account = files_setup
    with pytest.raises(FilesError):
        await service.prepare(
            scope, account, PrepareUpload(operation_id=uuid.uuid4(), provider="openai", size_bytes=20)
        )
    forged = account.model_copy(update={"provider": "openai"})
    with pytest.raises(FilesError):
        await service.prepare(scope, forged, PrepareUpload(operation_id=uuid.uuid4(), provider="openai", size_bytes=20))


async def test_byo_resolver_selects_requested_provider(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, _ = files_setup
    async with service.uow:
        for provider in ("anthropic", "openai"):
            await service.repo.save(
                OrgProviderKey(
                    organization_id=scope.organization_id,
                    provider=provider,
                    name=provider,
                    encrypted_api_key=encrypt_secret(f"{provider}-test-key"),
                )
            )
    selected = await FileAccountResolver(service.uow).select_byo(scope, provider="openai")
    assert selected is not None and selected.provider == "openai"
    assert selected.api_key.get_secret_value() == "openai-test-key"
    async with service.uow:
        generation = await service.repo.account(selected.generation_id)
    assert generation is not None and generation.provider == "openai"


async def test_native_list_cursor_and_purpose_filter(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, _ = files_setup
    account = await second_account(service, scope)
    for index in range(3):
        await put(
            service,
            scope,
            account,
            metadata(f"file_{index}").model_copy(update={"purpose": "user_data" if index != 1 else "batch"}),
        )
    first = await service.list_files(
        scope, FileListRequest(provider="openai", order="asc", limit=1, purpose="user_data")
    )
    assert [item.id for item in first.data] == ["file_0"]
    second = await service.list_files(
        scope, FileListRequest(provider="openai", order="asc", limit=1, purpose="user_data", after_id="file_0")
    )
    assert [item.id for item in second.data] == ["file_2"]


async def test_unknown_generated_size_reserves_maximum(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    from gateway.services.provider_files.contracts import OutputPrepare
    from gateway.services.provider_files.outputs import ProviderFileOutputs

    service, scope, _ = files_setup
    account = await second_account(service, scope)
    outputs = ProviderFileOutputs(service)
    operation = await outputs.prepare(
        scope,
        account,
        OutputPrepare(
            operation_id=uuid.uuid4(),
            request_id="request",
            attempt_id="attempt",
            generation_id=account.generation_id,
        ),
    )
    data = FileMetadata(id="file_generated", purpose="user_data")
    assert await outputs.register(scope, operation.id, data) == data
    async with service.uow:
        row = await service.repo.by_provider_id(account.generation_id, data.id)
    assert row is not None and row.size_bytes == service.max_bytes


async def test_native_order_uses_provider_timestamp(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    from datetime import UTC, datetime, timedelta

    service, scope, _ = files_setup
    account = await second_account(service, scope)
    now = datetime.now(UTC)
    await put(service, scope, account, metadata("file_a").model_copy(update={"created_at": now}))
    await put(service, scope, account, metadata("file_b").model_copy(update={"created_at": now - timedelta(seconds=5)}))
    cases: list[tuple[Literal["asc", "desc"], list[str]]] = [
        ("asc", ["file_b", "file_a"]),
        ("desc", ["file_a", "file_b"]),
    ]
    for order, expected in cases:
        result = await service.list_files(
            scope, FileListRequest(provider="openai", sort_by="provider_created_at", order=order)
        )
        assert [item.id for item in result.data] == expected
        after = await service.list_files(
            scope, FileListRequest(provider="openai", sort_by="provider_created_at", order=order, after_id=expected[0])
        )
        assert [item.id for item in after.data] == expected[1:]
