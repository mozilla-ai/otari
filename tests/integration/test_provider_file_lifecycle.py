"""Provider file state survives failed finalization and tenant isolation attempts."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.provider_files import ProviderAccountGeneration
from gateway.models.users import User
from gateway.repositories.tenancy.organization_repository import OrganizationRepository
from gateway.repositories.tenancy.workspace_repository import WorkspaceRepository
from gateway.services.provider_files.contracts import (
    FileAccount,
    FileListRequest,
    FileMetadata,
    FileScope,
    FilesError,
    PrepareUpload,
)
from gateway.services.provider_files.lifecycle import ProviderFileService
from gateway.services.secret_box import generate_secret_key

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def files_setup(
    async_db: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProviderFileService, FileScope, FileAccount]:
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    organization = await OrganizationRepository(async_db).create_organization(
        name="Files", slug="files", created_by_user_id=None
    )
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Files", organization_id=organization.id, created_by_user_id=None
    )
    account = ProviderAccountGeneration(
        organization_id=organization.id, credential_source="organization_key", credential_ref=str(uuid.uuid4())
    )
    async_db.add_all([User(user_id="uploader"), User(user_id="other")])
    async_db.add(account)
    await async_db.commit()
    scope = FileScope(
        organization_id=organization.id, workspace_id=workspace.id, user_id="uploader", gateway_id="gateway"
    )
    service = ProviderFileService(async_db, max_bytes=1024, max_files=10, max_outstanding_bytes=10240)
    return service, scope, FileAccount(generation_id=account.id, api_key=SecretStr("test-key"))


def metadata(file_id: str = "file_uploaded") -> FileMetadata:
    return FileMetadata(
        id=file_id,
        filename="private.csv",
        mime_type="text/csv",
        size_bytes=10,
        created_at=datetime.now(UTC),
        downloadable=False,
    )


async def test_finalize_is_idempotent_and_metadata_encrypted(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, account = files_setup
    operation = await service.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    data = metadata()
    assert await service.finalize(scope, operation.id, data) == data
    assert await service.finalize(scope, operation.id, data) == data
    row = await service.repo.get(operation.id)
    assert row is not None and row.encrypted_metadata is not None
    assert "private.csv" not in row.encrypted_metadata
    with pytest.raises(FilesError, match="conflict"):
        await service.finalize(scope, operation.id, metadata("file_different"))


async def test_foreign_owner_and_workspace_hidden(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, account = files_setup
    operation = await service.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    await service.finalize(scope, operation.id, metadata())
    for foreign in [
        scope.model_copy(update={"user_id": "other"}),
        scope.model_copy(update={"workspace_id": uuid.uuid4()}),
    ]:
        with pytest.raises(FilesError) as error:
            await service.resolve(foreign, "file_uploaded", "metadata")
        assert error.value.status_code == 404
    result = await service.list_files(
        scope.model_copy(update={"user_id": "other"}), FileListRequest(ids=["file_uploaded"])
    )
    assert result.data == []


async def test_retired_upload_cannot_reactivate(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, account = files_setup
    operation = await service.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    generation = await service.repo.account(account.generation_id)
    assert generation is not None
    generation.status = "retiring"
    await service.db.commit()
    with pytest.raises(FilesError, match="revoked"):
        await service.finalize(scope, operation.id, metadata())
    row = await service.repo.get(operation.id)
    assert row is not None and row.state == "pending_cleanup" and row.provider_file_id == "file_uploaded"


async def test_delete_revokes_before_provider_and_retries_survive(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    service, scope, account = files_setup
    operation = await service.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    await service.finalize(scope, operation.id, metadata())
    resolved = await service.resolve(scope, "file_uploaded", "delete", account)
    assert resolved.cleanup_token is not None
    with pytest.raises(FilesError):
        await service.resolve(scope, "file_uploaded", "metadata")
    await service.cleanup_result(operation.id, scope.gateway_id, resolved.cleanup_token.get_secret_value(), False)
    row = await service.repo.get(operation.id)
    assert row is not None and row.state == "pending_cleanup" and row.cleanup_attempts == 1
    await service.cleanup_result(operation.id, scope.gateway_id, resolved.cleanup_token.get_secret_value(), True)
    assert row.state == "deleted"


async def test_expired_files_are_hidden(files_setup: tuple[ProviderFileService, FileScope, FileAccount]) -> None:
    service, scope, account = files_setup
    operation = await service.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    await service.finalize(scope, operation.id, metadata())
    row = await service.repo.get(operation.id)
    assert row is not None
    row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    await service.db.commit()
    with pytest.raises(FilesError):
        await service.references(scope, ["file_uploaded"])


async def test_output_only_registration_and_collision(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    from gateway.services.provider_files.contracts import OutputPrepare
    from gateway.services.provider_files.outputs import ProviderFileOutputs

    service, scope, account = files_setup
    outputs = ProviderFileOutputs(service)
    operation = await outputs.prepare(
        scope,
        account,
        OutputPrepare(
            operation_id=uuid.uuid4(), request_id="request", attempt_id="attempt", generation_id=account.generation_id
        ),
    )
    data = metadata("file_generated").model_copy(update={"downloadable": True})
    assert await outputs.register(scope, operation.id, data) == data
    assert await outputs.register(scope, operation.id, data) == data
    other_scope = scope.model_copy(update={"user_id": "other"})
    other = await outputs.prepare(
        other_scope,
        account,
        OutputPrepare(
            operation_id=uuid.uuid4(),
            request_id="other-request",
            attempt_id="other-attempt",
            generation_id=account.generation_id,
        ),
    )
    with pytest.raises(FilesError, match="ownership conflict"):
        await outputs.register(other_scope, other.id, data)
    with pytest.raises(FilesError, match="ownership conflict"):
        await outputs.abandon(other.id, scope.gateway_id, other.cleanup_token.get_secret_value(), data)
    resolved = await service.resolve(scope, data.id, "metadata")
    assert resolved.metadata.downloadable


async def test_output_cleanup_survives_user_revocation(
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    from gateway.services.provider_files.contracts import OutputPrepare
    from gateway.services.provider_files.outputs import ProviderFileOutputs

    service, scope, account = files_setup
    outputs = ProviderFileOutputs(service)
    operation = await outputs.prepare(
        scope,
        account,
        OutputPrepare(
            operation_id=uuid.uuid4(), request_id="request", attempt_id="attempt", generation_id=account.generation_id
        ),
    )
    await service.repo.revoke_user(scope.user_id, datetime.now(UTC))
    user = await service.db.get(User, scope.user_id)
    assert user is not None
    user.deleted_at = datetime.now(UTC)
    await service.db.commit()
    cleanup = await outputs.abandon(
        operation.id, scope.gateway_id, operation.cleanup_token.get_secret_value(), None, "file_late"
    )
    row = await service.repo.get(cleanup.operation_id)
    assert row is not None and row.state == "pending_cleanup" and row.provider_file_id == "file_late"
    await service.cleanup_result(row.id, scope.gateway_id, cleanup.cleanup_token.get_secret_value(), True)
    assert row.state == "deleted"


async def test_cursor_scope_and_snapshot(files_setup: tuple[ProviderFileService, FileScope, FileAccount]) -> None:
    service, scope, account = files_setup
    for index in range(3):
        operation = await service.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
        await service.finalize(scope, operation.id, metadata(f"file_{index}"))
    first = await service.list_files(scope, FileListRequest(limit=1))
    assert first.next_page is not None
    with pytest.raises(FilesError, match="Invalid file page"):
        await service.list_files(
            scope.model_copy(update={"user_id": "other"}), FileListRequest(limit=1, page=first.next_page)
        )
    second = await service.list_files(scope, FileListRequest(limit=1, page=first.next_page))
    assert first.data[0].id != second.data[0].id


async def test_cleanup_lease_fencing(files_setup: tuple[ProviderFileService, FileScope, FileAccount]) -> None:
    from gateway.services.provider_files.cleanup import ProviderFileCleanup
    from gateway.services.provider_files.contracts import LeaseResult

    service, scope, account = files_setup
    operation = await service.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    await service.finalize(scope, operation.id, metadata())
    await service.resolve(scope, "file_uploaded", "delete", account)
    cleanup = ProviderFileCleanup(service)

    async def resolve(generation: uuid.UUID) -> FileAccount:
        assert generation == account.generation_id
        return account

    lease = await cleanup.claim(scope.organization_id, scope.gateway_id, resolve_account=resolve)
    assert lease is not None and len(lease.items) == 1
    assert await cleanup.claim(scope.organization_id, scope.gateway_id, resolve_account=resolve) is None
    with pytest.raises(FilesError, match="lease unavailable"):
        await cleanup.complete(
            scope.organization_id,
            "foreign-gateway",
            lease.id,
            LeaseResult(token=lease.token, results={operation.id: True}),
        )
    await cleanup.complete(
        scope.organization_id, scope.gateway_id, lease.id, LeaseResult(token=lease.token, results={operation.id: True})
    )
    row = await service.repo.get(operation.id)
    assert row is not None and row.state == "deleted"
    assert not await service.repo.account_busy(account.generation_id, datetime.now(UTC))


async def test_postgres_file_migration_round_trip(postgres_url: str) -> None:
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    config.set_main_option("sqlalchemy.url", postgres_url)
    try:
        command.downgrade(config, "d5f8b2a4c6e9")
    finally:
        command.upgrade(config, "c3e5a7b9d1f4")
