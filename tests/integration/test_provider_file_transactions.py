"""Files transaction boundaries preserve revocation without partially replacing secrets."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.provider_files import ProviderFileOutputOperation
from gateway.models.provider_keys import OrgProviderKey, OrgProviderKeyUpdateRequest
from gateway.models.tenancy import Organization
from gateway.models.users import User
from gateway.repositories.tenancy.provider_file_repository import ProviderFileRepository
from gateway.services.provider_files.accounts import FileAccountResolver
from gateway.services.provider_files.cleanup import ProviderFileCleanup
from gateway.services.provider_files.contracts import (
    FileAccount,
    FileScope,
    FilesError,
    LeaseResult,
    OutputPrepare,
    PrepareUpload,
)
from gateway.services.provider_files.lifecycle import ProviderFileService
from gateway.services.provider_files.outputs import ProviderFileOutputs
from gateway.services.secret_box import encrypt_secret
from gateway.services.tenancy.errors import OrgProviderKeyAlreadyExistsError, TenancyConflictError
from gateway.services.tenancy.org_provider_key_service import OrgProviderKeyService

from .test_org_provider_keys import _member
from .test_provider_file_lifecycle import files_setup as files_setup
from .test_provider_file_lifecycle import metadata

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("action", ["replace", "restore", "delete"])
async def test_refused_secret_release_commits_revocation(
    async_db: AsyncSession, files_setup: tuple[ProviderFileService, FileScope, FileAccount], action: str
) -> None:
    files, scope, account = files_setup
    repo = ProviderFileRepository(async_db)
    organization = await async_db.get(Organization, scope.organization_id)
    assert organization is not None
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    generation = await repo.account(account.generation_id)
    assert generation is not None
    key_id = uuid.UUID(generation.credential_ref)
    secret = encrypt_secret("original-credential")
    await repo.save(
        OrgProviderKey(
            id=key_id,
            organization_id=scope.organization_id,
            provider="anthropic",
            name="Files",
            encrypted_api_key=secret,
            archived_at=datetime.now(UTC) if action != "replace" else None,
        )
    )
    await async_db.commit()
    operation = await files.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    await files.finalize(scope, operation.id, metadata())
    keys = OrgProviderKeyService(async_db)
    with pytest.raises(TenancyConflictError, match="cleanup must finish"):
        if action == "replace":
            await keys.update_key_for_user(
                user=owner, key_id=key_id, request=OrgProviderKeyUpdateRequest(api_key="replacement")
            )
        elif action == "restore":
            await keys.restore_key_for_user(user=owner, key_id=key_id)
        else:
            await keys.delete_key_for_user(user=owner, key_id=key_id)
    await async_db.rollback()
    stored_key = await repo.provider_key(key_id)
    stored_generation = await repo.account(account.generation_id)
    binding = await repo.get(operation.id)
    assert stored_key is not None and stored_key.encrypted_api_key == secret
    assert (stored_key.archived_at is not None) == (action != "replace")
    assert stored_generation is not None and stored_generation.status == "retiring"
    assert binding is not None and binding.state == "pending_cleanup"


async def test_secret_update_failure_rolls_back_retirement(
    async_db: AsyncSession,
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, scope, account = files_setup
    repo = ProviderFileRepository(async_db)
    organization = await async_db.get(Organization, scope.organization_id)
    assert organization is not None
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    generation = await repo.account(account.generation_id)
    assert generation is not None
    key_id = uuid.UUID(generation.credential_ref)
    secret = encrypt_secret("original-credential")
    await repo.save(
        OrgProviderKey(
            id=key_id,
            organization_id=scope.organization_id,
            provider="anthropic",
            name="Files",
            encrypted_api_key=secret,
        )
    )
    await async_db.commit()
    keys = OrgProviderKeyService(async_db)
    monkeypatch.setattr(
        keys.keys, "update_key", AsyncMock(side_effect=IntegrityError("injected", {}, ValueError("injected conflict")))
    )
    with pytest.raises(OrgProviderKeyAlreadyExistsError):
        await keys.update_key_for_user(
            user=owner, key_id=key_id, request=OrgProviderKeyUpdateRequest(api_key="replacement")
        )
    stored_generation = await repo.account(account.generation_id)
    stored_key = await repo.provider_key(key_id)
    assert stored_generation is not None and stored_generation.status == "active"
    assert stored_key is not None and stored_key.encrypted_api_key == secret


async def test_account_resolution_rechecks_revoked_scope(
    async_db: AsyncSession, files_setup: tuple[ProviderFileService, FileScope, FileAccount]
) -> None:
    files, scope, account = files_setup
    repo = ProviderFileRepository(async_db)
    generation = await repo.account(account.generation_id)
    assert generation is not None
    await repo.save(
        OrgProviderKey(
            id=uuid.UUID(generation.credential_ref),
            organization_id=scope.organization_id,
            provider="anthropic",
            name="Files",
            encrypted_api_key=encrypt_secret("original-credential"),
        )
    )
    user = await async_db.get(User, scope.user_id)
    assert user is not None
    user.deleted_at = datetime.now(UTC)
    await async_db.commit()
    resolver = FileAccountResolver(files.uow)
    with pytest.raises(FilesError, match="unavailable"):
        await resolver.resolve(scope, account.generation_id)
    cleanup = await resolver.resolve(scope, account.generation_id, cleanup=True)
    assert cleanup.generation_id == account.generation_id


async def test_rejected_output_registration_commits_cleanup(
    async_db: AsyncSession, files_setup: tuple[ProviderFileService, FileScope, FileAccount]
) -> None:
    service, scope, account = files_setup
    outputs = ProviderFileOutputs(service)
    operation = await outputs.prepare(
        scope,
        account,
        OutputPrepare(
            operation_id=uuid.uuid4(), request_id="request", attempt_id="attempt", generation_id=account.generation_id
        ),
    )
    repo = ProviderFileRepository(async_db)
    generation = await repo.account(account.generation_id)
    assert generation is not None
    generation.status = "retiring"
    await async_db.commit()
    with pytest.raises(FilesError, match="revoked"):
        await outputs.register(scope, operation.id, metadata("file_late"))
    await async_db.rollback()
    binding = await repo.by_provider_id(account.generation_id, "file_late")
    assert binding is not None and binding.state == "pending_cleanup"


async def _blocked_replacement(
    async_db: AsyncSession, files: ProviderFileService, scope: FileScope, account: FileAccount
) -> uuid.UUID:
    """Refuse a secret replacement while the account is busy; return the key id."""
    repo = ProviderFileRepository(async_db)
    organization = await async_db.get(Organization, scope.organization_id)
    assert organization is not None
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    generation = await repo.account(account.generation_id)
    assert generation is not None
    key_id = uuid.UUID(generation.credential_ref)
    await repo.save(
        OrgProviderKey(
            id=key_id,
            organization_id=scope.organization_id,
            provider="anthropic",
            name="Files",
            encrypted_api_key=encrypt_secret("original-credential"),
        )
    )
    await async_db.commit()
    with pytest.raises(TenancyConflictError, match="cleanup must finish"):
        await OrgProviderKeyService(async_db).update_key_for_user(
            user=owner, key_id=key_id, request=OrgProviderKeyUpdateRequest(api_key="replacement")
        )
    await async_db.rollback()
    with pytest.raises(FilesError, match="retiring"):
        await FileAccountResolver(UnitOfWork(async_db)).select_byo(scope)
    return key_id


async def test_blocked_retirement_finalizes_when_cleanup_completes(
    async_db: AsyncSession, files_setup: tuple[ProviderFileService, FileScope, FileAccount]
) -> None:
    files, scope, account = files_setup
    operation = await files.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    await files.finalize(scope, operation.id, metadata())
    await _blocked_replacement(async_db, files, scope, account)

    async def resolve(generation_id: uuid.UUID) -> FileAccount:
        return account

    cleanup = ProviderFileCleanup(files)
    lease = await cleanup.claim(scope.organization_id, scope.gateway_id, resolve_account=resolve)
    assert lease is not None and len(lease.items) == 1
    await cleanup.complete(
        scope.organization_id, scope.gateway_id, lease.id, LeaseResult(token=lease.token, results={operation.id: True})
    )
    retired = await ProviderFileRepository(async_db).account(account.generation_id)
    assert retired is not None and retired.status == "retired" and retired.retired_at is not None
    selected = await FileAccountResolver(UnitOfWork(async_db)).select_byo(scope)
    assert selected is not None and selected.generation_id != account.generation_id


async def test_blocked_retirement_finalizes_on_selection_after_outputs_expire(
    async_db: AsyncSession, files_setup: tuple[ProviderFileService, FileScope, FileAccount]
) -> None:
    """An account busy only with an output operation never gets a cleanup lease, so selection finishes it."""
    files, scope, account = files_setup
    operation = await ProviderFileOutputs(files).prepare(
        scope,
        account,
        OutputPrepare(
            operation_id=uuid.uuid4(), request_id="request", attempt_id="attempt", generation_id=account.generation_id
        ),
    )
    await _blocked_replacement(async_db, files, scope, account)
    row = await async_db.get(ProviderFileOutputOperation, operation.id)
    assert row is not None
    row.deadline = datetime.now(UTC) - timedelta(seconds=1)
    await async_db.commit()
    selected = await FileAccountResolver(UnitOfWork(async_db)).select_byo(scope)
    assert selected is not None and selected.generation_id != account.generation_id
    retired = await ProviderFileRepository(async_db).account(account.generation_id)
    assert retired is not None and retired.status == "retired"
