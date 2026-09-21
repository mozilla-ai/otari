"""Tenant mutation listeners revoke Files in the caller's transaction."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel

from gateway.api.deps import get_attribution_user_service, get_org_provider_key_service, get_workspace_service
from gateway.models.api_keys import APIKey
from gateway.models.provider_files import ProviderFileBinding, ProviderFileOutputOperation
from gateway.models.provider_keys import OrgProviderKey, WorkspaceProviderKeyOverrideRequest
from gateway.models.tenancy import Organization, Workspace
from gateway.models.users import User
from gateway.repositories.tenancy.provider_file_repository import ProviderFileRepository
from gateway.services.provider_files.contracts import FileAccount, FileScope, OutputPrepare, PrepareUpload
from gateway.services.provider_files.lifecycle import ProviderFileService
from gateway.services.provider_files.outputs import ProviderFileOutputs
from gateway.services.secret_box import encrypt_secret
from gateway.services.tenancy.errors import TenancyNotFoundError, WorkspaceInUseError

from .test_org_provider_keys import _member, _workspace
from .test_provider_file_lifecycle import files_setup as files_setup
from .test_provider_file_lifecycle import metadata

pytestmark = pytest.mark.asyncio


async def _live_files(
    files: ProviderFileService, scope: FileScope, account: FileAccount
) -> tuple[uuid.UUID, uuid.UUID]:
    upload = await files.prepare(scope, account, PrepareUpload(operation_id=uuid.uuid4(), size_bytes=20))
    await files.finalize(scope, upload.id, metadata())
    output = await ProviderFileOutputs(files).prepare(
        scope,
        account,
        OutputPrepare(
            operation_id=uuid.uuid4(), request_id="request", attempt_id="attempt", generation_id=account.generation_id
        ),
    )
    return upload.id, output.id


async def _assert_states(db: AsyncSession, binding_id: uuid.UUID, output_id: uuid.UUID, *, revoked: bool) -> None:
    db.expire_all()
    binding = await db.get(ProviderFileBinding, binding_id)
    output = await db.get(ProviderFileOutputOperation, output_id)
    assert binding is not None and output is not None
    assert binding.state == ("pending_cleanup" if revoked else "active")
    assert output.state == ("revoked" if revoked else "active")


@pytest.mark.parametrize("fail", [False, True])
async def test_user_deletion_and_file_revocation_are_atomic(
    async_db: AsyncSession,
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    files, scope, account = files_setup
    binding_id, output_id = await _live_files(files, scope, account)
    key_id = str(uuid.uuid4())
    async_db.add(APIKey(id=key_id, key_hash=key_id, user_id=scope.user_id, workspace_id=scope.workspace_id))
    await async_db.commit()
    service = get_attribution_user_service(async_db)
    if fail:
        original = service.repo.soft_delete

        async def refuse(user: User, now: datetime) -> None:
            # Fail after both domains have written, before the only commit.
            await original(user, now)
            raise RuntimeError("injected failure")

        monkeypatch.setattr(service.repo, "soft_delete", refuse)
        with pytest.raises(RuntimeError, match="injected"):
            await service.delete(scope.user_id, scope.organization_id)
    else:
        await service.delete(scope.user_id, scope.organization_id)
    await _assert_states(async_db, binding_id, output_id, revoked=not fail)
    user = await async_db.get(User, scope.user_id)
    key = await async_db.get(APIKey, key_id)
    assert user is not None and key is not None
    assert (user.deleted_at is not None) is (not fail)
    assert key.is_active is fail


async def test_user_deletion_rechecks_organization_scope(
    async_db: AsyncSession,
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    files, scope, account = files_setup
    binding_id, output_id = await _live_files(files, scope, account)
    key_id = str(uuid.uuid4())
    async_db.add(APIKey(id=key_id, key_hash=key_id, user_id=scope.user_id, workspace_id=scope.workspace_id))
    await async_db.commit()
    with pytest.raises(TenancyNotFoundError):
        await get_attribution_user_service(async_db).delete(scope.user_id, uuid.uuid4())
    await _assert_states(async_db, binding_id, output_id, revoked=False)


@pytest.mark.parametrize("fail", [False, True])
async def test_workspace_deletion_and_file_revocation_are_atomic(
    async_db: AsyncSession,
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
) -> None:
    files, scope, account = files_setup
    binding_id, output_id = await _live_files(files, scope, account)
    organization = await async_db.get(Organization, scope.organization_id)
    assert organization is not None
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    await _workspace(async_db, organization, name="Survivor", owner=owner)
    await async_db.commit()
    service = get_workspace_service(async_db)
    if fail:
        monkeypatch.setattr(
            service.workspaces, "delete_workspace", AsyncMock(side_effect=IntegrityError("injected", {}, ValueError()))
        )
        with pytest.raises(WorkspaceInUseError):
            await service.delete_workspace(user=owner, workspace_id=scope.workspace_id)
    else:
        await service.delete_workspace(user=owner, workspace_id=scope.workspace_id)
    await _assert_states(async_db, binding_id, output_id, revoked=not fail)
    assert (await async_db.get(Workspace, scope.workspace_id) is not None) is fail


async def test_disabling_workspace_key_revokes_its_files(
    async_db: AsyncSession,
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
) -> None:
    files, scope, account = files_setup
    binding_id, output_id = await _live_files(files, scope, account)
    repo = ProviderFileRepository(async_db)
    generation = await repo.account(account.generation_id)
    organization = await async_db.get(Organization, scope.organization_id)
    assert generation is not None and organization is not None
    key_id = uuid.UUID(generation.credential_ref)
    async_db.add(
        OrgProviderKey(
            id=key_id,
            organization_id=scope.organization_id,
            provider="anthropic",
            name="Files",
            encrypted_api_key=encrypt_secret("key"),
        )
    )
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    await async_db.commit()
    await get_org_provider_key_service(async_db).set_workspace_override_for_user(
        user=owner,
        workspace_id=scope.workspace_id,
        key_id=key_id,
        request=WorkspaceProviderKeyOverrideRequest(disabled=True),
    )
    await _assert_states(async_db, binding_id, output_id, revoked=True)


@pytest.mark.parametrize(
    ("table_name", "column"),
    [
        ("provider_account_generations", "credential_source"),
        ("provider_account_generations", "status"),
        ("provider_file_bindings", "state"),
        ("provider_file_output_operations", "state"),
    ],
)
async def test_postgres_rejects_invalid_file_lifecycle_values(
    async_db: AsyncSession,
    files_setup: tuple[ProviderFileService, FileScope, FileAccount],
    table_name: str,
    column: str,
) -> None:
    files, scope, account = files_setup
    await _live_files(files, scope, account)
    table = SQLModel.metadata.tables[table_name]
    with pytest.raises(IntegrityError):
        await async_db.execute(update(table).values(**{column: "typo"}))
    await async_db.rollback()
