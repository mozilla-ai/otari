"""The api-keys service answers which workspace owns a key and which keys sit in a set of workspaces."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.unit_of_work import OutsideUnitOfWorkError, UnitOfWork
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import Workspace
from gateway.repositories.api_keys import ApiKeyRepository
from gateway.repositories.tenancy import OrganizationRepository, WorkspaceRepository
from gateway.services.api_keys import ApiKeyService

pytestmark = pytest.mark.asyncio


async def _two_workspaces_with_keys(db: AsyncSession) -> tuple[Workspace, Workspace]:
    organization = await OrganizationRepository(db).create_organization(
        name="Acme", slug="acme", created_by_user_id=None
    )
    workspaces = WorkspaceRepository(db)
    first = await workspaces.create_workspace(name="First", organization_id=organization.id, created_by_user_id=None)
    second = await workspaces.create_workspace(name="Second", organization_id=organization.id, created_by_user_id=None)
    for key_id, workspace in (("sk-first-a", first), ("sk-first-b", first), ("sk-second", second)):
        db.add(APIKey(id=key_id, key_hash=f"hash-{key_id}", workspace_id=workspace.id))
    await db.flush()
    return first, second


def _service(uow: UnitOfWork) -> ApiKeyService:
    return ApiKeyService(ApiKeyRepository(uow))


async def test_get_workspace_id_for_key_names_the_owning_workspace(async_db: AsyncSession) -> None:
    first, second = await _two_workspaces_with_keys(async_db)
    uow = UnitOfWork(async_db)

    async with uow:
        assert await _service(uow).get_workspace_id_for_key("sk-first-a") == first.id
        assert await _service(uow).get_workspace_id_for_key("sk-second") == second.id


async def test_get_workspace_id_for_key_is_none_for_an_unknown_key(async_db: AsyncSession) -> None:
    await _two_workspaces_with_keys(async_db)
    uow = UnitOfWork(async_db)

    async with uow:
        assert await _service(uow).get_workspace_id_for_key("sk-unknown") is None


async def test_get_key_ids_in_workspaces_returns_only_their_keys(async_db: AsyncSession) -> None:
    first, second = await _two_workspaces_with_keys(async_db)
    uow = UnitOfWork(async_db)

    async with uow:
        service = _service(uow)
        assert sorted(await service.get_key_ids_in_workspaces([first.id])) == ["sk-first-a", "sk-first-b"]
        assert sorted(await service.get_key_ids_in_workspaces([first.id, second.id])) == [
            "sk-first-a",
            "sk-first-b",
            "sk-second",
        ]
        assert await service.get_key_ids_in_workspaces([]) == []


async def test_lookups_raise_outside_a_unit_of_work_block(async_db: AsyncSession) -> None:
    first, _ = await _two_workspaces_with_keys(async_db)
    service = _service(UnitOfWork(async_db))

    with pytest.raises(OutsideUnitOfWorkError):
        await service.get_workspace_id_for_key("sk-first-a")
    with pytest.raises(OutsideUnitOfWorkError):
        await service.get_key_ids_in_workspaces([first.id])
