"""Organizations answers which organization owns a workspace or a membership, and lists the IDs it holds."""

import uuid
from dataclasses import dataclass

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import Organization, OrganizationMember, User, Workspace, WorkspaceMember
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.organization_service import OrganizationService

from .tenancy_helpers import create_member, create_organization, create_workspace

pytestmark = pytest.mark.asyncio


@dataclass(frozen=True)
class _Tenant:
    organization: Organization
    owner: User
    workspace: Workspace
    organization_member: OrganizationMember
    workspace_member: WorkspaceMember


async def _tenant(db: AsyncSession, slug: str) -> _Tenant:
    organization = await create_organization(db, slug=slug)
    owner = await create_member(db, organization, role="owner", full_name=f"{slug} owner")
    workspace = await create_workspace(db, organization, name=f"{slug} workspace", owner=owner)
    organization_member = await OrganizationMemberRepository(db).get_by_organization_and_user(organization.id, owner.id)
    workspace_member = await WorkspaceMemberRepository(db).get_by_workspace_and_user(workspace.id, owner.id)
    assert organization_member is not None
    assert workspace_member is not None
    return _Tenant(organization, owner, workspace, organization_member, workspace_member)


async def _identity(db: AsyncSession, organization: Organization) -> User:
    return await UserRepository(db).create_local_identity(full_name="Leaver", active_organization_id=organization.id)


def _service(db: AsyncSession) -> OrganizationService:
    return OrganizationService(db, membership_listener=None)


async def test_a_workspace_resolves_to_its_organization(async_db: AsyncSession) -> None:
    acme = await _tenant(async_db, "acme")
    globex = await _tenant(async_db, "globex")
    service = _service(async_db)

    assert await service.get_organization_id_for_workspace(acme.workspace.id) == acme.organization.id
    assert await service.get_organization_id_for_workspace(globex.workspace.id) == globex.organization.id
    assert await service.get_organization_id_for_workspace(uuid.uuid4()) is None


async def test_a_deleted_workspace_resolves_to_no_organization(async_db: AsyncSession) -> None:
    """A deleted workspace has no organization, even after an earlier lookup found one."""
    acme = await _tenant(async_db, "acme")
    service = _service(async_db)
    workspace_id = acme.workspace.id
    assert await service.get_organization_id_for_workspace(workspace_id) == acme.organization.id

    await WorkspaceRepository(async_db).delete_workspace(acme.workspace)

    assert await service.get_organization_id_for_workspace(workspace_id) is None


async def test_an_organization_membership_resolves_to_its_organization(async_db: AsyncSession) -> None:
    acme = await _tenant(async_db, "acme")
    globex = await _tenant(async_db, "globex")
    service = _service(async_db)

    assert (
        await service.get_organization_id_for_organization_member(acme.organization_member.id)
        == acme.organization.id
    )
    assert (
        await service.get_organization_id_for_organization_member(globex.organization_member.id)
        == globex.organization.id
    )
    assert await service.get_organization_id_for_organization_member(uuid.uuid4()) is None


async def test_a_workspace_membership_resolves_to_its_workspace(async_db: AsyncSession) -> None:
    acme = await _tenant(async_db, "acme")
    globex = await _tenant(async_db, "globex")
    service = _service(async_db)

    assert await service.get_workspace_id_for_workspace_member(acme.workspace_member.id) == acme.workspace.id
    assert await service.get_workspace_id_for_workspace_member(globex.workspace_member.id) == globex.workspace.id
    assert await service.get_workspace_id_for_workspace_member(uuid.uuid4()) is None


async def test_only_an_existing_organization_is_found(async_db: AsyncSession) -> None:
    acme = await _tenant(async_db, "acme")
    service = _service(async_db)

    assert await service.has_organization(acme.organization.id) is True
    assert await service.has_organization(uuid.uuid4()) is False


async def test_an_organization_lists_only_its_own_workspaces(async_db: AsyncSession) -> None:
    acme = await _tenant(async_db, "acme")
    globex = await _tenant(async_db, "globex")
    second = await create_workspace(async_db, acme.organization, name="acme second", owner=acme.owner)
    service = _service(async_db)

    assert sorted(await service.get_workspace_ids_in_organization(acme.organization.id)) == sorted(
        [acme.workspace.id, second.id]
    )
    assert await service.get_workspace_ids_in_organization(globex.organization.id) == [globex.workspace.id]
    assert await service.get_workspace_ids_in_organization(uuid.uuid4()) == []


async def test_an_organization_lists_every_membership_whatever_its_status(async_db: AsyncSession) -> None:
    acme = await _tenant(async_db, "acme")
    globex = await _tenant(async_db, "globex")
    leaver = await _identity(async_db, acme.organization)
    suspended = await OrganizationMemberRepository(async_db).create_membership(
        organization_id=acme.organization.id,
        user_id=leaver.id,
        role="member",
        status="suspended",
    )
    service = _service(async_db)

    assert sorted(await service.get_organization_member_ids(acme.organization.id)) == sorted(
        [acme.organization_member.id, suspended.id]
    )
    assert await service.get_organization_member_ids(globex.organization.id) == [globex.organization_member.id]
    assert await service.get_organization_member_ids(uuid.uuid4()) == []


async def test_an_organization_lists_every_workspace_membership_whatever_its_status(async_db: AsyncSession) -> None:
    acme = await _tenant(async_db, "acme")
    globex = await _tenant(async_db, "globex")
    second = await create_workspace(async_db, acme.organization, name="acme second", owner=acme.owner)
    second_owner = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(second.id, acme.owner.id)
    assert second_owner is not None
    leaver = await _identity(async_db, acme.organization)
    suspended = await WorkspaceMemberRepository(async_db).create(
        workspace_id=acme.workspace.id,
        user_id=leaver.id,
        status="suspended",
    )
    service = _service(async_db)

    assert sorted(await service.get_workspace_member_ids_in_organization(acme.organization.id)) == sorted(
        [acme.workspace_member.id, second_owner.id, suspended.id]
    )
    assert await service.get_workspace_member_ids_in_organization(globex.organization.id) == [
        globex.workspace_member.id
    ]
    assert await service.get_workspace_member_ids_in_organization(uuid.uuid4()) == []
