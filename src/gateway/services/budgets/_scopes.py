import uuid
from typing import assert_never

from gateway.models.budgets import ScopeType
from gateway.repositories.budgets import ScopeIdSets
from gateway.services.api_keys import ApiKeyService
from gateway.services.tenancy.organization_service import OrganizationService


def _uuid_or_none(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


class ScopeOwnership:
    """Which organization a budget scope belongs to, answered by the domains that own the scopes."""

    def __init__(self, organizations: OrganizationService, api_keys: ApiKeyService) -> None:
        self._organizations = organizations
        self._api_keys = api_keys

    async def _get_organization_id_for_workspace(self, workspace_id: uuid.UUID | None) -> uuid.UUID | None:
        if workspace_id is None:
            return None
        return await self._organizations.get_organization_id_for_workspace(workspace_id)

    async def get_organization_id_for(self, scope_type: ScopeType, scope_id: str) -> uuid.UUID | None:
        """Return the ID of the organization a scope sits in, or None when the scope resolves to nothing.

        A scope ID that is not a UUID where one is required resolves to nothing rather than raising,
        so a typo and another tenant's row are one answer.
        """
        match scope_type:
            case "organization":
                organization_id = _uuid_or_none(scope_id)
                if organization_id is None or not await self._organizations.has_organization(organization_id):
                    return None
                return organization_id
            case "workspace":
                return await self._get_organization_id_for_workspace(_uuid_or_none(scope_id))
            case "workspace_member":
                member_id = _uuid_or_none(scope_id)
                if member_id is None:
                    return None
                workspace_id = await self._organizations.get_workspace_id_for_workspace_member(member_id)
                return await self._get_organization_id_for_workspace(workspace_id)
            case "org_member":
                member_id = _uuid_or_none(scope_id)
                if member_id is None:
                    return None
                return await self._organizations.get_organization_id_for_organization_member(member_id)
            case "api_token":
                workspace_id = await self._api_keys.get_workspace_id_for_key(scope_id)
                return await self._get_organization_id_for_workspace(workspace_id)
            case _:
                assert_never(scope_type)

    async def get_scope_ids_in(self, organization_id: uuid.UUID) -> ScopeIdSets:
        """Return the IDs of every scope of every kind inside an organization."""
        workspace_ids = await self._organizations.get_workspace_ids_in_organization(organization_id)
        organization_member_ids = await self._organizations.get_organization_member_ids(organization_id)
        workspace_member_ids = await self._organizations.get_workspace_member_ids_in_organization(organization_id)
        api_key_ids = await self._api_keys.get_key_ids_in_workspaces(workspace_ids)
        return ScopeIdSets(
            organization_ids=(str(organization_id),),
            workspace_ids=tuple(str(value) for value in workspace_ids),
            organization_member_ids=tuple(str(value) for value in organization_member_ids),
            workspace_member_ids=tuple(str(value) for value in workspace_member_ids),
            api_key_ids=tuple(api_key_ids),
        )


async def lock_workspace_for_scope(organizations: OrganizationService, scope_type: ScopeType, scope_id: str) -> None:
    """Take the row lock on the workspace a workspace or membership scope sits in.

    The lock is held for the rest of the transaction, so a check of the scope and a write that
    follows it cannot straddle that workspace's deletion.
    Any other scope type, and any ID naming no workspace, locks nothing, and the set locked here
    must stay the set a workspace's deletion sweeps.

    Precondition: ``organizations`` must run in the transaction the following write commits in.
    """
    workspace_id: uuid.UUID | None
    match scope_type:
        case "workspace":
            workspace_id = _uuid_or_none(scope_id)
        case "workspace_member":
            member_id = _uuid_or_none(scope_id)
            workspace_id = (
                None if member_id is None else await organizations.get_workspace_id_for_workspace_member(member_id)
            )
        case "organization" | "org_member" | "api_token":
            return
        case _:
            assert_never(scope_type)
    if workspace_id is not None:
        await organizations.lock_workspace(workspace_id)


__all__ = ["ScopeOwnership", "lock_workspace_for_scope"]
