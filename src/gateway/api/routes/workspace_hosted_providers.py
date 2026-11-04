"""Per-workspace access to the deployment's hosted providers.

Thin composition over the providers service's `WorkspaceHostedAccessService`.
Whether the organization may use a hosted provider is the deployment's decision;
this surface says what one workspace keeps of it. Path-scoped by workspace, like
``org_provider_keys.workspace_router``, the same workspace's view of its
organization's own keys.
"""

import uuid

from fastapi import APIRouter, Depends

from gateway.api.deps import CurrentIdentity, WorkspaceHostedAccessServiceDep, verify_master_key
from gateway.schemas.providers import (
    WorkspaceHostedProviderAccessRequest,
    WorkspaceHostedProviderPublic,
    WorkspaceHostedProvidersPublic,
    WorkspaceProviderPolicyPublic,
    WorkspaceProviderPolicyRequest,
)

# Auth is declared on the router: every handler needs the master key, and a
# future one that forgot the decorator would be unauthenticated.
router = APIRouter(
    prefix="/workspaces/{workspace_id}/hosted-providers",
    tags=["workspace-hosted-providers"],
    dependencies=[Depends(verify_master_key)],
)

# What one workspace requires of every provider serving it, its organization's
# keys and configured instances as well as the hosted providers.
policy_router = APIRouter(
    prefix="/workspaces/{workspace_id}/provider-policy",
    tags=["workspace-hosted-providers"],
    dependencies=[Depends(verify_master_key)],
)


@router.get("")
async def list_workspace_hosted_providers(
    service: WorkspaceHostedAccessServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceHostedProvidersPublic:
    """Every hosted provider this workspace's organization may use, with what this workspace keeps of it.

    Any member of the workspace may read it.
    """
    return await service.list_for_workspace(user=current_identity, workspace_id=workspace_id)


@router.put("/{provider}")
async def set_workspace_hosted_provider(
    service: WorkspaceHostedAccessServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    provider: str,
    body: WorkspaceHostedProviderAccessRequest,
) -> WorkspaceHostedProviderPublic:
    """Switch a hosted provider off for this workspace, or narrow it to some of its models.

    An organization owner/admin, or an owner/admin of this workspace, may write
    it. It can only narrow what the organization may use: a model the
    deployment does not serve on the provider is refused with 400. Sending the
    inherited state (enabled, every model) is a reset.
    """
    return await service.set_access(user=current_identity, workspace_id=workspace_id, provider=provider, request=body)


@router.delete("/{provider}")
async def reset_workspace_hosted_provider(
    service: WorkspaceHostedAccessServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    provider: str,
) -> WorkspaceHostedProviderPublic:
    """Drop this workspace's rule for a hosted provider, so it inherits the provider whole. Idempotent."""
    return await service.reset_access(user=current_identity, workspace_id=workspace_id, provider=provider)


@policy_router.get("")
async def get_workspace_provider_policy(
    service: WorkspaceHostedAccessServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceProviderPolicyPublic:
    """What this workspace requires of every provider serving it. Any member of the workspace may read it."""
    return await service.get_policy(user=current_identity, workspace_id=workspace_id)


@policy_router.put("")
async def set_workspace_provider_policy(
    service: WorkspaceHostedAccessServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: WorkspaceProviderPolicyRequest,
) -> WorkspaceProviderPolicyPublic:
    """Require zero data retention of every provider serving this workspace, or stop requiring it.

    An organization owner/admin, or an owner/admin of this workspace, may write
    it. Required, a request is refused with 403 on an organization key or a
    hosted provider not declared zero data retention, and on any provider
    configured in config.yml or endpoint the workspace or its users own,
    neither of which can be declared.
    """
    return await service.set_policy(user=current_identity, workspace_id=workspace_id, request=body)
