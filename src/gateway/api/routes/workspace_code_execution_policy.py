"""Per-workspace code-execution policy (standalone mode only).

The deployment-wide sandbox configuration (its URL, its purpose hint) stays on
``/api/v1/tool-settings``; this surface says which workspaces on that deployment may
use it and within which limits. Thin composition over the tools service's
`WorkspaceCodeExecutionPolicyService`, following
`routes/workspace_member_budget_policies.py`'s shape (master key on the router,
plus the caller's tenancy identity for the per-workspace role checks).
"""

import uuid

from fastapi import APIRouter, Depends

from gateway.api.deps import CurrentIdentity, WorkspaceCodeExecutionPolicyServiceDep, verify_master_key
from gateway.services.tools import WorkspaceCodeExecutionPolicyPublic, WorkspaceCodeExecutionPolicyUpdate

# Auth is declared on the router, matching `routes/workspace_member_budget_policies.py`:
# every handler here needs the master key, and a future one that forgot the
# decorator would be unauthenticated with nothing to notice.
router = APIRouter(
    prefix="/workspaces/{workspace_id}/code-execution-policy",
    tags=["workspace-code-execution-policy"],
    dependencies=[Depends(verify_master_key)],
)


@router.get("")
async def get_workspace_code_execution_policy(
    service: WorkspaceCodeExecutionPolicyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceCodeExecutionPolicyPublic:
    """Read a workspace's code-execution policy.

    Takes the same role as setting it (an organization owner/admin, or an
    owner/admin of this workspace), because the policy describes the
    workspace's security and billing posture rather than one member's
    allowance. A workspace with no policy answers with the unconfigured one
    (``configured: false``), which is the deployment's own behavior described
    in the same shape rather than a 404.
    """
    return await service.get_policy(user=current_identity, workspace_id=workspace_id)


@router.put("")
async def set_workspace_code_execution_policy(
    service: WorkspaceCodeExecutionPolicyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: WorkspaceCodeExecutionPolicyUpdate,
) -> WorkspaceCodeExecutionPolicyPublic:
    """Set a workspace's code-execution policy, replacing any existing one.

    An organization owner/admin, or an owner/admin of this workspace, may
    write it. The policy can only narrow what the deployment permits: turning
    code execution off for the workspace, lowering the loop and execution
    ceilings, and removing tool kinds from what the sandbox backend serves. It
    never turns a sandbox the deployment has not configured on, and ``image``
    may only name one the operator curated (``allowed_images`` on the response
    reports the set); anything else is refused with 400.
    """
    return await service.set_policy(user=current_identity, workspace_id=workspace_id, request=body)


@router.delete("")
async def clear_workspace_code_execution_policy(
    service: WorkspaceCodeExecutionPolicyServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceCodeExecutionPolicyPublic:
    """Drop a workspace's policy, returning it to the deployment's behavior.

    Idempotent: a workspace that has no policy is already in the state this
    asks for, so it answers with the unconfigured policy rather than a 404.
    """
    return await service.clear_policy(user=current_identity, workspace_id=workspace_id)
