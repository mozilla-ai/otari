"""Per-workspace code-execution policy (standalone mode only).

The deployment-wide sandbox configuration (its URL, its purpose hint) stays on
``/v1/tool-settings``; this surface says which workspaces on that deployment may
use it and within which limits. Thin composition over
`gateway.services.tenancy.workspace_code_execution_policy_service`, following
`routes/workspace_member_budget_policies.py`'s shape (master key on the router,
plus the caller's tenancy identity for the per-workspace role checks).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.core.env import otari_env
from gateway.services.tenancy.workspace_code_execution_policy_service import (
    WorkspaceCodeExecutionPolicyPublic,
    WorkspaceCodeExecutionPolicyService,
    WorkspaceCodeExecutionPolicyUpdate,
)

# Auth is declared on the router, matching `routes/workspace_member_budget_policies.py`:
# every handler here needs the master key, and a future one that forgot the
# decorator would be unauthenticated with nothing to notice.
router = APIRouter(
    prefix="/v1/workspaces/{workspace_id}/code-execution-policy",
    tags=["workspace-code-execution-policy"],
    dependencies=[Depends(verify_master_key)],
)


def get_workspace_code_execution_policy_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceCodeExecutionPolicyService:
    """Build the service on the request's session.

    The sandbox-presence check is the same one ``GET /v1/tools`` makes when it
    decides whether to advertise code execution, so the page and the discovery
    endpoint agree about whether this deployment can run any.

    ``allowed_images`` is the operator's curated image list, read from the
    deployment config here for the same reason: which images exist is a property
    of the running gateway, not of the workspace, and the service is handed the
    answer rather than reaching for the config itself.
    """
    return WorkspaceCodeExecutionPolicyService(
        db,
        sandbox_configured=bool(config.sandbox_url or otari_env("SANDBOX_URL")),
        allowed_images=config.pinnable_sandbox_images(),
    )


WorkspaceCodeExecutionPolicyServiceDep = Annotated[
    WorkspaceCodeExecutionPolicyService, Depends(get_workspace_code_execution_policy_service)
]


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
