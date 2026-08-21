"""The dashboard's first-request setup guide (standalone mode only).

Thin composition over
`gateway.services.tenancy.workspace_activation_service`, following
`routes/workspaces.py`'s shape (master key or a dashboard session on the router,
plus the caller's tenancy identity for the per-workspace role checks). Its own
module rather than three more handlers on `routes/workspaces.py`, matching
`routes/workspace_member_budget_policies.py`: a sub-resource of a workspace with
its own service gets its own file.

The platform's equivalent lives on the same paths
(`otari-ai` `backend/app/api/routes/workspaces.py`), with one rename. Minting
the key is ``POST .../activation/key`` here and ``.../activation/presentation``
there, because the platform stores the key's plaintext (encrypted) and re-shows
the *same* key for one browser session, which is what the presentation id in its
body identifies. Nothing here stores a plaintext, so there is no presentation to
identify: the call issues a key and rotates the previous one.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_config, get_db, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.core.config import GatewayConfig
from gateway.services.tenancy.workspace_activation_service import (
    ActivationApiKeyPublic,
    WorkspaceActivationPublic,
    WorkspaceActivationService,
)

router = APIRouter(
    prefix="/v1/workspaces/{workspace_id}/activation",
    tags=["workspace-activation"],
    dependencies=[Depends(verify_master_key)],
)


def get_workspace_activation_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> WorkspaceActivationService:
    """Build the service on the request's session and this app's config."""
    return WorkspaceActivationService(db, config)


WorkspaceActivationServiceDep = Annotated[WorkspaceActivationService, Depends(get_workspace_activation_service)]


@router.get("")
async def get_workspace_activation(
    service: WorkspaceActivationServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> WorkspaceActivationPublic:
    """Where a workspace stands on its first successful request.

    Readable by any member who can see the workspace. Whether the guide should
    actually be offered to this caller is ``experience_eligible``, which also
    answers false when the deployment has the flow turned off
    (``activation_guide``), so a dashboard left open stops offering it without
    needing to be reloaded.
    """
    return await service.get_status(user=current_identity, workspace_id=workspace_id)


@router.post("/key", status_code=status.HTTP_201_CREATED)
async def create_workspace_activation_key(
    service: WorkspaceActivationServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> ActivationApiKeyPublic:
    """Issue the workspace's setup API key, rotating the one the guide already issued.

    Workspace owners and admins only, like every other action that changes a
    workspace. The plaintext is returned once and never stored, so a reopened
    guide rotates the same key row rather than collecting a second one, and
    answers 409 once the workspace has activated or the guide was dismissed.
    """
    return await service.issue_api_key(user=current_identity, workspace_id=workspace_id)


@router.post("/dismiss")
async def dismiss_workspace_activation(
    service: WorkspaceActivationServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
) -> Message:
    """Retire the guide for this workspace. Permanent, and idempotent.

    Workspace owners and admins only. It retires the card and nothing else: the
    key the guide issued keeps working, because the operator asked for it and
    may well have pasted it somewhere already. Revoking one is the Keys page's
    job.
    """
    await service.dismiss(user=current_identity, workspace_id=workspace_id)
    return Message(message="Setup guide dismissed")
