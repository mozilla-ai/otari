"""Workspace per-member budget defaults (standalone mode only).

A default is a workspace-level template for the per-member ``scoped_budgets``
ceiling; the materialized per-member rows live on the existing
``/v1/scoped-budgets`` surface. Thin composition over
`gateway.services.tenancy.workspace_budget_default_service`, following
`routes/workspaces.py`'s own shape (master-key on the router, plus the
caller's tenancy identity for the per-workspace role checks).
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.api.routes.organizations import Message
from gateway.services.tenancy.workspace_budget_default_service import (
    WorkspaceBudgetDefaultService,
    WorkspaceMemberBudgetPoliciesPublic,
    WorkspaceMemberBudgetPolicyCreate,
    WorkspaceMemberBudgetPolicyPublic,
    WorkspaceMemberBudgetPolicyUpdate,
)

# Auth is declared on the router, matching `routes/workspaces.py`: every
# handler here needs the master key, and a future one that forgot the
# decorator would be unauthenticated with nothing to notice.
router = APIRouter(
    prefix="/v1/workspaces/{workspace_id}/member-budget-policies",
    tags=["workspace-member-budget-policies"],
    dependencies=[Depends(verify_master_key)],
)


def get_workspace_budget_default_service(db: Annotated[AsyncSession, Depends(get_db)]) -> WorkspaceBudgetDefaultService:
    """Build the service on the request's session."""
    return WorkspaceBudgetDefaultService(db)


WorkspaceBudgetDefaultServiceDep = Annotated[
    WorkspaceBudgetDefaultService, Depends(get_workspace_budget_default_service)
]


@router.get("")
async def list_workspace_budget_defaults(
    service: WorkspaceBudgetDefaultServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> WorkspaceMemberBudgetPoliciesPublic:
    """List the budget defaults attached to a workspace. Any member may read it."""
    return await service.list_defaults(user=current_identity, workspace_id=workspace_id, skip=skip, limit=limit)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_workspace_budget_default(
    service: WorkspaceBudgetDefaultServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: WorkspaceMemberBudgetPolicyCreate,
) -> WorkspaceMemberBudgetPolicyPublic:
    """Create a budget default.

    Materializes it into a per-member ``scoped_budgets`` row for every
    existing active member of the workspace; a member who joins afterwards is
    materialized when they join.
    """
    return await service.create_default(user=current_identity, workspace_id=workspace_id, request=body)


@router.patch("/{default_id}")
async def update_workspace_budget_default(
    service: WorkspaceBudgetDefaultServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    default_id: str,
    body: WorkspaceMemberBudgetPolicyUpdate,
) -> WorkspaceMemberBudgetPolicyPublic:
    """Update a budget default's label or limit.

    Not retroactive: members already materialized from this default keep
    their existing ceiling; only a member materialized afterwards sees the
    new one.
    """
    return await service.update_default(
        user=current_identity,
        workspace_id=workspace_id,
        default_id=default_id,
        request=body,
    )


@router.delete("/{default_id}")
async def delete_workspace_budget_default(
    service: WorkspaceBudgetDefaultServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    default_id: str,
) -> Message:
    """Delete a budget default.

    The per-member ``scoped_budgets`` rows it already materialized are kept;
    a member joining afterwards no longer gets one from it.
    """
    await service.delete_default(user=current_identity, workspace_id=workspace_id, default_id=default_id)
    return Message(message="Workspace budget default deleted")
