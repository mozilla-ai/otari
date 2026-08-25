"""Deployment-wide account administration (standalone mode only).

Thin composition over `gateway.services.tenancy.deployment_user_service`, which
carries the whole of the reasoning: who counts as an operator, why a refusal is a
404, and which two changes are guarded against locking a deployment out of
itself.

The one prefix in the management API that is scoped to the *deployment* rather
than to an organization or a workspace, which is what it is for: every other
identity surface reads through a membership and so cannot see an account whose
memberships are all suspended. It is standalone-only like the rest of the
management plane; a hybrid gateway's accounts live on otari.ai.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.models.tenancy import (
    DeploymentAdminAccessPublic,
    DeploymentUserPublic,
    DeploymentUsersPublic,
    DeploymentUserUpdateRequest,
)
from gateway.services.tenancy.deployment_user_service import DeploymentUserService

# Auth on the router for the reason `routes/organizations.py` declares it there:
# every handler happens to take an identity today, and one that did not would be
# unauthenticated with nothing to notice. The operator gate is a second check on
# top, inside the service, because it is about who the caller *is* rather than
# whether they authenticated.
router = APIRouter(
    prefix="/v1/admin",
    tags=["admin"],
    dependencies=[Depends(verify_master_key)],
)


def get_deployment_user_service(db: Annotated[AsyncSession, Depends(get_db)]) -> DeploymentUserService:
    """Build the deployment user service on the request's session."""
    return DeploymentUserService(db)


DeploymentUserServiceDep = Annotated[DeploymentUserService, Depends(get_deployment_user_service)]


@router.get("/access")
async def get_administration_access(
    service: DeploymentUserServiceDep,
    current_identity: CurrentIdentity,
) -> DeploymentAdminAccessPublic:
    """Report whether the caller may use the deployment administration surface.

    The one endpoint here that answers 200 for everybody. The rest refuse a
    non-operator with 404 so they do not confirm they exist, which leaves a
    dashboard nothing to gate its navigation on but a failed request; this says
    the same thing without one. It publishes only the caller's own standing,
    which they could establish by trying an endpoint anyway.
    """
    return DeploymentAdminAccessPublic(granted=await service.has_administration_access(current_identity))


@router.get("/users")
async def list_deployment_users(
    service: DeploymentUserServiceDep,
    current_identity: CurrentIdentity,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000, description="Maximum number of records to return")] = 100,
) -> DeploymentUsersPublic:
    """List every account on this deployment, with the organizations each belongs to.

    Deployment-wide, so it is not the same list as ``GET /v1/organizations/me/members``:
    that one is the caller's organization roster and drops a suspended
    membership, while this one carries every identity at whatever standing,
    including one whose memberships are all suspended. Each row also reports when
    the account last signed in to the dashboard, and null there means never.
    """
    return await service.list_users(actor=current_identity, skip=skip, limit=limit)


@router.patch("/users/{user_id}")
async def update_deployment_user(
    service: DeploymentUserServiceDep,
    current_identity: CurrentIdentity,
    user_id: uuid.UUID,
    body: DeploymentUserUpdateRequest,
) -> DeploymentUserPublic:
    """Deactivate or reactivate an account, or change whether it may administer this deployment.

    Both fields are optional and omitting one leaves it alone; a body naming
    neither is refused rather than treated as a no-op. Deactivating also ends
    that account's dashboard sessions immediately, so a lost laptop stops
    working now rather than when its cookie is next presented.

    Two changes are refused to keep a deployment reachable: an operator may not
    deactivate their own account or drop their own operator access, and neither
    may be taken from the deployment's bootstrap operator, which is the identity
    master-key sign-in resolves to. Granting either is unguarded.
    """
    return await service.update_user(actor=current_identity, user_id=user_id, request=body)
