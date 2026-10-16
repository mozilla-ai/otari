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

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import CurrentIdentity, IdentityServiceDep, get_db, verify_master_key
from gateway.core.surface import Surface
from gateway.exceptions.identity_exceptions import DeploymentAdministrationUnavailableError
from gateway.models.tenancy import (
    DeploymentAdminAccessPublic,
    DeploymentUserPublic,
    DeploymentUsersPublic,
    DeploymentUserUpdateRequest,
)
from gateway.schemas.identity import PasswordLinkPublic
from gateway.services.tenancy.deployment_user_service import DeploymentUserService

# Auth on the router for the reason `routes/organizations.py` declares it there:
# every handler happens to take an identity today, and one that did not would be
# unauthenticated with nothing to notice. The operator gate is a second check on
# top, inside the service, because it is about who the caller *is* rather than
# whether they authenticated.
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(verify_master_key)],
)

SURFACE = Surface("admin")


def get_deployment_user_service(db: Annotated[AsyncSession, Depends(get_db)]) -> DeploymentUserService:
    """Build the deployment user service on the request's session."""
    return DeploymentUserService(db)


DeploymentUserServiceDep = Annotated[DeploymentUserService, Depends(get_deployment_user_service)]


async def require_deployment_administration(
    service: DeploymentUserServiceDep, current_identity: CurrentIdentity
) -> None:
    """Answer 404 unless the caller operates this deployment, as every route the service gates does."""
    if not await service.has_administration_access(current_identity):
        raise DeploymentAdministrationUnavailableError


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
    search: Annotated[
        str | None,
        Query(
            max_length=200,
            description="Narrow to accounts whose name or email contains this text, case-insensitively.",
        ),
    ] = None,
) -> DeploymentUsersPublic:
    """List every account on this deployment, with the organizations each belongs to.

    Deployment-wide, so it is not the same list as ``GET /api/v1/organizations/me/members``:
    that one is the caller's organization roster and drops a suspended
    membership, while this one carries every identity at whatever standing,
    including one whose memberships are all suspended. Each row also reports when
    the account last signed in to the dashboard, and null there means never.
    ``search`` narrows the page and the count together.
    """
    return await service.list_users(actor=current_identity, skip=skip, limit=limit, search=search)


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


@router.post(
    "/users/{user_id}/password-link",
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_deployment_administration)],
)
async def issue_password_link(identities: IdentityServiceDep, user_id: uuid.UUID) -> PasswordLinkPublic:
    """Give an account a single-use link that sets its password, for someone who cannot sign in.

    Where this deployment can mail links, the link goes to the account's address and is not returned, the
    same as "Forgot your password?". Where it cannot, or the send fails, the response carries the link for
    you to pass on, which is how a member who forgot their password gets back in on a deployment without
    mail. A password chosen through a link that was not mailed records the address as vouched for rather
    than proven. A new link replaces any earlier reset link for the account.

    Refused with 409 for a deactivated account, one with no address, one that has no password and signs in
    with a provider or a passkey (its owner adds a password once signed in), and, where open_signup is
    disabled, an unverified address no organization still has as a member or invitee.
    """
    link = await identities.issue_password_link(user_id)
    return PasswordLinkPublic(mail_sent=link.mail_sent, link=link.link, expires_at=link.expires_at)
