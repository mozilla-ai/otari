"""Which tenancy identity a request acts as, and whether it may act deployment-wide."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    MembershipListenerDep,
    UnitOfWorkDep,
)
from gateway.api.deps.auth import (
    get_session_identity,
    record_auth_failure,
    verify_master_key,
)
from gateway.core.database import get_db
from gateway.exceptions.identity_exceptions import DeploymentAdministrationUnavailableError
from gateway.models.tenancy import User as TenancyUser
from gateway.services.tenancy.deployment_user_service import DeploymentAdministrationAccess
from gateway.services.tenancy.provisioning_service import ensure_bootstrap_identity


async def get_current_identity(
    db: Annotated[AsyncSession, Depends(get_db)],
    session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)],
    _master_key: Annotated[str | None, Depends(verify_master_key)],
    uow: UnitOfWorkDep,
    membership_listener: MembershipListenerDep,
) -> TenancyUser:
    """Resolve the tenancy identity acting on this request.

    A dashboard session names the identity it was minted for, so a
    cookie-authenticated request resolves that identity and, through its
    ``active_organization_id``, the organization it is acting in. No second
    lookup: ``verify_master_key`` already resolved it through the same cached
    dependency to decide the cookie authenticates at all.

    A header master key names nobody, so it falls back to the deployment's
    bootstrap operator, provisioned on first use (otari-ai#1716 option A, see
    `gateway.services.tenancy.provisioning_service`). That is also the identity
    master-key sign-in binds a session to, so both credentials resolve to the
    same operator; the difference matters once the per-user sign-in flows mint
    sessions for other identities.

    Depending on ``verify_master_key`` keeps the credential check exactly where
    the rest of the management API has it, and keeps a request with no credential
    at all from provisioning anything.
    """
    if session_identity is not None:
        return session_identity
    return await ensure_bootstrap_identity(
        db,
        uow=uow,
        membership_listener=membership_listener,
    )


CurrentIdentity = Annotated[TenancyUser, Depends(get_current_identity)]


async def require_deployment_operator_or_absent(
    db: Annotated[AsyncSession, Depends(get_db)],
    identity: Annotated[TenancyUser, Depends(get_current_identity)],
) -> None:
    """Refuse a non-operator with the 404 ``/api/v1/admin`` answers, so the surface does not confirm it exists.

    The same predicate as :func:`require_deployment_operator`, with the other
    status: a deployment-wide surface that holds the deployment's own
    credentials is one worth not confirming to a signed-in member, who gets
    the answer an unentitled deployment gets. A header master key is the
    deployment credential itself, resolved to the bootstrap operator, so it
    passes. Declared on a router rather than per route, for the reason the 403
    gate gives.
    """
    if not await DeploymentAdministrationAccess(db).has_administration_access(identity):
        record_auth_failure("not_deployment_operator")
        raise DeploymentAdministrationUnavailableError
