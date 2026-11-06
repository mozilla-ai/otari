"""The caller's organization's own OIDC sign-in connection.

Thin composition over `gateway.services.identity.OIDCConnectionService`: resolve the
caller's identity, call the service, and return its typed result. The role gate and
the domain errors the service raises carry their own statuses (see
`gateway.exceptions.identity_exceptions`), so nothing here catches them.

Scoped to ``/me`` for the reason `routes/organization_guardrails.py` is: a deployment
with no overlay has exactly one organization and the caller's identity already points
at it.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from gateway.api.deps import CurrentIdentity, OIDCConnectionServiceDep, get_config, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.schemas.identity import (
    OIDCConnectionPublic,
    OIDCConnectionRequest,
    OrganizationSsoEnforcementPublic,
    OrganizationSsoEnforcementRequest,
    OrganizationSsoPreflightPublic,
    OrganizationSsoPublic,
)
from gateway.services.oauth_service import redirect_uri as _oauth_redirect_uri

# Master key on the router, as the rest of the ``/api/v1/organizations/me`` surface
# declares it. What the caller may change is decided per request by the service's own
# role gate, which is why the deployment operator gate does not belong here.
router = APIRouter(
    prefix="/organizations/me/sso",
    tags=["organization-sso"],
    dependencies=[Depends(verify_master_key)],
)


def _settings(
    config: GatewayConfig,
    connection: OIDCConnectionPublic | None,
    enforcement: OrganizationSsoEnforcementPublic,
) -> OrganizationSsoPublic:
    return OrganizationSsoPublic(
        redirect_uri=_oauth_redirect_uri(config, "sso"), connection=connection, enforcement=enforcement
    )


@router.get("")
async def get_organization_sso(
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> OrganizationSsoPublic:
    """Read the caller's organization's own sign-in connection. Owners and admins only.

    ``connection`` is null until the organization saves one.
    """
    connection = await connections.get_for_user(user=identity)
    enforcement = await connections.get_enforcement_for_user(user=identity)
    return _settings(config, connection, enforcement)


@router.put("")
async def put_organization_sso(
    request: OIDCConnectionRequest,
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> OrganizationSsoPublic:
    """Discover the submitted issuer and save the caller's organization's own sign-in connection.

    Owners and admins only. Replaces any existing connection. The client secret is
    encrypted at rest and never served back.
    """
    connection = await connections.save_for_user(user=identity, request=request)
    enforcement = await connections.get_enforcement_for_user(user=identity)
    return _settings(config, connection, enforcement)


@router.post("/test")
async def test_organization_sso(
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> OrganizationSsoPublic:
    """Re-discover the caller's organization's own issuer and store the result. Owners and admins only.

    Refused with 404 when the organization has not saved a connection yet.
    """
    connection = await connections.test_for_user(user=identity)
    enforcement = await connections.get_enforcement_for_user(user=identity)
    return _settings(config, connection, enforcement)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization_sso(
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
) -> None:
    """Remove the caller's organization's own sign-in connection. Owners and admins only.

    Refused with 404 when the organization has not saved a connection yet, and with 400 while
    the organization still requires sign-in through it: turn that requirement off first.
    """
    await connections.delete_for_user(user=identity)


@router.get("/preflight")
async def get_organization_sso_preflight(
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
) -> OrganizationSsoPreflightPublic:
    """Report who requiring sign-in through the caller's organization's own connection would affect.

    Owners and admins only. Refused with 404 when the organization has not configured a connection yet.
    """
    return await connections.preflight_sso_enforcement_for_user(user=identity)


@router.put("/enforcement", status_code=status.HTTP_204_NO_CONTENT)
async def put_organization_sso_enforcement(
    request: OrganizationSsoEnforcementRequest,
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
) -> None:
    """Require, or stop requiring, sign-in through the caller's organization's own connection. Owners and admins only.

    Requiring sign-in is refused with 404 when the organization has not configured a connection yet,
    and with 400 before anybody has signed in through the connection, when it names no owner who keeps
    another way in and waives none, or when the named account is not one of the organization's own
    owners. Turning enforcement off has none of these preconditions and always succeeds.
    """
    await connections.set_sso_enforcement_for_user(
        user=identity,
        required=request.required,
        break_glass_user_id=request.break_glass_user_id,
        break_glass_waived=request.break_glass_waived,
    )
