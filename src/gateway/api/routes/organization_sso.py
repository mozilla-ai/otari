"""The caller's organization's own OIDC sign-in connection.

Thin composition over `gateway.services.identity.OIDCConnectionService`: resolve the
caller's identity, call the service, and return its typed result. The role gate and
the domain errors the service raises carry their own statuses (see
`gateway.exceptions.identity_exceptions`), so nothing here catches them.

Scoped to ``/me`` for the reason `routes/organization_guardrails.py` is: a deployment
with no overlay has exactly one organization and the caller's identity already points
at it.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from gateway.api.deps import CurrentIdentity, OIDCConnectionServiceDep, get_config, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.schemas.identity import OIDCConnectionPublic, OIDCConnectionRequest, OrganizationSsoPublic
from gateway.services.oauth_service import redirect_uri as _oauth_redirect_uri

# Master key on the router, as the rest of the ``/api/v1/organizations/me`` surface
# declares it. What the caller may change is decided per request by the service's own
# role gate, which is why the deployment operator gate does not belong here.
router = APIRouter(
    prefix="/organizations/me/sso",
    tags=["organization-sso"],
    dependencies=[Depends(verify_master_key)],
)


def _settings(config: GatewayConfig, connection: OIDCConnectionPublic | None) -> OrganizationSsoPublic:
    return OrganizationSsoPublic(redirect_uri=_oauth_redirect_uri(config, "sso"), connection=connection)


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
    return _settings(config, connection)


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
    return _settings(config, connection)


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
    return _settings(config, connection)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization_sso(
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
) -> None:
    """Remove the caller's organization's own sign-in connection. Owners and admins only.

    Refused with 404 when the organization has not saved a connection yet.
    """
    await connections.delete_for_user(user=identity)


@router.delete("/links/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization_sso_link(
    user_id: uuid.UUID,
    identity: CurrentIdentity,
    connections: OIDCConnectionServiceDep,
) -> None:
    """Remove a member's own sign-in link at the caller's organization's connection. Owners and admins only.

    Refused with 404 for a member with no such link, including a user ID outside the caller's
    organization.
    """
    await connections.unlink_member_for_user(user=identity, member_user_id=user_id)
