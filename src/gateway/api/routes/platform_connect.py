"""Connect this deployment to otari.ai from the dashboard.

An operator starts an attempt, approves it on otari.ai, and the gateway stores
the key otari.ai issues as the ``otari-ai`` hosted provider. The wire contract
with otari.ai is ``docs/platform-connect-protocol.md``.

Gated as the hosted-providers surface is, on the router: the result is one of
the deployment's own credentials. No route returns the device code or the key.
"""

from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from gateway.api.deps import (
    CurrentIdentity,
    PlatformConnectServiceDep,
    get_config,
    require_deployment_operator_or_absent,
    verify_master_key,
)
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.exceptions.providers_exceptions import PlatformUnreachableError
from gateway.rate_limit import RateLimiter
from gateway.schemas.providers import PlatformConnectStartPublic, PlatformConnectStatusPublic

router = APIRouter(
    prefix="/platform/connect",
    tags=["platform-connect"],
    dependencies=[Depends(verify_master_key), Depends(require_deployment_operator_or_absent)],
)

_FALLBACK_GATEWAY_NAME = "Otari gateway"


def _gateway_name(request: Request, config: GatewayConfig) -> str:
    """What otari.ai shows the operator this gateway as: the host it is reached at."""
    host = urlsplit(config.public_base_url or "").hostname or request.url.hostname
    return host or _FALLBACK_GATEWAY_NAME


@router.post(
    "",
    responses={
        429: {"description": "Too many attempts. Retry-After gives the wait in seconds."},
        503: {"description": "otari.ai could not be reached."},
    },
)
async def start_platform_connect(
    request: Request,
    service: PlatformConnectServiceDep,
    identity: CurrentIdentity,
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> PlatformConnectStartPublic:
    """Ask otari.ai for a code to approve, and return the code and the page that approves it.

    This is the first request this deployment sends to otari.ai. While an
    attempt is pending it is returned as it stands, and otari.ai is not asked
    again. The provider a connection stores is kept to the caller's
    organization.
    """
    limiter: RateLimiter | None = getattr(request.app.state, "platform_connect_rate_limiter", None)
    if limiter is not None:
        limiter.check(str(identity.id))
    try:
        return await service.start(
            gateway_name=_gateway_name(request, config), organization_id=identity.active_organization_id
        )
    except PlatformUnreachableError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "otari.ai could not be reached. Try again in a moment."
        ) from None


@router.get("")
async def get_platform_connect(service: PlatformConnectServiceDep) -> PlatformConnectStatusPublic:
    """Where the connection stands: idle, pending, connected, expired or denied.

    While pending, a read asks otari.ai whether the code was approved, at most
    once per the interval otari.ai gave. On approval the ``otari-ai`` hosted
    provider is stored with the issued key, its models are listed, and the
    status reads connected.
    """
    return await service.status()


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_platform_connect(service: PlatformConnectServiceDep) -> Response:
    """Forget the current attempt. A stored ``otari-ai`` provider is left as it is."""
    await service.cancel()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


FEATURE = CoreFeature(
    name="platform_connect",
    surface=None,
    enabled=lambda config: config.platform_connect_available,
    routers=lambda config: (router,),
)
