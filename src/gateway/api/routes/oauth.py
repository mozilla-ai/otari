"""The token endpoint a workload exchanges its identity token at, as the Anthropic SDKs and Claude Code expect.

A client configured with ``ANTHROPIC_BASE_URL=<gateway>/api`` and workload identity posts its identity token here and
receives a short-lived access token it then presents in place of a key. Unauthenticated, since the identity token is
the credential, and mounted only where ``federation_issuers`` lists an issuer or ``federation_member_rules`` is on.
"""

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from gateway.api.deps import FederationTokenExchangeDep, record_auth_failure
from gateway.core.feature import CoreFeature
from gateway.schemas.api_keys import FederationTokenRequest, FederationTokenResponse, OAuthErrorResponse
from gateway.services.api_keys import FederationGrantError

router = APIRouter(prefix="/oauth", tags=["oauth"])

# RFC 6749 section 5.1: a response carrying a token must not be cached.
_NO_STORE = {"Cache-Control": "no-store", "Pragma": "no-cache"}


@router.post(
    "/token",
    response_model=FederationTokenResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "model": OAuthErrorResponse,
            "description": "invalid_request or unsupported_grant_type",
        },
        status.HTTP_401_UNAUTHORIZED: {"model": OAuthErrorResponse, "description": "invalid_grant"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": OAuthErrorResponse, "description": "temporarily_unavailable"},
    },
)
async def exchange_token(grant: FederationTokenRequest, exchange: FederationTokenExchangeDep) -> JSONResponse:
    """Exchange an identity token for an access token (the RFC 7523 ``jwt-bearer`` grant).

    The identity token must match the federation rule ``federation_rule_id`` names and verify against its issuer:
    one on ``federation_issuers`` for an operator's rule, the one the rule names for a member's. ``organization_id``,
    where set, must be the UUID of the rule's key's organization. ``workspace_id`` and ``service_account_id`` are
    accepted and ignored, so a client configured for Anthropic needs only its rule and organization changed. The
    access token authenticates as that key until ``expires_in`` runs out, the rule is deleted or the key stops passing
    its checks.

    Every refusal of the identity token answers 401 ``invalid_grant`` with the same description.
    """
    try:
        token, expires_in = await exchange.exchange(
            grant_type=grant.grant_type,
            assertion=grant.assertion,
            federation_rule_id=grant.federation_rule_id,
            organization_id=grant.organization_id,
        )
    except FederationGrantError as exc:
        record_auth_failure(exc.reason)
        body = OAuthErrorResponse(error=exc.error, error_description=exc.description)
        return JSONResponse(body.model_dump(), status_code=exc.status_code, headers=_NO_STORE)
    response = FederationTokenResponse(access_token=token, expires_in=expires_in)
    return JSONResponse(response.model_dump(), headers=_NO_STORE)


# Switched by ``federation_issuers`` and ``federation_member_rules``, startup settings: with no issuer trusted and no
# member able to name one, no identity token could be exchanged, so the route is absent rather than refusing.
FEATURE = CoreFeature(
    name="workload_identity_exchange",
    surface=None,
    enabled=lambda config: config.federation_enabled,
    routers=lambda config: (router,),
)
