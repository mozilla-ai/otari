"""Sign in through an organization's own OIDC connection.

Three public routes: a lookup by email domain, an authorization redirect by organization slug,
and a callback that exchanges a provider's authorization code for a dashboard session.

The lookup and the authorize routes each answer alike for an organization that does not exist and
one that exists but has no enabled connection, so neither can be read as a directory of which
organizations use Otari.

The callback requires an HttpOnly cookie scoped to this router's own path, binding it to the
browser that started the sign-in. It also checks the authorization response's issuer (RFC 9207)
against what the provider's own redirect carried: substituting this deployment's configured
issuer for it would make the check trivially true.
"""

from typing import Annotated

from apron_auth import AuthorizationCallback, StoredStateClient
from apron_auth.errors import AuthorizationResponseError, IssuerValidationError, NonceValidationError, StateError
from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, HTTPException, Request, Response, status

from gateway.api.deps import (
    GrowthSignalPortDep,
    IdentityServiceDep,
    MaintenanceModeDep,
    OIDCConnectionServiceDep,
    get_config,
    record_auth_failure,
)
from gateway.api.routes._helpers import database_error
from gateway.api.routes._public_auth import throttle_public_auth
from gateway.api.routes.auth_session import MAINTENANCE_MODE_REFUSAL
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.exceptions import TenancyError
from gateway.exceptions.identity_exceptions import (
    OAuthExchangeError,
    OAuthIdentityLinkConflictError,
    OAuthProviderUnavailableError,
    OAuthStateError,
    OIDCSignInUnavailableError,
    SignInBlockedError,
    TermsNotAcceptedError,
)
from gateway.log_config import logger
from gateway.models.identity import EmailAssertion, OIDCConnection, ProviderIdentity, oidc_identity_provider
from gateway.schemas.identity import (
    SsoAuthorizeResponse,
    SsoCallbackRequest,
    SsoLookupRequest,
    SsoLookupResponse,
    SsoSessionPublic,
)
from gateway.services.dashboard_session_service import apply_session_cookie, request_is_https
from gateway.services.oauth_service import (
    BrowserAuthorization,
    apply_flow_cookie,
    browser_token_for,
    drop_finished_flow,
    provider_refused_the_code,
    redirect_uri,
)
from gateway.services.tenancy import report_new_account

router = APIRouter(prefix="/auth/sso", tags=["auth"])

# This router's own flow cookie, scoped to its own path. A browser only attaches a cookie to a
# request under the path it was scoped to, so a cookie scoped elsewhere would never arrive here.
SSO_FLOW_COOKIE_NAME = "otari_sso_flow"
SSO_FLOW_COOKIE_PATH = f"{API_ROOT}/auth/sso"

SsoFlowCookie = Annotated[str | None, Cookie(alias=SSO_FLOW_COOKIE_NAME, include_in_schema=False)]


def _sso_redirect_uri(config: GatewayConfig) -> str:
    """This deployment's one fixed callback path for every organization's own connection.

    Every organization's connection registers and is sent back to this one path. The connection a
    given callback belongs to is resolved from the pending state's own ``state``, never from the
    path.
    """
    return redirect_uri(config, "sso")


@router.post("/lookup", response_model=SsoLookupResponse)
async def lookup_sso(
    body: SsoLookupRequest, request: Request, connections: OIDCConnectionServiceDep
) -> SsoLookupResponse:
    """Say whether an organization's own connection admits a sign-in at this address.

    An unknown domain, one no organization has proven, and a proven one whose organization holds
    no enabled connection all answer alike: ``found`` is false and nothing else is named.
    """
    throttle_public_auth(request)
    organization = await connections.organization_for_sign_in(email=body.email)
    if organization is None:
        return SsoLookupResponse(found=False)
    return SsoLookupResponse(found=True, organization_name=organization.name, slug=organization.slug)


@router.get("/{slug}/authorize", response_model=SsoAuthorizeResponse)
async def authorize_sso(
    slug: str,
    request: Request,
    response: Response,
    config: Annotated[GatewayConfig, Depends(get_config)],
    connections: OIDCConnectionServiceDep,
    flow_cookie: SsoFlowCookie = None,
) -> SsoAuthorizeResponse:
    """Start a sign-in through the organization's own connection at ``slug``.

    An unknown slug and a slug naming an organization with no enabled connection answer alike
    (404), so this surface cannot be read as a directory of which organizations use Otari.

    NOTE: This GET writes. It records a pending authorization and sets an HttpOnly cookie, both
    scoped to this router's own path.
    """
    throttle_public_auth(request)
    connection = await connections.connection_for_sign_in(slug=slug)
    if connection is None:
        raise OIDCSignInUnavailableError
    metadata = await connections.metadata_for_sign_in(connection)
    try:
        bound = await connections.start_sign_in(
            connection,
            metadata,
            redirect_uri=_sso_redirect_uri(config),
            allow_private_hosts=config.oidc_allow_private_hosts,
        )
    except DATABASE_ERRORS as exc:
        raise database_error("record a pending SSO sign-in", exc) from None
    apply_flow_cookie(
        response,
        flow_cookie,
        BrowserAuthorization(url=bound.url, state=bound.state, browser_token=bound.browser_token),
        secure=request_is_https(request),
        cookie_name=SSO_FLOW_COOKIE_NAME,
        cookie_path=SSO_FLOW_COOKIE_PATH,
    )
    return SsoAuthorizeResponse(authorization_url=bound.url, state=bound.state)


async def _exchange_sso_code(
    client: StoredStateClient,
    *,
    code: str,
    state: str,
    iss: str | None,
    browser_token: str | None,
    connection: OIDCConnection,
) -> ProviderIdentity:
    """Trade an authorization code for the identity this connection's provider vouches for.

    The resulting identity's provider names the issuer, never the connection, so every
    organization sharing one issuer resolves one person to one account.
    """
    try:
        tokens = await client.exchange_code(
            AuthorizationCallback(code=code, state=state, iss=iss), browser_token=browser_token
        )
        profile = await client.fetch_identity(tokens)
    except StateError as error:
        if browser_token is None:
            logger.warning(
                "Refused an SSO callback the browser presented no flow token for: its cookie did not arrive, "
                "or no longer holds that sign-in"
            )
        else:
            logger.warning("Refused an SSO callback whose state names no sign-in this deployment is waiting on")
        raise OAuthStateError from error
    except AuthorizationResponseError as error:
        logger.warning("Refused an SSO callback that states no authorization code")
        raise OAuthExchangeError(connection.display_name) from error
    except IssuerValidationError as error:
        logger.warning("Refused an SSO callback whose iss did not match this connection's issuer")
        raise OAuthExchangeError(connection.display_name) from error
    except NonceValidationError as error:
        logger.warning("Refused an SSO token response whose ID token did not carry the request's nonce")
        raise OAuthExchangeError(connection.display_name) from error
    except Exception as error:
        logger.warning("SSO code exchange failed", exc_info=True)
        if provider_refused_the_code(error):
            raise OAuthExchangeError(connection.display_name) from error
        raise OAuthProviderUnavailableError(connection.display_name) from error

    try:
        return ProviderIdentity(
            provider=oidc_identity_provider(connection.resolved_issuer),
            subject=profile.subject or "",
            email=profile.email,
            full_name=profile.name,
            email_verified=EmailAssertion.from_claim(profile.email_verified),
        )
    except ValueError as error:
        logger.warning("Refused an SSO identity that the link table cannot hold")
        raise OAuthExchangeError(connection.display_name) from error


@router.post("/callback", response_model=SsoSessionPublic)
async def callback_sso(
    body: SsoCallbackRequest,
    request: Request,
    response: Response,
    identities: IdentityServiceDep,
    connections: OIDCConnectionServiceDep,
    growth: GrowthSignalPortDep,
    background_tasks: BackgroundTasks,
    maintenance_mode: MaintenanceModeDep,
    config: Annotated[GatewayConfig, Depends(get_config)],
    flow_cookie: SsoFlowCookie = None,
) -> SsoSessionPublic:
    """Exchange an authorization code and set the HttpOnly session cookie.

    The session is bound to the identity the organization's connection resolves to. A refusal is
    counted as a failed sign-in, and a sign-in that creates the account, or is the first way in to
    one, is reported to ``GrowthSignalPort`` once it has committed.

    **Maintenance mode freezes this.** A frozen deployment makes no outbound call and spends
    nobody's authorization code.
    """
    throttle_public_auth(request)
    if maintenance_mode:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=MAINTENANCE_MODE_REFUSAL)

    connection = await connections.connection_for_pending_state(body.state)
    if connection is None:
        logger.warning(
            "Refused an SSO callback whose state names no pending sign-in through an organization's own connection"
        )
        raise OAuthStateError
    try:
        metadata = await connections.metadata_for_sign_in(connection)
        client = await connections.client_for_sign_in(
            connection,
            metadata,
            redirect_uri=_sso_redirect_uri(config),
            allow_private_hosts=config.oidc_allow_private_hosts,
        )
        external = await _exchange_sso_code(
            client,
            code=body.code,
            state=body.state,
            iss=body.iss,
            browser_token=browser_token_for(flow_cookie, body.state),
            connection=connection,
        )
        signed_in = await identities.sign_in_with_oauth(external, terms_accepted=body.terms_accepted)
    except (OAuthIdentityLinkConflictError, SignInBlockedError, TermsNotAcceptedError):
        # A concurrent sign-in finished first, the address is blocked, or the terms are still to accept, so
        # nobody failed to authenticate.
        raise
    except OAuthProviderUnavailableError as exc:
        # Rendered here so the 502 keeps its message, which would otherwise be blanked for a 5xx.
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from None
    except TenancyError:
        record_auth_failure("invalid_sso")
        raise
    except DATABASE_ERRORS as exc:
        raise database_error("persist an SSO sign-in", exc) from None

    identity = signed_in.account
    logger.info("Signed in %s through organization %s's own connection", identity.id, connection.organization_id)
    await report_new_account(growth, identity, background_tasks=background_tasks, new_account=signed_in.new_account)
    secure = request_is_https(request)
    drop_finished_flow(
        response,
        flow_cookie,
        body.state,
        secure=secure,
        cookie_name=SSO_FLOW_COOKIE_NAME,
        cookie_path=SSO_FLOW_COOKIE_PATH,
    )
    apply_session_cookie(response, signed_in.session_token, signed_in.session_expires_at, secure=secure)
    return SsoSessionPublic(
        expires_at=signed_in.session_expires_at,
        user_id=identity.id,
        active_organization_id=signed_in.active_organization_id,
        password_removed=signed_in.password_removed,
    )


__all__ = ["router"]
