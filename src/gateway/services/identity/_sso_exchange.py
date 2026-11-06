"""Trading an SSO authorization code for the identity an organization's own provider vouches for."""

from apron_auth import AuthorizationCallback, StoredStateClient
from apron_auth.errors import AuthorizationResponseError, IssuerValidationError, NonceValidationError, StateError

from gateway.exceptions.identity_exceptions import (
    OAuthExchangeError,
    OAuthProviderUnavailableError,
    OAuthStateError,
)
from gateway.log_config import logger
from gateway.models.identity import EmailAssertion, OIDCConnection, ProviderIdentity, oidc_identity_provider
from gateway.services.oauth_service import provider_refused_the_code

async def exchange_sso_code(
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


__all__ = ["exchange_sso_code"]
