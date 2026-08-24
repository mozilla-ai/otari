"""The OAuth half of dashboard sign-in: an authorization URL, then a code exchange.

Protocol mechanics come from apron-auth, which owns the provider endpoints, the
code exchange and the userinfo fetch that normalizes a provider response into an
``IdentityProfile``. The platform moved onto it in mozilla-ai/otari-ai#1740 and
this module follows, so neither side keeps a hand-rolled copy of a protocol
neither of us owns. What stays here is this deployment's own part: which
providers are configured, which scopes are asked for, and where the provider is
told to send the browser back to.

**Who the identity turns out to be is not here.** That decision is behind
``IdentityProviderPort``, because it is the one an edition varies: this build
resolves an identity against its roster and an overlay may provision instead.
This module stops at "the provider says this is who they are", and the sign-in
route hands that across the seam.

**PKCE is deliberately off, and the authorization URLs are built by hand.**
Authorize and callback are two independent HTTP requests with nothing kept
server-side between them, so a verifier minted while building the authorization
URL has nowhere to live until the exchange. That is why ``_without_pkce``
clears the flag and why ``authorization_url`` assembles the query itself rather
than calling ``OAuthClient.get_authorization_url``, which is the method that
reads the flag and would start sending a code challenge this flow cannot
answer. Turning PKCE on is a real change with a real prerequisite (somewhere for
the verifier to live), not a default to restore.

The CSRF ``state`` is minted here and checked in the browser, which is the same
split for the same reason: the dashboard puts it in ``sessionStorage`` when it
sends somebody to the provider and compares it when the provider sends them
back, so the value survives the round trip without this deployment storing
anything. See ``web/src/features/auth/OAuthCallbackPage.tsx``.
"""

import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

from apron_auth import OAuthClient, ProviderConfig
from apron_auth.providers import github as apron_github
from apron_auth.providers import google as apron_google

from gateway.core.config import OAUTH_PROVIDERS, GatewayConfig
from gateway.log_config import logger
from gateway.services.tenancy.errors import OAuthExchangeError, OAuthNotConfiguredError

# Where each provider's consent screen lives, and which scopes to ask it for.
#
# The scope sets must cover what the matching apron-auth identity handler reads
# back at callback time: Google's OIDC userinfo endpoint, and GitHub's ``/user``
# plus ``/user/emails``. Each preset merges its own base scopes on top of these,
# so a provider config's scope set is a superset of what the authorization URL
# asks for.
#
# ``extra_params`` is what an authorization URL carries beyond the four the
# protocol requires. Google's ``access_type=offline`` is the platform's own and
# is kept for parity of the consent screen a person sees; nothing here stores a
# refresh token.
_GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"


@dataclass(frozen=True)
class _Provider:
    """One provider's authorization endpoint, scopes, and apron-auth wiring."""

    label: str
    authorize_url: str
    scopes: tuple[str, ...]
    extra_authorize_params: tuple[tuple[str, str], ...] = ()


_PROVIDERS: dict[str, _Provider] = {
    "google": _Provider(
        label="Google",
        authorize_url=_GOOGLE_AUTHORIZE_URL,
        scopes=("openid", "email", "profile"),
        extra_authorize_params=(("access_type", "offline"),),
    ),
    "github": _Provider(
        label="GitHub",
        authorize_url=_GITHUB_AUTHORIZE_URL,
        scopes=("read:user", "user:email"),
    ),
}
# Kept honest by a unit test as well, but asserted at import so a provider added
# to one of the two lists and not the other fails on the way up rather than on
# the first request that names it.
assert set(_PROVIDERS) == set(OAUTH_PROVIDERS), "OAUTH_PROVIDERS and _PROVIDERS must name the same providers"


@dataclass(frozen=True)
class OAuthIdentity:
    """What a completed exchange says about the person who just consented.

    Narrower than apron-auth's ``IdentityProfile`` on purpose: this is the
    subset that crosses ``IdentityProviderPort``, so what an adapter may key on
    is visible here rather than being whatever the library happened to return.
    """

    provider: str
    email: str | None
    full_name: str | None
    email_verified: bool


def provider_label(provider: str) -> str:
    """The provider's own name, cased the way it writes it, for a message.

    An unknown provider answers with the string it was given rather than
    raising: the only caller is an error message, and a refusal that fails to
    render is worse than one naming a provider nobody configured.
    """
    known = _PROVIDERS.get(provider)
    return known.label if known else provider


def redirect_uri(config: GatewayConfig, provider: str) -> str:
    """Where the provider sends the browser back to, derived and never supplied.

    Derived from ``public_base_url`` rather than taken from the request, so the
    URI in the authorization request and the URI in the exchange are the same
    string by construction. The provider checks it against the one registered
    for the client, so a mismatch is caught at the provider either way; deriving
    it just means a browser cannot choose what this server sends.

    **It carries no fragment, which is why it is not a dashboard hash path.**
    Every page the dashboard serves in front of a session is a hash route, and
    RFC 6749 forbids a fragment in a redirection URI (Google rejects one
    outright). So this names an ordinary path, which ``gateway.main`` serves
    with a redirect into the hash route that finishes the sign-in.
    """
    return f"{(config.public_base_url or '').rstrip('/')}/auth/{provider}/callback"


def new_state() -> str:
    """A fresh CSRF ``state`` for one authorization request."""
    return secrets.token_urlsafe(32)


def authorization_url(config: GatewayConfig, provider: str, *, state: str) -> str:
    """The provider consent screen to send the browser to.

    Assembled here rather than by ``OAuthClient.get_authorization_url``; see the
    module docstring on PKCE for why that matters.

    Raises:
        OAuthNotConfiguredError: If this deployment configured no client
            credentials for ``provider``, or does not know its own address.

    """
    client_id, _ = _credentials(config, provider)
    known = _PROVIDERS[provider]
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(config, provider),
        "response_type": "code",
        "scope": " ".join(known.scopes),
        "state": state,
        **dict(known.extra_authorize_params),
    }
    return f"{known.authorize_url}?{urlencode(params)}"


async def exchange_code(config: GatewayConfig, provider: str, *, code: str) -> OAuthIdentity:
    """Trade an authorization code for the identity the provider vouches for.

    Raises:
        OAuthNotConfiguredError: If this deployment configured no client
            credentials for ``provider``, or does not know its own address.
        OAuthExchangeError: If the exchange or the identity fetch fails, for any
            reason. The provider's own words stay on the traceback and out of
            the response; see that error's docstring.

    """
    client = _client(config, provider)
    uri = redirect_uri(config, provider)
    try:
        tokens = await client.exchange_code(code=code, redirect_uri=uri)
        profile = await client.fetch_identity(tokens)
    except Exception as error:
        # Logged with the exception so an operator can see the provider's own
        # error and description, which the response deliberately does not carry.
        logger.warning("OAuth code exchange with %s failed", provider, exc_info=True)
        raise OAuthExchangeError(provider_label(provider)) from error

    return OAuthIdentity(
        provider=provider,
        email=profile.email,
        full_name=profile.name,
        # apron-auth reports email_verified as tri-state (True, False, or
        # unasserted). This edition resolves identity on a bool, so an
        # unasserted value collapses to unverified rather than being laundered
        # into a verified identity. mozilla-ai/otari-ai#1551 moves resolution
        # onto the tri-state model, once, on the platform; it is deliberately
        # not anticipated here.
        email_verified=profile.email_verified is True,
    )


def require_configured(config: GatewayConfig, provider: str) -> None:
    """Refuse unless this deployment can actually sign somebody in with ``provider``.

    The gate the sign-in routes apply ahead of everything else, so "is this
    provider on offer" is one decision rather than a side effect of whichever
    call below happens to need the credentials first.

    Raises:
        OAuthNotConfiguredError: If ``provider`` is unknown here, either half of
            its client credentials is missing, or ``public_base_url`` is.

    """
    _credentials(config, provider)


def _credentials(config: GatewayConfig, provider: str) -> tuple[str, str]:
    """This deployment's client ID and secret for ``provider``.

    Raises:
        OAuthNotConfiguredError: If either is missing, or ``public_base_url`` is.

    """
    if provider not in _PROVIDERS:
        raise OAuthNotConfiguredError(provider)
    credentials = config.oauth_client_credentials(provider)
    if credentials is None:
        raise OAuthNotConfiguredError(provider)
    return credentials


def _client(config: GatewayConfig, provider: str) -> OAuthClient:
    """The apron-auth client that performs ``provider``'s code exchange."""
    client_id, client_secret = _credentials(config, provider)
    known = _PROVIDERS[provider]
    preset = apron_google.preset if provider == "google" else apron_github.preset
    identity_handler = (
        apron_google.GoogleIdentityHandler() if provider == "google" else apron_github.GitHubIdentityHandler()
    )
    provider_config, _revocation_handler = preset(
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(known.scopes),
        redirect_uri=redirect_uri(config, provider),
    )
    return OAuthClient(_without_pkce(provider_config), identity_handler=identity_handler)


def _without_pkce(provider_config: ProviderConfig) -> ProviderConfig:
    """Clear PKCE on a config this flow uses.

    apron-auth reads ``use_pkce`` only inside ``get_authorization_url``, which
    this module does not call, so clearing it changes nothing about the exchange
    today. It is cleared so that adopting the library's URL builder, once
    somewhere exists for a verifier to live, is a deliberate step rather than
    one that silently starts sending a code challenge this flow cannot answer.
    See the module docstring.
    """
    return provider_config.model_copy(update={"use_pkce": False})


__all__ = [
    "OAuthIdentity",
    "authorization_url",
    "exchange_code",
    "new_state",
    "provider_label",
    "redirect_uri",
    "require_configured",
]
