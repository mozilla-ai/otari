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

**PKCE is on, and the ``state`` is checked here.** Both rest on one thing: a
``models.tenancy.OAuthPendingState`` row per authorization in flight, written
when the authorization URL is built and consumed by the exchange. apron-auth's
``StateStore`` protocol is the seam it plugs into, so ``get_authorization_url``
mints the verifier and ``exchange_code`` reads it back, and neither the code
challenge nor the state comparison is this module's own arithmetic.

**The authorization is bound to the browser as well as to the flow.** A
``code`` and a ``state`` travel together in one redirect URL, and that URL is
written to the access log and to browser history, so a flow keyed on the state
alone would let whoever reads either finish the sign-in for the full TTL
(RFC 9700, section 4.7.1). apron-auth issues a browser token per authorization
request and refuses a callback that token does not match. The token lives in an
HttpOnly cookie, one entry per sign-in in flight, which is what lets two tabs
each finish their own.

What is this module's own is the two overrides in ``_as_configured_here``: the
presets widen scopes and, for Google, ask for offline access, and neither is
wanted here. See that function.

The browser keeps its ``sessionStorage`` copy of the state and still compares
it, which is not redundant. That check binds a callback to the *tab* that
started the flow, which a server-side row cannot do; this one binds it to a
flow this deployment actually started, which the browser cannot do. They fail
in different directions. See ``web/src/features/auth/OAuthCallbackPage.tsx``.
"""

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from apron_auth import AuthorizationCallback, ProviderConfig, StoredStateClient
from apron_auth.errors import (
    INVALID_GRANT,
    AuthorizationResponseError,
    ConfigurationError,
    NonceValidationError,
    StateError,
    TokenExchangeError,
)
from apron_auth.models import OAuthPendingState as PendingState
from apron_auth.providers import github as apron_github
from apron_auth.providers import google as apron_google
from fastapi import Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import API_ROOT, OAUTH_PROVIDERS, GatewayConfig
from gateway.exceptions.identity_exceptions import (
    OAuthExchangeError,
    OAuthNotConfiguredError,
    OAuthProviderUnavailableError,
    OAuthStateError,
)
from gateway.log_config import logger
from gateway.models.identity import EmailAssertion, ProviderIdentity
from gateway.models.tenancy import OAUTH_STATE_TTL_SECONDS, OAuthPendingState

# Which scopes each provider is asked for, and what it calls itself.
#
# The scope sets must cover what the matching apron-auth identity handler reads
# back at callback time: Google's OIDC userinfo endpoint, and GitHub's ``/user``
# plus ``/user/emails``. They are the exact set the authorization request asks
# for, which ``_as_configured_here`` is what makes true: a preset would
# otherwise merge its own base scopes over them.
#
# The consent-screen endpoints are the presets' own and are not restated here.


@dataclass(frozen=True)
class BrowserAuthorization:
    """One authorization request, built for the browser that asked for it.

    ``state`` names the flow wherever one is recorded or looked up, and reveals
    nothing about ``browser_token``.

    NOTE: ``browser_token`` belongs in ``FLOW_COOKIE_NAME`` and nowhere script
    can read it.
    """

    url: str
    state: str
    browser_token: str


@dataclass(frozen=True)
class _Provider:
    """One provider's scopes, the name it writes itself under, and whether its email claim can be trusted.

    ``email_claim_trusted`` is true only when no tenant can set what the provider says about an address.
    """

    label: str
    scopes: tuple[str, ...]
    email_claim_trusted: bool


# The resource the OAuth routes are mounted at. The router takes its prefix from
# here and the flow cookie its path, so the cookie cannot end up scoped to
# somewhere the routes are not.
OAUTH_ROUTE_PREFIX = "/auth/oauth"
# The cookie that binds each pending authorization to the browser that started
# it. Scoped to the OAuth routes, which are the only ones that read it.
FLOW_COOKIE_NAME = "otari_oauth_flow"
FLOW_COOKIE_PATH = f"{API_ROOT}{OAUTH_ROUTE_PREFIX}"
# How many sign-ins one browser may have in flight. A browser holds one cookie,
# so an unbounded one would grow for as long as somebody keeps starting flows,
# and three concurrent sign-ins is already more than a person does.
MAX_LIVE_FLOWS = 3
# Three live flows encode to under 500 characters, so a longer value is one this
# deployment never wrote. The bound is what an unauthenticated caller can make
# this process decode.
MAX_FLOW_COOKIE_CHARS = 1024

_PROVIDERS: dict[str, _Provider] = {
    "google": _Provider(label="Google", scopes=("openid", "email", "profile"), email_claim_trusted=True),
    "github": _Provider(label="GitHub", scopes=("read:user", "user:email"), email_claim_trusted=True),
}
# Kept honest by a unit test as well, but asserted at import so a provider added
# to one of the two lists and not the other fails on the way up rather than on
# the first request that names it.
assert set(_PROVIDERS) == set(OAUTH_PROVIDERS), "OAUTH_PROVIDERS and _PROVIDERS must name the same providers"


def provider_label(provider: str) -> str:
    """The provider's own name, cased the way it writes it, for a message.

    An unknown provider answers with the string it was given rather than
    raising: the only caller is an error message, and a refusal that fails to
    render is worse than one naming a provider nobody configured.
    """
    known = _PROVIDERS.get(provider)
    return known.label if known else provider


def trusts_email_claim(provider: str) -> bool:
    """Say whether no tenant can set what this provider says about an address.

    An unknown provider is not trusted.
    """
    known = _PROVIDERS.get(provider)
    return known is not None and known.email_claim_trusted


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
    return f"{base_url(config)}/auth/{provider}/callback"


def base_url(config: GatewayConfig) -> str:
    """This deployment's own address with no trailing slash, or an empty string.

    One reading of ``public_base_url`` for everything that has to name this
    process, so the authorization request and the exchange name one redirect
    URI between them. It may carry a **path prefix**: a gateway served at
    ``https://example.com/otari`` is a supported shape, so a root-absolute
    answer here would send such a deployment's callback to the wrong origin
    path.
    """
    return (config.public_base_url or "").rstrip("/")


def callback_landing_target(config: GatewayConfig, provider: str, query: str) -> str:
    """Where ``/auth/{provider}/callback`` sends the browser to finish signing in.

    The interface's own hash route, carrying whatever query the provider
    appended. Built from ``effective_ui_base_url`` and not the redirect URI's
    base: the ``state`` in that query is checked against a value scoped to the
    origin that minted it, so a browser landed elsewhere cannot finish.

    Relative to that base rather than to the request, because the request's path
    is what a reverse proxy may already have rewritten, and this has to name a
    URL in the browser's address bar rather than in this process.
    """
    route = f"/#/auth/{quote(provider, safe='')}/callback"
    return config.ui_link(f"{route}?{query}" if query else route)


class DatabaseStateStore:
    """apron-auth's ``StateStore``, backed by ``oauth_pending_state``.

    Two methods and no repository, because it is not the shape a repository
    serves: apron-auth calls these, and each is a single statement against one
    table. A caller outside this module builds one with a provider value of its
    own, such as an organization's own OIDC connection, so the table is shared
    rather than owned by the built-in sign-in alone.

    Rows are staged on the caller's transaction and never committed here. The
    caller owns the commit boundary, which is what lets a callback that fails
    after consuming a state roll the consumption back and leave the person a
    flow to retry.
    """

    def __init__(self, db: AsyncSession, provider: str) -> None:
        self._db = db
        self._provider = provider

    async def save(self, state: PendingState) -> None:
        """Stage one pending authorization, and sweep whatever has expired."""
        if state.context:
            # The row has no column for it, so a caller that starts relying on
            # apron-auth's save/consume round trip finds out at the write rather
            # than by reading an empty ``TokenSet.context`` later.
            msg = "A pending sign-in cannot carry state beyond the fields recorded here"
            raise ValueError(msg)
        now = datetime.now(UTC)
        # The sweep rides here rather than on a scheduler because this is the
        # only write the table takes, so it is the only place that can grow it.
        await self._db.execute(delete(OAuthPendingState).where(col(OAuthPendingState.expires_at) <= now))
        self._db.add(
            OAuthPendingState(
                state_hash=state_hash(state.state),
                provider=self._provider,
                code_verifier=state.code_verifier,
                nonce=state.nonce,
                redirect_uri=state.redirect_uri,
                expires_at=now + timedelta(seconds=OAUTH_STATE_TTL_SECONDS),
            )
        )
        await self._db.flush()

    async def consume(self, state_key: str) -> PendingState | None:
        """Claim one pending authorization, or answer ``None``.

        **One conditional DELETE, not a SELECT and then a delete**, the rule
        ``webauthn_service.consume_challenge`` states at length and for the same
        reason: single use is the property, and a read-then-write cannot provide
        it. Two callbacks replaying one state would both find the row; deleting
        by primary key and reading what came back means exactly one of them
        does.

        ``provider`` is compared after the row is claimed rather than added to
        the WHERE clause, so a state minted for one provider and returned to
        another is refused as itself rather than as an unknown state. The
        delete is staged on the request's transaction and the refusal keeps
        that transaction from committing, so the row survives for the callback
        it was minted for.
        """
        row = (
            await self._db.execute(
                delete(OAuthPendingState)
                .where(col(OAuthPendingState.state_hash) == state_hash(state_key))
                .returning(
                    col(OAuthPendingState.provider),
                    col(OAuthPendingState.code_verifier),
                    col(OAuthPendingState.nonce),
                    col(OAuthPendingState.redirect_uri),
                    col(OAuthPendingState.created_at),
                    col(OAuthPendingState.expires_at),
                )
            )
        ).first()
        if row is None:
            return None
        provider, code_verifier, nonce, redirect_uri, created_at, expires_at = row
        if provider != self._provider:
            logger.warning("OAuth state minted for %s was returned to %s", provider, self._provider)
            return None
        if expires_at <= datetime.now(UTC):
            return None
        if nonce is None and "openid" in _PROVIDERS[provider].scopes:
            # Minted before the column existed. apron-auth would refuse it as a
            # nonce failure; as a state no longer pending, the person is told to
            # start again.
            logger.warning("OAuth %s state carried no nonce for an OpenID sign-in", provider)
            return None
        return PendingState(
            state=state_key,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            nonce=nonce,
            created_at=created_at.timestamp(),
        )


def state_hash(state: str) -> str:
    """The key a state is stored under: its SHA-256, hex.

    The state is already unguessable, so this is a lookup key and not a
    password hash; a single round is right and a KDF would be theater. What it
    buys is that a reader of the table holds digests, and the callback compares
    against the preimage.
    """
    return hashlib.sha256(state.encode()).hexdigest()


async def pending_flow_provider(db: AsyncSession, state: str) -> str | None:
    """The provider of the pending authorization ``state`` names, without consuming it.

    A caller that must resolve which client's credentials to spend before it can build the client
    that consumes the row, such as an organization's own OIDC connection, reads it here: single use
    is still enforced by the conditional DELETE ``DatabaseStateStore.consume`` performs, and a wrong
    guess here only fails the exchange that follows.
    """
    result = await db.execute(
        select(col(OAuthPendingState.provider)).where(col(OAuthPendingState.state_hash) == state_hash(state))
    )
    return result.scalar_one_or_none()


def browser_token_for(flow_cookie: str | None, state: str) -> str | None:
    """The browser token ``flow_cookie`` holds for ``state``, or ``None`` where it holds none.

    This selects which of the browser's live flows to present and decides
    nothing about it: apron-auth makes the one binding check, against the state
    the callback carried.
    """
    return _live_flows(flow_cookie).get(state)


def apply_flow_cookie(
    response: Response,
    flow_cookie: str | None,
    authorization: BrowserAuthorization,
    *,
    secure: bool,
    cookie_name: str = FLOW_COOKIE_NAME,
    cookie_path: str = FLOW_COOKIE_PATH,
) -> None:
    """Add ``authorization`` to the flows the browser has in flight, and write the cookie back.

    The oldest flow is evicted once a browser holds more than ``MAX_LIVE_FLOWS``. ``cookie_name``
    and ``cookie_path`` default to this module's own flow cookie; pass a different pair to scope
    the cookie elsewhere.
    """
    flows = _live_flows(flow_cookie)
    flows[authorization.state] = authorization.browser_token
    _write_flow_cookie(
        response,
        dict(list(flows.items())[-MAX_LIVE_FLOWS:]),
        secure=secure,
        cookie_name=cookie_name,
        cookie_path=cookie_path,
    )


def drop_finished_flow(
    response: Response,
    flow_cookie: str | None,
    state: str,
    *,
    secure: bool,
    cookie_name: str = FLOW_COOKIE_NAME,
    cookie_path: str = FLOW_COOKIE_PATH,
) -> None:
    """Write the cookie back without ``state``, or delete it where that was the last flow.

    A redeemed flow's token can buy nothing and must not sit in the browser for the rest of the
    cookie's life. ``cookie_name`` and ``cookie_path`` default to this module's own flow cookie.
    """
    kept = {flow: token for flow, token in _live_flows(flow_cookie).items() if flow != state}
    _write_flow_cookie(response, kept, secure=secure, cookie_name=cookie_name, cookie_path=cookie_path)


def _live_flows(flow_cookie: str | None) -> dict[str, str]:
    """The browser token of each sign-in this cookie has in flight, keyed by its state, oldest first.

    A cookie that cannot be read carries no flows, so a mangled, stale or
    over-long one costs a person a sign-in to start again rather than a failed
    request. Refusing one on the way in would be worse than reading none: it is
    HttpOnly, so nothing but a new authorization request can rewrite it.
    """
    if not flow_cookie or len(flow_cookie) > MAX_FLOW_COOKIE_CHARS:
        return {}
    try:
        flows = json.loads(base64.urlsafe_b64decode(flow_cookie + "=" * (-len(flow_cookie) % 4)))
    except ValueError:
        return {}
    if not isinstance(flows, dict):
        return {}
    return {state: token for state, token in flows.items() if isinstance(state, str) and isinstance(token, str)}


def _write_flow_cookie(
    response: Response,
    flows: Mapping[str, str],
    *,
    secure: bool,
    cookie_name: str = FLOW_COOKIE_NAME,
    cookie_path: str = FLOW_COOKIE_PATH,
) -> None:
    """Set ``cookie_name`` to exactly ``flows`` at ``cookie_path``, or delete it where ``flows`` is empty.

    The cookie is HttpOnly and SameSite=Lax. Its value is base64url of compact JSON, so braces
    and quotes stay out of a header the browser has to hand back byte for byte. It conceals
    nothing.
    """
    if not flows:
        response.delete_cookie(cookie_name, path=cookie_path, httponly=True, secure=secure, samesite="lax")
        return
    encoded = base64.urlsafe_b64encode(json.dumps(flows, separators=(",", ":")).encode()).decode().rstrip("=")
    response.set_cookie(
        cookie_name,
        encoded,
        max_age=OAUTH_STATE_TTL_SECONDS,
        httponly=True,
        secure=secure,
        samesite="lax",
        path=cookie_path,
    )


async def authorization_url(config: GatewayConfig, provider: str, *, db: AsyncSession) -> BrowserAuthorization:
    """The provider consent screen to send the browser to, the state to keep, and the token that binds them.

    Staged, not committed: the caller commits, so an authorization URL is never
    handed back over a row that failed to write.

    Raises:
        OAuthNotConfiguredError: If this deployment configured no client
            credentials for ``provider``, or does not know its own address.

    """
    client = _client(config, provider, db)
    bound = await client.get_authorization_url(redirect_uri=redirect_uri(config, provider))
    return BrowserAuthorization(url=bound.url, state=bound.state, browser_token=bound.browser_token)


async def exchange_code(
    config: GatewayConfig,
    provider: str,
    *,
    code: str,
    state: str,
    browser_token: str | None,
    db: AsyncSession,
) -> ProviderIdentity:
    """Trade an authorization code for the identity the provider vouches for.

    ``state`` is claimed before the code is sent anywhere, so a callback this
    deployment never started costs one indexed delete and no outbound call. The
    claim also yields the PKCE verifier and the redirect URI the authorization
    request was built with, which is why neither is passed here.

    ``browser_token`` is the token the browser's flow cookie holds for
    ``state``, or ``None`` where it holds none. The ``state`` must be the one
    that token mints, which is checked before the row is claimed and before the
    provider is called, so a callback presented from anywhere else redeems
    nothing and leaves the authorization live.

    Raises:
        OAuthNotConfiguredError: If this deployment configured no client
            credentials for ``provider``, or does not know its own address.
        OAuthStateError: If ``state`` names no authorization this deployment is
            still waiting on, for ``provider``, from this browser.
        OAuthExchangeError: If the callback states no code to redeem, the provider says the code is invalid,
            expired or spent, the ID token does not carry the nonce the authorization request sent, or the
            identity does not fit the link table.
        OAuthProviderUnavailableError: If the provider fails or cannot be reached during the exchange or the
            identity fetch.
            The provider's own words stay on the traceback and out of the response, as that error's docstring says.

    """
    client = _client(config, provider, db)
    try:
        tokens = await client.exchange_code(AuthorizationCallback(code=code, state=state), browser_token=browser_token)
        profile = await client.fetch_identity(tokens)
    except StateError as error:
        # Ahead of the catch-all below, which would otherwise render a refused
        # state as "the provider did not complete the sign-in" and send an
        # operator looking at a provider that was never contacted.
        if browser_token is None:
            logger.warning(
                "Refused a %s callback the browser presented no flow token for: its cookie did not arrive, "
                "or no longer holds that sign-in",
                provider,
            )
        else:
            logger.warning("Refused a %s callback whose state names no sign-in this deployment is waiting on", provider)
        raise OAuthStateError from error
    except AuthorizationResponseError as error:
        # Also ahead of the catch-all: the response states no code to redeem, so
        # nothing was asked of the provider and nothing there failed.
        logger.warning("Refused a %s callback that states no authorization code", provider)
        raise OAuthExchangeError(provider_label(provider)) from error
    except NonceValidationError as error:
        # Also ahead of the catch-all: a token response that does not answer the
        # request this deployment made is a refused sign-in, not an outage.
        logger.warning("Refused a %s token response whose ID token did not carry the request's nonce", provider)
        raise OAuthExchangeError(provider_label(provider)) from error
    except Exception as error:
        # Logged with the exception so an operator can see the provider's own
        # error and description, which the response deliberately does not carry.
        logger.warning("OAuth code exchange with %s failed", provider, exc_info=True)
        if provider_refused_the_code(error):
            raise OAuthExchangeError(provider_label(provider)) from error
        raise OAuthProviderUnavailableError(provider_label(provider)) from error

    try:
        return ProviderIdentity(
            provider=provider,
            subject=profile.subject or "",
            email=profile.email,
            full_name=profile.name,
            email_verified=EmailAssertion.from_claim(profile.email_verified),
        )
    except ValueError as error:
        logger.warning("Refused a %s identity that the link table cannot hold", provider)
        raise OAuthExchangeError(provider_label(provider)) from error


def provider_refused_the_code(error: Exception) -> bool:
    """Whether the token endpoint said the authorization code is invalid, expired or already spent.

    Only ``invalid_grant`` says that. Every other failure is the provider's or this deployment's: no answer, a
    server error under whatever code it carries, a failed identity fetch, or ``invalid_client`` and
    ``unauthorized_client``, which mean the configured client credentials were refused.
    """
    return isinstance(error, TokenExchangeError) and error.error_code == INVALID_GRANT


def require_configured(config: GatewayConfig, provider: str) -> None:
    """Refuse unless this deployment can actually sign somebody in with ``provider``.

    The gate the sign-in routes apply ahead of everything else, so "is this
    provider on offer" is one decision rather than a side effect of whichever
    call below happens to need the credentials first.

    Raises:
        OAuthNotConfiguredError: If ``provider`` is unknown here, either half of
            its client credentials is missing, ``public_base_url`` is, or the
            redirect URI it builds is neither https nor a loopback address.

    """
    client_id, client_secret = _credentials(config, provider)
    try:
        _provider_config(config, provider, client_id, client_secret)
    except ConfigurationError:
        raise OAuthNotConfiguredError.insecure_address(provider) from None


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


def _provider_config(config: GatewayConfig, provider: str, client_id: str, client_secret: str) -> ProviderConfig:
    """Build ``provider``'s configuration from this deployment's own settings.

    Raises:
        ConfigurationError: If the redirect URI this builds is neither https nor
            a loopback address.

    """
    preset = apron_google.preset if provider == "google" else apron_github.preset
    provider_config = preset(
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(_PROVIDERS[provider].scopes),
        redirect_uri=redirect_uri(config, provider),
    )
    return _as_configured_here(provider_config, provider)


def _client(config: GatewayConfig, provider: str, db: AsyncSession) -> StoredStateClient:
    """The apron-auth client that builds ``provider``'s authorization URL and spends its code."""
    client_id, client_secret = _credentials(config, provider)
    identity_handler = (
        apron_google.GoogleIdentityHandler() if provider == "google" else apron_github.GitHubIdentityHandler()
    )
    return StoredStateClient(
        _provider_config(config, provider, client_id, client_secret),
        DatabaseStateStore(db, provider),
        identity_handler=identity_handler,
    )


def _as_configured_here(provider_config: ProviderConfig, provider: str) -> ProviderConfig:
    """Undo the two preset defaults this deployment does not want.

    Both matter only because the authorization URL is apron-auth's to build
    now. While it was assembled by hand, neither field was ever read, so both
    departures were invisible; adopting ``get_authorization_url`` is what makes
    them settings rather than omissions.

    **Scopes are pinned to the exact set in ``_PROVIDERS``.** Each preset merges
    its own ``BASE_SCOPES`` over what it is given, which for Google adds the
    long-form ``userinfo.email`` alongside the ``email`` already asked for. It
    grants nothing new, and it is one more line on the consent screen naming a
    scope this gateway did not choose.

    **Google's ``extra_params`` are cleared.** The preset sets
    ``access_type=offline`` and ``prompt=consent``, and the preset's own
    ``extra_params`` argument merges *over* those rather than replacing them, so
    this is the only place to drop them. Offline access exists to obtain a
    refresh token; a sign-in here reads an identity once and mints Otari's own
    session, so a refresh token would be a durable credential Google issued,
    this deployment discarded, and nobody ever revoked. ``prompt=consent`` goes
    with it: re-consenting on every sign-in is what asking for offline access
    obliges, and nothing here needs it.
    """
    return provider_config.model_copy(update={"scopes": list(_PROVIDERS[provider].scopes), "extra_params": {}})


__all__ = [
    "BrowserAuthorization",
    "DatabaseStateStore",
    "FLOW_COOKIE_NAME",
    "FLOW_COOKIE_PATH",
    "MAX_FLOW_COOKIE_CHARS",
    "MAX_LIVE_FLOWS",
    "OAUTH_ROUTE_PREFIX",
    "apply_flow_cookie",
    "authorization_url",
    "base_url",
    "browser_token_for",
    "callback_landing_target",
    "drop_finished_flow",
    "exchange_code",
    "pending_flow_provider",
    "provider_label",
    "provider_refused_the_code",
    "redirect_uri",
    "require_configured",
    "state_hash",
    "trusts_email_claim",
]
