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

**The row is bound to the browser as well as to the flow.** A ``code`` and a
``state`` travel together in one redirect URL, and that URL is written to the
access log and to browser history, so a row keyed on the state alone would let
whoever reads either finish the sign-in for the full TTL. So ``/authorize``
also sets an HttpOnly cookie carrying a random flow secret, the row keeps that
secret's digest, and the callback must present the cookie (RFC 9700, section
4.7.1). One secret per browser rather than per flow: a second tab starting a
sign-in reuses the cookie it finds, so neither tab's callback breaks the other.

What is this module's own is the two overrides in ``_provider_config``: the
presets widen scopes and, for Google, ask for offline access, and neither is
wanted here. See that function.

The browser keeps its ``sessionStorage`` copy of the state and still compares
it, which is not redundant. That check binds a callback to the *tab* that
started the flow, which a server-side row cannot do; this one binds it to a
flow this deployment actually started, which the browser cannot do. They fail
in different directions. See ``web/src/features/auth/OAuthCallbackPage.tsx``.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import quote

from apron_auth import OAuthClient, ProviderConfig
from apron_auth.errors import StateError
from apron_auth.models import OAuthPendingState as PendingState
from apron_auth.providers import github as apron_github
from apron_auth.providers import google as apron_google
from fastapi import Response
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.core.config import OAUTH_PROVIDERS, GatewayConfig
from gateway.log_config import logger
from gateway.models.tenancy import OAUTH_STATE_TTL_SECONDS, OAuthPendingState
from gateway.services.tenancy.errors import (
    OAuthExchangeError,
    OAuthNotConfiguredError,
    OAuthStateError,
)

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
class _Provider:
    """One provider's scopes, and the name it writes itself under."""

    label: str
    scopes: tuple[str, ...]


# The cookie that binds a pending authorization to the browser that started it.
# Scoped to the OAuth routes, which are the only ones that read it.
FLOW_COOKIE_NAME = "otari_oauth_flow"
_FLOW_COOKIE_PATH = "/v1/auth/oauth"
# What ``secrets.token_urlsafe(32)`` produces; anything else in the cookie is
# not ours and is replaced rather than reused.
_FLOW_SECRET_LENGTH = 43

_PROVIDERS: dict[str, _Provider] = {
    "google": _Provider(label="Google", scopes=("openid", "email", "profile")),
    "github": _Provider(label="GitHub", scopes=("read:user", "user:email")),
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
    return f"{base_url(config)}/auth/{provider}/callback"


def base_url(config: GatewayConfig) -> str:
    """This deployment's own address with no trailing slash, or an empty string.

    One reading of ``public_base_url`` for everything OAuth derives from it, so
    the authorization request, the exchange, and the redirect that lands a
    browser back all agree. It may carry a **path prefix**: a gateway served at
    ``https://example.com/otari`` is a supported shape, and ``Mailer.link``
    already builds its links that way, so a root-absolute answer here would send
    such a deployment's callback to the wrong origin path.
    """
    return (config.public_base_url or "").rstrip("/")


def callback_landing_target(config: GatewayConfig, provider: str, query: str) -> str:
    """Where ``/auth/{provider}/callback`` sends the browser to finish signing in.

    The dashboard's own hash route, carrying whatever query the provider
    appended. Built here rather than in the route so it reads the same
    ``public_base_url`` the redirect URI does, path prefix included.

    Relative to that base rather than to the request, because the request's path
    is what a reverse proxy may already have rewritten, and this has to name a
    URL in the browser's address bar rather than in this process.
    """
    target = f"{base_url(config)}/#/auth/{quote(provider, safe='')}/callback"
    return f"{target}?{query}" if query else target


class _DatabaseStateStore:
    """apron-auth's ``StateStore``, backed by ``oauth_pending_state``.

    Two methods and no repository, because it is not the shape a repository
    serves: apron-auth calls these, and both are a single statement against one
    table that nothing else reads.

    Rows are staged on the caller's transaction and never committed here. The
    route owns the commit boundary, which is what lets a callback that fails
    after consuming a state roll the consumption back and leave the person a
    flow to retry.
    """

    def __init__(self, db: AsyncSession, provider: str, flow_hash: str) -> None:
        self._db = db
        self._provider = provider
        self._flow_hash = flow_hash

    async def save(self, state: PendingState) -> None:
        """Stage one pending authorization, and sweep whatever has expired."""
        if state.metadata:
            # The row has no column for it, so a caller that starts relying on
            # apron-auth's save/consume round trip finds out at the write rather
            # than by reading an empty ``TokenSet.context`` later.
            msg = "oauth_pending_state does not carry apron-auth state metadata"
            raise ValueError(msg)
        now = datetime.now(UTC)
        # The sweep rides here rather than on a scheduler because this is the
        # only write the table takes, so it is the only place that can grow it.
        # Same reason ``create_dashboard_session`` prunes on the way in.
        await self._db.execute(delete(OAuthPendingState).where(col(OAuthPendingState.expires_at) <= now))
        self._db.add(
            OAuthPendingState(
                state_hash=_state_hash(state.state),
                provider=self._provider,
                flow_hash=self._flow_hash,
                code_verifier=state.code_verifier,
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

        ``provider`` and the flow secret are compared after the row is claimed
        rather than added to the WHERE clause, so a state minted for one
        provider and returned to another, or presented from a browser other
        than the one that started it, is refused as itself rather than as an
        unknown state. The delete is staged on the request's transaction and
        the refusal keeps that transaction from committing, so the row survives
        for the callback it was minted for.
        """
        row = (
            await self._db.execute(
                delete(OAuthPendingState)
                .where(col(OAuthPendingState.state_hash) == _state_hash(state_key))
                .returning(
                    col(OAuthPendingState.provider),
                    col(OAuthPendingState.flow_hash),
                    col(OAuthPendingState.code_verifier),
                    col(OAuthPendingState.redirect_uri),
                    col(OAuthPendingState.created_at),
                    col(OAuthPendingState.expires_at),
                )
            )
        ).first()
        if row is None:
            return None
        provider, flow_hash, code_verifier, redirect_uri, created_at, expires_at = row
        if provider != self._provider:
            logger.warning("OAuth state minted for %s was returned to %s", provider, self._provider)
            return None
        if not hmac.compare_digest(flow_hash, self._flow_hash):
            logger.warning("OAuth %s callback came from a browser other than the one that started it", provider)
            return None
        if expires_at <= datetime.now(UTC):
            return None
        return PendingState(
            state=state_key,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            created_at=created_at.timestamp(),
        )


def _flow_hash(flow_secret: str) -> str:
    """The digest a row keeps of the browser's flow secret; same reasoning as ``_state_hash``."""
    return hashlib.sha256(flow_secret.encode()).hexdigest()


def _state_hash(state: str) -> str:
    """The key a state is stored under: its SHA-256, hex.

    The state is a 256-bit random value from ``secrets``, so this is a lookup
    key and not a password hash; a single round is right and a KDF would be
    theater. What it buys is that a reader of the table holds digests, and the
    callback compares against the preimage.
    """
    return hashlib.sha256(state.encode()).hexdigest()


def flow_secret_for(existing: str | None) -> str:
    """The flow secret this browser's authorizations are bound under.

    Reused when the browser already holds one of ours, so a second tab starting
    a sign-in does not invalidate the first tab's; minted otherwise. A cookie
    that is not the shape this module issues is somebody else's and is replaced.
    """
    if existing and len(existing) == _FLOW_SECRET_LENGTH and _is_urlsafe(existing):
        return existing
    return secrets.token_urlsafe(32)


def _is_urlsafe(value: str) -> bool:
    return all(character.isalnum() or character in "-_" for character in value)


def apply_flow_cookie(response: Response, secret: str, *, secure: bool) -> None:
    """Set (or refresh) the flow cookie with the attributes the session cookie uses.

    ``Lax`` rather than ``Strict`` because the request that spends it is a
    same-origin POST from the dashboard, which either setting carries; ``Lax``
    just does not depend on that staying true. The path keeps it off every
    request that is not an OAuth one.
    """
    response.set_cookie(
        FLOW_COOKIE_NAME,
        secret,
        max_age=OAUTH_STATE_TTL_SECONDS,
        httponly=True,
        secure=secure,
        samesite="lax",
        path=_FLOW_COOKIE_PATH,
    )


async def authorization_url(
    config: GatewayConfig, provider: str, *, db: AsyncSession, flow_secret: str
) -> tuple[str, str]:
    """The provider consent screen to send the browser to, and the state to keep.

    Staged, not committed: the caller commits, so an authorization URL is never
    handed back over a row that failed to write. ``flow_secret`` is the value
    the browser will hold in ``FLOW_COOKIE_NAME``; only its digest is stored.

    Raises:
        OAuthNotConfiguredError: If this deployment configured no client
            credentials for ``provider``, or does not know its own address.

    """
    client = _client(config, provider, db, flow_secret)
    url, pending = await client.get_authorization_url(redirect_uri=redirect_uri(config, provider))
    return url, pending.state


async def exchange_code(
    config: GatewayConfig,
    provider: str,
    *,
    code: str,
    state: str,
    flow_secret: str | None,
    db: AsyncSession,
) -> OAuthIdentity:
    """Trade an authorization code for the identity the provider vouches for.

    ``state`` is claimed before the code is sent anywhere, so a callback this
    deployment never started costs one indexed delete and no outbound call. The
    claim also yields the PKCE verifier and the redirect URI the authorization
    request was built with, which is why neither is passed here.

    ``flow_secret`` is the browser's ``FLOW_COOKIE_NAME`` cookie, or ``None``
    when it sent none. A callback without it is refused before the database is
    touched: there is no row it could match.

    Raises:
        OAuthNotConfiguredError: If this deployment configured no client
            credentials for ``provider``, or does not know its own address.
        OAuthStateError: If ``state`` names no authorization this deployment is
            still waiting on, for ``provider``, from this browser.
        OAuthExchangeError: If the exchange or the identity fetch fails, for any
            reason. The provider's own words stay on the traceback and out of
            the response; see that error's docstring.

    """
    if flow_secret is None:
        logger.warning("Refused a %s callback that carried no flow cookie", provider)
        raise OAuthStateError
    client = _client(config, provider, db, flow_secret)
    try:
        tokens = await client.exchange_code(code=code, state=state)
        profile = await client.fetch_identity(tokens)
    except StateError as error:
        # Ahead of the catch-all below, which would otherwise render a refused
        # state as "the provider did not complete the sign-in" and send an
        # operator looking at a provider that was never contacted.
        logger.warning("Refused a %s callback whose state matched no pending sign-in", provider)
        raise OAuthStateError from error
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


def _client(config: GatewayConfig, provider: str, db: AsyncSession, flow_secret: str) -> OAuthClient:
    """The apron-auth client that builds ``provider``'s authorization URL and spends its code."""
    client_id, client_secret = _credentials(config, provider)
    preset = apron_google.preset if provider == "google" else apron_github.preset
    identity_handler = (
        apron_google.GoogleIdentityHandler() if provider == "google" else apron_github.GitHubIdentityHandler()
    )
    provider_config, _revocation_handler = preset(
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(_PROVIDERS[provider].scopes),
        redirect_uri=redirect_uri(config, provider),
    )
    return OAuthClient(
        _as_configured_here(provider_config, provider),
        state_store=_DatabaseStateStore(db, provider, _flow_hash(flow_secret)),
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
    "FLOW_COOKIE_NAME",
    "OAuthIdentity",
    "apply_flow_cookie",
    "authorization_url",
    "base_url",
    "callback_landing_target",
    "exchange_code",
    "flow_secret_for",
    "provider_label",
    "redirect_uri",
    "require_configured",
]
