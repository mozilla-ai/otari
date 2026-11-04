"""Build the apron-auth client an organization's own OIDC connection signs in through.

The configuration and the identity handler both come from one connection's discovered metadata,
never from hand-assembled pieces, so a handler can never answer for a config naming another issuer.
Discovery metadata is this deployment's record of a host it does not control, so every outbound call
here, the same as during discovery, goes through the checked federation transport.
"""

from collections import OrderedDict
from datetime import UTC, datetime
from enum import StrEnum

from apron_auth import (
    INVALID_CLIENT,
    UNAUTHORIZED_CLIENT,
    AuthorizationCallback,
    IdentityHandler,
    OAuthClient,
    OAuthError,
    ProviderConfig,
    ServerMetadata,
    StoredStateClient,
    TokenExchangeError,
)
from apron_auth.models import OAuthPendingState as PendingState
from apron_auth.providers import oidc
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.identity import OIDCConnection, oidc_flow_provider
from gateway.services.identity._discovery import DiscoveredProvider
from gateway.services.oauth_service import DatabaseStateStore, pending_flow_provider, provider_refused_the_code
from gateway.services.secret_box import decrypt_secret
from gateway.services.web_retrieval_network import CheckedFederationTransport

# Requested beyond the ``openid`` scope every preset merges in, for the display fields apron-auth's
# generic OIDC identity handler fills from the ID token and userinfo.
_SIGN_IN_SCOPES = ("email", "profile")

_HANDLER_CACHE_MAX = 256
# Keyed on (resolved_issuer, discovered_at) rather than the connection id, so every connection that
# resolves to one issuer's current metadata shares a handler, and a refreshed discovery is a new
# cache entry rather than an explicit eviction nobody can forget to call. Not keyed on
# allow_private_hosts, which is this deployment's own setting and fixed for a worker's whole life.
_handler_cache: OrderedDict[tuple[str, datetime], IdentityHandler] = OrderedDict()

# A deliberately invalid code and state for the credential probe below, which redeems nothing and
# never reaches a browser.
_PROBE_STATE = "connection-credential-check"
_PROBE_CODE = "connection-credential-check"
_CREDENTIAL_REFUSAL_CODES = frozenset({INVALID_CLIENT, UNAUTHORIZED_CLIENT})


class ConnectionCredentialCheck(StrEnum):
    """Whether a provider's token endpoint accepted one connection's client ID and secret.

    ``INCONCLUSIVE`` covers every answer that proves neither: an outage, a refusal this probe did
    not anticipate, or a token endpoint that redeemed the deliberately invalid code it was sent.
    """

    ACCEPTED = "accepted"
    INCONCLUSIVE = "inconclusive"
    REFUSED = "refused"


def identity_handler_for(metadata: DiscoveredProvider, *, allow_private_hosts: bool) -> IdentityHandler:
    """The identity handler for one issuer's discovered metadata, memoized per worker.

    Bounded at ``_HANDLER_CACHE_MAX`` entries, with the least recently used dropped once a new one
    would exceed it.
    """
    key = (metadata.resolved_issuer, metadata.discovered_at)
    cached = _handler_cache.get(key)
    if cached is not None:
        _handler_cache.move_to_end(key)
        return cached
    handler = oidc.identity_handler(
        ServerMetadata.model_validate(metadata.document),
        transport_factory=lambda _url: CheckedFederationTransport(allow_private_hosts=allow_private_hosts),
    )
    _handler_cache[key] = handler
    if len(_handler_cache) > _HANDLER_CACHE_MAX:
        _handler_cache.popitem(last=False)
    return handler


def _provider_config_for(
    connection: OIDCConnection, metadata: DiscoveredProvider, *, redirect_uri: str
) -> ProviderConfig:
    """Build one connection's apron-auth provider configuration from its discovered metadata.

    ``metadata.document`` already carries the issuer and whether it advertises the ``iss``
    authorization-response parameter, so ``oidc.preset`` sets ``issuer`` and ``require_iss`` from it
    (RFC 9207): the mix-up check this config enforces is real rather than the no-op it is where
    neither is set.
    """
    return oidc.preset(
        client_id=connection.client_id,
        client_secret=decrypt_secret(connection.encrypted_client_secret),
        scopes=list(_SIGN_IN_SCOPES),
        redirect_uri=redirect_uri,
        metadata=ServerMetadata.model_validate(metadata.document),
    )


def client_for(
    connection: OIDCConnection,
    metadata: DiscoveredProvider,
    *,
    db: AsyncSession,
    redirect_uri: str,
    allow_private_hosts: bool,
) -> StoredStateClient:
    """Build the client one organization's own OIDC connection signs in through.

    Returns a ``StoredStateClient``, the shape the dashboard's built-in sign-in already uses: the
    pending authorization has to survive between the authorization redirect and a separate callback
    request, which only a stored state admits. The configuration and the identity handler both come
    from ``metadata``, so the handler can never answer for a config naming another issuer.

    NOTE: apron-auth sends the token request through authlib, which resolves onto ``httpx2`` rather
    than the ``httpx`` this transport is built on whenever ``httpx2`` is importable, which it is in
    this deployment. A real exchange then fails inside ``httpx2``'s own response handling with an
    unrelated ``AssertionError`` rather than a classified apron-auth error.
    """
    config = _provider_config_for(connection, metadata, redirect_uri=redirect_uri)
    handler = identity_handler_for(metadata, allow_private_hosts=allow_private_hosts)
    state_store = DatabaseStateStore(db, oidc_flow_provider(connection.id))
    return StoredStateClient(
        config,
        state_store,
        identity_handler=handler,
        transport_factory=lambda _url: CheckedFederationTransport(allow_private_hosts=allow_private_hosts),
    )


async def check_connection_credentials(
    connection: OIDCConnection, metadata: DiscoveredProvider, *, redirect_uri: str, allow_private_hosts: bool
) -> ConnectionCredentialCheck:
    """Probe whether the provider's token endpoint accepts this connection's client ID and secret.

    Sends a deliberately invalid authorization code, so nothing is redeemed and no browser is
    involved. ``invalid_grant`` means the provider rejected the code while accepting the client,
    which is what this probe is asking; ``invalid_client`` or ``unauthorized_client`` means it
    rejected the client. Every other answer, including a token endpoint that cannot be reached, is
    reported as inconclusive rather than risking a false green.

    NOTE: This is a real token-endpoint request carrying a code nothing issued, so it can appear in
    the provider's own access or audit logs as a failed exchange.

    NOTE: Also blocked by the authlib/``httpx2`` mismatch ``client_for`` documents: in this
    deployment, the request below fails with an unrelated ``AssertionError`` rather than reaching
    the token endpoint at all.
    """
    config = _provider_config_for(connection, metadata, redirect_uri=redirect_uri)
    client = OAuthClient(
        config, transport_factory=lambda _url: CheckedFederationTransport(allow_private_hosts=allow_private_hosts)
    )
    pending = PendingState(
        state=_PROBE_STATE,
        redirect_uri=redirect_uri,
        nonce=_PROBE_STATE,
        created_at=datetime.now(UTC).timestamp(),
    )
    callback = AuthorizationCallback(code=_PROBE_CODE, state=_PROBE_STATE, iss=config.issuer)
    try:
        await client.exchange_code(callback, pending_state=pending)
    except TokenExchangeError as error:
        if provider_refused_the_code(error):
            return ConnectionCredentialCheck.ACCEPTED
        if error.error_code in _CREDENTIAL_REFUSAL_CODES:
            return ConnectionCredentialCheck.REFUSED
        return ConnectionCredentialCheck.INCONCLUSIVE
    except OAuthError:
        return ConnectionCredentialCheck.INCONCLUSIVE
    else:
        return ConnectionCredentialCheck.INCONCLUSIVE


__all__ = [
    "ConnectionCredentialCheck",
    "check_connection_credentials",
    "client_for",
    "identity_handler_for",
    "pending_flow_provider",
]
