"""Build the apron-auth client an organization's own OIDC connection signs in through.

The configuration and the identity handler both come from one connection's discovered metadata,
never from hand-assembled pieces, so a handler can never answer for a config naming another issuer.
Discovery metadata is this deployment's record of a host it does not control, so every outbound call
here, the same as during discovery, goes through the checked federation transport.
"""

from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime

import httpx
from apron_auth import (
    IdentityHandler,
    ProviderConfig,
    ServerMetadata,
    StoredStateClient,
)
from apron_auth.errors import ConfigurationError
from apron_auth.providers import oidc

from gateway.exceptions.identity_exceptions import SsoNotAddressableError
from gateway.models.identity import OIDCConnection, oidc_flow_provider
from gateway.repositories.identity import OAuthPendingStateRepository
from gateway.services.identity._discovery import DiscoveredProvider
from gateway.services.oauth_service import DatabaseStateStore, pending_flow_provider
from gateway.services.secret_box import decrypt_secret
from gateway.services.web_retrieval_network import (
    MAX_FEDERATION_FETCH_BYTES,
    BoundedFederationTransport,
    CheckedFederationTransport,
)

# Requested beyond the ``openid`` scope every preset merges in, for the display fields apron-auth's
# generic OIDC identity handler fills from the ID token and userinfo.
_SIGN_IN_SCOPES = ("email", "profile")

_HANDLER_CACHE_MAX = 256
# Keyed on (resolved_issuer, discovered_at) rather than the connection id, so every connection that
# resolves to one issuer's current metadata shares a handler, and a refreshed discovery is a new
# cache entry rather than an explicit eviction nobody can forget to call. Not keyed on
# allow_private_hosts, which is this deployment's own setting and fixed for a worker's whole life.
_handler_cache: OrderedDict[tuple[str, datetime], IdentityHandler] = OrderedDict()


def _transport_factory_for(*, allow_private_hosts: bool) -> Callable[[str], httpx.AsyncBaseTransport]:
    """The ``transport_factory`` every apron-auth call for a connection is built with.

    One transport for every URL the library fetches, so the outbound checks and the byte ceiling
    apply to the token endpoint, the signing key set and the userinfo document alike.
    """

    def factory(_url: str) -> httpx.AsyncBaseTransport:
        return BoundedFederationTransport(
            CheckedFederationTransport(allow_private_hosts=allow_private_hosts),
            max_bytes=MAX_FEDERATION_FETCH_BYTES,
        )

    return factory


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
        transport_factory=_transport_factory_for(allow_private_hosts=allow_private_hosts),
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
    pending: OAuthPendingStateRepository,
    redirect_uri: str,
    allow_private_hosts: bool,
) -> StoredStateClient:
    """Build the client one organization's own OIDC connection signs in through.

    Returns a ``StoredStateClient``, the shape the dashboard's built-in sign-in already uses: the
    pending authorization has to survive between the authorization redirect and a separate callback
    request, which only a stored state admits. The configuration and the identity handler both come
    from ``metadata``, so the handler can never answer for a config naming another issuer.
    """
    try:
        config = _provider_config_for(connection, metadata, redirect_uri=redirect_uri)
    except ConfigurationError as error:
        # The library holds the rule, RFC 8252, so its refusal is translated here rather than
        # restated: a deployment with no https address of its own cannot be signed in to. The
        # provider's own addresses are held to the same rule by discovery, which refuses them there,
        # so the metadata reaching this is already known to pass and the redirect URI is what is
        # left to fail.
        raise SsoNotAddressableError from error
    handler = identity_handler_for(metadata, allow_private_hosts=allow_private_hosts)
    state_store = DatabaseStateStore(pending, oidc_flow_provider(connection.id))
    return StoredStateClient(
        config,
        state_store,
        identity_handler=handler,
        transport_factory=_transport_factory_for(allow_private_hosts=allow_private_hosts),
    )


__all__ = [
    "client_for",
    "identity_handler_for",
    "pending_flow_provider",
]
