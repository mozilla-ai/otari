"""The second credential an endpoint behind a proxy requires, and the rules a tenant's choice is held to.

A tenant endpoint often sits behind a CDN or identity-aware proxy that wants
its own credential, separate from the provider's. The method is a closed set
of ways to produce request headers. TLS client authentication is deliberately
not a member: it is presented when the connection is made, and the pinned
client's pools are keyed by address alone, so one organization's certificate
would reach another organization's connection to the same origin.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

# The namespace Otari's own headers live in, which a tenant may not write into.
_OTARI_PREFIX = "otari-"

# No colon, so a pseudo-header such as ``:authority`` cannot be spelled.
_HEADER_NAME = re.compile(r"^[A-Za-z0-9-]{1,64}$")

# Reserved for the connection itself rather than for a credential. RFC 9110
# section 7.6.1 and RFC 9112 section 6 name them.
_FRAMING_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# Headers a provider SDK already sends its own credential in. A tenant that set
# one would replace the provider credential with a value of its own choosing.
_CREDENTIAL_HEADERS = frozenset({"api-key", "authorization", "x-api-key", "x-goog-api-key"})


# The overlay entry key the resolved headers travel under. They are instance
# metadata rather than a provider argument, so the dispatch path strips the key
# before the rest of the entry reaches the provider call.
ENDPOINT_AUTH_HEADERS_KEY = "endpoint_auth_headers"

# How long before its expiry a credential is reported as due for rotation.
EXPIRY_WARNING_WINDOW = timedelta(days=14)


class EndpointAuthState(StrEnum):
    """How a stored credential stands against its own expiry.

    Reported so every reader draws the line in the same place. None of the
    members refuses a request.
    """

    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    VALID = "valid"


class EndpointAuthMethod(StrEnum):
    """How the credential for one endpoint is produced.

    Every member produces request headers, so what a dispatch or discovery
    caller receives is a header mapping whichever member a key names.
    """

    STATIC_HEADER = "static_header"


def refused_header_name_reason(name: str) -> str | None:
    """Why a tenant may not send a header called ``name``, or None.

    ``host`` is refused for a clear answer at write time. The pinned transport
    is what enforces it, by refusing a request whose Host is not the address it
    validated.
    """
    if not _HEADER_NAME.match(name):
        return "a name outside the allowed characters, which are letters, digits and hyphens"
    folded = name.lower()
    if folded == "host":
        return "the Host header, which names the endpoint the request is pinned to"
    if folded in _FRAMING_HEADERS:
        return "a header reserved for the connection itself"
    if folded in _CREDENTIAL_HEADERS:
        return "a header a provider sends its own credential in"
    if folded.startswith(_OTARI_PREFIX):
        return "a header in Otari's own namespace"
    return None


@dataclass(frozen=True)
class EndpointAuth:
    """One endpoint's second credential as it is stored.

    The fields only mean anything together, so they travel as a unit. A value
    with no ``method`` is the ordinary case: a key that dials an endpoint
    needing nothing but its provider credential.
    """

    method: str | None = None
    header_name: str | None = None
    encrypted_secret: str | None = None
    last4: str | None = None
    expires_at: datetime | None = None

    @property
    def declared(self) -> bool:
        """Whether this key carries an endpoint credential at all."""
        return self.method is not None

    def refused_reason(self, *, has_api_base: bool) -> str | None:
        """Why this credential cannot be served, or None.

        The expiry is deliberately not read here. It is a prediction about the
        answer the proxy gives on every request, so it must never be what
        refuses one.
        """
        if self.method is None:
            return None
        if not has_api_base:
            return "an endpoint credential on a key that dials the provider's own endpoint"
        if self.method != EndpointAuthMethod.STATIC_HEADER:
            return f"an unknown endpoint authentication method '{self.method}'"
        if not self.header_name:
            return "an endpoint credential with no header name"
        if self.encrypted_secret is None:
            return "an endpoint credential with no value"
        return refused_header_name_reason(self.header_name)

    def refused_expiry_reason(self, *, now: datetime) -> str | None:
        """Why this expiry cannot be stored, or None.

        An expiry is required so the organization has a rotation date to work
        to, and must be in the future so storing one is a decision rather than
        a formality. ``now`` is passed rather than read so a caller holds the
        clock.
        """
        if self.method is None:
            return None
        if self.expires_at is None:
            return "an endpoint credential with no expiry"
        if self.expires_at <= now:
            return "an endpoint credential that expired before it was stored"
        return None

    def state(self, *, now: datetime) -> EndpointAuthState | None:
        """How this credential stands against its expiry, or None where it declares none.

        ``now`` is passed rather than read so a caller holds the clock.
        """
        if self.expires_at is None:
            return None
        if self.expires_at <= now:
            return EndpointAuthState.EXPIRED
        if self.expires_at - now <= EXPIRY_WARNING_WINDOW:
            return EndpointAuthState.EXPIRING_SOON
        return EndpointAuthState.VALID

    def headers(self, *, secret: str | None) -> dict[str, str]:
        """The headers this credential adds to every request to its endpoint.

        Empty where the credential is not whole, so a caller adds the result
        unconditionally.
        """
        if self.method != EndpointAuthMethod.STATIC_HEADER or not self.header_name or secret is None:
            return {}
        return {self.header_name: secret}


def endpoint_auth_columns(auth: EndpointAuth | None) -> dict[str, object | None]:
    """Every column one endpoint credential occupies, cleared where there is none.

    A write names all of them whichever it is doing, so a row cannot keep a
    declaration with no value behind it.
    """
    return {
        "encrypted_endpoint_auth_secret": auth.encrypted_secret if auth is not None else None,
        "endpoint_auth_expires_at": auth.expires_at if auth is not None else None,
        "endpoint_auth_header_name": auth.header_name if auth is not None else None,
        "endpoint_auth_last4": auth.last4 if auth is not None else None,
        "endpoint_auth_method": auth.method if auth is not None else None,
    }
