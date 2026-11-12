"""Discover a customer's own OpenID Connect issuer configuration.

The issuer is a URL an organization administrator typed into a form, so it
names a host this deployment does not control and may not trust. The fetch
goes through the checked federation transport, with the response stream
capped in size and the request bounded by this deployment's own timeout
policy, so a hostile, slow, or oversized provider cannot exhaust memory or
hang the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from apron_auth.errors import ConfigurationError, OidcDiscoveryError
from apron_auth.models import TimeoutPolicy
from apron_auth.providers import oidc

from gateway.exceptions.identity_exceptions import OIDCDiscoveryError
from gateway.log_config import logger
from gateway.services.web_retrieval_network import (
    MAX_FEDERATION_FETCH_BYTES,
    BoundedFederationTransport,
    CheckedFederationTransport,
)

DISCOVERY_MAX_BYTES = MAX_FEDERATION_FETCH_BYTES
DISCOVERY_TIMEOUT_SECONDS = 10.0
# The connect limit stays at the policy's own default, which the policy refuses
# to let exceed the request limit.
DISCOVERY_TIMEOUTS = TimeoutPolicy(request_seconds=DISCOVERY_TIMEOUT_SECONDS)
DISCOVERY_TTL = timedelta(hours=12)


@dataclass(frozen=True)
class DiscoveredProvider:
    """A provider's configuration, read once and ready to store for ``DISCOVERY_TTL``."""

    resolved_issuer: str
    document: dict[str, Any]
    discovered_at: datetime


async def discover_connection_metadata(issuer: str, *, allow_private_hosts: bool) -> DiscoveredProvider:
    """Read and validate the configuration an organization's own OIDC issuer publishes.

    The document is fetched once, through the checked federation transport, and refused if it
    names a different issuer, advertises a code-challenge method set that excludes ``S256``, grows
    past ``DISCOVERY_MAX_BYTES``, or the exchange runs past ``DISCOVERY_TIMEOUT_SECONDS``.

    Raises:
        OIDCDiscoveryError: If the issuer could not be read into a usable configuration.
    """

    def transport_factory(_url: str) -> httpx.AsyncBaseTransport:
        return BoundedFederationTransport(
            CheckedFederationTransport(allow_private_hosts=allow_private_hosts),
            max_bytes=DISCOVERY_MAX_BYTES,
        )

    try:
        metadata = await oidc.discover(issuer, transport_factory=transport_factory, timeouts=DISCOVERY_TIMEOUTS)
    except (OidcDiscoveryError, ConfigurationError):
        logger.warning("OIDC issuer discovery failed for issuer %s", issuer, exc_info=True)
        raise OIDCDiscoveryError from None

    # NOTE: This refuses a provider naming no S256 method, including one that omits the field,
    # which RFC 8414 allows.
    if "S256" not in metadata.code_challenge_methods:
        logger.warning("OIDC issuer discovery refused issuer %s: no S256 code-challenge method advertised", issuer)
        raise OIDCDiscoveryError from None

    if metadata.issuer is None:  # pragma: no cover
        raise OIDCDiscoveryError from None

    return DiscoveredProvider(
        resolved_issuer=metadata.issuer,
        document=metadata.model_dump(mode="json"),
        discovered_at=datetime.now(UTC),
    )
