"""What the gateway checks itself of a URL a provider fetch instance fetches.

A provider follows redirects and resolves hostnames on its own side, so the
gateway can check only the URL it asks for and the URL the provider reports
finding the page at: the host must pass the domain policy, and must not name an
address the built-in fetcher refuses, as an IP address in any form or as
``localhost``. The fetch tool and a search's enrichment both apply them.
"""

import ipaddress
import socket

from gateway.services.tools._web_fetch._builtin import DOMAIN_POLICY_TAG, URL_INVALID_TAG
from gateway.services.web_retrieval_network import blocked_address_reason
from gateway.services.web_retrieval_policy import CanonicalHost, CanonicalWebURL, DomainPolicy, IPAddress


def provider_url_refusal(url: CanonicalWebURL, policy: DomainPolicy) -> str | None:
    """The tag that refuses ``url`` to a provider fetch instance, ``None`` when it may be fetched."""
    host = url.origin.host
    if not policy.permits(host):
        return DOMAIN_POLICY_TAG
    # The provider resolves a hostname itself, so only an address the URL names, and
    # localhost, can be checked here, by the built-in fetcher's rule for an address.
    address = _named_address(host)
    if _is_localhost(host) or (address is not None and blocked_address_reason(address) is not None):
        return URL_INVALID_TAG
    return None


def _named_address(host: CanonicalHost) -> IPAddress | None:
    """The address a host names: an IP literal, or an IPv4 shorthand a resolver reads too, such as ``127.1``."""
    if host.ip is not None:
        return host.ip
    try:
        return ipaddress.ip_address(socket.inet_aton(host.value))
    except OSError:
        return None


def _is_localhost(host: CanonicalHost) -> bool:
    """``localhost`` or a name under it, which resolves to loopback (RFC 6761)."""
    return host.value == "localhost" or host.value.endswith(".localhost")
