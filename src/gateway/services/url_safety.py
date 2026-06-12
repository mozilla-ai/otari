"""URL safety checks for outbound HTTP fetches the gateway makes on behalf of a request.

Two distinct call sites with overlapping but not identical threat models:

* **MCP server endpoints** (:func:`validate_mcp_url`) — URL comes from the
  request body. We block private/link-local/reserved IPs to prevent SSRF.
  Loopback is allowed by default (useful for same-host sidecar deployments)
  and gated by ``OTARI_MCP_ALLOW_LOOPBACK``. Also enforces TLS when a
  bearer token is supplied.

* **Web-search result URLs** (:func:`validate_outbound_fetch_url`) — URL
  comes from a third-party search engine via the configured search backend.
  Tighter defaults: loopback is blocked too (the gateway has no legitimate
  reason to fetch search results from itself). Gated by
  ``OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS`` for operators with unusual
  setups (private indexes etc.).

These checks are intentionally conservative: DNS rebinding can defeat host-based
allowlists. Production deployments should also enforce egress policy at the
network layer.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

from gateway.core.env import otari_env


class UnsafeURLError(ValueError):
    """Raised when an MCP server URL is rejected by the safety checks."""


def _allow_loopback() -> bool:
    return otari_env("MCP_ALLOW_LOOPBACK", "true").lower() not in {"0", "false", "no"}


def _allow_private_hosts() -> bool:
    return otari_env("MCP_ALLOW_PRIVATE_HOSTS", "false").lower() in {"1", "true", "yes"}


def _resolve_all(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        try:
            out.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return out


def validate_mcp_url(url: str, *, has_authorization_token: bool) -> None:
    """Reject URLs that are unsafe for the gateway to fetch.

    Raises :class:`UnsafeURLError` on rejection. Returns ``None`` on accept.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeURLError(f"MCP server URL must use http or https, got {scheme!r}")
    if scheme == "http" and has_authorization_token:
        raise UnsafeURLError("MCP server URL must use https when an authorization_token is set")

    host = parsed.hostname
    if not host:
        raise UnsafeURLError("MCP server URL must include a hostname")

    if _allow_private_hosts():
        return

    try:
        literal = ipaddress.ip_address(host)
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [literal]
    except ValueError:
        addresses = _resolve_all(host)
        if not addresses:
            # Couldn't resolve the host at validation time. Rejecting is the
            # safer default: a hostname that fails to resolve here could
            # later resolve to an internal address at fetch time (the
            # classic DNS-rebinding TOCTOU). Operators that explicitly want
            # to allow unresolvable hostnames (private DNS,
            # hosts-file-driven setups, etc.) can opt in via
            # OTARI_MCP_ALLOW_PRIVATE_HOSTS, which short-circuits this
            # whole function above.
            raise UnsafeURLError(
                f"MCP server host {host!r} could not be resolved at validation time; "
                "rejecting to avoid DNS-rebinding (a later lookup could resolve to a "
                "private address). Set OTARI_MCP_ALLOW_PRIVATE_HOSTS=true to override."
            )

    for addr in addresses:
        if addr.is_loopback and _allow_loopback():
            continue
        reason = _blocked_reason(addr)
        if reason is not None:
            raise UnsafeURLError(
                f"MCP server host {host!r} resolves to {addr} which is {reason}; "
                "rejecting to prevent SSRF. Set OTARI_MCP_ALLOW_PRIVATE_HOSTS=true to override."
            )


def _blocked_reason(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    # Order matters: is_private returns True for unspecified/loopback/link-local too,
    # so more specific labels go first to produce useful error messages.
    if addr.is_unspecified:
        return "unspecified (0.0.0.0/::)"
    if addr.is_loopback:
        return "loopback"
    if addr.is_link_local:
        return "link-local"
    if addr.is_multicast:
        return "multicast"
    if addr.is_private:
        return "in a private range (RFC 1918 / ULA)"
    if addr.is_reserved:
        return "in a reserved range"
    return None


def _allow_web_search_private_hosts() -> bool:
    return otari_env("WEB_SEARCH_ALLOW_PRIVATE_HOSTS", "false").lower() in {"1", "true", "yes"}


async def _resolve_all_async(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Async DNS resolution. Off-loads to the loop's default resolver so the
    event loop isn't blocked while we wait — critical when the per-fetch
    fan-out can trigger many lookups concurrently.
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        try:
            out.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return out


async def validate_outbound_fetch_url(url: str) -> None:
    """Reject URLs that are unsafe for the gateway to fetch on behalf of a request.

    Used for per-page fetches that the gateway initiates against URLs supplied
    by third-party content (search-engine results, etc.). Stricter than
    :func:`validate_mcp_url`: loopback is blocked by default because the
    gateway has no legitimate reason to fetch user-search results from itself.

    Async to keep the event loop unblocked under fan-out — see
    :func:`_resolve_all_async`. Raises :class:`UnsafeURLError` on rejection.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeURLError(f"fetch URL must use http or https, got {scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("fetch URL must include a hostname")

    if _allow_web_search_private_hosts():
        return

    try:
        literal = ipaddress.ip_address(host)
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [literal]
    except ValueError:
        addresses = await _resolve_all_async(host)
        if not addresses:
            raise UnsafeURLError(
                f"fetch host {host!r} could not be resolved; rejecting to avoid "
                "DNS-rebinding. Set OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS=true to override."
            ) from None

    for addr in addresses:
        reason = _blocked_reason(addr)
        if reason is not None:
            raise UnsafeURLError(
                f"fetch host {host!r} resolves to {addr} which is {reason}; "
                "rejecting to prevent SSRF. Set OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS=true to override."
            )
