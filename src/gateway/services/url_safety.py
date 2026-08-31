"""URL safety checks for outbound HTTP fetches the gateway makes on behalf of a request.

Three call sites with overlapping but not identical threat models:

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

* **Provider ``api_base``** (:func:`validate_provider_api_base`): the URL is
  operator-supplied (master-key gated, standalone-only), the same trust level
  as a config.yml provider endpoint. This one defaults to *allow-all*: the
  home-lab / self-hosted use case depends on private-network endpoints
  (``localhost``, RFC 1918), so the check is off unless an operator opts in via
  ``OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false``, at which point it blocks the
  same private/link-local/reserved ranges as the web-search path.

These checks are intentionally conservative: DNS rebinding can defeat host-based
allowlists. Production deployments should also enforce egress policy at the
network layer.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlsplit, urlunsplit

from gateway.core.config import parse_bool_env
from gateway.core.env import otari_env


def redact_url_secrets(value: str) -> str:
    """Mask credentials in a URL while keeping its shape recognizable.

    A ``user:pass@`` password is masked while the username stays visible. A
    single-component userinfo (``token@host``) is masked whole, since a lone
    userinfo component is more likely a bearer token than a benign username and
    the two cannot be told apart. Every query-parameter *value* is masked too
    (a denylist of "credential-looking" keys cannot be complete, and a viewer
    cannot know which value is a secret), while the keys stay visible so the
    reader can still see what is set. Scheme, host, and path are preserved.

    Lives here rather than beside the settings route that first needed it
    because the checks in this module are the other half of the same concern,
    and a service may not import the API layer: the workspace MCP list redacts
    a stored endpoint for a caller who may read it but not manage it.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return value

    netloc = parts.netloc
    if parts.username is not None or parts.password is not None:
        host = parts.hostname or ""
        # An IPv6 literal must keep its brackets or the rebuilt URL is malformed.
        if ":" in host:
            host = f"[{host}]"
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        if parts.password is not None:
            netloc = f"{parts.username or ''}:***@{host}"
        else:
            netloc = f"***@{host}"

    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        # quote_via with '*' left safe keeps the mask readable ('***', not '%2A%2A%2A').
        query = urlencode([(key, "***") for key, _ in pairs], quote_via=quote, safe="*")

    if netloc == parts.netloc and query == parts.query:
        return value
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


class UnsafeURLError(ValueError):
    """Raised when an MCP server URL is rejected by the safety checks."""


def _allow_loopback() -> bool:
    return otari_env("MCP_ALLOW_LOOPBACK", "true").lower() not in {"0", "false", "no"}


def _allow_private_hosts() -> bool:
    return otari_env("MCP_ALLOW_PRIVATE_HOSTS", "false").lower() in {"1", "true", "yes"}


async def validate_mcp_url(url: str, *, has_authorization_token: bool) -> None:
    """Reject URLs that are unsafe for the gateway to fetch.

    Async because DNS resolution (:func:`_resolve_all_async`) must not block
    the event loop: this is called from the request pipeline, not from
    request-body parsing, so other concurrent requests keep making progress
    while a slow/unresolvable hostname is looked up.

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
        addresses = await _resolve_all_async(host)
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

    await _reject_internal_host(host, host_label="fetch", override_var="OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS")


async def _reject_internal_host(host: str, *, host_label: str, override_var: str) -> None:
    """Reject a host that is (or resolves to) a private/link-local/reserved address.

    The shared literal-or-resolve + :func:`_blocked_reason` loop behind the
    web-search and provider-``api_base`` gates. Callers do their own scheme/host
    validation and allow-flag short-circuit first, then hand the bare hostname
    here. An unresolvable host is rejected (DNS-rebinding TOCTOU). ``host_label``
    names the host in error messages; ``override_var`` is the OTARI_ env var
    quoted in the rejection hint. Note ``validate_mcp_url`` does not use this: its
    loop has extra loopback-allowance semantics.
    """
    try:
        literal = ipaddress.ip_address(host)
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [literal]
    except ValueError:
        addresses = await _resolve_all_async(host)
        if not addresses:
            raise UnsafeURLError(
                f"{host_label} host {host!r} could not be resolved; rejecting to avoid "
                f"DNS-rebinding. Set {override_var}=true to override."
            ) from None

    for addr in addresses:
        reason = _blocked_reason(addr)
        if reason is not None:
            raise UnsafeURLError(
                f"{host_label} host {host!r} resolves to {addr} which is {reason}; "
                f"rejecting to prevent SSRF. Set {override_var}=true to override."
            )


def _allow_provider_private_hosts() -> bool:
    # Defaults to True (allow-all), the opposite of the MCP/web-search gates: an
    # operator-supplied api_base is master-key gated and the home-lab use case
    # depends on private endpoints, so the check is off until an operator opts in.
    # Uses the shared config bool parser so a spelling like `off` disables
    # allow-all (enables the gate) instead of silently falling open.
    return parse_bool_env(otari_env("PROVIDER_ALLOW_PRIVATE_HOSTS", "true"))


async def validate_provider_api_base(url: str) -> None:
    """Reject a provider ``api_base`` that resolves to an internal address.

    Opt-in and default-allow: an operator-supplied ``api_base`` is master-key
    gated and standalone-only, the same trust level as a config.yml provider
    endpoint, and the home-lab / self-hosted use case depends on private-network
    endpoints (``localhost``, RFC 1918). So this is a no-op unless an operator
    sets ``OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false``, which turns on the same
    private/link-local/reserved-range check the web-search path uses (loopback
    blocked too).

    The allow-flag short-circuit runs *before* scheme/host validation, inverting
    the order :func:`validate_mcp_url` and :func:`validate_outbound_fetch_url`
    use. That is deliberate: in the default allow-all state this must not impose
    any shape on the operator's ``api_base`` (any-llm validates it), so the whole
    check is skipped; scheme/host validation applies only once the gate is on.

    Scope: this gate covers the paths that *report* on an ``api_base`` (the
    connection-test endpoints and model discovery) and the credential write path
    that persists it (``POST /v1/provider-credentials`` and
    ``PATCH /v1/provider-credentials/{instance}``). It does not
    gate the chat dispatch that dials the endpoint for real on every request, so
    it is not a general egress control.

    Async to keep the event loop unblocked during DNS resolution (see
    :func:`_resolve_all_async`). Raises :class:`UnsafeURLError` on rejection;
    returns ``None`` on accept (including the default allow-all case).
    """
    if _allow_provider_private_hosts():
        return

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeURLError(f"provider api_base must use http or https, got {scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("provider api_base must include a hostname")

    await _reject_internal_host(host, host_label="provider api_base", override_var="OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS")
