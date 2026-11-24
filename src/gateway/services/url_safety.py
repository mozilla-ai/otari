"""URL safety checks for outbound HTTP fetches the gateway makes on behalf of a request.

Five call sites with overlapping but not identical threat models:

* **MCP server endpoints** (:func:`validate_mcp_url`): a workspace, an
  organization or the caller chose the URL, never the operator, so it must reach
  a public address and nothing else. No setting relaxes that. Also enforces TLS
  when a bearer token is supplied. This check is what a person sees when they
  save or send a URL; the MCP clients dial through the pinned, no-redirect
  transport in ``services/providers/_owned_endpoint_network.py``, which applies
  the same rule to the address it connects to.

* **Guardrail endpoints** (:func:`validate_mcp_url` again, with a label of its
  own): an organization admin or the caller writes them, so the same rule
  applies. TLS is enforced unconditionally for a stored definition, which holds
  a vendor credential beside its endpoint.

* **Web-search result URLs** (:func:`validate_outbound_fetch_url`) — URL
  comes from a third-party search engine via the configured search backend.
  Tighter defaults: loopback is blocked too (the gateway has no legitimate
  reason to fetch search results from itself). Gated by
  ``OTARI_WEB_SEARCH_ALLOW_PRIVATE_HOSTS`` for operators with unusual
  setups (private indexes etc.).

* **Provider ``api_base``** (:func:`validate_provider_api_base`): the URL is
  operator-supplied (master-key gated), the same trust level
  as a config.yml provider endpoint. This one defaults to *allow-all*: the
  home-lab / self-hosted use case depends on private-network endpoints
  (``localhost``, RFC 1918), so the check is off unless an operator opts in via
  ``OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS=false``, at which point it blocks the
  same private/link-local/reserved ranges as the web-search path. An
  ``api_base`` a tenant chose (an organization's provider or search key, an
  owned endpoint) is never checked here: it is held to public addresses and
  dialed through the pinned client in ``services/providers/_owned_endpoint_network.py``.

* **Workload identity discovery and JWKS** (:func:`validate_federation_url`):
  the issuer is operator-configured and its signing-key location comes from
  its discovery document. https only, and private hosts are blocked unless
  the ``federation_allow_private_hosts`` setting is on.

Each check validates a URL's shape before its address: a host no HTTP client can
dial is refused as invalid, so an address check only ever reports on the host
the caller wrote.

A check here resolves the host once, so on its own it does not stop DNS
rebinding: a client that must not be re-pointed dials the address that was
checked (see ``web_retrieval_network``). Production deployments should also
enforce egress policy at the network layer.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import ParseResult, parse_qsl, quote, urlencode, urlparse, urlsplit, urlunsplit

from gateway.core.config import parse_bool_env
from gateway.core.env import otari_env
from gateway.core.settings.providers import https_or_loopback_problem
from gateway.core.settings.tools import PrivateHostEntry, canonical_host, parse_mcp_private_hosts


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
        # Everything after the last "@" is host[:port], taken as written rather
        # than rebuilt from `hostname` and `port`. Deliberately not
        # `SplitResult.port`, which parses lazily and raises ValueError on a
        # malformed port ("host:bad"): this runs on a URL that is already stored,
        # so it must not raise. Taking the text also keeps an IPv6 literal's
        # brackets without a special case.
        host_and_port = parts.netloc.rpartition("@")[2]
        if parts.password is not None:
            netloc = f"{parts.username or ''}:***@{host_and_port}"
        else:
            netloc = f"***@{host_and_port}"

    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        # quote_via with '*' left safe keeps the mask readable ('***', not '%2A%2A%2A').
        query = urlencode([(key, "***") for key, _ in pairs], quote_via=quote, safe="*")

    if netloc == parts.netloc and query == parts.query:
        return value
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


class UnsafeURLError(ValueError):
    """Raised when a URL is rejected by the safety checks."""


class UnresolvableHostError(UnsafeURLError):
    """A URL's host did not resolve when it was checked, which a later check may answer differently."""


def _authority_detail(label: str) -> str:
    return (
        f"{label} URL is not valid: write the host as a name, an IPv4 address, or an IPv6 "
        "address in brackets, for example https://[2400:8901::1]:8787/"
    )


def _parse_safe_url(url: str, *, label: str) -> ParseResult:
    """Parse ``url``, refusing an authority no HTTP client can dial.

    Validates the shape before the address checks read the host.
    An unbracketed IPv6 host ends at its first colon,
    which leaves a hostname the caller never wrote for a resolver to answer.

    ``label`` names the surface the URL was written for, as in :func:`validate_mcp_url`.
    Raises :class:`UnsafeURLError` on rejection.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        raise UnsafeURLError(_authority_detail(label)) from None

    try:
        # NOTE: The port is validated only when it is read.
        _ = parsed.port
    except ValueError:
        host_and_port = parsed.netloc.rpartition("@")[2]
        # A well-formed unbracketed authority holds one colon at most, so
        # anything beyond that is an IPv6 address written without brackets.
        if "[" not in host_and_port and host_and_port.count(":") > 1:
            raise UnsafeURLError(_authority_detail(label)) from None
        raise UnsafeURLError(f"{label} URL port must be a number from 0 to 65535") from None

    return parsed


_DEFAULT_PORTS = {"http": 80, "https": 443}
# Metadata services reachable at an address none of the range tests below calls
# link-local: AWS's IPv6 one sits in a ULA range, Alibaba Cloud's in shared space.
_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_METADATA_ADDRESSES = frozenset({ipaddress.ip_address("fd00:ec2::254"), ipaddress.ip_address("100.100.100.200")})


@dataclass(frozen=True, slots=True)
class PrivateHosts:
    """The operator's ``mcp_private_hosts``: exact hosts an MCP server URL may name on a private address."""

    entries: frozenset[PrivateHostEntry]

    def admits(self, host: str | None, port: int | None, scheme: str) -> bool:
        """Whether a URL's host, and its port where the entry names one, is listed exactly."""
        if not host:
            return False
        name = canonical_host(host)
        effective_port = port if port is not None else _DEFAULT_PORTS.get(scheme.lower())
        return (name, None) in self.entries or (name, effective_port) in self.entries


def mcp_private_hosts() -> PrivateHosts:
    """The deployment's ``mcp_private_hosts``, read from the environment the config bridges it into."""
    return _parsed_private_hosts(otari_env("MCP_PRIVATE_HOSTS", ""))


@lru_cache(maxsize=4)
def _parsed_private_hosts(raw: str) -> PrivateHosts:
    return PrivateHosts(parse_mcp_private_hosts(raw))


def never_admitted_reason(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Why ``addr`` is refused even for a host the operator listed, or ``None``.

    Link-local (the cloud metadata range among it), unspecified, multicast and the
    metadata addresses outside link-local. An IPv6 address that carries an IPv4
    one (IPv4-mapped, 6to4 in ``2002::/16``, NAT64 in ``64:ff9b::/96``) is read as
    the IPv4 address it carries, since a translator on the path delivers it there.
    """
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        elif addr.sixtofour is not None:
            addr = addr.sixtofour
        elif addr in _NAT64_PREFIX:
            addr = ipaddress.IPv4Address(int(addr) & 0xFFFFFFFF)
    if addr in _METADATA_ADDRESSES:
        return "a cloud metadata address"
    if addr.is_link_local:
        return "link-local"
    if addr.is_unspecified:
        return "unspecified (0.0.0.0/::)"
    if addr.is_multicast:
        return "multicast"
    return None


async def validate_mcp_url(
    url: str,
    *,
    has_authorization_token: bool,
    label: str = "MCP server",
    private_hosts: PrivateHosts | None = None,
) -> None:
    """Refuse a URL a tenant chose unless it is http(s) and every address its host resolves to is public.

    Async because DNS resolution (:func:`_resolve_all_async`) must not block
    the event loop: this is called from the request pipeline, not from
    request-body parsing, so other concurrent requests keep making progress
    while a slow/unresolvable hostname is looked up.

    ``label`` names the surface the URL was written for, so a caller that is
    not an MCP server does not tell its own admin about one: the organization
    guardrail definition store passes ``guardrail endpoint``.

    A host that does not resolve is refused, since a later lookup could answer
    with an internal address. Passing this check does not make a later
    connection safe on its own: the name can be re-pointed afterwards, which is
    why the clients that dial these URLs pin the address they connect to.

    ``private_hosts`` is the operator's list, passed only for an MCP server's URL
    and the sign-in addresses it leads to. A URL whose host (and port, where the
    entry names one) is listed exactly may resolve to a loopback or private
    address, never to one :func:`never_admitted_reason` refuses.

    Raises :class:`UnsafeURLError` on rejection. Returns ``None`` on accept.
    """
    parsed = _parse_safe_url(url, label=label)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeURLError(f"{label} URL must use http or https, got {scheme!r}")
    if scheme == "http" and has_authorization_token:
        raise UnsafeURLError(f"{label} URL must use https when a credential travels with it")

    host = parsed.hostname
    if not host:
        raise UnsafeURLError(f"{label} URL must include a hostname")
    if host.endswith("."):
        # The pinned transport dials the name without the dot, so a URL written with one never connects.
        raise UnsafeURLError(f"{label} host {host!r} ends with a dot; write the host without it")
    listed = private_hosts is not None and private_hosts.admits(host, parsed.port, scheme)

    try:
        literal = ipaddress.ip_address(host)
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [literal]
    except ValueError:
        # Before resolving: platforms read ``0177.0.0.1`` or ``127.1`` differently, and a
        # single-label or ``.local`` name only means something on the gateway's own network.
        problem = None if listed else _host_text_problem(host.rstrip(".").lower())
        if problem is not None:
            raise UnsafeURLError(
                f"{label} {problem}; rejecting to prevent SSRF. The {label} must be on a public address."
            ) from None
        addresses = await _resolve_all_async(host)
        if not addresses:
            raise UnresolvableHostError(
                f"{label} host {host!r} could not be resolved at validation time; "
                "rejecting to avoid DNS-rebinding (a later lookup could resolve to a "
                "private address)."
            )

    for addr in addresses:
        if listed:
            reason = never_admitted_reason(addr)
            if reason is not None:
                raise UnsafeURLError(
                    f"{label} host {host!r} resolves to {addr} which is {reason}, which is refused even for "
                    "a host in mcp_private_hosts."
                )
            continue
        reason = _blocked_reason(addr)
        if reason is not None:
            raise UnsafeURLError(
                f"{label} host {host!r} resolves to {addr} which is {reason}; "
                f"rejecting to prevent SSRF. The {label} must be on a public address."
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
    # Catches what the tests above miss, such as RFC 6598 shared space (100.64.0.0/10),
    # which CGNAT, Tailscale and some cluster networks use.
    if not addr.is_global:
        return "outside the public address space"
    return None


async def public_addresses_of(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """The publicly routable addresses ``host`` resolves to now; empty when none, or when it does not resolve."""
    return [addr for addr in await _resolve_all_async(host) if _blocked_reason(addr) is None]


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
    by third-party content (search-engine results, etc.). Loopback is blocked
    because the gateway has no legitimate reason to fetch user-search results
    from itself.

    Async to keep the event loop unblocked under fan-out — see
    :func:`_resolve_all_async`. Raises :class:`UnsafeURLError` on rejection.
    """
    parsed = _parse_safe_url(url, label="fetch")
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
    quoted in the rejection hint. ``validate_mcp_url`` does not use this, since
    no setting relaxes it and so it has no override to quote.
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
    gated, the same trust level as a config.yml provider
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
    that persists it (creating or editing a hosted provider). It does not
    gate the chat dispatch that dials the endpoint for real on every request, so
    it is not a general egress control.

    Async to keep the event loop unblocked during DNS resolution (see
    :func:`_resolve_all_async`). Raises :class:`UnsafeURLError` on rejection;
    returns ``None`` on accept (including the default allow-all case).
    """
    if _allow_provider_private_hosts():
        return

    parsed = _parse_safe_url(url, label="provider api_base")
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeURLError(f"provider api_base must use http or https, got {scheme!r}")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("provider api_base must include a hostname")

    await _reject_internal_host(host, host_label="provider api_base", override_var="OTARI_PROVIDER_ALLOW_PRIVATE_HOSTS")


async def validate_federation_url(url: str, *, allow_private_hosts: bool = False) -> None:
    """Reject an OpenID discovery or JWKS URL that is not https or that reaches an internal address.

    The issuer is operator-configured, but the JWKS location comes from the
    issuer's discovery document, so it is checked like any URL the gateway did
    not write itself. Private hosts are refused unless ``allow_private_hosts``,
    the ``federation_allow_private_hosts`` setting, is on for an issuer on the
    deployment's own network. Raises :class:`UnsafeURLError` on rejection.
    """
    parsed = _parse_safe_url(url, label="federation")
    if parsed.scheme.lower() != "https":
        raise UnsafeURLError("federation URL must use https")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("federation URL must include a hostname")
    if allow_private_hosts:
        return
    await _reject_internal_host(host, host_label="federation", override_var="OTARI_FEDERATION_ALLOW_PRIVATE_HOSTS")


def validate_public_https_url_shape(url: str, *, label: str) -> None:
    """Refuse a URL whose text alone shows the gateway should not offer it, without resolving its host.

    For a URL read in bulk from a public listing, where a lookup per entry is
    not affordable: it must be https, carry no credentials, and name a host
    that is neither an internal address literal nor a local name. It is a
    screen, not the SSRF check: whoever dials the URL still runs
    :func:`validate_mcp_url`, which resolves the host.

    Raises :class:`UnsafeURLError` on rejection.
    """
    parsed = _parse_safe_url(url, label=label)
    if parsed.scheme.lower() != "https":
        raise UnsafeURLError(f"{label} URL must use https")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError(f"{label} URL must not carry credentials")
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeURLError(f"{label} URL must include a hostname")
    problem = _host_text_problem(host)
    if problem is not None:
        raise UnsafeURLError(f"{label} {problem}")


def _host_text_problem(host: str) -> str | None:
    """Why ``host``, as written and unresolved, is not a public one, or ``None``."""
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        if "." not in host or host.endswith((".localhost", ".local", ".internal")):
            return f"host {host!r} is a local name"
        # No top-level domain is a number, so this is an address in a short or non-decimal form (127.1, 0x7f.1).
        if host.rpartition(".")[2].isdigit() or host.rpartition(".")[2].startswith("0x"):
            return f"host {host!r} is an address written as a name"
        return None
    reason = _blocked_reason(literal)
    return None if reason is None else f"host {host!r} is {reason}"


def validate_credential_address(url: str, *, label: str) -> None:
    """Refuse an address a credential travels to or a browser opens unless it is https, or http on this machine.

    Raises :class:`UnsafeURLError` on rejection.
    """
    _parse_safe_url(url, label=label)
    problem = https_or_loopback_problem(url)
    if problem is not None:
        raise UnsafeURLError(f"{label} URL {problem}")
