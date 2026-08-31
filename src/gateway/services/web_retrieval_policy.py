"""Canonical URL and domain-policy primitives for gateway web retrieval.

The public Fetch tool is introduced in a later change. This module is deliberately
independent of tool declarations and request models so Search and Fetch can use one
canonical identity for policy checks, redirects, DNS pinning, and provenance.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import TypeAlias

import httpx
import idna

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
MAX_WEB_URL_LENGTH = 8192


class WebURLValidationError(ValueError):
    """Raised when a web-retrieval URL cannot be safely canonicalized."""


class DomainRuleValidationError(ValueError):
    """Raised when a workspace domain rule is not a bare valid hostname."""


class DisjointDomainAllowListsError(ValueError):
    """Raised when two present allow-lists have no common host scope."""


@dataclass(frozen=True, slots=True)
class CanonicalHost:
    """One canonical DNS A-label or compressed IP literal."""

    value: str
    ip: IPAddress | None = None

    @property
    def is_ip(self) -> bool:
        return self.ip is not None

    @property
    def url_host(self) -> str:
        """Authority rendering suitable for URLs and HTTP Host values."""
        if isinstance(self.ip, ipaddress.IPv6Address):
            return f"[{self.value}]"
        return self.value


@dataclass(frozen=True, slots=True)
class CanonicalOrigin:
    """Canonical network origin, including its effective port."""

    scheme: str
    host: CanonicalHost
    port: int

    @property
    def authority(self) -> str:
        default_port = _DEFAULT_PORTS[self.scheme]
        if self.port == default_port:
            return self.host.url_host
        return f"{self.host.url_host}:{self.port}"


@dataclass(frozen=True, slots=True)
class CanonicalWebURL:
    """A normalized request URL and the identities derived from it."""

    url: httpx.URL
    origin: CanonicalOrigin

    @property
    def redirect_loop_key(self) -> tuple[str, str, int, bytes]:
        """Identity used for redirect-loop detection, excluding the fragment."""
        return (
            self.origin.scheme,
            self.origin.host.value,
            self.origin.port,
            self.url.raw_path,
        )

    @property
    def display_url(self) -> str:
        """Return scheme, canonical authority, and path, never query or fragment."""
        return str(self.url.copy_with(query=None, fragment=None))


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def canonicalize_host(value: str, *, error_type: type[ValueError] = WebURLValidationError) -> CanonicalHost:
    """Canonicalize a DNS name or IP literal without doing DNS resolution.

    DNS names use UTS #46 processing with STD3 rules. One terminal root dot is
    accepted and removed; any other empty label is invalid. IPv6 scope identifiers
    are rejected because they are interface-local and cannot be a portable policy
    identity.
    """
    if not value or _contains_control_character(value):
        raise error_type("hostname is empty or contains a control character")
    if value.endswith(".."):
        raise error_type("hostname may contain at most one terminal root dot")
    canonical_candidate = value[:-1] if value.endswith(".") else value
    if not canonical_candidate or ".." in canonical_candidate:
        raise error_type("hostname contains an empty label")
    if "%" in canonical_candidate and ":" in canonical_candidate:
        raise error_type("scoped IPv6 addresses are not allowed")

    try:
        parsed_ip = ipaddress.ip_address(canonical_candidate)
    except ValueError:
        parsed_ip = None
    if parsed_ip is not None:
        return CanonicalHost(value=parsed_ip.compressed, ip=parsed_ip)

    dns_name = canonical_candidate

    try:
        ascii_name = idna.encode(dns_name, uts46=True, std3_rules=True).decode("ascii").lower()
    except idna.IDNAError as exc:
        raise error_type("hostname is not valid IDNA") from exc
    if not ascii_name or ".." in ascii_name:
        raise error_type("hostname contains an empty label")
    return CanonicalHost(value=ascii_name)


def canonicalize_web_url(value: str, *, max_length: int = MAX_WEB_URL_LENGTH) -> CanonicalWebURL:
    """Parse an absolute HTTP(S) URL into the retrieval canonical form.

    Only the scheme, authority, and fragment are rewritten. HTTPX's serialized
    path and query are retained byte-for-byte so signed URLs keep their meaning.
    """
    if not isinstance(value, str) or not value:
        raise WebURLValidationError("URL must be a non-empty string")
    if len(value) > max_length:
        raise WebURLValidationError(f"URL exceeds the {max_length}-character limit")
    if _contains_control_character(value):
        raise WebURLValidationError("URL contains a control character")

    try:
        parsed = httpx.URL(value)
    except (httpx.InvalidURL, TypeError, ValueError) as exc:
        raise WebURLValidationError("URL is malformed") from exc

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise WebURLValidationError("URL must use http or https")
    if not parsed.is_absolute_url or not parsed.host:
        raise WebURLValidationError("URL must be absolute and include a hostname")
    if parsed.userinfo:
        raise WebURLValidationError("URL user information is not allowed")

    host = canonicalize_host(parsed.host)
    try:
        explicit_or_default_port = parsed.port
    except httpx.InvalidURL as exc:
        raise WebURLValidationError("URL has an invalid port") from exc
    effective_port = explicit_or_default_port or _DEFAULT_PORTS[scheme]
    if not 1 <= effective_port <= 65535:
        raise WebURLValidationError("URL port is outside the valid range")

    normalized_port = None if effective_port == _DEFAULT_PORTS[scheme] else effective_port
    try:
        normalized = parsed.copy_with(
            scheme=scheme,
            host=host.value,
            port=normalized_port,
            fragment=None,
        )
    except (httpx.InvalidURL, TypeError, ValueError) as exc:
        raise WebURLValidationError("URL authority is malformed") from exc

    return CanonicalWebURL(
        url=normalized,
        origin=CanonicalOrigin(scheme=scheme, host=host, port=effective_port),
    )


def resolve_redirect_url(current: CanonicalWebURL, location: str) -> CanonicalWebURL:
    """Resolve and canonicalize one redirect target relative to ``current``."""
    if not location or _contains_control_character(location):
        raise WebURLValidationError("redirect Location is missing or malformed")
    try:
        joined = current.url.join(location)
    except (httpx.InvalidURL, TypeError, ValueError) as exc:
        raise WebURLValidationError("redirect Location is malformed") from exc
    return canonicalize_web_url(str(joined))


def canonicalize_domain_rule(value: str) -> CanonicalHost:
    """Validate and canonicalize a workspace domain rule.

    Rules are bare DNS names or IP literals. DNS rules cover the named host and
    subdomains; IP rules match only the exact canonical address.
    """
    if not isinstance(value, str) or not value:
        raise DomainRuleValidationError("domain rule must be a non-empty string")
    if _contains_control_character(value):
        raise DomainRuleValidationError("domain rule contains a control character")
    if any(character in value for character in "/?#@"):
        raise DomainRuleValidationError("domain rule must not include URL syntax")
    if value.startswith("[") or value.endswith("]"):
        if not (value.startswith("[") and value.endswith("]")):
            raise DomainRuleValidationError("domain rule has malformed IPv6 brackets")
        value = value[1:-1]
    return canonicalize_host(value, error_type=DomainRuleValidationError)


def domain_rule_matches(rule: CanonicalHost, host: CanonicalHost) -> bool:
    """Return whether a canonical rule covers a canonical destination host."""
    if rule.is_ip or host.is_ip:
        return rule.ip is not None and host.ip is not None and rule.ip == host.ip
    return host.value == rule.value or host.value.endswith(f".{rule.value}")


def canonicalize_domain_rules(values: list[str] | tuple[str, ...]) -> tuple[CanonicalHost, ...]:
    """Canonicalize and de-duplicate domain rules while retaining stable order."""
    result: list[CanonicalHost] = []
    seen: set[tuple[str, bool]] = set()
    for value in values:
        rule = canonicalize_domain_rule(value)
        key = (rule.value, rule.is_ip)
        if key not in seen:
            result.append(rule)
            seen.add(key)
    return tuple(result)


def intersect_domain_allow_lists(
    first: tuple[CanonicalHost, ...] | None,
    second: tuple[CanonicalHost, ...] | None,
) -> tuple[CanonicalHost, ...] | None:
    """Suffix-aware intersection of optional domain allow-lists.

    ``None`` means that layer supplied no narrowing. For two non-empty lists,
    each overlapping pair contributes its narrower rule. A disjoint pair fails
    explicitly rather than returning an empty tuple that a caller might mistake
    for unrestricted access.
    """
    if first is None:
        return second
    if second is None:
        return first
    if not first:
        return second
    if not second:
        return first

    result: list[CanonicalHost] = []
    seen: set[tuple[str, bool]] = set()
    for left in first:
        for right in second:
            narrower: CanonicalHost | None = None
            if domain_rule_matches(left, right):
                narrower = right
            elif domain_rule_matches(right, left):
                narrower = left
            if narrower is not None:
                key = (narrower.value, narrower.is_ip)
                if key not in seen:
                    result.append(narrower)
                    seen.add(key)
    if not result:
        raise DisjointDomainAllowListsError("domain allow-lists do not overlap")
    return tuple(result)


def union_domain_block_lists(
    first: tuple[CanonicalHost, ...] | None,
    second: tuple[CanonicalHost, ...] | None,
) -> tuple[CanonicalHost, ...]:
    """Return the stable canonical union of two optional block-lists."""
    result: list[CanonicalHost] = []
    seen: set[tuple[str, bool]] = set()
    for rule in (*([] if first is None else first), *([] if second is None else second)):
        key = (rule.value, rule.is_ip)
        if key not in seen:
            result.append(rule)
            seen.add(key)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DomainPolicy:
    """Mandatory destination policy composed at request admission."""

    allowed: tuple[CanonicalHost, ...] = ()
    blocked: tuple[CanonicalHost, ...] = ()

    def permits(self, host: CanonicalHost) -> bool:
        if any(domain_rule_matches(rule, host) for rule in self.blocked):
            return False
        return not self.allowed or any(domain_rule_matches(rule, host) for rule in self.allowed)
