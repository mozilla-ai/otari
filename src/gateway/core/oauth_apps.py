"""Strict redirect validation for registered public OAuth clients."""

import ipaddress
from datetime import timedelta
from urllib.parse import SplitResult, urlsplit

from pydantic import HttpUrl

# How long an app key lives, by the names OpenRouter's PKCE sign-in uses for its ``expiration``.
APP_KEY_LIFETIMES: dict[str, timedelta | None] = {
    "1-hour": timedelta(hours=1),
    "1-day": timedelta(days=1),
    "7-days": timedelta(days=7),
    "30-days": timedelta(days=30),
    "90-days": timedelta(days=90),
    "180-days": timedelta(days=180),
    "1-year": timedelta(days=365),
    "never": None,
}

# The only hosts plain HTTP is admitted for, and where RFC 8252 lets the port vary.
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def parse_oauth_redirect(value: str) -> SplitResult | None:
    """Parse a callback without browser URL normalization.

    HTTP deliberately admits only localhost, 127.0.0.1 and ::1, not other loopback aliases or the full 127/8 range.
    """
    if not value or len(value) > 2048 or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        return None
    if any(char in value for char in ("\\", "?", "#")):
        return None
    try:
        HttpUrl(value)  # Validate browser host syntax without replacing the exact registered URI.
        parsed = urlsplit(value)
        port = parsed.port  # urlsplit alone accepts invalid and out-of-range ports.
    except ValueError:
        return None
    if (
        not parsed.hostname
        or "@" in parsed.netloc
        or "%" in parsed.netloc
        or parsed.netloc.endswith(":")
        or port == 0
        or (parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in _LOOPBACK_HOSTS))
    ):
        return None
    return parsed


def open_app_site(client_id: str, redirect_uri: str) -> str | None:
    """The site an unregistered app is, when its client ID is its callback's origin.

    The origin must be written as a browser writes ``location.origin``: ASCII with the host
    in punycode, lowercase, and the port in decimal unless it is the scheme's default.
    Anything else is refused, not normalized, because the site's name is shown on the
    consent screen and a non-ASCII host can be made to look like another site's.
    """
    parsed = parse_oauth_redirect(redirect_uri)
    if parsed is None or not client_id.isascii() or client_id != f"{parsed.scheme}://{parsed.netloc}":
        return None
    host = parsed.hostname or ""  # Never empty: parse_oauth_redirect refuses a callback without one.
    if ":" in host:
        address = ipaddress.IPv6Address(host)
        # A browser writes a mapped IPv4 address in hex, which ``compressed`` does not, and no site is served on one.
        if address.ipv4_mapped is not None:
            return None
        host = f"[{address.compressed}]"
    if parsed.port is not None and parsed.port != {"http": 80, "https": 443}[parsed.scheme]:
        host += f":{parsed.port}"
    return parsed.netloc if parsed.netloc == host else None


def redirect_matches(registered: str, requested: str) -> bool:
    """RFC 8252 loopback callbacks may choose a free port; every other component is exact."""
    expected, actual = parse_oauth_redirect(registered), parse_oauth_redirect(requested)
    if expected is None or actual is None:
        return False
    if registered == requested:
        return True
    if expected.scheme != "http" or expected.hostname not in _LOOPBACK_HOSTS:
        return False
    return actual.scheme == expected.scheme and actual.hostname == expected.hostname and actual.path == expected.path
