"""Strict redirect validation for registered public OAuth clients."""

from urllib.parse import SplitResult, urlsplit

from pydantic import HttpUrl

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
