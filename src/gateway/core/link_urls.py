"""What this codebase accepts as a link it publishes to a browser.

In ``core`` because the configuration and the API schemas both validate one:
the deployment's own menu links, and the policy links of a hosted provider.
"""

from urllib.parse import urlsplit


def normalize_link_url(value: str | None, field_name: str) -> str | None:
    """``value`` trimmed, or None when blank.

    A link is rendered as an anchor target, so a scheme other than http(s)
    would be a script URL. Userinfo is refused because these links reach
    unauthenticated readers. No message repeats the value, since it may carry a
    credential.

    Raises:
        ValueError: ``value`` is not an absolute http(s) URL, or carries a username or password.
    """
    normalized = (value or "").strip()
    if not normalized:
        return None
    msg = f"{field_name} must be an absolute http(s) URL"
    try:
        parsed = urlsplit(normalized)
        has_userinfo = parsed.username is not None or parsed.password is not None
        _ = parsed.port  # A port that is not a number raises here.
    except ValueError:
        raise ValueError(msg) from None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(msg)
    if has_userinfo:
        msg = f"{field_name} must carry no username or password"
        raise ValueError(msg)
    return normalized
