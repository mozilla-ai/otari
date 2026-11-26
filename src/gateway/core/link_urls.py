"""What this codebase accepts as a URL, as a destination and as a base.

In ``core`` because the configuration and the API schemas both validate one:
the deployment's own menu links, the policy links of a hosted provider, and the
addresses the deployment is reached at.

A destination is followed as written. A base has other URLs appended to it, so
it is held to more: anything belonging at a URL's end would land in the middle
of the result.
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
    # Ahead of the scheme and the host, so a value wrong in both ways names the credential, which
    # is the part the writer has to remove.
    if has_userinfo:
        msg = f"{field_name} must carry no username or password"
        raise ValueError(msg)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(msg)
    return normalized


def strip_path_slashes(url: str) -> str:
    """``url`` with trailing slashes dropped from its path, leaving any query as written.

    A slash ending a query value (``?edge=team/``) is part of that value, so stripping the whole
    string would hand every link built from it a different one.
    """
    location, separator, query = url.partition("?")
    return f"{location.rstrip('/')}{separator}{query}"


def normalize_base_url(value: str | None, field_name: str, *, allow_query: bool, strip_slashes: bool) -> str | None:
    """``value`` as a destination, held to what a base also needs, or None when blank.

    A base has other URLs appended to it, so it carries nothing that belongs at a URL's end: never
    a fragment, and never a query unless ``allow_query``. The delimiter counts on its own, because
    an empty one displaces an appended path exactly as a populated one does. Interior whitespace is
    refused because parsing drops it while storing keeps it, so the value that was checked and the
    value that is used would differ.

    ``strip_slashes`` drops trailing slashes from the path, for a base whose joining does not.

    Raises:
        ValueError: ``value`` cannot serve as a base, naming the field and the condition.
    """
    stripped = (value or "").strip()
    if not stripped:
        return None
    normalized = strip_path_slashes(stripped) if strip_slashes else stripped
    if not normalized:
        # Slashes alone, which would otherwise strip to empty and read as unset.
        msg = f"{field_name} must be an absolute http(s) URL"
        raise ValueError(msg)
    if any(character.isspace() for character in normalized):
        msg = f"{field_name} must carry no whitespace"
        raise ValueError(msg)
    if "#" in normalized:
        msg = f"{field_name} must carry no fragment"
        raise ValueError(msg)
    if not allow_query and "?" in normalized:
        msg = f"{field_name} must carry no query string"
        raise ValueError(msg)
    return normalize_link_url(normalized, field_name)
