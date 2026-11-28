"""How a stored endpoint URL is matched, whichever way a row spelled it.

In ``core`` because two domains match by it: the MCP catalog finds the row listed at
a workspace server's URL, and a member's request for a catalog server is found by
the URL a manager later adds.
"""


def _path_end(url: str) -> int:
    """Where a URL's path ends: at its query or fragment, or at its end."""
    ends = [index for index in (url.find("?"), url.find("#")) if index >= 0]
    return min(ends, default=len(url))


def bare_url(url: str) -> str:
    """A URL without the whitespace around it or the trailing slashes of its path. Its query is kept as it is.

    Two URLs name one endpoint where their bare spellings are equal.
    """
    url = url.strip()
    end = _path_end(url)
    return url[:end].rstrip("/") + url[end:]


def url_path(url: str) -> str:
    """A URL up to its query or fragment."""
    return url[: _path_end(url)]


def url_spellings(url: str) -> tuple[str, str]:
    """``url``'s endpoint spelled without and with a trailing slash on its path, the two spellings a row may hold."""
    bare = bare_url(url)
    end = _path_end(bare)
    return bare, f"{bare[:end]}/{bare[end:]}"
