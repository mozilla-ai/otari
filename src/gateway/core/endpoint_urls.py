"""How a stored endpoint URL is matched, whichever way a row spelled it.

In ``core`` because two domains match by it: the MCP catalog finds the row listed at
a workspace server's URL, and a member's request for a catalog server is found by
the URL a manager later adds.
"""


def url_spellings(url: str) -> tuple[str, str]:
    """``url`` without and with a trailing slash, the two spellings of one endpoint a row may hold."""
    bare = url.strip().rstrip("/")
    return bare, f"{bare}/"
