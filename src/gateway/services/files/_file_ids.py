"""The opaque cursor a listing pages with.

Anthropic's Files API resumes a page from a token rather than an offset, so the
token is one file's ID, encoded so that a caller reads it as opaque and does not
build one of its own.
"""

import base64

from gateway.repositories.files import could_name_a_file

_TOKEN_PREFIX = "page_"


def page_token(file_id: str) -> str:
    """The token that resumes a listing after ``file_id``."""
    return _TOKEN_PREFIX + base64.urlsafe_b64encode(file_id.encode()).decode().rstrip("=")


def file_id_in(token: str) -> str | None:
    """The file ID ``token`` resumes after, or None for a token this gateway did not issue."""
    if not token.startswith(_TOKEN_PREFIX):
        return None
    encoded = token.removeprefix(_TOKEN_PREFIX)
    try:
        file_id = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except ValueError:
        return None
    return file_id if could_name_a_file(file_id) else None
