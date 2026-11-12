"""The media type a file's name and its declared type resolve to."""

import mimetypes


def guess_mime_type(filename: str | None, declared: str | None = None) -> str:
    """The media type for ``filename``: the declared one when it says something, else by extension."""
    if declared and declared != "application/octet-stream":
        return declared
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return declared or "application/octet-stream"
