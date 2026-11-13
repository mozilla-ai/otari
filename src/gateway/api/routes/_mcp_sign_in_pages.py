"""The plain pages a person sees after opening an MCP sign-in link from a conversation.

Served by the gateway itself rather than the dashboard, because the person may have no dashboard
session: they came from a coding agent's chat. No script, nothing loaded from elsewhere.
"""

from enum import StrEnum
from html import escape

from fastapi.responses import HTMLResponse


class SignInPage(StrEnum):
    """Which of the pages to show."""

    CONNECTED = "connected"
    FAILED = "failed"
    LINK_EXPIRED = "link_expired"
    CANNOT_START = "cannot_start"


_TEXT: dict[SignInPage, tuple[str, str]] = {
    SignInPage.CONNECTED: (
        "You're signed in",
        "{server} is now connected to your account. Go back to your conversation and ask again.",
    ),
    SignInPage.FAILED: (
        "The sign-in did not finish",
        "Nothing was connected. Go back to your conversation and ask again to get a new link.",
    ),
    SignInPage.LINK_EXPIRED: (
        "This link has expired",
        "Sign-in links work for a short time only. Go back to your conversation and ask again to get a new one.",
    ),
    SignInPage.CANNOT_START: (
        "The sign-in could not start",
        "Try again in a few minutes. If it keeps happening, tell your Otari administrator.",
    ),
}

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Otari</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; padding: 4rem 1rem; background: #f6f6f4; color: #1d1d1b; }}
main {{ max-width: 32rem; margin: 0 auto; }}
h1 {{ font-size: 1.5rem; margin: 0 0 1rem; }}
p {{ line-height: 1.5; margin: 0; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #1d1d1b; color: #f6f6f4; }} }}
</style>
</head>
<body><main><h1>{title}</h1><p>{body}</p></main></body>
</html>
"""


def conversation_page(page: SignInPage, *, server_name: str | None = None) -> HTMLResponse:
    """One of the pages, with the server's name escaped into it where it has one."""
    title, body = _TEXT[page]
    # Element content only, where a quote is inert.
    html = _PAGE.format(
        title=escape(title, quote=False), body=escape(body.format(server=server_name or "The server"), quote=False)
    )
    return HTMLResponse(
        html,
        status_code=200 if page is SignInPage.CONNECTED else 400,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            "Referrer-Policy": "no-referrer",
        },
    )
