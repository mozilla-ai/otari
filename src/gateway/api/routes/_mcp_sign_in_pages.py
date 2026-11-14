"""The plain pages a person sees after opening an MCP sign-in link from a conversation.

Served by the gateway itself rather than the dashboard, because the person may have no dashboard
session: they came from a coding agent's chat. No script, nothing loaded from elsewhere, and no
other site may frame one. Every value is escaped into its page.
"""

from enum import StrEnum
from html import escape
from urllib.parse import urlsplit

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
<title>{title} · Otari</title>{head}
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; padding: 4rem 1rem; background: #f6f6f4; color: #1d1d1b; }}
main {{ max-width: 32rem; margin: 0 auto; }}
h1 {{ font-size: 1.5rem; margin: 0 0 1rem; }}
p {{ line-height: 1.5; margin: 0 0 1rem; }}
.warning {{ border-left: 4px solid #b45309; padding-left: 0.75rem; }}
button {{ font: inherit; padding: 0.6rem 1rem; border-radius: 0.4rem; border: 1px solid currentColor;
  background: #1d1d1b; color: #f6f6f4; cursor: pointer; }}
a {{ color: inherit; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #1d1d1b; color: #f6f6f4; }}
  button {{ background: #f6f6f4; color: #1d1d1b; }}
}}
</style>
</head>
<body><main><h1>{title}</h1>{body}</main></body>
</html>
"""

_CSP = "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'"


def _response(
    title: str, body: str, *, status_code: int, head: str = "", csp: str = _CSP, referrer_policy: str = "no-referrer"
) -> HTMLResponse:
    """A page from its title and its body, which is HTML the caller has already escaped every value into."""
    return HTMLResponse(
        _PAGE.format(title=escape(title, quote=False), head=head, body=body),
        status_code=status_code,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": csp,
            "Referrer-Policy": referrer_policy,
            "X-Frame-Options": "DENY",
        },
    )


def _paragraph(text: str, *, css_class: str | None = None) -> str:
    return f'<p class="{css_class}">{escape(text)}</p>' if css_class else f"<p>{escape(text)}</p>"


def conversation_page(page: SignInPage, *, server_name: str | None = None) -> HTMLResponse:
    """One of the pages, with the server's name escaped into it where it has one."""
    title, body = _TEXT[page]
    return _response(
        title,
        _paragraph(body.format(server=server_name or "The server")),
        status_code=200 if page is SignInPage.CONNECTED else 400,
    )


def confirm_page(
    *, ticket: str, server_name: str, server_host: str, user_label: str, opened_as: str | None
) -> HTMLResponse:
    """Ask before a link's sign-in starts: which server, and which Otari user the account will be attached to.

    Starts nothing itself. The form posts the ticket back to this same address, which is the only
    place the page may post to.
    """
    body = (
        _paragraph(
            f"You are about to sign in to {server_name} ({server_host}). The account you sign in with will be "
            f"attached to this Otari user: {user_label}"
        )
        + _paragraph(f"From then on, that user's Otari API keys act on {server_name} as the account you sign in with.")
        + _paragraph(
            f"Continue only if {user_label} is you and you asked for this link yourself. If someone sent you this "
            f"link, close this page: continuing would give them the use of your {server_name} account.",
            css_class="warning",
        )
        + (_paragraph(f"You are signed in to Otari as {opened_as}.") if opened_as else "")
        + f'<form method="post"><input type="hidden" name="ticket" value="{escape(ticket)}">'
        + '<button type="submit">This is me, continue</button></form>'
    )
    return _response(
        f"Connect {server_name} to your Otari user?",
        body,
        status_code=200,
        csp=f"{_CSP}; form-action 'self'",
        # The form's POST must carry this page's origin, which ``no-referrer`` would blank out.
        referrer_policy="same-origin",
    )


def other_user_page(*, server_name: str, user_label: str, opened_as: str) -> HTMLResponse:
    """Refuse a link opened in the dashboard session of somebody it was not made for. Offers no way on."""
    body = _paragraph(
        f"This link attaches a {server_name} account to the Otari user {user_label}. You are signed in to Otari "
        f"as {opened_as}, which is a different user, so nothing was started."
    ) + _paragraph(
        "If someone sent you this link, close this page and do not sign in. If it came from your own conversation, "
        f"the API key that conversation uses belongs to {user_label} and not to your dashboard account.",
        css_class="warning",
    )
    return _response("This sign-in link is not yours", body, status_code=403)


def leaving_page(authorization_url: str) -> HTMLResponse:
    """Send the browser on to the server's own sign-in page, once the person confirmed.

    A page rather than a redirect: a browser holds a redirect that follows a form to the page's
    ``form-action``, which names this gateway alone.
    """
    url = escape(authorization_url)
    host = escape(urlsplit(authorization_url).hostname or "the sign-in page")
    return _response(
        "Taking you to the sign-in",
        f'<p><a href="{url}" rel="noreferrer">Continue to {host}</a> if nothing happens.</p>',
        status_code=200,
        head=f'\n<meta http-equiv="refresh" content="0; url={url}">',
    )
