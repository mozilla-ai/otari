"""The HTTP client the web tools' providers are called through (decision 55).

The lifespan builds one per app, keeps it on ``app.state`` and closes it at
shutdown, and ``api/deps.py`` serves it, so no module holds a client of its
own. It is not ``services/providers/_provider_clients.py``, which reuses the
LLM providers' clients per event loop.

It keeps httpx's defaults, so environment proxies still apply, with two
exceptions, since one client serves every tenant's searches and keys:
- its cookie jar stores nothing, so a cookie set in reply to one tenant's call
  never rides on another's;
- it sets no cap on total connections, keeping httpx's 20 idle keep-alive ones,
  so a burst of calls is not queued into pool timeouts by httpx's default of
  100.
"""

from http.cookiejar import Cookie, CookieJar
from typing import Any

import httpx


class _NoCookieJar(CookieJar):
    """A jar that keeps no cookie, whatever a provider sets."""

    def set_cookie(self, cookie: Cookie) -> None:
        return None

    def extract_cookies(self, response: Any, request: Any) -> None:
        return None


def build_web_provider_client() -> httpx.AsyncClient:
    """A new web provider client; its owner closes it. Timeouts are per request, set by each provider."""
    return httpx.AsyncClient(
        cookies=_NoCookieJar(),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
    )
