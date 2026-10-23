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

A call to an ``api_base`` an organization's key names goes through
:func:`web_provider_client_for` instead, which dials it as an endpoint a tenant
owns: public addresses only, pinned to the one checked, no redirects.
"""

import httpx

from gateway.services.providers import owned_endpoint_http_client
from gateway.services.web_retrieval_network import NoCookieJar


def build_web_provider_client() -> httpx.AsyncClient:
    """A new web provider client; its owner closes it. Timeouts are per request, set by each provider."""
    return httpx.AsyncClient(
        cookies=NoCookieJar(),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
    )


def web_provider_client_for(shared: httpx.AsyncClient | None, *, tenant_endpoint: bool) -> httpx.AsyncClient | None:
    """The client a web provider call goes through: ``shared``, unless a tenant chose the endpoint it reaches."""
    return owned_endpoint_http_client() if tenant_endpoint else shared
