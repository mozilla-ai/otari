"""The SearXNG-shaped search backend a data-plane gateway calls.

Served by a deployment that holds a licensed search key and runs its inference
somewhere else, which is what a hosted control plane is: the gateway has no
credential and must not be given one, so it sends the query here and this
process makes the provider call. Standalone deployments need none of this and
call :mod:`gateway.services.web_search_providers` in-process instead; the two
share that module, so both surfaces answer a query the same way.

The contract is the one ``WebSearchBackend`` already speaks to a SearXNG
container, so the client side is one setting: a hybrid gateway's
``web_search_url`` pointed at ``{control-plane}/v1/web-search``. It has to be
that host and path, under the gateway's own ``PLATFORM_BASE_URL``, because
``url_targets_platform`` is what decides whether the platform token is forwarded
at all, and that token is the credential below. A standalone gateway forwards no
such token and cannot call this; it configures a provider of its own.

**Authenticated, and only mounted when it can be.** The route spends the
deployment's own search quota, and a control plane is internet-reachable, so it
requires ``X-Gateway-Token`` to match the configured ``web_search_backend_token``.
That is the deployment's own data-plane gateway and nothing else: a gateway
someone self-hosts presents a credential of its own, and serving it here would
put a deployment-owned credential's *use* behind a caller the managed-key
boundary keeps it from ever holding. Nothing returned here is a credential; the
response carries public search results.

The query travels in the request target, as it does to a SearXNG container,
because that is the contract ``WebSearchBackend`` speaks. A deployment whose
access log records request targets therefore records search queries; a control
plane that must not keep them turns that log off or redacts this path.
"""

from __future__ import annotations

from hashlib import sha256
from secrets import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from gateway.api.deps import get_config
from gateway.core.config import GatewayConfig
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.services.search_backend import get_search_client
from gateway.services.web_search_backend import MAX_RESULTS_CAP, WEB_SEARCH_TOOL_NAME
from gateway.services.web_search_providers import WebSearchProviderError, provider_search

router = APIRouter(prefix="/v1/web-search", tags=["web-search"])

ConfigDep = Annotated[GatewayConfig, Depends(get_config)]

SEARCH_ENDPOINT = "/v1/web-search/search"


class WebSearchBackendResult(BaseModel):
    """One hit, in the shape ``WebSearchBackend`` reads off a SearXNG response."""

    url: str
    title: str = ""
    content: str = ""
    extracted_content: str | None = Field(
        default=None,
        description="The page's own text, when the provider returned it, so the caller can skip fetching the page.",
    )
    published_date: str | None = Field(
        default=None,
        description=(
            "The provider's own recency string for the page, forwarded unparsed. Declared so a search "
            "over this hop renders the same date an in-process one does."
        ),
    )


class WebSearchBackendResponse(BaseModel):
    """A SearXNG-compatible ``/search`` body."""

    results: list[WebSearchBackendResult]


def _digest(token: str) -> bytes:
    return sha256(token.encode("utf-8", errors="surrogateescape")).digest()


def _authorize(config: GatewayConfig, presented: str | None) -> None:
    """Refuse anything but the deployment's own gateway.

    Hashed before comparing, then compared under :func:`secrets.compare_digest`,
    so neither the length nor the leading characters of the configured token leak
    through the comparison's timing. Hashed rather than compared directly because
    ``compare_digest`` raises on a non-ASCII string, and Starlette decodes a
    header as latin-1: a byte above 0x7f in ``X-Gateway-Token`` would otherwise
    be an unhandled 500 from an unauthenticated caller rather than a refusal.
    """
    expected = config.web_search_backend_token
    if not expected or not presented or not compare_digest(_digest(presented), _digest(expected)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Web search backend authentication failed")


@router.get("/search", response_model=WebSearchBackendResponse, response_model_exclude_none=True)
async def web_search(
    request: Request,
    config: ConfigDep,
    q: Annotated[str, Query(min_length=1, description="The search query.")],
    x_gateway_token: Annotated[str | None, Header()] = None,
    max_results: Annotated[int | None, Query(ge=1, le=MAX_RESULTS_CAP)] = None,
    search_depth: Annotated[str | None, Query(pattern="^(basic|advanced)$")] = None,
    topic: Annotated[str | None, Query(pattern="^(general|news|finance)$")] = None,
    time_range: Annotated[str | None, Query(pattern="^(day|week|month|year|d|w|m|y)$")] = None,
    include_answer: Annotated[bool | None, Query()] = None,
) -> WebSearchBackendResponse:
    """Run one search through this deployment's configured search provider.

    The declared query params are the ``provider_options`` keys the providers
    understand. Everything else the caller sends, ``format`` and ``engines``
    included, is ignored rather than forwarded: the bag is opaque to the gateway
    that filled it, so passing it upstream unread would let a workspace set
    provider request fields this deployment never chose.
    """
    _authorize(config, x_gateway_token)
    provider = config.web_search_provider
    api_key = config.web_search_provider_api_key
    if not provider or not api_key:
        # Only reachable if the provider was unconfigured after boot, since the
        # route is not mounted without one. 503 rather than 400: nothing about
        # the request is wrong.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No web-search provider is configured on this deployment",
        )

    # Every gate has passed and the provider is about to be called, so the search
    # is genuinely in flight from here. Registered for the same reason the direct
    # search path registers: a paid call nobody can see running is a paid call
    # nobody can see hanging. The entry is dropped by ``InFlightMiddleware``.
    # There is no key or user to attribute it to; the caller is the deployment's
    # own gateway, which reports its own usage.
    track_request(request, endpoint=SEARCH_ENDPOINT, model=WEB_SEARCH_TOOL_NAME, provider=provider)

    options = {
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
        "time_range": time_range,
        "include_answer": include_answer,
    }
    try:
        results = await provider_search(
            provider=provider,
            api_key=api_key,
            query=q,
            options={key: value for key, value in options.items() if value is not None},
            client=get_search_client(),
        )
    except WebSearchProviderError as exc:
        # Logged here because nothing else on this path does: the exception only
        # becomes an ``HTTPException``'s ``__cause__``, which FastAPI renders and
        # discards, leaving an operator debugging a 502 with nothing. The message
        # names the provider and the upstream status and never its body, which
        # can echo back what was sent to it.
        logger.error("Web search backend call to '%s' failed: %s", provider, exc)
        # The caller is told only that the upstream failed.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The {provider} search provider could not serve this query",
        ) from exc

    return WebSearchBackendResponse(results=[WebSearchBackendResult(**hit) for hit in results])
