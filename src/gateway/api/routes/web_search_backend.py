"""The SearXNG-shaped search backend a data-plane gateway calls.

Served by a deployment that holds a licensed search key and runs its inference
somewhere else, which is what a hosted control plane is: the gateway has no
credential and must not be given one, so it sends the query here and this
process makes the provider call. Standalone deployments need none of this and
call :mod:`gateway.services.web_search_providers` in-process instead; the two
share that module, so both surfaces answer a query the same way.

The contract is the one ``WebSearchBackend`` already speaks to a SearXNG
container, so pointing ``web_search_url`` at ``{control-plane}/v1/web-search``
is the whole of the client-side configuration.

**Authenticated, and only mounted when it can be.** The route spends the
deployment's own search quota, and a control plane is internet-reachable, so it
requires ``X-Gateway-Token`` to match the configured ``web_search_backend_token``.
That is the deployment's own data-plane gateway and nothing else: a gateway
someone self-hosts presents a credential of its own, and serving it here would
put a deployment-owned credential's *use* behind a caller the managed-key
boundary keeps it from ever holding. Nothing returned here is a credential; the
response carries public search results.
"""

from __future__ import annotations

from secrets import compare_digest
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from gateway.api.deps import get_config
from gateway.core.config import GatewayConfig
from gateway.services.search_backend import get_search_client
from gateway.services.web_search_providers import WebSearchProviderError, provider_search

router = APIRouter(prefix="/v1/web-search", tags=["web-search"])

ConfigDep = Annotated[GatewayConfig, Depends(get_config)]


class WebSearchBackendResult(BaseModel):
    """One hit, in the shape ``WebSearchBackend`` reads off a SearXNG response."""

    url: str
    title: str = ""
    content: str = ""
    extracted_content: str | None = Field(
        default=None,
        description="The page's own text, when the provider returned it, so the caller can skip fetching the page.",
    )


class WebSearchBackendResponse(BaseModel):
    """A SearXNG-compatible ``/search`` body."""

    results: list[WebSearchBackendResult]


def _authorize(config: GatewayConfig, presented: str | None) -> None:
    """Refuse anything but the deployment's own gateway.

    Compared as digests through :func:`secrets.compare_digest`, so neither the
    length nor the leading characters of the configured token leak through the
    comparison's timing.
    """
    expected = config.web_search_backend_token
    if not expected or not presented or not compare_digest(presented, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Web search backend authentication failed")


@router.get("/search", response_model=WebSearchBackendResponse, response_model_exclude_none=True)
async def web_search(
    config: ConfigDep,
    q: Annotated[str, Query(min_length=1, description="The search query.")],
    x_gateway_token: Annotated[str | None, Header()] = None,
    max_results: Annotated[int | None, Query(ge=1)] = None,
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
        # The provider's own status and body stay in the log through the raised
        # message; the caller is told only that the upstream failed.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"The {provider} search provider could not serve this query",
        ) from exc

    return WebSearchBackendResponse(results=[WebSearchBackendResult(**hit) for hit in results])
