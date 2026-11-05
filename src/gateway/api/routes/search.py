"""Search pass-through endpoint: a direct, billed search request.

``otari_web_search`` answers a model's tool call mid-completion. This endpoint
is its direct counterpart: the caller asks, the gateway asks the search
provider, and the result comes straight back, having gone through the same
auth, rate limit, budget reservation, and usage log as a completion. It exists
so a client that proxies a search API can keep one gateway on the billing path
instead of holding a second provider key of its own.

The request and response follow LiteLLM's ``/v1/search`` (itself shaped after
Perplexity's Search API), so a caller moving off the LiteLLM proxy keeps its
request shape; ``include_raw`` and ``provider_options`` are otari's own. Two
Perplexity features are deliberately not modeled: ``query`` is a single string
rather than the multi-query array, and the filters with no field below
(recency, context size, published-date) are ignored rather than rejected. Both
are called out in ``docs/api-reference.md`` so a migrating caller can check for
them.

Which tool, which key and which count a request gets, and the call itself, are
the tools services' (:func:`gateway.services.tools.resolve_direct_search` and
:func:`gateway.services.tools.run_direct_search`); the gates, the billing and
the usage row are :class:`gateway.services.tools.DirectToolService`'s.

Both the body-selected (``POST /api/v1/search``) and path-selected
(``POST /api/v1/search/{search_tool_name}``) forms log ``endpoint="/v1/search"``,
a frozen label rather than a path, so one Activity filter covers every search
regardless of how the tool was named.

Billing: search is priced per request, not per token, so a usage row carries
zero tokens and a cost taken from the provider's own reported charge when it
reports one, else the call's flat per-request rate. A search on an instance is
priced at the tool's own search price row, then at the ``<provider>:<tool>`` row
it was priced under before tools had a kind; only the operator's price or
override sets either, since the deployment pays for the call. A search a
workspace's organization key served is priced under ``otari:web_search``, which
the organization's own override may set, since a deployment cannot price a
tenant's key. With ``require_pricing`` on, a search whose rate is missing is
refused with a 402 after the budget checks, even from a provider that reports
its own cost; a key exempt from budgets is still served.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, Response

from gateway.api.deps import DirectToolServiceDep, verify_api_key_or_master_key
from gateway.api.routes._helpers import answer_direct_tool_call
from gateway.models.api_keys import APIKey
from gateway.schemas.tools import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])

SEARCH_ENDPOINT = "/v1/search"


@router.post("/search")
async def create_search(
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    service: DirectToolServiceDep,
) -> SearchResponse:
    """Run a search against a configured search tool.

    The tool is taken from ``search_tool_name``, which may be omitted when the
    deployment's ``web_search_default_tool`` names one, or when exactly one
    tool is configured. Such a call runs on the workspace's organization search
    key when it has one.

    Authentication modes:
    - Master key: the ``user`` field is required and may name any existing user.
    - API key: usage and spend bind to the key's own user. A ``user`` field
      naming a different user is rejected with 403 (or ignored, when the key's
      own ``reject_user_mismatch`` is false, or the deployment-wide setting is
      disabled and the key does not override it); it is never billed to that
      user.
    - Service key: a ``user`` field names one of the key owner's end users,
      created on first use on the budget ``Otari-End-User-Budget`` names (or
      the key's default), and is billed and rate limited as that end user, as
      on chat completions.
    """
    return await _dispatch_search(
        raw_request=raw_request,
        response=response,
        request=request,
        tool_name=request.search_tool_name,
        auth_result=auth_result,
        service=service,
    )


@router.post("/search/{search_tool_name}")
async def create_search_for_tool(
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    search_tool_name: Annotated[str, Path(description="Configured search tool to run against")],
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    service: DirectToolServiceDep,
) -> SearchResponse:
    """Run a search against the search tool named in the path.

    Identical to ``POST /api/v1/search`` except that the path names the tool, which
    is the form LiteLLM clients use. Any ``search_tool_name`` in the body is
    ignored.

    Authentication modes:
    - Master key: the ``user`` field is required and may name any existing user.
    - API key: usage and spend bind to the key's own user. A ``user`` field
      naming a different user is rejected with 403 (or ignored, when the key's
      own ``reject_user_mismatch`` is false, or the deployment-wide setting is
      disabled and the key does not override it); it is never billed to that
      user.
    - Service key: a ``user`` field names one of the key owner's end users,
      created on first use on the budget ``Otari-End-User-Budget`` names (or
      the key's default), and is billed and rate limited as that end user, as
      on chat completions.
    """
    return await _dispatch_search(
        raw_request=raw_request,
        response=response,
        request=request,
        tool_name=search_tool_name,
        auth_result=auth_result,
        service=service,
    )


async def _dispatch_search(
    *,
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    tool_name: str | None,
    auth_result: tuple[APIKey | None, bool],
    service: DirectToolServiceDep,
) -> SearchResponse:
    return await answer_direct_tool_call(
        raw_request,
        response,
        auth_result=auth_result,
        user=request.user,
        run=lambda caller: service.search(request, caller=caller, tool_name=tool_name, endpoint=SEARCH_ENDPOINT),
    )
