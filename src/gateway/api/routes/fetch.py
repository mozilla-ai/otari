"""Fetch endpoint: a direct, billed fetch of one page.

``otari_web_fetch`` answers a model's tool call mid-completion. This endpoint
is its direct counterpart, as ``POST /api/v1/search`` is ``otari_web_search``'s:
the caller names a URL, the gateway reads the page through its fetch tool, with
the checks that tool makes of the URL, and the text comes straight back, having
gone through the same auth, rate limit, budget reservation and usage log as a
completion. A client that runs its own agent loop reads pages through the
gateway this way, without a completion for each page.

Which tool, which key and which checks a request gets, and the fetch itself,
are the tools services' (:func:`gateway.services.tools.resolve_direct_fetch`
and :func:`gateway.services.tools.run_direct_fetch`); the gates, the billing
and the usage row are :class:`gateway.services.tools.DirectToolService`'s.

Both the body-selected (``POST /api/v1/fetch``) and path-selected
(``POST /api/v1/fetch/{fetch_tool_name}``) forms log ``endpoint="/v1/fetch"``,
one label for every form of the route, so one Activity filter covers every
fetch regardless of how the tool was named.

Billing: a fetch is priced per page, not per token, so a usage row carries zero
tokens and a cost taken from the provider's own reported charge when it reports
one, else the tool's own fetch price row, ``builtin_fetch``'s included; only the
operator's price or override sets it, since the deployment pays for the call. A
fetch a workspace's organization fetch key served is priced under
``otari:web_fetch``, which the organization's own override may set. With
``require_pricing`` on, a fetch whose rate is missing is refused with a 402
after the budget checks; a key exempt from budgets is still served.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, Response

from gateway.api.deps import DirectToolServiceDep, verify_api_key_or_master_key
from gateway.api.routes._helpers import answer_direct_tool_call
from gateway.models.api_keys import APIKey
from gateway.schemas.tools import FetchRequest, FetchResponse

router = APIRouter(tags=["fetch"])

FETCH_ENDPOINT = "/v1/fetch"


@router.post("/fetch")
async def create_fetch(
    raw_request: Request,
    response: Response,
    request: FetchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    service: DirectToolServiceDep,
) -> FetchResponse:
    """Fetch one page through a configured fetch tool.

    The tool is taken from ``fetch_tool_name``, which may be omitted: the
    deployment's ``web_fetch_default_tool`` then serves, else ``builtin_fetch``,
    the gateway's own fetcher. Such a call runs on the workspace's organization
    fetch key when it has one. While the deployment's ``web_fetch_enabled`` is
    off, the call is refused, as ``otari_web_fetch`` is. The URL must pass the
    workspace's domain lists and the gateway's own checks: a public http or
    https URL that names no private address. The built-in fetcher also checks
    where each host resolves.

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
    return await _dispatch_fetch(
        raw_request=raw_request,
        response=response,
        request=request,
        tool_name=request.fetch_tool_name,
        auth_result=auth_result,
        service=service,
    )


@router.post("/fetch/{fetch_tool_name}")
async def create_fetch_for_tool(
    raw_request: Request,
    response: Response,
    request: FetchRequest,
    fetch_tool_name: Annotated[str, Path(description="Configured fetch tool to run on, builtin_fetch included")],
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    service: DirectToolServiceDep,
) -> FetchResponse:
    """Fetch one page through the fetch tool named in the path.

    Identical to ``POST /api/v1/fetch`` except that the path names the tool,
    which then runs on its own key. Any ``fetch_tool_name`` in the body is
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
    return await _dispatch_fetch(
        raw_request=raw_request,
        response=response,
        request=request,
        tool_name=fetch_tool_name,
        auth_result=auth_result,
        service=service,
    )


async def _dispatch_fetch(
    *,
    raw_request: Request,
    response: Response,
    request: FetchRequest,
    tool_name: str | None,
    auth_result: tuple[APIKey | None, bool],
    service: DirectToolServiceDep,
) -> FetchResponse:
    return await answer_direct_tool_call(
        raw_request,
        response,
        auth_result=auth_result,
        user=request.user,
        run=lambda caller: service.fetch(request, caller=caller, tool_name=tool_name, endpoint=FETCH_ENDPOINT),
    )
