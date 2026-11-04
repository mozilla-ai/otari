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
and :func:`gateway.services.tools.run_direct_fetch`). The gates, the billing
and the usage row are the direct tool scaffold's
(:func:`gateway.api.routes._passthrough.run_direct_tool`); this module reads
the request and hands the scaffold the fetch's own pieces.

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

import httpx
from fastapi import APIRouter, Depends, Path, Request, Response

from gateway.api.deps import WebProviderClientDep, get_config, verify_api_key_or_master_key
from gateway.api.routes._passthrough import DirectToolCall, DirectToolScaffold, get_direct_tool_scaffold
from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import ToolInstance
from gateway.models.api_keys import APIKey
from gateway.models.tools import ResolvedWebSearchConfig, WebTool
from gateway.schemas.tools import FetchRequest, FetchResponse
from gateway.services.tools import (
    pick_direct_fetch_instance,
    resolve_direct_fetch,
    run_direct_fetch,
    unpriced_direct_fetch_detail,
)

router = APIRouter(tags=["fetch"])

FETCH_ENDPOINT = "/v1/fetch"


@router.post("/fetch")
async def create_fetch(
    raw_request: Request,
    response: Response,
    request: FetchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    scaffold: Annotated[DirectToolScaffold, Depends(get_direct_tool_scaffold)],
    web_provider_client: WebProviderClientDep,
) -> FetchResponse:
    """Fetch one page through a configured fetch tool.

    The tool is taken from ``fetch_tool_name``, which may be omitted: the
    deployment's ``web_fetch_default_tool`` then serves, else ``builtin_fetch``,
    the gateway's own fetcher. Such a call runs on the workspace's organization
    fetch key when it has one. The deployment's ``web_fetch_enabled`` must be
    on, and the URL must pass the workspace's domain lists and the fetcher's
    own checks: a public http or https address, never a private one.

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
        config=config,
        scaffold=scaffold,
        web_provider_client=web_provider_client,
    )


@router.post("/fetch/{fetch_tool_name}")
async def create_fetch_for_tool(
    raw_request: Request,
    response: Response,
    request: FetchRequest,
    fetch_tool_name: Annotated[str, Path(description="Configured fetch tool to run on, builtin_fetch included")],
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    scaffold: Annotated[DirectToolScaffold, Depends(get_direct_tool_scaffold)],
    web_provider_client: WebProviderClientDep,
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
        config=config,
        scaffold=scaffold,
        web_provider_client=web_provider_client,
    )


async def _dispatch_fetch(
    *,
    raw_request: Request,
    response: Response,
    request: FetchRequest,
    tool_name: str | None,
    auth_result: tuple[APIKey | None, bool],
    config: GatewayConfig,
    scaffold: DirectToolScaffold,
    web_provider_client: httpx.AsyncClient,
) -> FetchResponse:
    """Run the scaffold around the fetch's own pieces: the instance, the resolved call and the fetch."""

    def resolve(tool: ToolInstance, policy: ResolvedWebSearchConfig | None) -> DirectToolCall[FetchResponse]:
        # A workspace whose organization brought its own fetch key reads pages with it
        # on a call that names no tool, so its own account pays. The tool is still what
        # the request is allowlisted, rate-limited and timed out as.
        fetch = resolve_direct_fetch(config, tool, policy, request, named=tool_name is not None)
        return DirectToolCall(
            provider=fetch.resolved.provider,
            label=fetch.resolved.label,
            rate_keys=fetch.resolved.rate_keys,
            deployment_paid=not fetch.resolved.on_credential,
            unpriced_detail=unpriced_direct_fetch_detail(fetch),
            run=lambda: run_direct_fetch(
                fetch, web_provider_client, trust_env_proxy=config.web_retrieval_trust_env_proxy
            ),
        )

    return await scaffold.run(
        endpoint=FETCH_ENDPOINT,
        kind="fetch",
        raw_request=raw_request,
        response=response,
        auth_result=auth_result,
        user=request.user,
        tool_name=tool_name,
        pick=lambda: pick_direct_fetch_instance(config, tool_name),
        policy_tools=(WebTool.FETCH,),
        resolve=resolve,
    )
