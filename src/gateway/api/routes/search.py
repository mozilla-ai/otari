"""Search pass-through endpoint: a direct, billed search request.

``otari_web_search`` answers a model's tool call mid-completion. This endpoint
is the standalone counterpart: the caller asks, the gateway asks the search
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

Which tool, which key and which count a request gets, and the call itself, on
any-search or on an old client until any-search has the provider's adapter, are
the tools services' (:func:`gateway.services.tools.resolve_direct_search` and
:func:`gateway.services.tools.run_direct_search`); this module reads the
request and its dependencies, gates, bills and logs.

Both the body-selected (``POST /api/v1/search``) and path-selected
(``POST /api/v1/search/{search_tool_name}``) forms log ``endpoint="/v1/search"``,
a frozen label rather than a path, so one Activity filter covers every search
regardless of how the tool was named.

A request the gateway itself turns away (an unknown or ambiguous tool name, a
workspace that has web search switched off, a tool the caller's key may not use,
an option the provider does not take) writes a usage row too, through the shared
:func:`gateway.api.routes._pipeline.log_gateway_rejection` the pass-through
routes use: refused traffic is dropped traffic, and it should be visible in
Activity and countable as a failure rather than invisible outside the caller's
own logs.

Billing: search is priced per request, not per token, so a usage row carries
zero tokens and a cost taken from the provider's own reported charge when it
reports one. Otherwise the cost is the flat per-request rate configured for
``<provider>:<tool>`` under the same convention moderations uses
(:func:`flat_request_cost`), or, for a search a workspace's organization key
served, the ``otari:web_search`` rate, since a deployment cannot price a
tenant's key. Like moderations and audio, search is exempt from
``require_pricing``: a provider that reports its own cost is metered whether or
not a rate is configured, so failing closed would reject calls the gateway can
bill precisely.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import (
    WebProviderClientDep,
    WebSearchPolicyPortDep,
    get_budget_service,
    get_config,
    get_db_if_needed,
    get_log_writer,
    verify_api_key_or_master_key,
)
from gateway.api.routes._passthrough import (
    PASSTHROUGH_PROVIDER_ERROR_DETAIL,
    resolve_passthrough_user_id,
)
from gateway.api.routes._pipeline import (
    _elapsed_ms,
    _names_end_user,
    failure_status_code,
    log_gateway_rejection,
    rate_limit_headers,
)
from gateway.core.config import END_USER_BUDGET_HEADER, GatewayConfig
from gateway.core.database import release_session
from gateway.core.metered_pricing import quantize_cost
from gateway.exceptions.tools_exceptions import (
    DirectSearchFailedError,
    DirectSearchFieldRefusedError,
    SearchToolMisconfiguredError,
    SearchToolNotFoundError,
    SearchToolSelectionError,
    WebSearchCredentialRefusedError,
    WebSearchNotEnabledError,
    WebSearchOptionRefusedError,
    WebSearchPolicyResolutionFailedError,
)
from gateway.inflight import track_request
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.tools import WebTool
from gateway.models.usage import UsageLog
from gateway.ports.web_search_policy_port import WebSearchPolicyPort, WebSearchPolicyScope
from gateway.rate_limit import admit_rate_limit_rules, check_rate_limit
from gateway.schemas.tools import SearchRequest, SearchResponse
from gateway.services.budgets import (
    BudgetScopeRequest,
    BudgetService,
    reconcile_reservation,
    refund_reservation,
    reserve_budget,
)
from gateway.services.log_writer import LogWriter
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.pricing_service import find_model_pricing, flat_request_cost
from gateway.services.tools import pick_direct_search_instance, resolve_direct_search, run_direct_search
from gateway.services.workspace_scope import organization_for_key_id, workspace_for_key_id

router = APIRouter(tags=["search"])

SEARCH_ENDPOINT = "/v1/search"

# Stands in for the usage row's model when the request named no tool that could
# be resolved (unknown name, or none given with several configured). The column
# is not nullable, and "unknown" reads better in Activity than an empty string.
_UNRESOLVED_TOOL = "unknown"


@router.post("/search")
async def create_search(
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    budget_service: Annotated[BudgetService, Depends(get_budget_service)],
    web_search_policy: WebSearchPolicyPortDep,
    web_provider_client: WebProviderClientDep,
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
        db=db,
        config=config,
        log_writer=log_writer,
        budget_service=budget_service,
        web_search_policy=web_search_policy,
        web_provider_client=web_provider_client,
    )


@router.post("/search/{search_tool_name}")
async def create_search_for_tool(
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    search_tool_name: Annotated[str, Path(description="Configured search tool to run against")],
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession | None, Depends(get_db_if_needed)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
    budget_service: Annotated[BudgetService, Depends(get_budget_service)],
    web_search_policy: WebSearchPolicyPortDep,
    web_provider_client: WebProviderClientDep,
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
        db=db,
        config=config,
        log_writer=log_writer,
        budget_service=budget_service,
        web_search_policy=web_search_policy,
        web_provider_client=web_provider_client,
    )


async def _dispatch_search(
    *,
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    tool_name: str | None,
    auth_result: tuple[APIKey | None, bool],
    db: AsyncSession | None,
    config: GatewayConfig,
    log_writer: LogWriter,
    budget_service: BudgetService,
    web_search_policy: WebSearchPolicyPort,
    web_provider_client: httpx.AsyncClient,
) -> SearchResponse:
    """Run the search request scaffold: reserve, call, log, settle.

    The shape mirrors :func:`gateway.api.routes._passthrough.run_passthrough`,
    which cannot be reused directly: it resolves the request's model against
    the any-llm provider instances, and a search tool is neither a model nor an
    any-llm provider.
    """
    # The session the policy port reads on too, so one release before the provider call
    # frees the request's one connection. Never None: the endpoint is mounted only where
    # a process serves both planes, which get_db_if_needed hands a session.
    assert db is not None
    started_at = time.monotonic()
    api_key, _ = auth_result
    api_key_id = api_key.id if api_key else None
    # A key flagged exclude_from_budget logs cost but is never reserved or folded
    # into users.spend, matching every other billed endpoint.
    budget_exempt = api_key is not None and api_key.exclude_from_budget

    if api_key is not None and request.user and _names_end_user(api_key, request.user):
        # As on chat: the owner's own rpm is checked before an end user is created for it.
        rate_limit_info = await check_rate_limit(raw_request, str(api_key.user_id))
        end_user = await budget_service.resolve_end_user(
            api_key=api_key,
            external_id=request.user,
            requested_budget_id=raw_request.headers.get(END_USER_BUDGET_HEADER) or None,
        )
        user_id = end_user.user_id
        if end_user.budget_id is not None:
            response.headers[END_USER_BUDGET_HEADER] = end_user.budget_id
    else:
        user_id = resolve_passthrough_user_id(auth_result, request.user, reject_mismatch=config.reject_user_mismatch)
        rate_limit_info = await check_rate_limit(raw_request, user_id)

    async def log_rejection(detail: str, *, row_model: str, row_provider: str | None, status_code: int) -> None:
        """Record a search the gateway itself refused.

        The shared writer owns the row shape (``status="error"``, no cost,
        ``counts_toward_budget`` pinned True) and swallows a writer failure, so a
        sick log writer cannot turn this route's 400 or 403 into a 500.
        ``status_code`` is the status this refusal returns, which the row keeps so
        the failure taxonomy can tell a refusal from a provider fault.
        """
        await log_gateway_rejection(
            db=db,
            log_writer=log_writer,
            api_key_id=api_key_id,
            user_id=user_id,
            model=row_model,
            provider=row_provider,
            endpoint=SEARCH_ENDPOINT,
            detail=detail,
            status_code=status_code,
            started_at=started_at,
        )

    # Picked before any reservation is taken, so an unknown tool costs the
    # caller nothing to be told about. The refusal is still logged: a request
    # the gateway turned away is dropped traffic, and it belongs in Activity and
    # in the failure count rather than reaching the operator as a complaint.
    try:
        tool = pick_direct_search_instance(config, tool_name)
    except (SearchToolSelectionError, SearchToolMisconfiguredError) as exc:
        # No tool was resolved, so there is no provider to attribute the row to
        # and the name is whatever the caller asked for, matching how the
        # pass-through routes log a selector that did not resolve.
        await log_rejection(
            exc.message,
            row_model=tool_name or _UNRESOLVED_TOOL,
            row_provider=None,
            status_code=status.HTTP_400_BAD_REQUEST,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message) from exc

    # Resolved before the gates below rather than beside the usage row further
    # down, because the first of them is a question about the workspace.
    usage_workspace_id = await workspace_for_key_id(db, api_key_id)

    # The workspace's web search policy, read through the port the in-loop tool
    # reads it through. A workspace that has turned web search off has turned it
    # off: leaving this endpoint open would leave the switch bypassable by any key
    # in that workspace, which is a policy that fails open. Of the rest, only the
    # result ceiling and the organization key apply here.
    try:
        policy = await web_search_policy.resolve(
            WebSearchPolicyScope(workspace_id=usage_workspace_id, user_token=None), [WebTool.SEARCH]
        )
    except WebSearchPolicyResolutionFailedError as exc:
        await log_rejection(
            exc.message,
            row_model=tool.name,
            row_provider=tool.provider,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.message) from exc
    if policy is not None and not policy.enabled:
        refusal = WebSearchNotEnabledError()
        await log_rejection(
            refusal.message,
            row_model=tool.name,
            row_provider=tool.provider,
            status_code=status.HTTP_403_FORBIDDEN,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=refusal.message)

    # A workspace whose organization brought its own search key searches with it on a
    # call that names no tool, so its own account pays. The tool is still what the
    # request is allowlisted, rate-limited and timed out as.
    try:
        search = resolve_direct_search(config, tool, policy, request, named=tool_name is not None)
    except (
        WebSearchOptionRefusedError,
        DirectSearchFieldRefusedError,
        WebSearchCredentialRefusedError,
        # The tool was removed or changed by another writer since it was picked.
        SearchToolNotFoundError,
        SearchToolMisconfiguredError,
    ) as exc:
        await log_rejection(exc.message, row_model=tool.name, row_provider=tool.provider, status_code=exc.status_code)
        # A 5xx names no credential state to the caller, as the tenancy error handler does.
        detail = PASSTHROUGH_PROVIDER_ERROR_DETAIL if exc.status_code >= 500 else exc.message
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc

    tool_key = f"{tool.provider}:{tool.name}"

    # Model access control (per-key), keyed on the same <provider>:<tool> string
    # pricing uses. A key with no list of its own inherits its user's default, so
    # an unrestricted key is unaffected; a restricted one must name the search
    # tool, which is the fail-closed side to be on for a brand-new spend surface.
    key_allowlist = await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None and not is_model_allowed(key_allowlist, tool_key):
        not_allowed_detail = model_not_allowed_detail(tool_key)
        await log_rejection(
            not_allowed_detail,
            row_model=tool.name,
            row_provider=tool.provider,
            status_code=status.HTTP_403_FORBIDDEN,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=not_allowed_detail)

    # A search tool is not a model, so the community default-pricing dataset can
    # only produce a false match on the tool's name.
    price_provider, price_model = search.price_key
    pricing = await find_model_pricing(
        db,
        price_provider,
        price_model,
        use_defaults=False,
        organization_id=await organization_for_key_id(db, api_key_id),
    )
    # After the gates above, so a search they refuse is counted by no rule, and
    # before the reservation, so one the rules refuse holds no budget to refund.
    rate_limit_grant = await admit_rate_limit_rules(
        raw_request,
        key_id=api_key_id,
        user_id=user_id,
        estimated_tokens=0,
        budget_limits=None
        if budget_exempt
        else await budget_service.minute_limits(user_id, strategy=config.budget_strategy),
    )
    if rate_limit_grant is not None:
        # A `per: model` rule names a search tool by its pricing key, <provider>:<tool>.
        await rate_limit_grant.admit_model(tool.provider, tool.name)
    reservation = await reserve_budget(
        db,
        user_id,
        flat_request_cost(pricing),
        # Deliberately not the pricing key: ``model`` exists only to drive
        # reserve_budget's free-model shortcut, which splits the string through
        # any-llm. A search tool is not an any-llm model, so passing it logs a
        # warning on every request, and for a tool whose name did happen to be an
        # any-llm provider the shortcut would run an unguarded default-pricing
        # lookup on the tool name and could skip the reservation outright.
        model=None,
        strategy=config.budget_strategy,
        counts_toward_budget=not budget_exempt,
        # A search tool is not a model, but it is billed to a workspace like one,
        # so the scope carries the tool's provider as the narrowing axis.
        scope=BudgetScopeRequest(api_key=api_key, provider_instance=tool.provider),
    )

    def usage_row(**overrides: Any) -> UsageLog:
        """Build the row for a search that reached the provider.

        Unlike a refusal row, this one carries a cost and so honors the key's
        ``exclude_from_budget`` flag.
        """
        return UsageLog(
            id=str(uuid.uuid4()),
            workspace_id=usage_workspace_id,
            api_key_id=api_key_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=tool.name,
            provider=tool.provider,
            endpoint=SEARCH_ENDPOINT,
            latency_ms=_elapsed_ms(started_at),
            counts_toward_budget=not budget_exempt,
            **overrides,
        )

    # Every gate has passed and the search provider is about to be called, so the
    # request is genuinely in flight from here until its response has been sent.
    # Registered for the same reason as on the chat and pass-through paths, and a
    # dropped search cannot be seen settling in Activity either. The entry is
    # dropped by InFlightMiddleware, not here.
    track_request(
        raw_request,
        endpoint=SEARCH_ENDPOINT,
        # The same pair the usage row carries for search: the tool name and its
        # provider, so a search does not appear to change model when it settles.
        model=tool.name,
        provider=tool.provider,
        user_id=user_id,
        api_key_id=api_key_id,
    )
    # The policy port read on this same session, so this frees the request's one
    # connection before the provider call; settlement checks one back out.
    await release_session(db)

    try:
        result = await run_direct_search(search, web_provider_client)
    except Exception as exc:
        if isinstance(exc, DirectSearchFailedError):
            # The failure's safe fields ride as headers (decision 67); the detail stays generic.
            failed, row_status, headers = exc.message, exc.upstream_status or status.HTTP_502_BAD_GATEWAY, exc.headers
        else:
            # Raised by something other than the provider, and named by its type alone,
            # since an adapter's exception text could carry the query.
            failed, row_status, headers = type(exc).__name__, failure_status_code(exc), None
        await log_writer.put(usage_row(status="error", error_message=failed, status_code=row_status))
        await refund_reservation(db, reservation)
        logger.error("Search failed for tool '%s' (%s): %s", tool.name, search.resolved.provider, failed)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=PASSTHROUGH_PROVIDER_ERROR_DETAIL, headers=headers
        ) from exc
    except BaseException:
        # A cancelled request, its client gone, is refunded all the same.
        await refund_reservation(db, reservation)
        raise

    # The provider's own charge is the true cost; the configured flat rate is
    # the fallback for a provider that reports none.
    # Rounded once, here, so the row and the reconciled spend agree.
    cost = quantize_cost(result.cost if result.cost is not None else flat_request_cost(pricing))
    await log_writer.put(
        usage_row(
            status="success",
            # Search bills per request; there are no tokens to report, and a
            # zero is truthful where a null would read as "unknown".
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=cost,
        )
    )
    await reconcile_reservation(db, reservation, cost)

    if rate_limit_info:
        for header, value in rate_limit_headers(rate_limit_info).items():
            response.headers[header] = value

    return result.response
