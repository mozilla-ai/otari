"""Search pass-through endpoint: a direct, billed search request.

``otari_web_search`` answers a model's tool call mid-completion. This endpoint
is the standalone counterpart: the caller asks, the gateway asks the search
provider, and the result comes straight back, having gone through the same
auth, rate limit, budget reservation, and usage log as a completion. It exists
so a client that proxies a search API can keep one gateway on the billing path
instead of holding a second provider key of its own.

The request and response follow LiteLLM's ``/v1/search`` (itself shaped after
Perplexity's Search API), so a caller moving off the LiteLLM proxy keeps its
request shape; :mod:`gateway.services.search_backend` translates to and from the
provider's native shape and owns the per-provider adapters. Two Perplexity
features are deliberately not modeled: ``query`` is a single string rather than
the multi-query array, and the filters with no field below (recency, context
size, published-date) are ignored rather than rejected. Both are called out in
``docs/api-reference.md`` so a migrating caller can check for them.

Both the body-selected (``POST /v1/search``) and path-selected
(``POST /v1/search/{search_tool_name}``) forms log ``endpoint="/v1/search"``,
so one Activity filter covers every search regardless of how the tool was
named.

A request the gateway itself turns away (an unknown or ambiguous tool name, a
tool the caller's key may not use) writes a usage row too, through the shared
:func:`gateway.api.routes._pipeline.log_gateway_rejection` the pass-through
routes use: refused traffic is dropped traffic, and it should be visible in
Activity and countable as a failure rather than invisible outside the caller's
own logs.

Billing: search is priced per request, not per token, so a usage row carries
zero tokens and a cost taken from the provider's own reported charge when it
reports one. Otherwise the cost is the flat per-request rate configured for
``<provider>:<tool>`` under the same convention moderations uses
(:func:`flat_request_cost`). Like moderations and audio, search is exempt from
``require_pricing``: a provider that reports its own cost is metered whether or
not a rate is configured, so failing closed would reject calls the gateway can
bill precisely.
"""

import time
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_log_writer, verify_api_key_or_master_key
from gateway.api.routes._passthrough import (
    PASSTHROUGH_PROVIDER_ERROR_DETAIL,
    resolve_passthrough_user_id,
)
from gateway.api.routes._pipeline import (
    _elapsed_ms,
    failure_status_code,
    log_gateway_rejection,
    rate_limit_headers,
)
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, UsageLog
from gateway.rate_limit import check_rate_limit
from gateway.services.budget_service import reconcile_reservation, refund_reservation, reserve_budget
from gateway.services.log_writer import LogWriter
from gateway.services.model_access import is_model_allowed, model_not_allowed_detail, resolve_request_allowlist
from gateway.services.pricing_service import find_model_pricing, flat_request_cost
from gateway.services.search_backend import (
    MAX_RESULTS_CAP,
    SearchHit,
    SearchQuery,
    SearchTool,
    SearchToolError,
    resolve_search_tool,
    run_search,
)

router = APIRouter(prefix="/v1", tags=["search"])

SEARCH_ENDPOINT = "/v1/search"

# Perplexity's documented ceiling on search_domain_filter, mirrored so an
# oversized list is rejected here rather than by the upstream provider.
_MAX_DOMAIN_FILTERS = 20

# Stands in for the usage row's model when the request named no tool that could
# be resolved (unknown name, or none given with several configured). The column
# is not nullable, and "unknown" reads better in Activity than an empty string.
_UNRESOLVED_TOOL = "unknown"


class SearchRequest(BaseModel):
    """A search request, following LiteLLM's ``/v1/search`` body."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"query": "otari llm gateway", "max_results": 5}},
    )

    query: str = Field(min_length=1, description="The search query")
    search_tool_name: str | None = Field(
        default=None,
        description=(
            "Configured search tool to run against. Optional when exactly one tool is "
            "configured, and ignored on POST /v1/search/{search_tool_name}."
        ),
    )
    max_results: int | None = Field(
        default=None, ge=1, le=MAX_RESULTS_CAP, description="Maximum number of results to return"
    )
    search_domain_filter: list[str] | None = Field(
        default=None,
        max_length=_MAX_DOMAIN_FILTERS,
        description="Restrict results to these domains; prefix a domain with '-' to exclude it instead",
    )
    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Two-letter ISO country code to localize results to",
    )
    max_tokens_per_page: int | None = Field(
        default=None, ge=1, description="Approximate cap on the page content returned per result"
    )
    user: str | None = Field(default=None, description="User ID for usage attribution")


class SearchResultItem(BaseModel):
    """One search result."""

    title: str | None = None
    url: str
    snippet: str | None = None
    date: str | None = None


class SearchResponse(BaseModel):
    """A completed search."""

    object: Literal["search"] = "search"
    search_tool: str = Field(description="The configured search tool that served the request")
    results: list[SearchResultItem]


@router.post("/search")
async def create_search(
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> SearchResponse:
    """Run a search against a configured search tool.

    The tool is taken from ``search_tool_name``, which may be omitted when
    exactly one tool is configured.

    Authentication modes:
    - Master key: the ``user`` field is required and may name any existing user.
    - API key: usage and spend always bind to the key's own user. A ``user``
      field naming a different user is rejected with 403 (or ignored, when
      ``reject_user_mismatch`` is disabled); it is never billed to that user.
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
    )


@router.post("/search/{search_tool_name}")
async def create_search_for_tool(
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    search_tool_name: Annotated[str, Path(description="Configured search tool to run against")],
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    log_writer: Annotated[LogWriter, Depends(get_log_writer)],
) -> SearchResponse:
    """Run a search against the search tool named in the path.

    Identical to ``POST /v1/search`` except that the path names the tool, which
    is the form LiteLLM clients use. Any ``search_tool_name`` in the body is
    ignored.

    Authentication modes:
    - Master key: the ``user`` field is required and may name any existing user.
    - API key: usage and spend always bind to the key's own user. A ``user``
      field naming a different user is rejected with 403 (or ignored, when
      ``reject_user_mismatch`` is disabled); it is never billed to that user.
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
    )


async def _dispatch_search(
    *,
    raw_request: Request,
    response: Response,
    request: SearchRequest,
    tool_name: str | None,
    auth_result: tuple[APIKey | None, bool],
    db: AsyncSession,
    config: GatewayConfig,
    log_writer: LogWriter,
) -> SearchResponse:
    """Run the search request scaffold: reserve, call, log, settle.

    The shape mirrors :func:`gateway.api.routes._passthrough.run_passthrough`,
    which cannot be reused directly: it resolves the request's model against
    the any-llm provider instances, and a search tool is neither a model nor an
    any-llm provider.
    """
    started_at = time.monotonic()
    api_key, _ = auth_result
    api_key_id = api_key.id if api_key else None
    # A key flagged exclude_from_budget logs cost but is never reserved or folded
    # into users.spend, matching every other billed endpoint.
    budget_exempt = api_key is not None and api_key.exclude_from_budget

    user_id = resolve_passthrough_user_id(auth_result, request.user, reject_mismatch=config.reject_user_mismatch)
    rate_limit_info = check_rate_limit(raw_request, user_id)

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

    # Resolved before any reservation is taken, so an unknown tool costs the
    # caller nothing to be told about. The refusal is still logged: a request
    # the gateway turned away is dropped traffic, and it belongs in Activity and
    # in the failure count rather than reaching the operator as a complaint.
    try:
        tool = resolve_search_tool(config, tool_name)
    except SearchToolError as exc:
        tool_error_detail = str(exc)
        # No tool was resolved, so there is no provider to attribute the row to
        # and the name is whatever the caller asked for, matching how the
        # pass-through routes log a selector that did not resolve.
        await log_rejection(
            tool_error_detail,
            row_model=tool_name or _UNRESOLVED_TOOL,
            row_provider=None,
            status_code=status.HTTP_400_BAD_REQUEST,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=tool_error_detail) from exc

    pricing_key = f"{tool.provider}:{tool.name}"

    # Model access control (per-key), keyed on the same <provider>:<tool> string
    # pricing uses. A key with no list of its own inherits its user's default, so
    # an unrestricted key is unaffected; a restricted one must name the search
    # tool, which is the fail-closed side to be on for a brand-new spend surface.
    key_allowlist = await resolve_request_allowlist(db, api_key)
    if key_allowlist is not None and not is_model_allowed(key_allowlist, pricing_key):
        not_allowed_detail = model_not_allowed_detail(pricing_key)
        await log_rejection(
            not_allowed_detail,
            row_model=tool.name,
            row_provider=tool.provider,
            status_code=status.HTTP_403_FORBIDDEN,
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=not_allowed_detail)

    # A search tool is not a model, so the community default-pricing dataset can
    # only produce a false match on the tool's name.
    pricing = await find_model_pricing(db, tool.provider, tool.name, use_defaults=False)
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
    )

    def usage_row(**overrides: Any) -> UsageLog:
        """Build the row for a search that reached the provider.

        Unlike a refusal row, this one carries a cost and so honors the key's
        ``exclude_from_budget`` flag.
        """
        return UsageLog(
            id=str(uuid.uuid4()),
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

    try:
        outcome = await run_search(tool, _search_query(request))
        # The provider's own charge is the true cost; the configured flat rate is
        # the fallback for a provider that reports none.
        cost = outcome.cost_usd if outcome.cost_usd is not None else flat_request_cost(pricing)
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
    except HTTPException:
        await refund_reservation(db, reservation)
        raise
    except Exception as exc:
        await log_writer.put(usage_row(status="error", error_message=str(exc), status_code=failure_status_code(exc)))
        await refund_reservation(db, reservation)
        # The raw provider message is kept on the usage log and the gateway log,
        # never in the response: it can carry upstream internals, and the
        # pass-through routes hold the same line.
        logger.error("Search failed for tool '%s' (%s): %s", tool.name, tool.provider, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=PASSTHROUGH_PROVIDER_ERROR_DETAIL,
        ) from exc

    if rate_limit_info:
        for header, value in rate_limit_headers(rate_limit_info).items():
            response.headers[header] = value

    return _to_response(tool, outcome.results)


def _search_query(request: SearchRequest) -> SearchQuery:
    return SearchQuery(
        query=request.query,
        max_results=request.max_results,
        domain_filter=tuple(request.search_domain_filter or ()),
        country=request.country,
        max_tokens_per_page=request.max_tokens_per_page,
    )


def _to_response(tool: SearchTool, hits: list[SearchHit]) -> SearchResponse:
    return SearchResponse(
        search_tool=tool.name,
        results=[SearchResultItem(title=hit.title, url=hit.url, snippet=hit.snippet, date=hit.date) for hit in hits],
    )
