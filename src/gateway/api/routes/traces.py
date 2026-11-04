"""Agent traces as a core feature: the session list, one session with its turns, the per-hour counts, and retention.

Two routers serve the same reads over different scopes. ``/traces`` is
operator-only, like ``/usage``, and reads every workspace of the operator's active
organization. ``/organizations/me/traces`` reads the caller's own organization, by
the rule the organization's usage reads use. The scope is derived from the caller, never
taken from the request; a filter narrows inside it and cannot widen it.
"""

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request, status

from gateway.api.deps import (
    NoSessionOwners,
    OrganizationBreakGlass,
    PlatformBreakGlass,
    TraceServiceDep,
    build_trace_service,
    get_config,
    get_deployment_trace_scope,
    get_organization_break_glass,
    get_organization_trace_scope,
    get_platform_break_glass,
    require_deployment_operator,
    require_signed_in_reader,
    verify_master_key,
)
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.schemas.traces import (
    BreakGlassCreate,
    BreakGlassReadListPublic,
    ContentReadListPublic,
    ContentShareCreate,
    ContentShareListPublic,
    ContentSharePublic,
    PlatformBreakGlassCreate,
    TraceContentPolicyPublic,
    TraceCountPublic,
    TraceDetailPublic,
    TraceListPublic,
    TraceSeriesPublic,
    TraceSummaryPublic,
    break_glass_read_public,
    detail_public,
    list_public,
    read_public,
    series_public,
    share_public,
    summary_public,
)
from gateway.services.traces import SpanContent, run_trace_retention
from gateway.types.traces import (
    BreakGlass,
    TraceBucketGrain,
    TraceFilter,
    TraceScope,
    TraceSort,
    TraceSortKey,
)

MAX_PAGE_SIZE = 100
MAX_SHARE_PAGE = 50
# Traces expire on the scale of days, so an hourly pass is soon enough.
_RETENTION_INTERVAL_S = 3600.0
_MAX_FILTER_VALUES = 50


def _filters(
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Only sessions in this workspace. Narrows the caller's scope; never widens it."),
    ] = None,
    start: Annotated[datetime | None, Query(description="Sessions active at or after this instant (ISO 8601).")] = None,
    end: Annotated[datetime | None, Query(description="Sessions started before this instant (ISO 8601).")] = None,
    user_id: Annotated[list[str] | None, Query(max_length=_MAX_FILTER_VALUES, description="Owning users.")] = None,
    api_key_id: Annotated[list[str] | None, Query(max_length=_MAX_FILTER_VALUES, description="API keys.")] = None,
    model: Annotated[
        list[str] | None,
        Query(
            max_length=_MAX_FILTER_VALUES,
            description="Sessions with an LLM call recorded under any of these model ids (exact match).",
        ),
    ] = None,
    harness: Annotated[list[str] | None, Query(max_length=_MAX_FILTER_VALUES, description="Agent harnesses.")] = None,
    session_source: Annotated[
        list[Literal["client", "harness", "otlp", "none"]] | None,
        Query(max_length=4, description="What grouped the session."),
    ] = None,
    has_error: Annotated[
        bool | None, Query(description="Only sessions with, or without, an unrecovered error.")
    ] = None,
    q: Annotated[
        str | None,
        Query(max_length=128, description="Part of a trace id, session name, harness or user id, in any case."),
    ] = None,
) -> TraceFilter:
    return TraceFilter(
        workspace_ids=(workspace_id,) if workspace_id is not None else (),
        start=start,
        end=end,
        user_ids=tuple(user_id or ()),
        api_key_ids=tuple(api_key_id or ()),
        models=tuple(model or ()),
        harnesses=tuple(harness or ()),
        session_sources=tuple(session_source or ()),
        has_error=has_error,
        search=(q or "").strip() or None,
    )


FiltersDep = Annotated[TraceFilter, Depends(_filters)]


def build_trace_router(router: APIRouter, scope_dependency: Callable[..., Any]) -> APIRouter:
    """Declare the trace reads on ``router``, each confined to the scope ``scope_dependency`` derives."""
    ScopeDep = Annotated[TraceScope, Depends(scope_dependency)]

    @router.get("", response_model=TraceListPublic)
    async def list_traces(
        traces: TraceServiceDep,
        scope: ScopeDep,
        filters: FiltersDep,
        skip: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
        sort: Annotated[
            TraceSortKey,
            Query(description="What to order by. Latency runs from a session's start to its last activity."),
        ] = "last_activity",
        order: Annotated[Literal["asc", "desc"], Query(description="The direction of `sort`.")] = "desc",
    ) -> TraceListPublic:
        """List agent sessions, most recently active first unless ``sort`` says otherwise."""
        return list_public(
            await traces.search(
                scope, filters, limit=limit, offset=skip, sort=TraceSort(key=sort, descending=order == "desc")
            )
        )

    @router.get("/count", response_model=TraceCountPublic)
    async def count_traces(traces: TraceServiceDep, scope: ScopeDep, filters: FiltersDep) -> TraceCountPublic:
        """Count the agent sessions that match."""
        return TraceCountPublic(count=await traces.count(scope, filters))

    @router.get("/series", response_model=TraceSeriesPublic)
    async def trace_series(
        traces: TraceServiceDep,
        scope: ScopeDep,
        filters: FiltersDep,
        bucket: Annotated[TraceBucketGrain, Query()] = "hour",
    ) -> TraceSeriesPublic:
        """Count the sessions active in each bucket, split by whether they failed.

        A session is active from its start to its last activity, so a long one counts in every bucket it ran
        through. Here ``start`` and ``end`` bound the buckets: a session that began before
        ``start`` or was still active at ``end`` counts in the buckets between them it overlaps. Buckets are
        aligned to their size in UTC and only the populated ones are returned. A session failed when it holds an
        unrecovered error other than a tool's. A tool's error goes back to the model, an MCP tool's included, so
        a session whose only errors are tools' counts as succeeded.
        """
        return series_public(bucket, await traces.series(scope, filters, bucket=bucket))

    @router.get("/requests/{request_id}", response_model=TraceSummaryPublic)
    async def trace_of_request(traces: TraceServiceDep, scope: ScopeDep, request_id: str) -> TraceSummaryPublic:
        """The session a gateway request was traced into, by the ``Otari-Request-ID`` it was sent.

        The usage log's ``request_group_id`` is that id, so a request log row opens its
        session here. A request outside the scope, or with no trace stored, is a 404.
        """
        return summary_public(await traces.for_request(scope, request_id))

    @router.get("/{trace_id}", response_model=TraceDetailPublic)
    async def get_trace(
        traces: TraceServiceDep,
        scope: ScopeDep,
        trace_id: str,
        workspace_id: Annotated[
            uuid.UUID | None, Query(description="The workspace the trace is in, as its list entry names it.")
        ] = None,
    ) -> TraceDetailPublic:
        """One agent session: its turns, and every span inside them.

        A trace outside the scope is a 404. Without ``workspace_id``, an id in more
        than one of the caller's workspaces is a 409 rather than a guess.
        """
        return detail_public(await traces.detail(scope, trace_id, workspace_id=workspace_id))

    return router


router = build_trace_router(
    APIRouter(prefix="/traces", tags=["traces"], dependencies=[Depends(require_deployment_operator)]),
    get_deployment_trace_scope,
)


@router.post(
    "/{trace_id}/break-glass", response_model=TraceDetailPublic, dependencies=[Depends(require_signed_in_reader)]
)
async def platform_break_glass(
    traces: TraceServiceDep,
    operator: Annotated[PlatformBreakGlass, Depends(get_platform_break_glass)],
    trace_id: str,
    body: PlatformBreakGlassCreate,
) -> TraceDetailPublic:
    """Open a session's content in any organization without its owner's share, as the deployment's operator.

    Only where ``trace_platform_break_glass_enabled`` is on; a 404 otherwise. The
    opening is recorded with its reason and reference before anything opens, and
    listed at ``GET /traces/audit/break-glass``.
    """
    break_glass = BreakGlass(
        kind="deployment",
        organization_id=operator.organization_id,
        reader_user_id=operator.reader.user_id,
        reader_email=operator.reader.email,
        reason=body.reason,
        reference=body.reference,
    )
    return detail_public(
        await traces.break_glass_detail(
            TraceScope.deployment(), trace_id, break_glass=break_glass, workspace_id=operator.workspace_id
        )
    )


@router.get("/audit/break-glass", response_model=BreakGlassReadListPublic)
async def list_break_glass_reads(
    traces: TraceServiceDep,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
) -> BreakGlassReadListPublic:
    """The operator's recorded break-glass openings, in any organization, newest first.

    An organization owner's openings are listed for that organization's owners
    alone, at ``GET /organizations/me/traces/audit/break-glass``: their reasons
    can name the people under investigation, which is no business of the operator's.
    """
    reads, has_more = await traces.break_glass_reads(organization_id=None, kind="deployment", limit=limit, offset=skip)
    return BreakGlassReadListPublic(items=[break_glass_read_public(read) for read in reads], has_more=has_more)


# Behind a sign-in rather than on the public bootstrap: whether a deployment keeps
# prompts, and whether its operator can open them, is nothing to tell a visitor.
policy_router = APIRouter(prefix="/trace-content", tags=["traces"], dependencies=[Depends(verify_master_key)])


@policy_router.get("/policy", response_model=TraceContentPolicyPublic)
async def get_trace_content_policy(
    request: Request, config: Annotated[GatewayConfig, Depends(get_config)]
) -> TraceContentPolicyPublic:
    """What this deployment keeps of agent sessions, and who besides a session's owner can open it."""
    content: SpanContent | None = getattr(request.app.state, "trace_content", None)
    return TraceContentPolicyPublic(
        # A deployment with no key to seal with keeps no content, whatever its settings say.
        content_available=config.trace_content_enabled and content is not None and content.sealing,
        platform_break_glass=config.trace_platform_break_glass_enabled,
        retention_days=config.trace_retention_days,
        session_max_age_days=config.trace_session_max_age_days,
    )


# Authentication only, like the rest of ``/organizations/me``: the scope dependency
# is what confines each read, from the caller's own memberships.
organization_router = build_trace_router(
    APIRouter(
        prefix="/organizations/me/traces", tags=["organization-traces"], dependencies=[Depends(verify_master_key)]
    ),
    get_organization_trace_scope,
)

OrganizationScopeDep = Annotated[TraceScope, Depends(get_organization_trace_scope)]
WorkspaceQuery = Annotated[
    uuid.UUID | None, Query(description="The workspace the trace is in, as its list entry names it.")
]


@organization_router.post(
    "/{trace_id}/break-glass", response_model=TraceDetailPublic, dependencies=[Depends(require_signed_in_reader)]
)
async def organization_break_glass(
    traces: TraceServiceDep,
    owner: Annotated[OrganizationBreakGlass, Depends(get_organization_break_glass)],
    trace_id: str,
    body: BreakGlassCreate,
    workspace_id: WorkspaceQuery = None,
) -> TraceDetailPublic:
    """Open a session's content without its owner's share, as an owner of the organization.

    For an investigation the session's owner cannot be asked to share for, a member
    who left included. The opening is recorded with its reason before anything
    opens, and listed for every owner at ``GET .../audit/break-glass``. The
    session's owner is not told.
    """
    break_glass = BreakGlass(
        kind="organization",
        organization_id=owner.organization_id,
        reader_user_id=owner.reader.user_id,
        reader_email=owner.reader.email,
        reason=body.reason,
    )
    return detail_public(
        await traces.break_glass_detail(owner.scope, trace_id, break_glass=break_glass, workspace_id=workspace_id)
    )


@organization_router.get("/audit/break-glass", response_model=BreakGlassReadListPublic)
async def list_organization_break_glass_reads(
    traces: TraceServiceDep,
    owner: Annotated[OrganizationBreakGlass, Depends(get_organization_break_glass)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
) -> BreakGlassReadListPublic:
    """The organization's owners' break-glass openings, newest first. Owners only."""
    reads, has_more = await traces.break_glass_reads(
        organization_id=owner.organization_id, kind="organization", limit=limit, offset=skip
    )
    return BreakGlassReadListPublic(items=[break_glass_read_public(read) for read in reads], has_more=has_more)


@organization_router.post("/{trace_id}/shares", response_model=ContentSharePublic, status_code=status.HTTP_201_CREATED)
async def share_trace_content(
    traces: TraceServiceDep,
    scope: OrganizationScopeDep,
    trace_id: str,
    body: ContentShareCreate,
    workspace_id: WorkspaceQuery = None,
) -> ContentSharePublic:
    """Let the organization's admins, or the deployment's operator, read what one of your sessions was about.

    A session's content is private to its own user. A share lets its audience read
    it until you end it or lose the workspace, and each read through it is recorded for you.
    Only the session's own user can share it.
    """
    return share_public(
        await traces.share_content(scope, trace_id, audience=body.audience, note=body.note, workspace_id=workspace_id)
    )


@organization_router.get("/{trace_id}/shares", response_model=ContentShareListPublic)
async def list_trace_content_shares(
    traces: TraceServiceDep,
    scope: OrganizationScopeDep,
    trace_id: str,
    workspace_id: WorkspaceQuery = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_SHARE_PAGE)] = 20,
) -> ContentShareListPublic:
    """The shares one of your sessions has had, ended ones included, newest first, each with its newest reads."""
    shares, has_more = await traces.content_shares(scope, trace_id, limit=limit, offset=skip, workspace_id=workspace_id)
    return ContentShareListPublic(items=[share_public(share) for share in shares], has_more=has_more)


@organization_router.get("/{trace_id}/shares/{share_id}/reads", response_model=ContentReadListPublic)
async def list_trace_content_share_reads(
    traces: TraceServiceDep,
    scope: OrganizationScopeDep,
    trace_id: str,
    share_id: uuid.UUID,
    workspace_id: WorkspaceQuery = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
) -> ContentReadListPublic:
    """Every read through one share of your session, newest first, a page at a time."""
    reads, has_more = await traces.content_share_reads(
        scope, trace_id, share_id, limit=limit, offset=skip, workspace_id=workspace_id
    )
    return ContentReadListPublic(items=[read_public(read) for read in reads], has_more=has_more)


@organization_router.delete("/{trace_id}/shares/{share_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_trace_content_share(
    traces: TraceServiceDep,
    scope: OrganizationScopeDep,
    trace_id: str,
    share_id: uuid.UUID,
    workspace_id: WorkspaceQuery = None,
) -> None:
    """Stop a share letting its audience read what your session was about."""
    await traces.end_content_share(scope, trace_id, share_id, workspace_id=workspace_id)


async def _retention(config: GatewayConfig) -> None:
    # Retention deletes by age and never reads content.
    await run_trace_retention(
        lambda uow: build_trace_service(uow, SpanContent.disabled(), NoSessionOwners()),
        retention_days=config.trace_retention_days,
        max_age_days=config.trace_session_max_age_days,
        interval=_RETENTION_INTERVAL_S,
    )


FEATURE = CoreFeature(
    name="traces",
    surface=Surface("traces"),
    enabled=lambda config: config.trace_capture_enabled,
    routers=lambda config: (router, organization_router, policy_router),
)

# Apart from capture, so a deployment that stops recording still deletes what
# it recorded, with its content, shares and reads, on the documented schedule.
RETENTION_FEATURE = CoreFeature(
    name="trace_retention",
    surface=None,
    enabled=lambda config: True,
    routers=lambda config: (),
    worker=_retention,
)
