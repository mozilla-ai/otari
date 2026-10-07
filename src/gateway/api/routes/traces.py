"""Reading agent traces: the session list, one session with its turns, and the per-hour counts.

Two routers serve the same four reads over different scopes. ``/traces`` is
deployment-wide and operator-only, like ``/usage``. ``/organizations/me/traces``
reads the caller's own organization, by the rule the organization's usage reads
use. The scope is derived from the caller, never
taken from the request; a filter narrows inside it and cannot widen it.
"""

import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from gateway.api.deps import (
    ContentAccessServiceDep,
    CurrentIdentity,
    TraceServiceDep,
    TraceSettingsServiceDep,
    get_deployment_trace_scope,
    get_organization_content_scope,
    get_organization_trace_scope,
    get_session_identity,
    require_deployment_operator,
    verify_master_key,
)
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.models.tenancy import User as TenancyUser
from gateway.ports.trace_storage_port import TraceFilter, TraceScope
from gateway.schemas.traces import (
    BreakGlassRequest,
    ContentAccessListPublic,
    ContentPurgePublic,
    SpanContentPublic,
    TraceCountPublic,
    TraceDetailPublic,
    TraceListPublic,
    TraceSeriesPublic,
    TraceSettingsPublic,
    TraceSettingsUpdate,
    content_access_public,
    detail_public,
    list_public,
    series_public,
    settings_public,
)

MAX_PAGE_SIZE = 100
_MAX_FILTER_VALUES = 50


def _filters(
    workspace_id: Annotated[
        uuid.UUID | None,
        Query(description="Only sessions in this workspace. Narrows the caller's scope; never widens it."),
    ] = None,
    start: Annotated[datetime | None, Query(description="Last activity at or after this instant (ISO 8601).")] = None,
    end: Annotated[datetime | None, Query(description="Last activity before this instant (ISO 8601).")] = None,
    user_id: Annotated[list[str] | None, Query(max_length=_MAX_FILTER_VALUES, description="Owning users.")] = None,
    api_key_id: Annotated[list[str] | None, Query(max_length=_MAX_FILTER_VALUES, description="API keys.")] = None,
    harness: Annotated[list[str] | None, Query(max_length=_MAX_FILTER_VALUES, description="Agent harnesses.")] = None,
    session_source: Annotated[
        list[Literal["client", "harness", "otlp", "none"]] | None,
        Query(max_length=4, description="What grouped the session."),
    ] = None,
    has_error: Annotated[
        bool | None, Query(description="Only sessions with, or without, an unrecovered error.")
    ] = None,
    q: Annotated[str | None, Query(max_length=64, description="A trace id, or the start of one.")] = None,
) -> TraceFilter:
    return TraceFilter(
        workspace_ids=(workspace_id,) if workspace_id is not None else (),
        start=start,
        end=end,
        user_ids=tuple(user_id or ()),
        api_key_ids=tuple(api_key_id or ()),
        harnesses=tuple(harness or ()),
        session_sources=tuple(session_source or ()),
        has_error=has_error,
        trace_id_prefix=q or None,
    )


FiltersDep = Annotated[TraceFilter, Depends(_filters)]

# What a content read's audit line names when a header master key, which names nobody, made it.
MASTER_KEY_READER = "master_key"


def _reader(session_identity: Annotated[TenancyUser | None, Depends(get_session_identity)]) -> str:
    """Who is reading, for the audit log: the signed-in user, or the deployment master key."""
    return f"user:{session_identity.id}" if session_identity is not None else MASTER_KEY_READER


ReaderDep = Annotated[str, Depends(_reader)]


def build_trace_router(router: APIRouter, scope_dependency: Callable[..., Any]) -> APIRouter:
    """Declare the trace metadata reads on ``router``, each confined to the scope ``scope_dependency`` derives.

    Content is not read here: each router declares its own content route, with its own rule.
    """
    ScopeDep = Annotated[TraceScope, Depends(scope_dependency)]

    @router.get("", response_model=TraceListPublic)
    async def list_traces(
        traces: TraceServiceDep,
        scope: ScopeDep,
        filters: FiltersDep,
        skip: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    ) -> TraceListPublic:
        """List agent sessions, most recently active first."""
        return list_public(await traces.search(scope, filters, limit=limit, offset=skip))

    @router.get("/count", response_model=TraceCountPublic)
    async def count_traces(traces: TraceServiceDep, scope: ScopeDep, filters: FiltersDep) -> TraceCountPublic:
        """Count the agent sessions that match."""
        return TraceCountPublic(count=await traces.count(scope, filters))

    @router.get("/series", response_model=TraceSeriesPublic)
    async def trace_series(
        traces: TraceServiceDep,
        scope: ScopeDep,
        filters: FiltersDep,
        bucket: Annotated[Literal["hour", "day"], Query()] = "hour",
    ) -> TraceSeriesPublic:
        """Count sessions per bucket of their start, split by whether any span failed unrecovered."""
        return series_public(bucket, await traces.series(scope, filters, bucket=bucket))

    @router.get("/{trace_id}", response_model=TraceDetailPublic)
    async def get_trace(traces: TraceServiceDep, scope: ScopeDep, trace_id: str) -> TraceDetailPublic:
        """One agent session: its turns, and every span inside them. A trace outside the scope is a 404."""
        return detail_public(await traces.detail(scope, trace_id))

    return router


router = build_trace_router(
    APIRouter(prefix="/traces", tags=["traces"], dependencies=[Depends(require_deployment_operator)]),
    get_deployment_trace_scope,
)


@router.post("/{trace_id}/spans/{span_id}/content/break-glass", response_model=SpanContentPublic)
async def break_glass_span_content(
    access: ContentAccessServiceDep, reader: ReaderDep, trace_id: str, span_id: str, body: BreakGlassRequest
) -> SpanContentPublic:
    """Read one span's content as a platform operator, for a stated reason such as a legal request.

    The only way an operator reads content. The read and its reason are recorded where the workspace's
    admins see them, and logged."""
    return SpanContentPublic(
        fields=await access.read_break_glass(reader=reader, reason=body.reason, trace_id=trace_id, span_id=span_id)
    )


# Authentication only, like the rest of ``/organizations/me``: the scope dependency
# is what confines each read, from the caller's own memberships.
organization_router = build_trace_router(
    APIRouter(
        prefix="/organizations/me/traces", tags=["organization-traces"], dependencies=[Depends(verify_master_key)]
    ),
    get_organization_trace_scope,
)


@organization_router.get("/{trace_id}/spans/{span_id}/content", response_model=SpanContentPublic)
async def get_span_content(
    access: ContentAccessServiceDep,
    identity: CurrentIdentity,
    visible: Annotated[TraceScope, Depends(get_organization_trace_scope)],
    administered: Annotated[TraceScope, Depends(get_organization_content_scope)],
    trace_id: str,
    span_id: str,
) -> SpanContentPublic:
    """One span's captured content: a request's input and output, or a tool's arguments and result.

    Readable by the session's own user, and by the organization's owners and admins where the workspace
    lets them. Every read is recorded. Anyone else who can see the session gets a 403."""
    return SpanContentPublic(
        fields=await access.read_as_member(
            reader_id=identity.id, visible=visible, administered=administered, trace_id=trace_id, span_id=span_id
        )
    )


# A workspace's content capture, set by an admin of that workspace. Master key on
# the router, then the caller's identity for the per-workspace role check, like
# the workspace code execution policy.
settings_router = APIRouter(
    prefix="/workspaces/{workspace_id}/trace-settings",
    tags=["workspace-trace-settings"],
    dependencies=[Depends(verify_master_key)],
)


@settings_router.get("", response_model=TraceSettingsPublic)
async def get_workspace_trace_settings(
    service: TraceSettingsServiceDep, current_identity: CurrentIdentity, workspace_id: uuid.UUID
) -> TraceSettingsPublic:
    """How much of its requests' content the workspace keeps. Off until one of its admins turns it on."""
    return settings_public(await service.get(user=current_identity, workspace_id=workspace_id))


@settings_router.put("", response_model=TraceSettingsPublic)
async def set_workspace_trace_settings(
    service: TraceSettingsServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    body: TraceSettingsUpdate,
) -> TraceSettingsPublic:
    """Set how much content the workspace keeps, up to what the deployment permits, and whether the organization's
    admins may read it. Either field may be left out. Who changed it is recorded."""
    view = await service.get(user=current_identity, workspace_id=workspace_id)
    if body.content_capture is not None:
        view = await service.set_content_capture(
            user=current_identity, workspace_id=workspace_id, level=body.content_capture
        )
    if body.admin_content_access is not None:
        view = await service.set_admin_content_access(
            user=current_identity, workspace_id=workspace_id, allowed=body.admin_content_access
        )
    return settings_public(view)


@settings_router.post("/purge-content", response_model=ContentPurgePublic)
async def purge_workspace_trace_content(
    service: TraceSettingsServiceDep, current_identity: CurrentIdentity, workspace_id: uuid.UUID
) -> ContentPurgePublic:
    """Delete every span's stored content in the workspace and destroy the keys that sealed it. Spans stay."""
    return ContentPurgePublic(removed=await service.purge_content(user=current_identity, workspace_id=workspace_id))


@settings_router.get("/content-access", response_model=ContentAccessListPublic)
async def list_workspace_content_reads(
    service: TraceSettingsServiceDep,
    current_identity: CurrentIdentity,
    workspace_id: uuid.UUID,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
) -> ContentAccessListPublic:
    """Every recorded read of the workspace's captured content, newest first: who read it, as what, and why."""
    reads = await service.content_reads(user=current_identity, workspace_id=workspace_id, limit=limit, offset=skip)
    return ContentAccessListPublic(items=[content_access_public(view) for view in reads])


FEATURE = CoreFeature(
    name="traces",
    surface=Surface("traces"),
    enabled=lambda config: config.trace_capture_enabled,
    routers=lambda config: (router, organization_router, settings_router),
)
