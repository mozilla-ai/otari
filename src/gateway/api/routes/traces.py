"""Reading agent traces: the session list, one session with its turns, and the per-hour counts.

Two routers serve the same four reads over different scopes. ``/traces`` is
deployment-wide and operator-only, like ``/usage``. ``/organizations/me/traces``
reads the caller's own organization, by the rule the organization's usage reads
use. The scope is derived from the caller, never
taken from the request; a filter narrows inside it and cannot widen it.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from gateway.api.deps import (
    TraceServiceDep,
    get_deployment_trace_scope,
    get_organization_trace_scope,
    require_deployment_operator,
    verify_master_key,
)
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.ports.trace_storage_port import TraceFilter, TraceScope
from gateway.schemas.traces import (
    TraceCountPublic,
    TraceDetailPublic,
    TraceListPublic,
    TraceSeriesPublic,
    detail_public,
    list_public,
    series_public,
)

MAX_PAGE_SIZE = 100
_MAX_FILTER_VALUES = 50


def _filters(
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


def build_trace_router(router: APIRouter, scope_dependency: Callable[..., Any]) -> APIRouter:
    """Declare the four trace reads on ``router``, each confined to the scope ``scope_dependency`` derives."""
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

# Authentication only, like the rest of ``/organizations/me``: the scope dependency
# is what confines each read, from the caller's own memberships.
organization_router = build_trace_router(
    APIRouter(
        prefix="/organizations/me/traces", tags=["organization-traces"], dependencies=[Depends(verify_master_key)]
    ),
    get_organization_trace_scope,
)

FEATURE = CoreFeature(
    name="traces",
    surface=Surface("traces"),
    enabled=lambda config: config.trace_capture_enabled,
    routers=lambda config: (router, organization_router),
)
