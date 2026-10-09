"""Agent traces as a core feature: the session list, one session with its turns, the per-hour counts, and retention.

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
    TraceServiceDep,
    build_trace_service,
    get_deployment_trace_scope,
    get_organization_trace_scope,
    require_deployment_operator,
    verify_master_key,
)
from gateway.core.config import GatewayConfig
from gateway.core.feature import CoreFeature
from gateway.core.surface import Surface
from gateway.schemas.traces import (
    TraceCountPublic,
    TraceDetailPublic,
    TraceListPublic,
    TraceSeriesPublic,
    detail_public,
    list_public,
    series_public,
)
from gateway.services.traces import run_trace_retention
from gateway.types.traces import TraceFilter, TraceScope

MAX_PAGE_SIZE = 100
# Traces expire on the scale of days, so an hourly pass is soon enough.
_RETENTION_INTERVAL_S = 3600.0
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

# Authentication only, like the rest of ``/organizations/me``: the scope dependency
# is what confines each read, from the caller's own memberships.
organization_router = build_trace_router(
    APIRouter(
        prefix="/organizations/me/traces", tags=["organization-traces"], dependencies=[Depends(verify_master_key)]
    ),
    get_organization_trace_scope,
)


async def _retention(config: GatewayConfig) -> None:
    await run_trace_retention(
        build_trace_service,
        retention_days=config.trace_retention_days,
        max_age_days=config.trace_session_max_age_days,
        interval=_RETENTION_INTERVAL_S,
    )


FEATURE = CoreFeature(
    name="traces",
    surface=Surface("traces"),
    enabled=lambda config: config.trace_capture_enabled,
    routers=lambda config: (router, organization_router),
    worker=_retention,
)
