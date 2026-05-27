from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import RouteTrace

router = APIRouter(prefix="/v1/route-traces", tags=["route-traces"])


class RouteTraceResponse(BaseModel):
    """Response model for route trace information."""

    trace_id: str
    timestamp: str
    api_key_id: str | None
    user_id: str | None
    project_id: str | None
    policy_id: str | None
    requested_model: str
    endpoint: str
    selected_model: str | None
    selected_provider: str | None
    strategy: str | None
    status: str
    error_message: str | None
    selected_reason: str | None
    estimated_prompt_tokens: int | None
    estimated_output_tokens: int | None
    estimated_cost: float | None
    fallback_enabled: bool
    policy_source: str | None
    tags: dict[str, Any]
    guardrails: dict[str, Any]
    context: dict[str, Any]
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]

    @classmethod
    def from_model(cls, trace: RouteTrace) -> "RouteTraceResponse":
        """Create a response from an ORM model."""
        return cls(
            trace_id=trace.trace_id,
            timestamp=trace.timestamp.isoformat(),
            api_key_id=trace.api_key_id,
            user_id=trace.user_id,
            project_id=trace.project_id,
            policy_id=trace.policy_id,
            requested_model=trace.requested_model,
            endpoint=trace.endpoint,
            selected_model=trace.selected_model,
            selected_provider=trace.selected_provider,
            strategy=trace.strategy,
            status=trace.status,
            error_message=trace.error_message,
            selected_reason=trace.selected_reason,
            estimated_prompt_tokens=trace.estimated_prompt_tokens,
            estimated_output_tokens=trace.estimated_output_tokens,
            estimated_cost=trace.estimated_cost,
            fallback_enabled=bool(trace.fallback_enabled),
            policy_source=trace.policy_source,
            tags=dict(trace.tags) if trace.tags else {},
            guardrails=dict(trace.guardrails) if trace.guardrails else {},
            context=dict(trace.context) if trace.context else {},
            candidates=list(trace.candidates) if trace.candidates else [],
            attempts=list(trace.attempts) if trace.attempts else [],
        )


class RouteTraceSummaryBucket(BaseModel):
    """Aggregated route trace metrics for one grouping key."""

    key: str
    count: int
    success_count: int
    error_count: int
    estimated_cost: float
    average_latency_ms: float | None


class RouteTraceSummaryResponse(BaseModel):
    """Aggregated route trace metrics."""

    total_count: int
    success_count: int
    error_count: int
    estimated_cost: float
    average_latency_ms: float | None
    by_model: list[RouteTraceSummaryBucket]
    by_policy: list[RouteTraceSummaryBucket]
    by_policy_source: list[RouteTraceSummaryBucket]
    by_endpoint: list[RouteTraceSummaryBucket]
    by_provider: list[RouteTraceSummaryBucket]
    by_strategy: list[RouteTraceSummaryBucket]


def _filtered_trace_stmt(
    project_id: str | None = None,
    user_id: str | None = None,
    policy_id: str | None = None,
    endpoint: str | None = None,
    status: str | None = None,
) -> Select[tuple[RouteTrace]]:
    stmt = select(RouteTrace)
    if project_id is not None:
        stmt = stmt.where(RouteTrace.project_id == project_id)
    if user_id is not None:
        stmt = stmt.where(RouteTrace.user_id == user_id)
    if policy_id is not None:
        stmt = stmt.where(RouteTrace.policy_id == policy_id)
    if endpoint is not None:
        stmt = stmt.where(RouteTrace.endpoint == endpoint)
    if status is not None:
        stmt = stmt.where(RouteTrace.status == status)
    return stmt


def _attempt_duration_ms(attempt: dict[str, Any]) -> float | None:
    duration = attempt.get("duration_ms")
    if isinstance(duration, bool):
        return None
    if isinstance(duration, int | float) and duration >= 0:
        return float(duration)
    return None


def _trace_latency_ms(trace: RouteTrace) -> float | None:
    attempts = trace.attempts if isinstance(trace.attempts, list) else []
    for attempt in attempts:
        if not isinstance(attempt, dict) or attempt.get("status") != "success":
            continue
        duration = _attempt_duration_ms(attempt)
        if duration is not None:
            return duration
    return None


def _new_bucket(key: str) -> dict[str, Any]:
    return {
        "key": key,
        "count": 0,
        "success_count": 0,
        "error_count": 0,
        "estimated_cost": 0.0,
        "latency_total_ms": 0.0,
        "latency_count": 0,
    }


def _add_trace_to_bucket(bucket: dict[str, Any], trace: RouteTrace, latency_ms: float | None) -> None:
    bucket["count"] += 1
    if trace.status == "success":
        bucket["success_count"] += 1
    elif trace.status == "error":
        bucket["error_count"] += 1
    if trace.estimated_cost:
        bucket["estimated_cost"] += trace.estimated_cost
    if latency_ms is not None:
        bucket["latency_total_ms"] += latency_ms
        bucket["latency_count"] += 1


def _bucket_response(bucket: dict[str, Any]) -> RouteTraceSummaryBucket:
    latency_count = int(bucket["latency_count"])
    average_latency_ms = None
    if latency_count:
        average_latency_ms = float(bucket["latency_total_ms"]) / latency_count
    return RouteTraceSummaryBucket(
        key=str(bucket["key"]),
        count=int(bucket["count"]),
        success_count=int(bucket["success_count"]),
        error_count=int(bucket["error_count"]),
        estimated_cost=float(bucket["estimated_cost"]),
        average_latency_ms=average_latency_ms,
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_route_traces(
    db: Annotated[AsyncSession, Depends(get_db)],
    project_id: str | None = None,
    user_id: str | None = None,
    policy_id: str | None = None,
    endpoint: str | None = None,
    status: str | None = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[RouteTraceResponse]:
    """List route traces with optional filters."""
    stmt = _filtered_trace_stmt(
        project_id=project_id,
        user_id=user_id,
        policy_id=policy_id,
        endpoint=endpoint,
        status=status,
    )
    result = await db.execute(stmt.order_by(RouteTrace.timestamp.desc()).offset(skip).limit(limit))
    traces = result.scalars().all()
    return [RouteTraceResponse.from_model(trace) for trace in traces]


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def summarize_route_traces(
    db: Annotated[AsyncSession, Depends(get_db)],
    project_id: str | None = None,
    user_id: str | None = None,
    policy_id: str | None = None,
    endpoint: str | None = None,
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> RouteTraceSummaryResponse:
    """Summarize route traces with optional filters."""
    stmt = _filtered_trace_stmt(
        project_id=project_id,
        user_id=user_id,
        policy_id=policy_id,
        endpoint=endpoint,
        status=status,
    )
    result = await db.execute(stmt.order_by(RouteTrace.timestamp.desc()).limit(limit))
    traces = list(result.scalars().all())

    model_buckets: dict[str, dict[str, Any]] = {}
    policy_buckets: dict[str, dict[str, Any]] = {}
    policy_source_buckets: dict[str, dict[str, Any]] = {}
    endpoint_buckets: dict[str, dict[str, Any]] = {}
    provider_buckets: dict[str, dict[str, Any]] = {}
    strategy_buckets: dict[str, dict[str, Any]] = {}
    total_bucket = _new_bucket("__total__")

    for trace in traces:
        latency_ms = _trace_latency_ms(trace)
        _add_trace_to_bucket(total_bucket, trace, latency_ms)

        model_key = trace.selected_model or "unknown"
        policy_key = trace.policy_id or "unknown"
        policy_source_key = trace.policy_source or "unknown"
        endpoint_key = trace.endpoint or "unknown"
        provider_key = trace.selected_provider or "unknown"
        strategy_key = trace.strategy or "unknown"
        for buckets, key in (
            (model_buckets, model_key),
            (policy_buckets, policy_key),
            (policy_source_buckets, policy_source_key),
            (endpoint_buckets, endpoint_key),
            (provider_buckets, provider_key),
            (strategy_buckets, strategy_key),
        ):
            bucket = buckets.setdefault(key, _new_bucket(key))
            _add_trace_to_bucket(bucket, trace, latency_ms)

    total = _bucket_response(total_bucket)
    return RouteTraceSummaryResponse(
        total_count=total.count,
        success_count=total.success_count,
        error_count=total.error_count,
        estimated_cost=total.estimated_cost,
        average_latency_ms=total.average_latency_ms,
        by_model=[_bucket_response(bucket) for bucket in model_buckets.values()],
        by_policy=[_bucket_response(bucket) for bucket in policy_buckets.values()],
        by_policy_source=[_bucket_response(bucket) for bucket in policy_source_buckets.values()],
        by_endpoint=[_bucket_response(bucket) for bucket in endpoint_buckets.values()],
        by_provider=[_bucket_response(bucket) for bucket in provider_buckets.values()],
        by_strategy=[_bucket_response(bucket) for bucket in strategy_buckets.values()],
    )


@router.get("/{trace_id}", dependencies=[Depends(verify_master_key)])
async def get_route_trace(
    trace_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RouteTraceResponse:
    """Get a route trace by id."""
    trace = await db.get(RouteTrace, trace_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Route trace '{trace_id}' not found",
        )
    return RouteTraceResponse.from_model(trace)
