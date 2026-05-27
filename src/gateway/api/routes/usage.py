"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import _as_utc, get_db, verify_master_key
from gateway.models.entities import UsageLog

router = APIRouter(prefix="/v1/usage", tags=["usage"])


def _format_timestamp(value: datetime) -> str:
    return (_as_utc(value) or value).isoformat()


class UsageEntry(BaseModel):
    """A single usage log entry."""

    id: str
    user_id: str | None
    api_key_id: str | None
    project_id: str | None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None
    tags: dict[str, Any]

    @classmethod
    def from_model(cls, log: UsageLog) -> "UsageEntry":
        tags = log.tags if isinstance(log.tags, dict) else {}
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
            project_id=log.project_id,
            timestamp=_format_timestamp(log.timestamp),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
            tags=tags,
        )


class UsageSummaryBucket(BaseModel):
    """Aggregated usage metrics for one grouping key."""

    key: str
    count: int
    success_count: int
    error_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class UsageSummaryResponse(BaseModel):
    """Aggregated usage metrics."""

    total_count: int
    success_count: int
    error_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    by_project: list[UsageSummaryBucket]
    by_user: list[UsageSummaryBucket]
    by_model: list[UsageSummaryBucket]
    by_provider: list[UsageSummaryBucket]
    by_endpoint: list[UsageSummaryBucket]
    by_status: list[UsageSummaryBucket]
    by_tag: list[UsageSummaryBucket]


def _filtered_usage_stmt(
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    user_id: str | None = None,
    project_id: str | None = None,
) -> Select[tuple[UsageLog]]:
    stmt = select(UsageLog)
    if start_date is not None:
        stmt = stmt.where(UsageLog.timestamp >= start_date)
    if end_date is not None:
        stmt = stmt.where(UsageLog.timestamp < end_date)
    if user_id is not None:
        stmt = stmt.where(UsageLog.user_id == user_id)
    if project_id is not None:
        stmt = stmt.where(UsageLog.project_id == project_id)
    return stmt


def _matches_tag_filter(log: UsageLog, tag_key: str | None, tag_value: str | None) -> bool:
    if tag_key is None:
        return True
    tags = log.tags if isinstance(log.tags, dict) else {}
    if tag_key not in tags:
        return False
    if tag_value is None:
        return True
    return str(tags[tag_key]) == tag_value


def _new_bucket(key: str) -> dict[str, Any]:
    return {
        "key": key,
        "count": 0,
        "success_count": 0,
        "error_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
    }


def _add_log_to_bucket(bucket: dict[str, Any], log: UsageLog) -> None:
    bucket["count"] += 1
    if log.status == "success":
        bucket["success_count"] += 1
    elif log.status == "error":
        bucket["error_count"] += 1
    bucket["prompt_tokens"] += log.prompt_tokens or 0
    bucket["completion_tokens"] += log.completion_tokens or 0
    bucket["total_tokens"] += log.total_tokens or 0
    bucket["cost"] += log.cost or 0.0


def _bucket_response(bucket: dict[str, Any]) -> UsageSummaryBucket:
    return UsageSummaryBucket(
        key=str(bucket["key"]),
        count=int(bucket["count"]),
        success_count=int(bucket["success_count"]),
        error_count=int(bucket["error_count"]),
        prompt_tokens=int(bucket["prompt_tokens"]),
        completion_tokens=int(bucket["completion_tokens"]),
        total_tokens=int(bucket["total_tokens"]),
        cost=float(bucket["cost"]),
    )


def _bucket_responses(buckets: dict[str, dict[str, Any]]) -> list[UsageSummaryBucket]:
    responses = [_bucket_response(bucket) for bucket in buckets.values()]
    return sorted(responses, key=lambda bucket: (-bucket.cost, -bucket.count, bucket.key))


def _tag_bucket_key(tag_key: str, tag_value: Any) -> str:
    return f"{tag_key}={tag_value}"


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)",
    ),
    user_id: str | None = Query(default=None, description="Filter to a single user"),
    project_id: str | None = Query(default=None, description="Filter to a gateway project"),
    tag_key: str | None = Query(default=None, description="Filter to logs containing this usage tag key"),
    tag_value: str | None = Query(
        default=None,
        description="When tag_key is set, filter to logs whose tag value string matches this value",
    ),
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range, user, project, and usage tags.
    Paginated via skip/limit. Timestamps accept either ISO 8601 strings or
    Unix epoch seconds (numeric).
    """

    stmt = _filtered_usage_stmt(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        project_id=project_id,
    ).order_by(UsageLog.timestamp.desc())
    if tag_key is None:
        stmt = stmt.offset(skip).limit(limit)

    result = await db.execute(stmt)
    logs = result.scalars().all()
    if tag_key is not None:
        matching_logs = [log for log in logs if _matches_tag_filter(log, tag_key, tag_value)]
        logs = matching_logs[skip : skip + limit]
    return [UsageEntry.from_model(log) for log in logs]


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def summarize_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)",
    ),
    user_id: str | None = Query(default=None, description="Filter to a single user"),
    project_id: str | None = Query(default=None, description="Filter to a gateway project"),
    tag_key: str | None = Query(default=None, description="Filter to logs containing this usage tag key"),
    tag_value: str | None = Query(
        default=None,
        description="When tag_key is set, filter to logs whose tag value string matches this value",
    ),
    limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
) -> UsageSummaryResponse:
    """Summarize usage logs with optional project and tag filters."""

    stmt = _filtered_usage_stmt(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        project_id=project_id,
    ).order_by(UsageLog.timestamp.desc())
    result = await db.execute(stmt.limit(limit))
    logs = [log for log in result.scalars().all() if _matches_tag_filter(log, tag_key, tag_value)]

    total_bucket = _new_bucket("__total__")
    project_buckets: dict[str, dict[str, Any]] = {}
    user_buckets: dict[str, dict[str, Any]] = {}
    model_buckets: dict[str, dict[str, Any]] = {}
    provider_buckets: dict[str, dict[str, Any]] = {}
    endpoint_buckets: dict[str, dict[str, Any]] = {}
    status_buckets: dict[str, dict[str, Any]] = {}
    tag_buckets: dict[str, dict[str, Any]] = {}

    for log in logs:
        _add_log_to_bucket(total_bucket, log)
        for buckets, key in (
            (project_buckets, log.project_id or "unknown"),
            (user_buckets, log.user_id or "unknown"),
            (model_buckets, log.model or "unknown"),
            (provider_buckets, log.provider or "unknown"),
            (endpoint_buckets, log.endpoint or "unknown"),
            (status_buckets, log.status or "unknown"),
        ):
            bucket = buckets.setdefault(key, _new_bucket(key))
            _add_log_to_bucket(bucket, log)

        tags = log.tags if isinstance(log.tags, dict) else {}
        for key, value in tags.items():
            tag_bucket_key = _tag_bucket_key(str(key), value)
            bucket = tag_buckets.setdefault(tag_bucket_key, _new_bucket(tag_bucket_key))
            _add_log_to_bucket(bucket, log)

    total = _bucket_response(total_bucket)
    return UsageSummaryResponse(
        total_count=total.count,
        success_count=total.success_count,
        error_count=total.error_count,
        prompt_tokens=total.prompt_tokens,
        completion_tokens=total.completion_tokens,
        total_tokens=total.total_tokens,
        cost=total.cost,
        by_project=_bucket_responses(project_buckets),
        by_user=_bucket_responses(user_buckets),
        by_model=_bucket_responses(model_buckets),
        by_provider=_bucket_responses(provider_buckets),
        by_endpoint=_bucket_responses(endpoint_buckets),
        by_status=_bucket_responses(status_buckets),
        by_tag=_bucket_responses(tag_buckets),
    )
