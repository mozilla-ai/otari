"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import UsageLog

router = APIRouter(prefix="/v1/usage", tags=["usage"])


class UsageEntry(BaseModel):
    """A single usage log entry."""

    id: str
    user_id: str | None
    api_key_id: str | None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None
    latency_ms: int | None

    @classmethod
    def from_model(cls, log: UsageLog) -> "UsageEntry":
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cache_read_tokens=log.cache_read_tokens,
            cache_write_tokens=log.cache_write_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
            latency_ms=log.latency_ms,
        )


class UsageCount(BaseModel):
    """Total number of usage logs matching a set of filters."""

    total: int


# Shared query descriptions so the list and count endpoints stay in lockstep.
_START_DESC = "Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)"
_END_DESC = "Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)"
_USER_DESC = "Filter to a single user"
_STATUS_DESC = "Filter to a single status (e.g. 'success' or 'error')"
_MODEL_DESC = "Filter to a single model"
_ENDPOINT_DESC = "Filter to a single endpoint (e.g. '/v1/chat/completions')"


def _usage_filters(
    *,
    start_date: datetime | None,
    end_date: datetime | None,
    user_id: str | None,
    status: str | None,
    model: str | None,
    endpoint: str | None,
) -> list[ColumnElement[bool]]:
    """Build the shared WHERE conditions for the list and count endpoints.

    Keeping this in one place guarantees the paginator's total (``/count``)
    always matches the rows ``list_usage`` returns for the same filters.
    """
    conditions: list[ColumnElement[bool]] = []
    if start_date is not None:
        conditions.append(UsageLog.timestamp >= start_date)
    if end_date is not None:
        conditions.append(UsageLog.timestamp < end_date)
    if user_id is not None:
        conditions.append(UsageLog.user_id == user_id)
    if status is not None:
        conditions.append(UsageLog.status == status)
    if model is not None:
        conditions.append(UsageLog.model == model)
    if endpoint is not None:
        conditions.append(UsageLog.endpoint == endpoint)
    return conditions


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range, user, status, model, and endpoint.
    Paginated via skip/limit. The return shape is a bare JSON array; external
    billing/analytics consumers depend on this, so the total row count for a
    paginated UI is served separately by ``GET /v1/usage/count`` rather than
    wrapped in an envelope here. Timestamps accept either ISO 8601 strings or
    Unix epoch seconds (numeric).
    """
    conditions = _usage_filters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        model=model,
        endpoint=endpoint,
    )
    stmt = (
        select(UsageLog)
        .where(*conditions)
        .order_by(UsageLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [UsageEntry.from_model(log) for log in logs]


@router.get("/count", dependencies=[Depends(verify_master_key)])
async def count_usage(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: str | None = Query(default=None, description=_USER_DESC),
    status: str | None = Query(default=None, description=_STATUS_DESC),
    model: str | None = Query(default=None, description=_MODEL_DESC),
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
) -> UsageCount:
    """Total number of usage logs matching the given filters.

    Serves the dashboard paginator's "N of M" total without changing the bare
    array contract of ``GET /v1/usage``. Runs only when the client asks (a
    separate request), so the ``COUNT(*)`` is not paid on every page load.
    """
    conditions = _usage_filters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        model=model,
        endpoint=endpoint,
    )
    stmt: Any = select(func.count()).select_from(UsageLog).where(*conditions)
    total = (await db.execute(stmt)).scalar_one()
    return UsageCount(total=total)
