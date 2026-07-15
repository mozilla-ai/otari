"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first, plus a database-side
aggregate for callers that need totals rather than rows. Intended for
external systems that need to sync usage data (billing, analytics).
"""

from datetime import UTC, datetime
from typing import Annotated, Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import Select, case, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_master_key
from gateway.core.config import GatewayConfig
from gateway.models.entities import UsageLog
from gateway.services.budget_service import ReservationHandle, reconcile_reservation
from gateway.services.pricing_service import find_model_pricing

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
        )


_SelectT = TypeVar("_SelectT", bound=Select[Any])


def _filtered_usage(
    stmt: _SelectT,
    start_date: datetime | None,
    end_date: datetime | None,
    user_id: str | None,
) -> _SelectT:
    """Apply the shared time-range / user filters to a usage query.

    Generic over the statement type so a filtered row query and a filtered
    aggregate query each keep their own result typing.
    """
    if start_date is not None:
        stmt = stmt.where(UsageLog.timestamp >= start_date)
    if end_date is not None:
        stmt = stmt.where(UsageLog.timestamp < end_date)
    if user_id is not None:
        stmt = stmt.where(UsageLog.user_id == user_id)
    return stmt


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
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range and user. Paginated via skip/limit.
    Timestamps accept either ISO 8601 strings or Unix epoch seconds (numeric).

    This is a page of raw rows, capped at ``limit``. For counts and totals over
    every matching row, use ``/v1/usage/summary`` instead of summing a page.
    """

    stmt = _filtered_usage(select(UsageLog), start_date, end_date, user_id)
    stmt = stmt.order_by(UsageLog.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [UsageEntry.from_model(log) for log in logs]


class UsageTotals(BaseModel):
    """Aggregate counters over every usage row matching the filters."""

    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    errors: int = Field(description="Requests whose status is not 'success'.")


class ModelUsage(BaseModel):
    """Aggregate counters for one (provider, model) pair."""

    key: str = Field(description="'provider:model', or the bare model when no provider was recorded.")
    model: str
    provider: str | None
    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float


class UsageSummary(BaseModel):
    """Usage totals plus a per-model breakdown."""

    totals: UsageTotals
    by_model: list[ModelUsage]


@router.get("/summary", dependencies=[Depends(verify_master_key)])
async def usage_summary(
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(
        default=None,
        description="Aggregate logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)",
    ),
    end_date: datetime | None = Query(
        default=None,
        description="Aggregate logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)",
    ),
    user_id: str | None = Query(default=None, description="Filter to a single user"),
) -> UsageSummary:
    """Aggregate usage over every matching row, plus a per-model breakdown.

    Counted in the database, so the result covers the whole table rather than a
    page of it: summing ``/v1/usage`` instead silently under-reports once the
    row count exceeds the page limit.
    """

    # SUM over zero matching rows is NULL, not 0, and token/cost columns are
    # themselves nullable, so every sum is coalesced before it reaches Pydantic.
    errors = func.sum(case((UsageLog.status != "success", 1), else_=0))
    totals_stmt = _filtered_usage(
        select(
            func.count(UsageLog.id),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(UsageLog.completion_tokens), 0),
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            func.coalesce(func.sum(UsageLog.cost), 0.0),
            func.coalesce(errors, 0),
        ),
        start_date,
        end_date,
        user_id,
    )
    requests, prompt, completion, total, cost, error_count = (await db.execute(totals_stmt)).one()
    totals = UsageTotals(
        requests=int(requests),
        prompt_tokens=int(prompt),
        completion_tokens=int(completion),
        total_tokens=int(total),
        cost=float(cost),
        errors=int(error_count),
    )

    by_model_stmt = _filtered_usage(
        select(
            UsageLog.provider,
            UsageLog.model,
            func.count(UsageLog.id),
            func.coalesce(func.sum(UsageLog.prompt_tokens), 0),
            func.coalesce(func.sum(UsageLog.completion_tokens), 0),
            func.coalesce(func.sum(UsageLog.total_tokens), 0),
            func.coalesce(func.sum(UsageLog.cost), 0.0),
        ),
        start_date,
        end_date,
        user_id,
    )
    # Group by provider as well as model so the same model name served by two
    # providers does not collapse into one misattributed bucket. The model name
    # breaks ties so the ordering is stable across calls.
    by_model_stmt = by_model_stmt.group_by(UsageLog.provider, UsageLog.model).order_by(
        func.count(UsageLog.id).desc(), UsageLog.model
    )
    by_model = [
        ModelUsage(
            key=f"{row_provider}:{row_model}" if row_provider else row_model,
            model=row_model,
            provider=row_provider,
            requests=int(row_requests),
            prompt_tokens=int(row_prompt),
            completion_tokens=int(row_completion),
            total_tokens=int(row_total),
            cost=float(row_cost),
        )
        for row_provider, row_model, row_requests, row_prompt, row_completion, row_total, row_cost in (
            await db.execute(by_model_stmt)
        ).all()
    ]

    return UsageSummary(totals=totals, by_model=by_model)


class BackfillCostRequest(BaseModel):
    """Request to backfill cost on previously-unpriced usage rows."""

    model_key: str = Field(
        description="Model to backfill, as 'provider:model' (or a bare model name when no provider was recorded)."
    )


class BackfillCostResponse(BaseModel):
    """Summary of a usage-cost backfill."""

    model_key: str
    rows_updated: int
    cost_added: float
    users_updated: int


@router.post("/backfill", dependencies=[Depends(verify_master_key)])
async def backfill_usage_cost(
    request: BackfillCostRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> BackfillCostResponse:
    """Recompute cost for a model's usage rows that were logged without one.

    For usage rows of ``model_key`` whose ``cost`` is null (they ran while the
    model was unpriced), compute cost from the recorded token counts and the
    model's current price, then add each user's backfilled total to their spend.
    Rows that already have a cost are left untouched, so this is safe to re-run.
    Requires master key authentication.
    """
    raw_provider, separator, raw_model = request.model_key.partition(":")
    provider: str | None = raw_provider if separator else None
    model = raw_model if separator else request.model_key

    pricing = await find_model_pricing(db, provider, model, as_of=datetime.now(UTC))
    if pricing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pricing configured for model '{request.model_key}'. Set a price before backfilling.",
        )

    conditions = [
        UsageLog.model == model,
        UsageLog.cost.is_(None),
        UsageLog.prompt_tokens.is_not(None),
    ]
    conditions.append(UsageLog.provider.is_(None) if provider is None else UsageLog.provider == provider)

    rows = (await db.execute(select(UsageLog).where(*conditions))).scalars().all()

    per_user: dict[str, float] = {}
    total_cost = 0.0
    for row in rows:
        prompt_tokens = row.prompt_tokens or 0
        completion_tokens = row.completion_tokens or 0
        cost = (prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
            completion_tokens / 1_000_000
        ) * pricing.output_price_per_million
        row.cost = cost
        total_cost += cost
        if row.user_id:
            per_user[row.user_id] = per_user.get(row.user_id, 0.0) + cost

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None

    # Fold each user's backfilled cost into their spend via the same path the
    # live request uses (no held reservation to release here).
    for user_id, user_cost in per_user.items():
        await reconcile_reservation(
            db,
            ReservationHandle(user_id=user_id, estimate=0.0, reserved=False, strategy=config.budget_strategy),
            user_cost,
        )

    return BackfillCostResponse(
        model_key=request.model_key,
        rows_updated=len(rows),
        cost_added=total_cost,
        users_updated=len(per_user),
    )
