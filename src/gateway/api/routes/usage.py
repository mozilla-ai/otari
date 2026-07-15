"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
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
    """

    stmt = select(UsageLog)
    if start_date is not None:
        stmt = stmt.where(UsageLog.timestamp >= start_date)
    if end_date is not None:
        stmt = stmt.where(UsageLog.timestamp < end_date)
    if user_id is not None:
        stmt = stmt.where(UsageLog.user_id == user_id)

    stmt = stmt.order_by(UsageLog.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    logs = result.scalars().all()
    return [UsageEntry.from_model(log) for log in logs]


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
