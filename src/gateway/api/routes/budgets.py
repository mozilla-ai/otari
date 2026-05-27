from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_db, verify_master_key
from gateway.models.entities import Budget, BudgetAlert
from gateway.services.budget_alert_webhook_service import dispatch_budget_alert_webhook
from gateway.services.budget_service import (
    TAG_BUDGET_SCOPE,
    calculate_next_reset,
    normalize_alert_thresholds,
    normalize_alert_webhook_url,
)

router = APIRouter(prefix="/v1/budgets", tags=["budgets"])


class CreateBudgetRequest(BaseModel):
    """Request model for creating a new budget."""

    max_budget: float | None = Field(default=None, ge=0, description="Maximum spending limit")
    budget_duration_sec: int | None = Field(
        default=None, gt=0, description="Budget duration in seconds (e.g., 86400 for daily, 604800 for weekly)"
    )
    scope_type: str = Field(default="entity", description="Budget scope: entity or tag")
    match_tags: dict[str, str] | None = Field(
        default=None,
        description="Exact request tags matched by tag-scoped budget groups",
    )
    alert_thresholds: list[float] | None = Field(
        default=None,
        description="Optional spend ratios that emit budget alert events when crossed, e.g. [0.5, 0.8, 0.95]",
    )
    alert_webhook_url: str | None = Field(
        default=None,
        description="Optional HTTP(S) endpoint that receives budget threshold alert webhooks",
    )
    blocked: bool = False
    is_active: bool = True

    @field_validator("scope_type")
    @classmethod
    def validate_scope_type(cls, value: str) -> str:
        """Validate supported budget scopes."""
        normalized = value.strip().lower()
        if normalized not in {"entity", TAG_BUDGET_SCOPE}:
            raise ValueError("scope_type must be 'entity' or 'tag'")
        return normalized

    @field_validator("alert_thresholds")
    @classmethod
    def validate_alert_thresholds(cls, value: list[float] | None) -> list[float]:
        """Validate optional budget alert thresholds."""
        return normalize_alert_thresholds(value)

    @field_validator("alert_webhook_url")
    @classmethod
    def validate_alert_webhook_url(cls, value: str | None) -> str | None:
        """Validate optional budget alert webhook URL."""
        return normalize_alert_webhook_url(value)


class BudgetResponse(BaseModel):
    """Response model for budget information."""

    budget_id: str
    max_budget: float | None
    budget_duration_sec: int | None
    scope_type: str
    match_tags: dict[str, Any]
    alert_thresholds: list[float]
    alert_webhook_url: str | None
    spend: float
    budget_started_at: str | None
    next_budget_reset_at: str | None
    blocked: bool
    is_active: bool
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, budget: "Budget") -> "BudgetResponse":
        """Create a BudgetResponse from a Budget ORM model."""
        scope_type = getattr(budget, "scope_type", "entity")
        match_tags = getattr(budget, "match_tags", {})
        alert_thresholds = getattr(budget, "alert_thresholds", [])
        alert_webhook_url = getattr(budget, "alert_webhook_url", None)
        spend = getattr(budget, "spend", 0.0)
        budget_started_at = getattr(budget, "budget_started_at", None)
        next_budget_reset_at = getattr(budget, "next_budget_reset_at", None)
        blocked = getattr(budget, "blocked", False)
        is_active = getattr(budget, "is_active", True)
        return cls(
            budget_id=budget.budget_id,
            max_budget=budget.max_budget,
            budget_duration_sec=budget.budget_duration_sec,
            scope_type=scope_type if isinstance(scope_type, str) else "entity",
            match_tags=match_tags if isinstance(match_tags, dict) else {},
            alert_thresholds=alert_thresholds if isinstance(alert_thresholds, list) else [],
            alert_webhook_url=alert_webhook_url if isinstance(alert_webhook_url, str) else None,
            spend=float(spend) if isinstance(spend, int | float) else 0.0,
            budget_started_at=budget_started_at.isoformat() if isinstance(budget_started_at, datetime) else None,
            next_budget_reset_at=(
                next_budget_reset_at.isoformat()
                if isinstance(next_budget_reset_at, datetime)
                else None
            ),
            blocked=blocked if isinstance(blocked, bool) else False,
            is_active=is_active if isinstance(is_active, bool) else True,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )


class UpdateBudgetRequest(BaseModel):
    """Request model for updating a budget."""

    max_budget: float | None = Field(default=None, ge=0)
    budget_duration_sec: int | None = Field(default=None, gt=0)
    scope_type: str | None = None
    match_tags: dict[str, str] | None = None
    alert_thresholds: list[float] | None = None
    alert_webhook_url: str | None = None
    spend: float | None = Field(default=None, ge=0)
    blocked: bool | None = None
    is_active: bool | None = None

    @field_validator("scope_type")
    @classmethod
    def validate_scope_type(cls, value: str | None) -> str | None:
        """Validate supported budget scopes."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in {"entity", TAG_BUDGET_SCOPE}:
            raise ValueError("scope_type must be 'entity' or 'tag'")
        return normalized

    @field_validator("alert_thresholds")
    @classmethod
    def validate_alert_thresholds(cls, value: list[float] | None) -> list[float]:
        """Validate optional budget alert thresholds."""
        return normalize_alert_thresholds(value)

    @field_validator("alert_webhook_url")
    @classmethod
    def validate_alert_webhook_url(cls, value: str | None) -> str | None:
        """Validate optional budget alert webhook URL."""
        return normalize_alert_webhook_url(value)


class BudgetAlertResponse(BaseModel):
    """Response model for budget threshold alert events."""

    id: int
    budget_id: str
    scope_type: str
    scope_id: str | None
    threshold: float
    spend: float
    max_budget: float
    budget_period_start: str | None
    webhook_url: str | None
    delivery_status: str
    delivery_attempts: int
    last_delivery_status_code: int | None
    last_delivery_error: str | None
    last_delivery_attempt_at: str | None
    next_delivery_attempt_at: str | None
    delivered_at: str | None
    dead_lettered_at: str | None
    created_at: str
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, alert: BudgetAlert) -> "BudgetAlertResponse":
        """Create a BudgetAlertResponse from a BudgetAlert ORM model."""
        return cls(
            id=alert.id,
            budget_id=alert.budget_id,
            scope_type=alert.scope_type,
            scope_id=alert.scope_id,
            threshold=alert.threshold,
            spend=alert.spend,
            max_budget=alert.max_budget,
            budget_period_start=(
                alert.budget_period_start.isoformat()
                if isinstance(alert.budget_period_start, datetime)
                else None
            ),
            webhook_url=alert.webhook_url,
            delivery_status=alert.delivery_status,
            delivery_attempts=alert.delivery_attempts,
            last_delivery_status_code=alert.last_delivery_status_code,
            last_delivery_error=alert.last_delivery_error,
            last_delivery_attempt_at=(
                alert.last_delivery_attempt_at.isoformat()
                if isinstance(alert.last_delivery_attempt_at, datetime)
                else None
            ),
            next_delivery_attempt_at=(
                alert.next_delivery_attempt_at.isoformat()
                if isinstance(alert.next_delivery_attempt_at, datetime)
                else None
            ),
            delivered_at=alert.delivered_at.isoformat() if isinstance(alert.delivered_at, datetime) else None,
            dead_lettered_at=(
                alert.dead_lettered_at.isoformat()
                if isinstance(alert.dead_lettered_at, datetime)
                else None
            ),
            created_at=alert.created_at.isoformat(),
            metadata=alert.metadata_ if isinstance(alert.metadata_, dict) else {},
        )


def _validate_tag_budget_shape(scope_type: str, match_tags: dict[str, str] | None) -> None:
    if scope_type == TAG_BUDGET_SCOPE and not match_tags:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Tag-scoped budgets require match_tags",
        )


def _budget_window_start(duration_sec: int | None) -> tuple[datetime, datetime | None]:
    now = datetime.now(UTC)
    next_reset_at = calculate_next_reset(now, duration_sec) if duration_sec else None
    return now, next_reset_at


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_budget(
    request: CreateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Create a new budget."""
    _validate_tag_budget_shape(request.scope_type, request.match_tags)
    budget_started_at = None
    next_budget_reset_at = None
    if request.scope_type == TAG_BUDGET_SCOPE:
        budget_started_at, next_budget_reset_at = _budget_window_start(request.budget_duration_sec)

    budget = Budget(
        max_budget=request.max_budget,
        budget_duration_sec=request.budget_duration_sec,
        scope_type=request.scope_type,
        match_tags=dict(request.match_tags or {}),
        alert_thresholds=normalize_alert_thresholds(request.alert_thresholds),
        alert_webhook_url=normalize_alert_webhook_url(request.alert_webhook_url),
        spend=0.0,
        budget_started_at=budget_started_at,
        next_budget_reset_at=next_budget_reset_at,
        blocked=request.blocked,
        is_active=request.is_active,
    )

    db.add(budget)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)

    return BudgetResponse.from_model(budget)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_budgets(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetResponse]:
    """List all budgets with pagination."""
    result = await db.execute(select(Budget).offset(skip).limit(limit))
    budgets = result.scalars().all()

    return [BudgetResponse.from_model(budget) for budget in budgets]


@router.get("/alerts", dependencies=[Depends(verify_master_key)])
async def list_budget_alerts(
    db: Annotated[AsyncSession, Depends(get_db)],
    budget_id: Annotated[str | None, Query()] = None,
    scope_type: Annotated[str | None, Query()] = None,
    scope_id: Annotated[str | None, Query()] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetAlertResponse]:
    """List budget alert events."""
    stmt = select(BudgetAlert)
    if budget_id is not None:
        stmt = stmt.where(BudgetAlert.budget_id == budget_id)
    if scope_type is not None:
        stmt = stmt.where(BudgetAlert.scope_type == scope_type)
    if scope_id is not None:
        stmt = stmt.where(BudgetAlert.scope_id == scope_id)
    stmt = stmt.order_by(BudgetAlert.created_at.desc(), BudgetAlert.id.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    return [BudgetAlertResponse.from_model(alert) for alert in result.scalars().all()]


@router.post("/alerts/{alert_id}/deliver", dependencies=[Depends(verify_master_key)])
async def deliver_budget_alert(
    alert_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetAlertResponse:
    """Retry webhook delivery for a budget alert event."""
    result = await db.execute(select(BudgetAlert.id).where(BudgetAlert.id == alert_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget alert with id '{alert_id}' not found",
        )
    await dispatch_budget_alert_webhook(alert_id)
    refreshed = await db.get(BudgetAlert, alert_id)
    if refreshed is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget alert with id '{alert_id}' not found",
        )
    return BudgetAlertResponse.from_model(refreshed)


@router.get("/{budget_id}/alerts", dependencies=[Depends(verify_master_key)])
async def list_alerts_for_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetAlertResponse]:
    """List alert events for a specific budget."""
    result = await db.execute(select(Budget.budget_id).where(Budget.budget_id == budget_id))
    if result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )
    stmt = (
        select(BudgetAlert)
        .where(BudgetAlert.budget_id == budget_id)
        .order_by(BudgetAlert.created_at.desc(), BudgetAlert.id.desc())
        .offset(skip)
        .limit(limit)
    )
    alerts = await db.execute(stmt)
    return [BudgetAlertResponse.from_model(alert) for alert in alerts.scalars().all()]


@router.get("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def get_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Get details of a specific budget."""
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    return BudgetResponse.from_model(budget)


@router.patch("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def update_budget(
    budget_id: str,
    request: UpdateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Update a budget."""
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    if request.max_budget is not None:
        budget.max_budget = request.max_budget
    if request.budget_duration_sec is not None:
        budget.budget_duration_sec = request.budget_duration_sec
    if request.scope_type is not None:
        budget.scope_type = request.scope_type
    if request.match_tags is not None:
        budget.match_tags = dict(request.match_tags)
    if "alert_thresholds" in request.model_fields_set:
        budget.alert_thresholds = normalize_alert_thresholds(request.alert_thresholds)
    if "alert_webhook_url" in request.model_fields_set:
        budget.alert_webhook_url = normalize_alert_webhook_url(request.alert_webhook_url)
    _validate_tag_budget_shape(budget.scope_type, budget.match_tags)
    if request.spend is not None:
        budget.spend = request.spend
    if request.blocked is not None:
        budget.blocked = request.blocked
    if request.is_active is not None:
        budget.is_active = request.is_active
    if budget.scope_type == TAG_BUDGET_SCOPE and budget.budget_started_at is None:
        budget.budget_started_at, budget.next_budget_reset_at = _budget_window_start(budget.budget_duration_sec)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)

    return BudgetResponse.from_model(budget)


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a budget."""
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    budget = result.scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    await db.delete(budget)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
