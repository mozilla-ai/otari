from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from fastapi import HTTPException, status
from pydantic import AnyHttpUrl, TypeAdapter
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.metrics import record_budget_exceeded
from gateway.models.entities import Budget, BudgetAlert, BudgetResetLog, Project, User
from gateway.repositories.users_repository import get_active_user
from gateway.services.pricing_service import find_model_pricing

TAG_BUDGET_SCOPE = "tag"
BUDGET_ALERT_SCOPE_PROJECT = "project"
BUDGET_ALERT_SCOPE_USER = "user"
_HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)


def calculate_next_reset(start: datetime, duration_sec: int) -> datetime:
    """Calculate next budget reset datetime.

    Args:
        start: Starting datetime for the budget period
        duration_sec: Duration in seconds

    Returns:
        datetime when the budget should next reset

    """
    return start + timedelta(seconds=duration_sec)


async def reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> None:
    """Reset user's budget spend and schedule next reset."""

    previous_spend = float(user.spend)
    user_id_str = user.user_id

    user.spend = 0.0
    user.budget_started_at = now

    if budget.budget_duration_sec:
        user.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)
    else:
        user.next_budget_reset_at = None

    reset_log = BudgetResetLog(
        user_id=user.user_id,
        budget_id=budget.budget_id,
        previous_spend=previous_spend,
        reset_at=now,
        next_reset_at=user.next_budget_reset_at,
    )
    db.add(reset_log)

    try:
        await db.commit()
        await db.refresh(user)
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for user '%s': %s", user_id_str, e)
        raise


async def _cas_reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> User:
    next_reset_at = calculate_next_reset(now, budget.budget_duration_sec) if budget.budget_duration_sec else None

    result = await db.execute(
        update(User)
        .where(
            User.user_id == user.user_id,
            User.deleted_at.is_(None),
            User.next_budget_reset_at.is_not(None),
            User.next_budget_reset_at <= now,
        )
        .values(
            spend=0.0,
            budget_started_at=now,
            next_budget_reset_at=next_reset_at,
        )
        .execution_options(synchronize_session=False)
    )

    rowcount = getattr(result, "rowcount", 0)
    if rowcount and rowcount > 0:
        reset_log = BudgetResetLog(
            user_id=user.user_id,
            budget_id=budget.budget_id,
            previous_spend=float(user.spend),
            reset_at=now,
            next_reset_at=next_reset_at,
        )
        db.add(reset_log)
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error("Failed to commit CAS budget reset for user '%s': %s", user.user_id, e)
            raise
        refreshed = await get_active_user(db, user.user_id)
        return refreshed or user

    await db.rollback()
    return user


async def reset_project_budget(db: AsyncSession, project: Project, budget: Budget, now: datetime) -> None:
    """Reset project's budget spend and schedule next reset."""

    previous_spend = float(project.spend)
    project_id_str = project.project_id

    project.spend = 0.0
    project.budget_started_at = now

    if budget.budget_duration_sec:
        project.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)
    else:
        project.next_budget_reset_at = None

    reset_log = BudgetResetLog(
        project_id=project.project_id,
        budget_id=budget.budget_id,
        previous_spend=previous_spend,
        reset_at=now,
        next_reset_at=project.next_budget_reset_at,
    )
    db.add(reset_log)

    try:
        await db.commit()
        await db.refresh(project)
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for project '%s': %s", project_id_str, e)
        raise


async def _cas_reset_project_budget(db: AsyncSession, project: Project, budget: Budget, now: datetime) -> Project:
    next_reset_at = calculate_next_reset(now, budget.budget_duration_sec) if budget.budget_duration_sec else None

    result = await db.execute(
        update(Project)
        .where(
            Project.project_id == project.project_id,
            Project.next_budget_reset_at.is_not(None),
            Project.next_budget_reset_at <= now,
        )
        .values(
            spend=0.0,
            budget_started_at=now,
            next_budget_reset_at=next_reset_at,
        )
        .execution_options(synchronize_session=False)
    )

    rowcount = getattr(result, "rowcount", 0)
    if rowcount and rowcount > 0:
        reset_log = BudgetResetLog(
            project_id=project.project_id,
            budget_id=budget.budget_id,
            previous_spend=float(project.spend),
            reset_at=now,
            next_reset_at=next_reset_at,
        )
        db.add(reset_log)
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error("Failed to commit CAS budget reset for project '%s': %s", project.project_id, e)
            raise
        refreshed = await db.get(Project, project.project_id)
        return refreshed or project

    await db.rollback()
    return project


async def _get_budget(db: AsyncSession, budget_id: str) -> Budget | None:
    result = await db.execute(select(Budget).where(Budget.budget_id == budget_id))
    return result.scalar_one_or_none()


def _normalize_budget_strategy(strategy: str) -> str:
    normalized_strategy = strategy or "for_update"
    normalized_strategy = normalized_strategy.strip().lower()
    if normalized_strategy not in {"for_update", "cas", "disabled"}:
        return "for_update"
    return normalized_strategy


def normalize_alert_thresholds(value: Any) -> list[float]:
    """Normalize budget alert thresholds as sorted spend ratios."""
    if value is None:
        return []
    if not isinstance(value, list | tuple):
        raise ValueError("alert_thresholds must be a list of ratios between 0 and 1")

    thresholds: set[float] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ValueError("alert_thresholds must contain only numeric ratios")
        threshold = float(item)
        if threshold <= 0 or threshold > 1:
            raise ValueError("alert_thresholds entries must be greater than 0 and less than or equal to 1")
        thresholds.add(threshold)
    return sorted(thresholds)


def normalize_alert_webhook_url(value: str | None) -> str | None:
    """Normalize and validate an optional budget alert webhook URL."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return str(_HTTP_URL_ADAPTER.validate_python(stripped))
    except ValueError as exc:
        raise ValueError("alert_webhook_url must be a valid http or https URL") from exc


def _tag_scope_id(budget: Budget) -> str | None:
    match_tags = budget.match_tags if isinstance(budget.match_tags, dict) else {}
    if not match_tags:
        return None
    return ",".join(f"{key}={match_tags[key]}" for key in sorted(match_tags))


async def record_budget_alerts(
    db: AsyncSession,
    *,
    budget: Budget,
    scope_type: str,
    scope_id: str | None,
    spend: float,
    period_start: datetime | None,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Record newly crossed alert thresholds for a budget scope."""
    max_budget = budget.max_budget
    if max_budget is None or max_budget <= 0 or spend <= 0:
        return []

    thresholds = normalize_alert_thresholds(budget.alert_thresholds)
    if not thresholds:
        return []

    created: list[BudgetAlert] = []
    for threshold in thresholds:
        if spend < max_budget * threshold:
            continue

        stmt = select(BudgetAlert).where(
            BudgetAlert.budget_id == budget.budget_id,
            BudgetAlert.scope_type == scope_type,
            BudgetAlert.scope_id == scope_id,
            BudgetAlert.threshold == threshold,
        )
        if period_start is None:
            stmt = stmt.where(BudgetAlert.budget_period_start.is_(None))
        else:
            stmt = stmt.where(BudgetAlert.budget_period_start == period_start)

        existing = await db.execute(stmt.limit(1))
        if existing.scalar_one_or_none() is not None:
            continue

        alert = BudgetAlert(
            budget_id=budget.budget_id,
            scope_type=scope_type,
            scope_id=scope_id,
            threshold=threshold,
            spend=spend,
            max_budget=max_budget,
            budget_period_start=period_start,
            webhook_url=budget.alert_webhook_url,
            delivery_status="pending" if budget.alert_webhook_url else "not_configured",
            metadata_=metadata or {},
        )
        db.add(alert)
        created.append(alert)
        logger.warning(
            "Budget alert %.0f%% crossed for %s '%s' on budget '%s' (spend %.6f / %.6f)",
            threshold * 100,
            scope_type,
            scope_id,
            budget.budget_id,
            spend,
            max_budget,
        )

    return created


async def record_user_budget_alerts_after_spend(
    db: AsyncSession,
    *,
    user_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Record alert events for a user's budget after usage spend increments."""
    user = await get_active_user(db, user_id)
    if user is None or not user.budget_id:
        return []
    budget = await _get_budget(db, user.budget_id)
    if budget is None:
        return []
    return await record_budget_alerts(
        db,
        budget=budget,
        scope_type=BUDGET_ALERT_SCOPE_USER,
        scope_id=user.user_id,
        spend=float(user.spend),
        period_start=user.budget_started_at,
        metadata=metadata,
    )


async def record_project_budget_alerts_after_spend(
    db: AsyncSession,
    *,
    project_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Record alert events for a project's budget after usage spend increments."""
    result = await db.execute(select(Project).where(Project.project_id == project_id))
    project = result.scalar_one_or_none()
    if project is None or not project.budget_id:
        return []
    budget = await _get_budget(db, project.budget_id)
    if budget is None:
        return []
    return await record_budget_alerts(
        db,
        budget=budget,
        scope_type=BUDGET_ALERT_SCOPE_PROJECT,
        scope_id=project.project_id,
        spend=float(project.spend),
        period_start=project.budget_started_at,
        metadata=metadata,
    )


def budget_matches_tags(budget: Budget, tags: dict[str, Any] | None) -> bool:
    """Return whether a tag-scoped budget applies to a request's tags."""
    if budget.scope_type != TAG_BUDGET_SCOPE or not budget.is_active:
        return False
    match_tags = budget.match_tags if isinstance(budget.match_tags, dict) else {}
    if not match_tags:
        return False
    request_tags = tags if isinstance(tags, dict) else {}
    return all(str(request_tags.get(key)) == str(value) for key, value in match_tags.items())


async def _matching_tag_budgets(
    db: AsyncSession,
    tags: dict[str, Any] | None,
    *,
    for_update: bool = False,
) -> list[Budget]:
    request_tags = tags if isinstance(tags, dict) else {}
    if not request_tags:
        return []
    stmt = select(Budget).where(Budget.scope_type == TAG_BUDGET_SCOPE, Budget.is_active.is_(True))
    if for_update:
        stmt = stmt.with_for_update()
    result = await db.execute(stmt)
    return [budget for budget in result.scalars().all() if budget_matches_tags(budget, request_tags)]


async def reset_tag_budget(db: AsyncSession, budget: Budget, now: datetime) -> None:
    """Reset a tag-scoped budget group's spend and schedule next reset."""
    previous_spend = float(budget.spend)
    budget.spend = 0.0
    budget.budget_started_at = now
    budget.next_budget_reset_at = (
        calculate_next_reset(now, budget.budget_duration_sec)
        if budget.budget_duration_sec
        else None
    )
    db.add(
        BudgetResetLog(
            budget_id=budget.budget_id,
            previous_spend=previous_spend,
            reset_at=now,
            next_reset_at=budget.next_budget_reset_at,
        )
    )
    try:
        await db.commit()
        await db.refresh(budget)
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for tag budget '%s': %s", budget.budget_id, e)
        raise


async def _cas_reset_tag_budget(db: AsyncSession, budget: Budget, now: datetime) -> Budget:
    next_reset_at = calculate_next_reset(now, budget.budget_duration_sec) if budget.budget_duration_sec else None
    result = await db.execute(
        update(Budget)
        .where(
            Budget.budget_id == budget.budget_id,
            Budget.scope_type == TAG_BUDGET_SCOPE,
            Budget.next_budget_reset_at.is_not(None),
            Budget.next_budget_reset_at <= now,
        )
        .values(
            spend=0.0,
            budget_started_at=now,
            next_budget_reset_at=next_reset_at,
        )
        .execution_options(synchronize_session=False)
    )
    rowcount = getattr(result, "rowcount", 0)
    if rowcount and rowcount > 0:
        db.add(
            BudgetResetLog(
                budget_id=budget.budget_id,
                previous_spend=float(budget.spend),
                reset_at=now,
                next_reset_at=next_reset_at,
            )
        )
        try:
            await db.commit()
        except SQLAlchemyError as e:
            await db.rollback()
            logger.error("Failed to commit CAS budget reset for tag budget '%s': %s", budget.budget_id, e)
            raise
        refreshed = await _get_budget(db, budget.budget_id)
        return refreshed or budget

    await db.rollback()
    return budget


async def validate_tag_budgets(
    db: AsyncSession,
    tags: dict[str, Any] | None,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> list[Budget]:
    """Validate matching tag-scoped budgets have available spend."""
    normalized_strategy = _normalize_budget_strategy(strategy)
    if normalized_strategy == "disabled":
        return []

    matching_budgets = await _matching_tag_budgets(
        db,
        tags,
        for_update=normalized_strategy == "for_update",
    )
    if not matching_budgets:
        return []

    now = datetime.now(UTC)
    validated: list[Budget] = []
    for budget in matching_budgets:
        if budget.blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Budget group '{budget.budget_id}' is blocked",
            )

        if budget.next_budget_reset_at and now >= budget.next_budget_reset_at:
            if normalized_strategy == "cas":
                budget = await _cas_reset_tag_budget(db, budget, now)
            else:
                await reset_tag_budget(db, budget, now)

        if budget.max_budget is not None and budget.spend >= budget.max_budget:
            if model and await _is_model_free(db, model):
                validated.append(budget)
                continue
            record_budget_exceeded()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Budget group '{budget.budget_id}' has exceeded budget limit",
            )
        validated.append(budget)
    return validated


async def increment_matching_tag_budget_spend(
    db: AsyncSession,
    *,
    tags: dict[str, Any] | None,
    cost: float | None,
    metadata: dict[str, Any] | None = None,
) -> list[BudgetAlert]:
    """Increment spend for active tag-scoped budgets matching usage tags."""
    if not cost or cost <= 0:
        return []
    matching_budgets = await _matching_tag_budgets(db, tags)
    if not matching_budgets:
        return []
    budget_ids = [budget.budget_id for budget in matching_budgets]
    await db.execute(
        update(Budget)
        .where(Budget.budget_id.in_(budget_ids))
        .values(spend=Budget.spend + cost)
        .execution_options(synchronize_session=False)
    )
    updated_result = await db.execute(
        select(Budget)
        .where(Budget.budget_id.in_(budget_ids))
        .execution_options(populate_existing=True)
    )
    created_alerts: list[BudgetAlert] = []
    for budget in updated_result.scalars().all():
        alerts = await record_budget_alerts(
            db,
            budget=budget,
            scope_type=TAG_BUDGET_SCOPE,
            scope_id=_tag_scope_id(budget),
            spend=float(budget.spend),
            period_start=budget.budget_started_at,
            metadata=metadata,
        )
        created_alerts.extend(alerts)
    return created_alerts


async def validate_user_budget(
    db: AsyncSession,
    user_id: str,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> User:
    """Validate user exists, is not blocked, and has available budget.

    Args:
        db: Database session
        user_id: User identifier
        model: Optional model identifier (e.g., "provider/model") to check if it's a free model

    Returns:
        User object if validation passes

    Raises:
        HTTPException: If user is blocked, doesn't exist, or exceeded budget

    """
    normalized_strategy = _normalize_budget_strategy(strategy)

    lock_for_update = normalized_strategy == "for_update"
    user = await get_active_user(db, user_id, for_update=lock_for_update)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found",
        )

    if user.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' is blocked",
        )

    if normalized_strategy == "disabled" or not user.budget_id:
        return user

    budget = await _get_budget(db, user.budget_id)
    if not budget:
        return user

    now = datetime.now(UTC)
    if user.next_budget_reset_at and now >= user.next_budget_reset_at:
        if normalized_strategy == "cas":
            user = await _cas_reset_user_budget(db, user, budget, now)
        else:
            await reset_user_budget(db, user, budget, now)

    if budget.max_budget is not None and user.spend >= budget.max_budget:
        if model and await _is_model_free(db, model):
            return user
        record_budget_exceeded()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' has exceeded budget limit",
        )

    return user


async def validate_project_budget(
    db: AsyncSession,
    project_id: str,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> Project:
    """Validate project exists, is active, is not blocked, and has available budget."""

    normalized_strategy = _normalize_budget_strategy(strategy)

    stmt = select(Project).where(Project.project_id == project_id)
    if normalized_strategy == "for_update":
        stmt = stmt.with_for_update()
    result = await db.execute(stmt)
    project = result.scalar_one_or_none()

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found",
        )

    if not project.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Project '{project_id}' is inactive",
        )

    if project.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Project '{project_id}' is blocked",
        )

    if normalized_strategy == "disabled" or not project.budget_id:
        return project

    budget = await _get_budget(db, project.budget_id)
    if not budget:
        return project

    now = datetime.now(UTC)
    if project.next_budget_reset_at and now >= project.next_budget_reset_at:
        if normalized_strategy == "cas":
            project = await _cas_reset_project_budget(db, project, budget, now)
        else:
            await reset_project_budget(db, project, budget, now)

    if budget.max_budget is not None and project.spend >= budget.max_budget:
        if model and await _is_model_free(db, model):
            return project
        record_budget_exceeded()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Project '{project_id}' has exceeded budget limit",
        )

    return project


async def _is_model_free(db: AsyncSession, model: str) -> bool:
    """Check if a model is free (both input and output prices are 0).

    Args:
        db: Database session
        model: Model identifier (e.g., "provider/model" or "model")

    Returns:
        True if the model is free, False otherwise or if pricing not found

    """
    try:
        provider, model_name = AnyLLM.split_model_provider(model)
        provider_str = provider.value if provider else None
        pricing = await find_model_pricing(db, provider_str, model_name)
        if pricing:
            return pricing.input_price_per_million == 0 and pricing.output_price_per_million == 0
    except (AnyLLMError, ValueError, SQLAlchemyError) as e:
        logger.warning("Failed to determine provider pricing: %s", e)

    return False
