from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.metrics import (
    record_budget_alert_webhook_dead_letter,
    record_budget_alert_webhook_delivery,
    record_budget_alert_webhook_retry_run,
    record_budget_alert_webhook_retry_selected,
)
from gateway.models.entities import BudgetAlert

_WEBHOOK_TIMEOUT_SECONDS = 5.0
_MAX_ERROR_CHARS = 500
_RETRYABLE_DELIVERY_STATUSES = ("pending", "failed")
_DEAD_LETTER_DELIVERY_STATUS = "dead_letter"


@dataclass(frozen=True)
class WebhookDeliveryResult:
    status_code: int | None
    error: str | None = None

    @property
    def delivered(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 300


def _trim_error(value: str) -> str:
    return value[:_MAX_ERROR_CHARS]


def _alert_payload(alert: BudgetAlert) -> dict[str, Any]:
    return {
        "event": "budget.threshold_crossed",
        "alert": {
            "id": alert.id,
            "budget_id": alert.budget_id,
            "scope_type": alert.scope_type,
            "scope_id": alert.scope_id,
            "threshold": alert.threshold,
            "spend": alert.spend,
            "max_budget": alert.max_budget,
            "budget_period_start": (
                alert.budget_period_start.isoformat()
                if alert.budget_period_start
                else None
            ),
            "created_at": alert.created_at.isoformat() if alert.created_at else None,
            "metadata": alert.metadata_ or {},
        },
    }


def _next_retry_at(
    now: datetime,
    *,
    delivery_attempts: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
) -> datetime:
    if backoff_seconds <= 0:
        return now
    multiplier = 2 ** max(delivery_attempts - 1, 0)
    delay_seconds = backoff_seconds * multiplier
    if max_backoff_seconds > 0:
        delay_seconds = min(delay_seconds, max_backoff_seconds)
    return now + timedelta(seconds=delay_seconds)


async def _post_budget_alert_webhook(
    *,
    webhook_url: str,
    payload: dict[str, Any],
    timeout_seconds: float = _WEBHOOK_TIMEOUT_SECONDS,
) -> WebhookDeliveryResult:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(webhook_url, json=payload)
    except httpx.HTTPError as exc:
        return WebhookDeliveryResult(status_code=None, error=_trim_error(str(exc)))

    if 200 <= response.status_code < 300:
        return WebhookDeliveryResult(status_code=response.status_code)
    return WebhookDeliveryResult(
        status_code=response.status_code,
        error=_trim_error(f"HTTP {response.status_code}: {response.text}"),
    )


async def dispatch_budget_alert_webhooks(
    alert_ids: Sequence[int],
    *,
    max_attempts: int | None = None,
    backoff_seconds: float = 0,
    max_backoff_seconds: float = 0,
) -> None:
    """Deliver pending or failed budget alert webhooks and persist delivery state."""
    for alert_id in alert_ids:
        await dispatch_budget_alert_webhook(
            alert_id,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )


async def dispatch_pending_budget_alert_webhooks(
    *,
    limit: int,
    max_attempts: int,
    backoff_seconds: float = 0,
    max_backoff_seconds: float = 0,
    now: datetime | None = None,
) -> int:
    """Retry pending or failed budget alert webhooks and return the number selected."""
    effective_now = now or datetime.now(UTC)
    async with create_session() as db:
        maxed_result = await db.execute(
            select(BudgetAlert)
            .where(
                BudgetAlert.webhook_url.is_not(None),
                BudgetAlert.delivery_status.in_(_RETRYABLE_DELIVERY_STATUSES),
                BudgetAlert.delivery_attempts >= max_attempts,
            )
            .order_by(BudgetAlert.created_at.asc(), BudgetAlert.id.asc())
            .limit(limit)
        )
        maxed_alerts = list(maxed_result.scalars().all())
        for alert in maxed_alerts:
            alert.delivery_status = _DEAD_LETTER_DELIVERY_STATUS
            alert.dead_lettered_at = effective_now
            alert.next_delivery_attempt_at = None
            record_budget_alert_webhook_dead_letter(alert.scope_type, "max_attempts_before_delivery")
        if maxed_alerts:
            await db.commit()

        result = await db.execute(
            select(BudgetAlert.id)
            .where(
                BudgetAlert.webhook_url.is_not(None),
                BudgetAlert.delivery_status.in_(_RETRYABLE_DELIVERY_STATUSES),
                BudgetAlert.delivery_attempts < max_attempts,
                or_(
                    BudgetAlert.next_delivery_attempt_at.is_(None),
                    BudgetAlert.next_delivery_attempt_at <= effective_now,
                ),
            )
            .order_by(BudgetAlert.created_at.asc(), BudgetAlert.id.asc())
            .limit(limit)
        )
        alert_ids = list(result.scalars().all())

    record_budget_alert_webhook_retry_selected(len(alert_ids))
    await dispatch_budget_alert_webhooks(
        alert_ids,
        max_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
    )
    return len(alert_ids)


async def dispatch_budget_alert_webhook(
    alert_id: int,
    *,
    max_attempts: int | None = None,
    backoff_seconds: float = 0,
    max_backoff_seconds: float = 0,
) -> BudgetAlert | None:
    """Deliver a single budget alert webhook if the alert has a webhook URL."""
    async with create_session() as db:
        alert = await db.get(BudgetAlert, alert_id)
        if alert is None:
            return None
        if not alert.webhook_url:
            alert.delivery_status = "not_configured"
            alert.next_delivery_attempt_at = None
            await db.commit()
            return alert
        if alert.delivery_status == "delivered":
            return alert

        payload = _alert_payload(alert)
        delivery_started_at = time.monotonic()
        result = await _post_budget_alert_webhook(webhook_url=alert.webhook_url, payload=payload)
        delivery_duration = time.monotonic() - delivery_started_at
        now = datetime.now(UTC)
        alert.delivery_attempts += 1
        alert.last_delivery_attempt_at = now
        alert.last_delivery_status_code = result.status_code
        alert.last_delivery_error = result.error
        if result.delivered:
            alert.delivery_status = "delivered"
            alert.delivered_at = now
            alert.dead_lettered_at = None
            alert.next_delivery_attempt_at = None
        else:
            alert.delivered_at = None
            if max_attempts is not None and alert.delivery_attempts >= max_attempts:
                alert.delivery_status = _DEAD_LETTER_DELIVERY_STATUS
                alert.dead_lettered_at = now
                alert.next_delivery_attempt_at = None
                record_budget_alert_webhook_dead_letter(alert.scope_type, "max_attempts_after_delivery")
            else:
                alert.delivery_status = "failed"
                alert.dead_lettered_at = None
                alert.next_delivery_attempt_at = _next_retry_at(
                    now,
                    delivery_attempts=alert.delivery_attempts,
                    backoff_seconds=backoff_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                )
            logger.warning(
                "Budget alert webhook delivery failed for alert %s: %s",
                alert_id,
                result.error or f"HTTP {result.status_code}",
            )
        record_budget_alert_webhook_delivery(
            scope_type=alert.scope_type,
            outcome=alert.delivery_status,
            duration_seconds=delivery_duration,
        )

        try:
            await db.commit()
        except SQLAlchemyError as exc:
            await db.rollback()
            logger.error("Failed to persist budget alert webhook delivery for alert %s: %s", alert_id, exc)
            raise
        await db.refresh(alert)
        return alert


class BudgetAlertWebhookRetryWorker:
    """Background worker that retries failed or pending budget alert webhooks."""

    def __init__(
        self,
        *,
        interval_seconds: float,
        max_attempts: int,
        backoff_seconds: float,
        max_backoff_seconds: float,
        batch_size: int,
    ) -> None:
        self._interval_seconds = interval_seconds
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._batch_size = batch_size
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._interval_seconds > 0 and self._max_attempts > 0 and self._batch_size > 0

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def run_once(self) -> int:
        try:
            retried = await dispatch_pending_budget_alert_webhooks(
                limit=self._batch_size,
                max_attempts=self._max_attempts,
                backoff_seconds=self._backoff_seconds,
                max_backoff_seconds=self._max_backoff_seconds,
            )
        except Exception:
            record_budget_alert_webhook_retry_run("error")
            raise
        record_budget_alert_webhook_retry_run("success")
        return retried

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval_seconds)
                retried = await self.run_once()
                if retried:
                    logger.info("Retried %d budget alert webhook deliveries", retried)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - defensive background loop logging
                logger.error("Budget alert webhook retry worker failed: %s", exc)
