"""Usage log writer implementations."""

from __future__ import annotations

import asyncio
import time
from typing import Protocol

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError

from gateway.core.database import create_session
from gateway.log_config import logger
from gateway.metrics import (
    log_writer_batch_size,
    log_writer_flush_duration,
    log_writer_queue_depth,
    log_writer_rows,
    record_budget_alert_created,
)
from gateway.models.entities import BudgetAlert, Project, UsageLog, User
from gateway.services.budget_alert_webhook_service import dispatch_budget_alert_webhooks
from gateway.services.budget_service import (
    increment_matching_tag_budget_spend,
    record_project_budget_alerts_after_spend,
    record_user_budget_alerts_after_spend,
)


class LogWriter(Protocol):
    async def put(self, log: UsageLog) -> None: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


def _budget_alert_metadata(log: UsageLog) -> dict[str, object]:
    return {
        "api_key_id": log.api_key_id,
        "cost": log.cost,
        "endpoint": log.endpoint,
        "model": log.model,
        "provider": log.provider,
        "status": log.status,
        "tags": log.tags if isinstance(log.tags, dict) else {},
    }


async def _dispatch_new_alert_webhooks(alerts: list[BudgetAlert]) -> None:
    alert_ids = [alert.id for alert in alerts if alert.id is not None and alert.webhook_url]
    if not alert_ids:
        return
    try:
        await dispatch_budget_alert_webhooks(alert_ids)
    except Exception as exc:  # pragma: no cover - post-commit defensive logging
        logger.error("Budget alert webhook dispatch failed after usage commit: %s", exc)


def _record_new_budget_alert_metrics(alerts: list[BudgetAlert]) -> None:
    for alert in alerts:
        record_budget_alert_created(alert.scope_type, alert.delivery_status)


class SingleLogWriter:
    """Write each usage log inline, one transaction per event."""

    async def put(self, log: UsageLog) -> None:
        created_alerts: list[BudgetAlert] = []
        async with create_session() as db:
            try:
                db.add(log)
                alert_metadata = _budget_alert_metadata(log)
                if log.cost and log.user_id:
                    await db.execute(
                        update(User)
                        .where(User.user_id == log.user_id, User.deleted_at.is_(None))
                        .values(spend=User.spend + log.cost)
                    )
                    user_alerts = await record_user_budget_alerts_after_spend(
                        db,
                        user_id=log.user_id,
                        metadata=alert_metadata,
                    )
                    created_alerts.extend(user_alerts)
                if log.cost and log.project_id:
                    await db.execute(
                        update(Project)
                        .where(Project.project_id == log.project_id)
                        .values(spend=Project.spend + log.cost)
                    )
                    project_alerts = await record_project_budget_alerts_after_spend(
                        db,
                        project_id=log.project_id,
                        metadata=alert_metadata,
                    )
                    created_alerts.extend(project_alerts)
                if log.cost:
                    tag_alerts = await increment_matching_tag_budget_spend(
                        db,
                        tags=log.tags if isinstance(log.tags, dict) else {},
                        cost=log.cost,
                        metadata=alert_metadata,
                    )
                    created_alerts.extend(tag_alerts)
                await db.commit()
                log_writer_rows.labels(writer="single", result="written").inc()
            except SQLAlchemyError as e:  # pragma: no cover - defensive logging
                await db.rollback()
                logger.error("SingleLogWriter failed: %s", e)
                log_writer_rows.labels(writer="single", result="dropped").inc()
                return

        _record_new_budget_alert_metrics(created_alerts)
        await _dispatch_new_alert_webhooks(created_alerts)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class BatchLogWriter:
    """Queue usage logs and flush in batches."""

    def __init__(self, max_batch: int = 100, flush_interval: float = 1.0) -> None:
        self._queue: asyncio.Queue[UsageLog] = asyncio.Queue()
        self._max_batch = max_batch
        self._flush_interval = flush_interval
        self._task: asyncio.Task[None] | None = None

    async def put(self, log: UsageLog) -> None:
        await self._queue.put(log)
        log_writer_queue_depth.set(self._queue.qsize())

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._flush_all()

    async def _run(self) -> None:
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._flush(batch)
            except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
                break
            except Exception as e:  # pragma: no cover - defensive logging
                logger.error("BatchLogWriter loop error: %s", e)

    async def _collect_batch(self) -> list[UsageLog]:
        batch: list[UsageLog] = []
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=self._flush_interval)
            batch.append(item)
            self._queue.task_done()
        except asyncio.TimeoutError:
            return batch

        while len(batch) < self._max_batch:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        return batch

    async def _flush(self, batch: list[UsageLog]) -> None:
        start = time.monotonic()
        log_writer_batch_size.labels(writer="batch").observe(len(batch))
        created_alerts: list[BudgetAlert] = []
        try:
            async with create_session() as db:
                for log in batch:
                    db.add(log)
                    alert_metadata = _budget_alert_metadata(log)
                    if log.cost and log.user_id:
                        await db.execute(
                            update(User)
                            .where(User.user_id == log.user_id, User.deleted_at.is_(None))
                            .values(spend=User.spend + log.cost)
                        )
                        user_alerts = await record_user_budget_alerts_after_spend(
                            db,
                            user_id=log.user_id,
                            metadata=alert_metadata,
                        )
                        created_alerts.extend(user_alerts)
                    if log.cost and log.project_id:
                        await db.execute(
                            update(Project)
                            .where(Project.project_id == log.project_id)
                            .values(spend=Project.spend + log.cost)
                        )
                        project_alerts = await record_project_budget_alerts_after_spend(
                            db,
                            project_id=log.project_id,
                            metadata=alert_metadata,
                        )
                        created_alerts.extend(project_alerts)
                    if log.cost:
                        tag_alerts = await increment_matching_tag_budget_spend(
                            db,
                            tags=log.tags if isinstance(log.tags, dict) else {},
                            cost=log.cost,
                            metadata=alert_metadata,
                        )
                        created_alerts.extend(tag_alerts)
                await db.commit()
                log_writer_rows.labels(writer="batch", result="written").inc(len(batch))
            log_writer_flush_duration.labels(writer="batch", result="ok").observe(time.monotonic() - start)
        except SQLAlchemyError as e:  # pragma: no cover - defensive logging
            logger.error("BatchLogWriter flush failed, dropping %d rows: %s", len(batch), e)
            log_writer_rows.labels(writer="batch", result="dropped").inc(len(batch))
            log_writer_flush_duration.labels(writer="batch", result="error").observe(time.monotonic() - start)
            return

        _record_new_budget_alert_metrics(created_alerts)
        await _dispatch_new_alert_webhooks(created_alerts)

    async def _flush_all(self) -> None:
        batch: list[UsageLog] = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._flush(batch)


def create_log_writer(strategy: str) -> LogWriter:
    if strategy == "batch":
        return BatchLogWriter()
    return SingleLogWriter()


class NoopLogWriter:
    """LogWriter implementation that discards all writes (used when DB is unavailable)."""

    async def put(self, log: UsageLog) -> None:  # noqa: D401,B027 - trivial no-op
        return None

    async def start(self) -> None:  # noqa: D401,B027
        return None

    async def stop(self) -> None:  # noqa: D401,B027
        return None
