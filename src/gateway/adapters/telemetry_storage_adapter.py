"""Core adapter for ``TelemetryStoragePort``: Otari's own database.

Satisfies :class:`gateway.ports.telemetry_storage_port.TelemetryStoragePort`
with the storage Otari has always used, the ``agent_telemetry`` table, so a
deployment with no overlay keeps exactly the behavior it had before the port
existed. This is a real adapter and not a Null Object: telemetry that arrives
with nowhere to go is telemetry lost, and the base has a perfectly good place
to put it.

Everything here is the query layer the read endpoints and the ingest service
used to hold inline. It moved rather than changed: the aggregates, the
uniqueness-collision escalation on insert, and the bucket grid are the same,
which is what makes binding this adapter indistinguishable from not having the
seam at all.
"""

from collections import defaultdict
from typing import Any, cast

from sqlalchemy import ColumnElement, case, delete, func, null, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.sql import bucket_expr, canonical_bucket, dialect_name, match_any, utc_bound
from gateway.log_config import logger
from gateway.models.entities import AgentTelemetry
from gateway.ports.telemetry_storage_port import (
    BehaviorCounts,
    BehaviorGroup,
    BucketedEventCount,
    GroupedBucketCount,
    GroupedRowCounts,
    GroupTotal,
    IngestResult,
    MetricPoint,
    TelemetryBucket,
    TelemetryFilter,
    TelemetryGroupBy,
    TelemetryRecord,
    TelemetryScanTooLargeError,
)

# TelemetryRecord fields that feed a dedup key but are not their own
# AgentTelemetry column (the key that derives from them is what is stored).
_DEDUP_ONLY_FIELDS = ("tool_use_id", "event_sequence")

_METRIC_KIND = "metric"


def _conditions(filters: TelemetryFilter) -> list[ColumnElement[bool]]:
    """The scope as WHERE conditions. An unset field adds nothing."""
    conditions: list[ColumnElement[bool]] = []
    if filters.start is not None:
        conditions.append(AgentTelemetry.timestamp >= utc_bound(filters.start))
    if filters.end is not None:
        conditions.append(AgentTelemetry.timestamp < utc_bound(filters.end))
    if filters.user_ids:
        conditions.append(match_any(AgentTelemetry.user_id, list(filters.user_ids)))
    if filters.api_key_ids:
        conditions.append(match_any(AgentTelemetry.api_key_id, list(filters.api_key_ids)))
    if filters.name is not None:
        conditions.append(AgentTelemetry.name == filters.name)
    if filters.session_label is not None:
        conditions.append(AgentTelemetry.session_label == filters.session_label)
    return conditions


def _build_row(api_key_id: str, user_id: str, record: TelemetryRecord) -> AgentTelemetry:
    row_fields = {k: v for k, v in record.__dict__.items() if k not in _DEDUP_ONLY_FIELDS}
    return AgentTelemetry(api_key_id=api_key_id, user_id=user_id, **row_fields)


async def _existing_dedup_keys(db: AsyncSession, source: str, dedup_keys: list[str]) -> set[str]:
    rows = (
        await db.execute(
            select(AgentTelemetry.dedup_key).where(
                AgentTelemetry.source == source,
                AgentTelemetry.dedup_key.in_(dedup_keys),
            )
        )
    ).scalars().all()
    return set(rows)


async def _insert_same_source_batch(db: AsyncSession, source: str, rows: list[AgentTelemetry]) -> IngestResult:
    """Insert a same-source batch, retrying only rows that do not collide.

    Mirrors ``external_usage_service._insert_rows``: one ``add_all`` + ``commit``
    for the whole batch; on a uniqueness collision, roll back, re-query which
    ``(source, dedup_key)`` pairs already exist, and retry only the survivors as
    one bulk insert; if that also collides, fall back to row-at-a-time for the
    still-colliding remainder.
    """
    db.add_all(rows)
    try:
        await db.commit()
        return IngestResult(accepted=len(rows))
    except IntegrityError:
        await db.rollback()

    existing = await _existing_dedup_keys(db, source, [row.dedup_key for row in rows])
    survivors = [row for row in rows if row.dedup_key not in existing]
    duplicate = len(rows) - len(survivors)
    if not survivors:
        return IngestResult(duplicate=duplicate)
    db.add_all(survivors)
    try:
        await db.commit()
        return IngestResult(accepted=len(survivors), duplicate=duplicate)
    except IntegrityError:
        await db.rollback()

    accepted = still_duplicate = 0
    for row in survivors:
        db.add(row)
        try:
            await db.commit()
            accepted += 1
        except IntegrityError:
            await db.rollback()
            still_duplicate += 1
    return IngestResult(accepted=accepted, duplicate=duplicate + still_duplicate)


class DatabaseTelemetryStorageAdapter:
    """Core adapter: telemetry lives in this deployment's own database.

    Session-bound, unlike the stateless core adapters beside it: every method
    is a query, so the request's session is the whole of what it needs. The
    surfaces that reach this port are standalone-only, so a ``None`` session
    means a caller mounted one somewhere it cannot work rather than an ordinary
    hybrid request, and it fails loudly instead of silently storing nothing.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        self._session = session

    @property
    def _db(self) -> AsyncSession:
        if self._session is None:
            msg = "Telemetry storage requires a database session, and this request has none"
            raise RuntimeError(msg)
        return self._session

    async def record(
        self,
        *,
        api_key_id: str,
        user_id: str,
        records: tuple[TelemetryRecord, ...],
    ) -> IngestResult:
        """Store the records, batching by source and treating collisions as replays.

        Batch-inserts by source (the unique constraint is
        ``(source, dedup_key)``) rather than one savepoint per record, so
        ingesting a large export issues a small, bounded number of round trips.

        Repeats inside this batch are dropped before the database is touched.
        The stored projection is lossy by design, so two records can collapse
        onto one dedup key; letting that reach the insert fails the whole batch
        and drops it into the row-at-a-time fallback, turning one bulk insert
        into one commit per record.
        """
        if not records:
            return IngestResult()
        db = self._db

        by_source: dict[str, list[AgentTelemetry]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        duplicate = 0
        for record in records:
            identity = (record.source, record.dedup_key)
            if identity in seen:
                duplicate += 1
                continue
            seen.add(identity)
            by_source[record.source].append(_build_row(api_key_id, user_id, record))

        accepted = 0
        try:
            for source, rows in by_source.items():
                result = await _insert_same_source_batch(db, source, rows)
                accepted += result.accepted
                duplicate += result.duplicate
        except SQLAlchemyError:
            await db.rollback()
            raise
        return IngestResult(accepted=accepted, duplicate=duplicate)

    async def metric_points(self, *, filters: TelemetryFilter, limit: int) -> tuple[MetricPoint, ...]:
        """Metric readings in scope, as stored, for the caller to diff.

        The scan is bounded by the window and served by the
        ``(series_key, timestamp)`` index. One row past ``limit`` is fetched so
        the ceiling can be detected without a second counting query.
        """
        db = self._db
        rows = (
            await db.execute(
                select(
                    AgentTelemetry.name,
                    AgentTelemetry.series_key,
                    AgentTelemetry.series_start,
                    AgentTelemetry.temporality,
                    AgentTelemetry.timestamp,
                    AgentTelemetry.value,
                )
                .where(*_conditions(filters), AgentTelemetry.kind == _METRIC_KIND, AgentTelemetry.value.is_not(None))
                .limit(limit + 1)
            )
        ).all()
        if len(rows) > limit:
            raise TelemetryScanTooLargeError(limit)
        return tuple(
            MetricPoint(
                name=name,
                series_key=series_key,
                series_start=series_start,
                temporality=temporality,
                timestamp=timestamp,
                value=float(value),
            )
            for name, series_key, series_start, temporality, timestamp, value in rows
        )

    async def behavior_counts(self, *, filters: TelemetryFilter) -> BehaviorCounts:
        """Behavioral-event counts from one grouped pass, plus the session count."""
        db = self._db
        behavioral = [*_conditions(filters), AgentTelemetry.kind.is_(None)]
        rows = (
            await db.execute(
                select(AgentTelemetry.name, AgentTelemetry.tool_name, AgentTelemetry.decision, func.count())
                .where(*behavioral)
                .group_by(AgentTelemetry.name, AgentTelemetry.tool_name, AgentTelemetry.decision)
            )
        ).all()
        sessions = (
            await db.execute(select(func.count(func.distinct(AgentTelemetry.session_label))).where(*behavioral))
        ).scalar_one()
        return BehaviorCounts(
            groups=tuple(
                BehaviorGroup(name=name, tool_name=tool_name, decision=decision, count=int(count))
                for name, tool_name, decision, count in rows
            ),
            sessions=int(sessions),
        )

    async def behavior_counts_by_bucket(
        self,
        *,
        filters: TelemetryFilter,
        bucket: TelemetryBucket,
    ) -> tuple[BucketedEventCount, ...]:
        """The same counts, grouped onto the shared UTC bucket grid."""
        db = self._db
        expr = bucket_expr(dialect_name(db), bucket, AgentTelemetry.timestamp)
        rows = (
            await db.execute(
                select(expr, AgentTelemetry.name, func.count())
                .where(*_conditions(filters), AgentTelemetry.kind.is_(None))
                .group_by(expr, AgentTelemetry.name)
            )
        ).all()
        return tuple(
            BucketedEventCount(bucket_start=canonical_bucket(raw_bucket, bucket), name=name, count=int(count))
            for raw_bucket, name, count in rows
        )

    async def count(self, *, filters: TelemetryFilter) -> int:
        """Count every row in scope, behavioral and metric alike."""
        stmt: Any = select(func.count()).select_from(AgentTelemetry).where(*_conditions(filters))
        return int((await self._db.execute(stmt)).scalar_one())

    async def grouped_row_counts(
        self,
        *,
        filters: TelemetryFilter,
        group_by: TelemetryGroupBy,
        bucket: TelemetryBucket,
        top_n: int,
    ) -> GroupedRowCounts:
        """Row volume over time, split by one dimension, with the tail folded.

        Groups past the top N collapse in SQL, so the grid stays bounded by
        buckets times (top N + 2) however high the dimension's cardinality is.
        The fold is encoded as (key NULL, flag) rather than a sentinel key,
        which no value could be trusted never to collide with; the flag then
        separates a real NULL group that ranked in the top N from the remainder.
        """
        db = self._db
        conditions = _conditions(filters)
        column = AgentTelemetry.user_id if group_by == "user_id" else AgentTelemetry.api_key_id

        row_count = func.count()
        group_rows = (
            await db.execute(
                select(column, row_count).where(*conditions).group_by(column).order_by(row_count.desc()).limit(top_n)
            )
        ).all()
        total_stmt: Any = select(func.count()).select_from(AgentTelemetry).where(*conditions)
        total = int((await db.execute(total_stmt)).scalar_one())
        groups = tuple(GroupTotal(key=row[0], rows=int(row[1])) for row in group_rows)

        named = {group.key for group in groups if group.key is not None}
        keeps_null = any(group.key is None for group in groups)
        key_expr = case((column.in_(named), column), else_=null())
        if keeps_null:
            fold_expr = case((column.is_(None), 0), (column.in_(named), 0), else_=1)
        else:
            fold_expr = case((column.in_(named), 0), else_=1)
        grid_bucket = bucket_expr(dialect_name(db), bucket, AgentTelemetry.timestamp)
        rows = (
            await db.execute(
                select(grid_bucket, key_expr, fold_expr, func.count())
                .where(*conditions)
                .group_by(grid_bucket, key_expr, fold_expr)
            )
        ).all()
        points = tuple(
            GroupedBucketCount(
                bucket_start=canonical_bucket(row[0], bucket),
                key=row[1],
                is_other=bool(row[2]),
                rows=int(row[3]),
            )
            for row in rows
        )
        return GroupedRowCounts(groups=groups, total=total, points=points)

    async def purge(self, *, ids: tuple[str, ...], filters: TelemetryFilter) -> int:
        """Delete by explicit id, or by filter when no ids are given."""
        conditions: list[ColumnElement[bool]] = (
            [cast("ColumnElement[bool]", AgentTelemetry.id.in_(ids))] if ids else _conditions(filters)
        )
        deleted = await self._delete(conditions)
        logger.info("agent_telemetry delete: removed=%d by_filter=%s", deleted, not ids)
        return deleted

    async def purge_user(self, *, user_id: str) -> int:
        """Delete every row attributed to one user."""
        deleted = await self._delete([AgentTelemetry.user_id == user_id])
        logger.info("agent_telemetry erase for user: removed=%d", deleted)
        return deleted

    async def _delete(self, conditions: list[ColumnElement[bool]]) -> int:
        db = self._db
        try:
            result = cast("CursorResult[Any]", await db.execute(delete(AgentTelemetry).where(*conditions)))
            await db.commit()
        except SQLAlchemyError:
            await db.rollback()
            logger.exception("agent_telemetry delete failed")
            raise
        return result.rowcount or 0
