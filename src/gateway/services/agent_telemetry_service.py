"""Content-free coding-agent telemetry mapping, ingestion, and aggregation helpers."""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import AgentTelemetry, APIKey
from gateway.repositories.users_repository import get_active_user

_MAX_NUMBER = 1_000_000_000
_EVENTS = {"tool_result", "tool_decision", "user_prompt", "api_error"}
# TelemetryRecord fields that feed event_dedup_key() but are not their own
# AgentTelemetry column (the dedup key that derives from them is what's stored).
_DEDUP_ONLY_FIELDS = ("tool_use_id", "event_sequence")

# The outcome counters a coding agent reports on the metrics signal that Otari has
# no other source for. Each becomes a content-free, non-billable metric row.
METRIC_LINES_OF_CODE = "claude_code.lines_of_code.count"
METRIC_COMMITS = "claude_code.commit.count"
METRIC_PULL_REQUESTS = "claude_code.pull_request.count"
METRIC_ACTIVE_TIME = "claude_code.active_time.total"
_METRICS = frozenset({METRIC_LINES_OF_CODE, METRIC_COMMITS, METRIC_PULL_REQUESTS, METRIC_ACTIVE_TIME})

# Metrics that duplicate a signal Otari already holds. Named rather than left to
# the generic unknown-name path so the intent is legible: token/cost usage is
# already billed from the api_request usage event (recording it here would double
# count spend), and code_edit_tool.decision is the same accept/reject signal the
# tool_decision behavioral event already carries.
_SKIPPED_METRICS = frozenset(
    {"claude_code.token.usage", "claude_code.cost.usage", "claude_code.code_edit_tool.decision"}
)

CUMULATIVE = "cumulative"
DELTA = "delta"


@dataclass(frozen=True)
class TelemetryRecord:
    name: str
    timestamp: datetime
    source: str
    dedup_key: str
    tool_name: str | None = None
    decision: str | None = None
    success: bool | None = None
    duration_ms: int | None = None
    status_code: int | None = None
    prompt_length: int | None = None
    session_label: str | None = None
    tool_use_id: str | None = None
    event_sequence: int | None = None
    # Metric-point fields; all None on a behavioral event. See AgentTelemetry.
    kind: str | None = None
    value: float | None = None
    temporality: str | None = None
    series_start: datetime | None = None
    series_key: str | None = None


@dataclass(frozen=True)
class IngestResult:
    accepted: int = 0
    duplicate: int = 0
    rejected: int = 0


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def _timestamp(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _bounded_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number) or abs(number) > _MAX_NUMBER:
        return None
    return number


def _bounded_int(value: Any) -> int | None:
    number = _bounded_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _bounded_bool(value: Any) -> bool | None:
    """Read a boolean attribute that may arrive as a real bool or as a string.

    Claude Code emits ``success`` on ``tool_result`` as the string ``"true"`` /
    ``"false"``, not an OTLP boolValue, so a strict isinstance check would null out
    the outcome field on every tool result.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    identifier = str(value).strip()
    if not identifier or len(identifier) > 128 or any(char.isspace() for char in identifier):
        return None
    return identifier


def _session(attrs: dict[str, Any]) -> str | None:
    return _identifier(attrs.get("session.id"))


def event_dedup_key(record: TelemetryRecord, user_id: str | None = None) -> str:
    """Return the documented natural idempotency key for a behavioral event."""
    return _hash(
        record.name,
        user_id or "",
        record.session_label or "",
        record.timestamp.isoformat(),
        record.tool_name or "",
        record.decision or "",
        record.success if record.success is not None else "",
        record.duration_ms if record.duration_ms is not None else "",
        record.status_code if record.status_code is not None else "",
        record.prompt_length if record.prompt_length is not None else "",
        record.tool_use_id or "",
        record.event_sequence if record.event_sequence is not None else "",
    )


def map_behavioral_event(
    name: str,
    attrs: dict[str, Any],
    *,
    timestamp: datetime,
    source: str,
    user_id: str | None,
) -> TelemetryRecord | None:
    """Map a known behavioral event, retaining only its typed allow-list fields."""
    if name not in _EVENTS:
        return None
    tool_name = _identifier(attrs.get("tool_name") or attrs.get("tool.name"))
    decision = _identifier(attrs.get("decision"))
    timestamp = _timestamp(timestamp)
    tool_use_id = _identifier(attrs.get("tool_use_id")) if name in {"tool_result", "tool_decision"} else None
    event_sequence = _bounded_int(attrs.get("event.sequence"))
    fields: dict[str, Any] = {}
    if name == "tool_decision":
        fields = {"tool_name": tool_name, "decision": decision if decision in {"accept", "reject"} else None}
    elif name == "tool_result":
        fields = {
            "tool_name": tool_name,
            "success": _bounded_bool(attrs.get("success")),
            "duration_ms": _bounded_int(attrs.get("duration_ms")),
        }
    elif name == "api_error":
        fields = {"status_code": _bounded_int(attrs.get("status_code") or attrs.get("http.status_code"))}
    else:
        fields = {"prompt_length": _bounded_int(attrs.get("prompt_length"))}
    provisional = TelemetryRecord(
        name=name,
        timestamp=timestamp,
        source=source,
        dedup_key="",
        session_label=_session(attrs),
        tool_use_id=tool_use_id,
        event_sequence=event_sequence,
        **fields,
    )
    return TelemetryRecord(**{**provisional.__dict__, "dedup_key": event_dedup_key(provisional, user_id)})


def metric_series_key(name: str, attrs: dict[str, Any]) -> str:
    """Return the OTLP series identity for a metric point.

    The metric name plus its full attribute set, sorted so the hash does not
    depend on attribute order. Deliberately carries no ``user_id``: this is pure
    OTLP identity, and the read-time per-series diff groups on it. A dimensioned
    metric (``lines_of_code.count`` split by ``type=added``/``removed``) is
    therefore two series, never one.
    """
    return _hash(name, sorted((str(key), str(value)) for key, value in attrs.items()))


def metric_dedup_key(record: TelemetryRecord, user_id: str | None = None) -> str:
    """Return the natural idempotency key for one metric data point.

    The metric-point sibling of ``event_dedup_key``, and folds ``user_id`` in for
    the same reason: the uniqueness constraint is ``(source, dedup_key)`` across
    the whole table, so two users reporting an identical series at the same
    instant would otherwise collide and one point would read as the other's replay.
    """
    return _hash(
        record.series_key or "",
        user_id or "",
        record.series_start.isoformat() if record.series_start else "",
        record.timestamp.isoformat(),
    )


def map_metric_point(
    name: str,
    value: Any,
    temporality: str,
    start_timestamp: datetime | None,
    attrs: dict[str, Any],
    *,
    timestamp: datetime,
    source: str,
    user_id: str | None,
) -> TelemetryRecord | None:
    """Map one OTLP metric data point onto a metric row, or None to skip it.

    Only the four outcome counters are recorded. Everything else is skipped: the
    metrics that duplicate an already-captured signal by name, and any other
    metric generically, so a newer agent version emitting names this does not know
    never breaks reception. The point is kept as reported (no delta conversion at
    ingest) and content-free: its attributes are folded into ``series_key``, not
    stored.
    """
    if name not in _METRICS:
        return None
    number = _bounded_number(value)
    if number is None:
        return None
    provisional = TelemetryRecord(
        name=name,
        timestamp=_timestamp(timestamp),
        source=source,
        dedup_key="",
        session_label=_session(attrs),
        kind="metric",
        value=number,
        temporality=CUMULATIVE if temporality == CUMULATIVE else DELTA,
        series_start=_timestamp(start_timestamp) if start_timestamp is not None else None,
        series_key=metric_series_key(name, attrs),
    )
    return TelemetryRecord(**{**provisional.__dict__, "dedup_key": metric_dedup_key(provisional, user_id)})


def series_point_increments(
    points: list[tuple[datetime, float]], temporality: str
) -> list[tuple[datetime, float]]:
    """Each point's own contribution to its series generation, in time order.

    The caller splits by generation first: pass the points of a single
    ``(series_key, series_start)`` pair. A counter reset arrives as a new
    ``series_start``, so diffing across one here would subtract the pre-reset
    total and report a negative increment.

    A ``delta`` series' points are already increments. A ``cumulative`` series
    carries running totals, so the growth between two readings is attributed to
    the later one, which is what makes a re-reported total add nothing and lets a
    caller bucket the increments by time.
    """
    if not points:
        return []
    ordered = sorted(points, key=lambda point: point[0])
    if temporality != CUMULATIVE:
        return ordered
    return [
        (later_time, max(later - earlier, 0.0))
        for (_earlier_time, earlier), (later_time, later) in zip(ordered, ordered[1:])
    ]


def compute_series_increment(points: list[tuple[datetime, float]], temporality: str) -> float:
    """How much one series generation grew across the points given, in total."""
    return float(sum(increment for _, increment in series_point_increments(points, temporality)))





def _build_row(api_key: APIKey, user_id: str | None, record: TelemetryRecord) -> AgentTelemetry:
    row_fields = {k: v for k, v in record.__dict__.items() if k not in _DEDUP_ONLY_FIELDS}
    return AgentTelemetry(api_key_id=api_key.id, user_id=user_id, **row_fields)


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


async def _insert_same_source_batch(
    db: AsyncSession, source: str, rows: list[AgentTelemetry]
) -> IngestResult:
    """Insert a same-source batch, retrying only rows that don't collide.

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


async def ingest(
    db: AsyncSession,
    records: Iterable[TelemetryRecord],
    *,
    api_key: APIKey,
) -> IngestResult:
    """Persist telemetry rows, treating uniqueness collisions as replay duplicates.

    Batch-inserts by source (the unique constraint is ``(source, dedup_key)``)
    rather than one savepoint per record, so ingesting a large export issues a
    small, bounded number of database round trips.
    """
    records = list(records)
    if not records:
        return IngestResult()
    user_id = api_key.user_id
    # Same active-user gate the usage path applies: a soft-deleted user's exporter
    # can still hold a live key, and its events must be rejected, not stored.
    if not user_id or await get_active_user(db, user_id) is None:
        return IngestResult(rejected=len(records))

    # Drop repeats inside this export before touching the database, the same guard
    # external_usage_service applies with its seen_in_batch set. The stored projection
    # is lossy by design, so two records can collapse onto one dedup key; letting that
    # reach the insert fails the whole batch and drops it into the row-at-a-time
    # fallback, turning one bulk insert into one commit per record.
    by_source: dict[str, list[TelemetryRecord]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    duplicate = 0
    for record in records:
        identity = (record.source, record.dedup_key)
        if identity in seen:
            duplicate += 1
            continue
        seen.add(identity)
        by_source[record.source].append(record)

    accepted = 0
    try:
        for source, source_records in by_source.items():
            rows = [_build_row(api_key, user_id, record) for record in source_records]
            result = await _insert_same_source_batch(db, source, rows)
            accepted += result.accepted
            duplicate += result.duplicate
    except SQLAlchemyError:
        await db.rollback()
        raise
    return IngestResult(accepted=accepted, duplicate=duplicate)
