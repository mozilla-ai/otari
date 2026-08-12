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
