"""Content-free coding-agent telemetry mapping, ingestion, and aggregation helpers."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.entities import AgentTelemetry, APIKey, User

_MAX_NUMBER = 1_000_000_000
_EVENTS = {"tool_result", "tool_decision", "user_prompt", "api_error"}


@dataclass(frozen=True)
class TelemetryRecord:
    kind: str
    name: str
    timestamp: datetime
    source: str
    dedup_key: str
    value: float | None = None
    temporality: str | None = None
    series_start: datetime | None = None
    series_key: str | None = None
    tool_name: str | None = None
    decision: str | None = None
    success: bool | None = None
    duration_ms: int | None = None
    status_code: int | None = None
    prompt_length: int | None = None
    session_label: str | None = None


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
        record.duration_ms or "",
        record.status_code or "",
        record.prompt_length or "",
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
    fields: dict[str, Any] = {}
    if name == "tool_decision":
        fields = {"tool_name": tool_name, "decision": decision if decision in {"accept", "reject"} else None}
    elif name == "tool_result":
        success = attrs.get("success")
        fields = {
            "tool_name": tool_name,
            "success": success if isinstance(success, bool) else None,
            "duration_ms": _bounded_int(attrs.get("duration_ms")),
        }
    elif name == "api_error":
        fields = {"status_code": _bounded_int(attrs.get("status_code") or attrs.get("http.status_code"))}
    else:
        fields = {"prompt_length": _bounded_int(attrs.get("prompt_length"))}
    provisional = TelemetryRecord(
        kind="event",
        name=name,
        timestamp=timestamp,
        source=source,
        dedup_key="",
        session_label=_session(attrs),
        **fields,
    )
    return TelemetryRecord(**{**provisional.__dict__, "dedup_key": event_dedup_key(provisional, user_id)})


async def ingest(
    db: AsyncSession,
    records: Iterable[TelemetryRecord],
    *,
    api_key: APIKey,
) -> IngestResult:
    """Persist telemetry rows, treating uniqueness collisions as replay duplicates."""
    records = list(records)
    if not records:
        return IngestResult()
    user_id = api_key.user_id
    user_exists = user_id and (
        await db.execute(select(User.user_id).where(User.user_id == user_id))
    ).scalar_one_or_none()
    if not user_exists:
        return IngestResult(rejected=len(records))
    accepted = duplicate = 0
    try:
        for record in records:
            row = AgentTelemetry(api_key_id=api_key.id, user_id=user_id, **record.__dict__)
            try:
                async with db.begin_nested():
                    db.add(row)
                    await db.flush()
                accepted += 1
            except IntegrityError:
                duplicate += 1
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise
    return IngestResult(accepted=accepted, duplicate=duplicate)
