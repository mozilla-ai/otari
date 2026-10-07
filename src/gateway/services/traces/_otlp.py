"""Spans an instrumented agent sends to the OTLP receiver, projected into trace documents.

Every span is kept, not only the ones that carry token usage: an agent's own
spans (``invoke_agent``, ``execute_tool``) are what the gateway cannot see for
itself. A span's session is the conversation it names
(``gen_ai.conversation.id``), or its OTel trace when it names none.

Nothing is passed through verbatim. Each name and value is projected the way the
gateway's own spans are, so text an agent put in a span name or attribute never
reaches the store, and the original OTLP body is never kept. Numbers and times
are clamped to what the store holds, and a span that cannot be projected is
dropped and counted, never allowed to fail the export that carried it.
"""

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from gateway.log_config import logger
from gateway.ports.trace_storage_port import SpanRecord, TraceWrite, is_identifier
from gateway.services.traces._collector import identifier_or_none
from gateway.services.traces._identity import SessionRef
from gateway.services.traces._writer import SPANS_DROPPED

# ``gen_ai.operation.name`` values, by the span kind they are.
_LLM_OPERATIONS = frozenset({"chat", "text_completion", "generate_content", "embeddings"})
_AGENT_OPERATIONS = frozenset({"invoke_agent", "create_agent", "invoke_workflow", "plan"})
_TOOL_OPERATIONS = frozenset({"execute_tool"})

# Token counts and durations are stored as 32-bit integers.
_INT32_MAX = 2_147_483_647
# Digits a numeric string attribute may have: anything longer is past the clamp anyway.
_MAX_INT_DIGITS = 19
# How far ahead of the gateway's clock an exporter's clock may run.
_FUTURE_SKEW = timedelta(minutes=5)

# Sessions one export may write. Each is a trace row of its own, so the span cap
# alone does not bound how many traces an export creates.
MAX_TRACES_PER_EXPORT = 1_000


@dataclass(frozen=True)
class OtlpSpan:
    """One received span, read out of its protobuf by the receiver."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: datetime | None
    end_time: datetime | None
    failed: bool
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class OtlpProjection:
    """The trace documents an export projects to, and how many of its spans were dropped."""

    writes: tuple[TraceWrite, ...]
    dropped: int = 0


@dataclass(frozen=True)
class _Window:
    floor: datetime
    ceiling: datetime

    def clamp(self, moment: datetime | None) -> datetime | None:
        if moment is None:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return min(max(moment, self.floor), self.ceiling)


def _str(attributes: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _int(attributes: Mapping[str, Any], key: str) -> int | None:
    """A non-negative count, clamped to what the store holds, or None when the value is not one."""
    value = attributes.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        # ``isdigit`` alone accepts non-ASCII digits such as "²", which ``int`` refuses.
        if not (value.isascii() and value.isdigit() and len(value) <= _MAX_INT_DIGITS):
            return None
        try:
            value = int(value)
        except ValueError:
            return None
    if not isinstance(value, int) or value < 0:
        return None
    return min(value, _INT32_MAX)


def _kind(operation: str | None) -> str:
    if operation in _LLM_OPERATIONS:
        return "llm"
    if operation in _TOOL_OPERATIONS:
        return "tool"
    if operation in _AGENT_OPERATIONS:
        return "agent"
    return "other"


def _session(span: OtlpSpan) -> SessionRef:
    key = _str(span.attributes, "gen_ai.conversation.id", "otari.session_label") or span.trace_id
    return SessionRef(source="otlp", key=key)


def _span_id(otel_trace_id: str, otel_span_id: str) -> str:
    """A received span's id: unique per OTel trace, since spans of several share one session.

    Hashed rather than concatenated, so it stays identifier-shaped and within the
    id length whatever the exporter sent, and apart from the gateway's own ids.
    """
    digest = hashlib.sha256(f"{otel_trace_id}\x00{otel_span_id}".encode()).hexdigest()[:32]
    return f"otlp-{digest}"


def _record(span: OtlpSpan, window: _Window) -> SpanRecord:
    attributes = span.attributes
    operation = identifier_or_none(_str(attributes, "gen_ai.operation.name"))
    tool_name = identifier_or_none(_str(attributes, "gen_ai.tool.name"))
    kind = _kind(operation)
    if is_identifier(span.name):
        name = span.name
    elif kind == "tool" and tool_name and is_identifier(f"execute_tool:{tool_name}"):
        name = f"execute_tool:{tool_name}"
    else:
        name = operation or "span"
    error_type = identifier_or_none(_str(attributes, "error.type"))
    start = window.clamp(span.start_time)
    end = window.clamp(span.end_time)
    if start is not None and end is not None and end < start:
        end = start
    duration = (
        min(round((end - start).total_seconds() * 1000), _INT32_MAX) if start is not None and end is not None else None
    )
    return SpanRecord(
        span_id=_span_id(span.trace_id, span.span_id),
        parent_span_id=_span_id(span.trace_id, span.parent_span_id) if span.parent_span_id else None,
        kind=kind,
        origin="otlp",
        name=name,
        operation=operation,
        outcome="error" if span.failed else "ok",
        error_class=(error_type or "error") if span.failed else None,
        start_time=start,
        end_time=end,
        duration_ms=duration,
        model=identifier_or_none(_str(attributes, "gen_ai.response.model", "gen_ai.request.model")),
        provider=identifier_or_none(_str(attributes, "gen_ai.provider.name", "gen_ai.system")),
        input_tokens=_int(attributes, "gen_ai.usage.input_tokens"),
        output_tokens=_int(attributes, "gen_ai.usage.output_tokens"),
        tool_name=tool_name,
        tool_type=identifier_or_none(_str(attributes, "gen_ai.tool.type")),
        tool_call_id=identifier_or_none(_str(attributes, "gen_ai.tool.call.id")),
        otel_trace_id=span.trace_id[:64],
        otel_span_id=span.span_id[:64],
        attributes={"error.type": error_type} if span.failed and error_type else {},
    )


def _project(
    spans: list[OtlpSpan],
    *,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
    window: _Window,
    max_traces: int,
) -> OtlpProjection:
    by_session: dict[SessionRef, list[SpanRecord]] = {}
    harness: dict[SessionRef, str | None] = {}
    invalid = 0
    over_cap = 0
    for span in spans:
        if not span.trace_id or not span.span_id:
            continue
        try:
            session = _session(span)
            if session not in by_session and len(by_session) >= max_traces:
                over_cap += 1
                continue
            record = _record(span, window)
        except Exception as exc:  # noqa: BLE001 - one bad span is dropped, never the export
            # The class only: an exception's message may quote what the exporter sent.
            logger.debug("OTLP span not recorded: %s", type(exc).__name__)
            invalid += 1
            continue
        by_session.setdefault(session, []).append(record)
        harness.setdefault(session, identifier_or_none(_str(span.attributes, "otari.client_name")))
    if invalid:
        SPANS_DROPPED.labels(reason="invalid").inc(invalid)
    if over_cap:
        SPANS_DROPPED.labels(reason="truncated").inc(over_cap)
    writes = tuple(
        TraceWrite(
            workspace_id=workspace_id,
            trace_id=session.trace_id(workspace_id, user_id, api_key_id),
            user_id=user_id,
            api_key_id=api_key_id,
            session_source="otlp",
            harness=harness[session],
            spans=tuple(records),
        )
        for session, records in by_session.items()
    )
    return OtlpProjection(writes=writes, dropped=invalid + over_cap)


def project_otlp_spans(
    spans: list[OtlpSpan],
    *,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
    retention: timedelta,
    now: datetime | None = None,
    max_traces: int = MAX_TRACES_PER_EXPORT,
) -> OtlpProjection:
    """Group received spans by session and project each one, attributed to the exporting key.

    Total: a span that cannot be projected is dropped and counted, and so is every
    span past the export's ``max_traces``-th session. Times are clamped to the
    window a trace can live in, from ``retention`` ago to a few minutes from ``now``.
    """
    now = now or datetime.now(UTC)
    window = _Window(floor=now - retention, ceiling=now + _FUTURE_SKEW)
    try:
        return _project(
            spans,
            workspace_id=workspace_id,
            user_id=user_id,
            api_key_id=api_key_id,
            window=window,
            max_traces=max_traces,
        )
    except Exception as exc:  # noqa: BLE001 - recording spans never fails the export
        logger.warning("OTLP spans not recorded: %s", type(exc).__name__)
        SPANS_DROPPED.labels(reason="invalid").inc(len(spans))
        return OtlpProjection(writes=(), dropped=len(spans))
