"""What a trace read returns: a session with its turns, as the traces service derives it.

A leaf type, so the traces service can build it and the response schemas can map
it without either importing the other.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from gateway.ports.trace_storage_port import SpanRecord, TraceSummary

TurnState = Literal["active", "completed", "failed", "incomplete"]
SessionState = Literal["active", "idle"]


@dataclass(frozen=True)
class TurnView:
    """One user prompt and the steps it caused."""

    index: int
    state: TurnState
    started_at: datetime | None
    ended_at: datetime | None
    step_ids: tuple[str, ...]
    tool_calls: int
    llm_calls: int
    errors: int
    input_tokens: int
    output_tokens: int
    cost: Decimal
    # True for the steps a session started with before any prompt was seen.
    continued: bool = False


@dataclass(frozen=True)
class TraceView:
    """A session as it is read: its totals, its turns and its spans."""

    summary: TraceSummary
    state: SessionState
    turns: tuple[TurnView, ...]
    spans: tuple[SpanRecord, ...]
    # Span ids whose start was inferred rather than measured.
    approximate: frozenset[str]
    truncated: bool
    # Span ids whose content was captured and may be read one at a time.
    content_span_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class TraceSettingsView:
    """A workspace's content capture: what it asked for, what it gets, and the deployment's limit."""

    content_capture: str
    effective: str
    ceiling: str
    admin_content_access: bool = False


@dataclass(frozen=True)
class ContentAccessView:
    """One recorded read of captured content."""

    accessed_at: datetime
    trace_id: str
    span_id: str
    reader_kind: str
    reader: str
    reason: str | None
