"""ORM tables for agent traces: one row per session, and the spans inside it.

A trace is one agent session. Its spans are a projection of what the gateway
observed or an instrumented agent sent, written best-effort after accounting, so
nothing here is a billing source: ``usage_logs`` stays authoritative, and the
cost on a span is a snapshot for display.

Every key starts with the workspace, so ids a client chose (an OTLP trace id, a
session label) can never reach another tenant's rows.
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, BigInteger, ForeignKey, ForeignKeyConstraint, Index, String, Uuid, false, func
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdCost

# The vocabulary lives with the records, so the records import nothing from here.
from gateway.types.traces import TRACE_ID_MAX_LENGTH


class Trace(Base):
    """One agent session, and the totals its spans add up to.

    The totals are what the session list reads, so it never scans spans. They grow
    only by spans whose insert actually inserted, so a replayed batch cannot count
    twice.
    """

    __tablename__ = "traces"
    __table_args__ = (
        Index("ix_traces_workspace_last_activity", "workspace_id", "last_activity_at"),
        Index("ix_traces_last_activity", "last_activity_at"),
        # Retention's age bound.
        Index("ix_traces_started_at", "started_at"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH), primary_key=True)
    # The trace's owner: the user whose span created it. Session ids hash the user
    # in, so a second user's spans never land here.
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), index=True)
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    session_source: Mapped[str] = mapped_column(String)
    harness: Mapped[str | None] = mapped_column(String)
    name: Mapped[str | None] = mapped_column(String)
    started_at: Mapped[datetime] = mapped_column(UtcDateTime())
    last_activity_at: Mapped[datetime] = mapped_column(UtcDateTime())
    step_count: Mapped[int] = mapped_column(default=0, server_default="0")
    span_count: Mapped[int] = mapped_column(default=0, server_default="0")
    # Spans that ended in an error a fallback did not recover.
    error_count: Mapped[int] = mapped_column(default=0, server_default="0")
    # A long session's sum outgrows int32; one span's count does not.
    input_tokens: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    cost_snapshot: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class TraceSpan(Base):
    """One span of a trace. Written once and never changed."""

    __tablename__ = "trace_spans"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "trace_id"],
            ["traces.workspace_id", "traces.trace_id"],
            ondelete="CASCADE",
            name="fk_trace_spans_trace",
        ),
        Index("ix_trace_spans_trace_start", "workspace_id", "trace_id", "start_time"),
        # Closes a client-run tool when a later request in the session answers it.
        Index("ix_trace_spans_trace_tool_call", "workspace_id", "trace_id", "tool_call_id"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH), primary_key=True)
    span_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH), primary_key=True)
    parent_span_id: Mapped[str | None] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    kind: Mapped[str] = mapped_column(String)
    origin: Mapped[str] = mapped_column(String)
    name: Mapped[str] = mapped_column(String)
    # ``gen_ai.operation.name``: chat, execute_tool, invoke_agent, ...
    operation: Mapped[str | None] = mapped_column(String)
    # Null for a client-run tool, whose start the requesting step's end stands in for.
    start_time: Mapped[datetime | None] = mapped_column(UtcDateTime())
    end_time: Mapped[datetime | None] = mapped_column(UtcDateTime())
    duration_ms: Mapped[int | None] = mapped_column()
    outcome: Mapped[str] = mapped_column(String)
    # A failed routing attempt a later candidate made up for.
    recovered: Mapped[bool] = mapped_column(default=False, server_default=false())
    error_class: Mapped[str | None] = mapped_column(String)
    # On a step: whether the request's last input was user text, which opens a turn.
    opens_turn: Mapped[bool] = mapped_column(default=False, server_default=false())
    model: Mapped[str | None] = mapped_column(String)
    provider: Mapped[str | None] = mapped_column(String)
    input_tokens: Mapped[int | None] = mapped_column()
    output_tokens: Mapped[int | None] = mapped_column()
    cost_snapshot: Mapped[Decimal | None] = mapped_column(UsdCost())
    tool_name: Mapped[str | None] = mapped_column(String)
    tool_type: Mapped[str | None] = mapped_column(String)
    tool_call_id: Mapped[str | None] = mapped_column(String)
    # The gateway's ``Otari-Request-ID``, the join to ``usage_logs.request_group_id``.
    request_id: Mapped[str | None] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    otel_trace_id: Mapped[str | None] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    otel_span_id: Mapped[str | None] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    # Allowlisted, typed attributes only (``types.traces.SPAN_ATTRIBUTES``).
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
