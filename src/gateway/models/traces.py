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
from typing import Any, Literal

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    LargeBinary,
    String,
    Uuid,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdCost

# The longest id the tables hold: a gateway request id (a UUID, 36), an OTel
# trace id (32 hex) or span id (16 hex), and the hashes derived from them.
TRACE_ID_MAX_LENGTH = 64

SpanKind = Literal["turn", "step", "llm", "tool", "routing_attempt", "guardrail", "mcp_connect", "agent", "other"]
SPAN_KINDS: tuple[SpanKind, ...] = (
    "turn",
    "step",
    "llm",
    "tool",
    "routing_attempt",
    "guardrail",
    "mcp_connect",
    "agent",
    "other",
)

# Where a span came from: measured by this gateway while serving a request, or sent
# by an instrumented agent to the OTLP receiver.
SpanOrigin = Literal["gateway", "otlp"]
SPAN_ORIGINS: tuple[SpanOrigin, ...] = ("gateway", "otlp")

# ``unknown`` is a client-run tool whose result no later request has carried yet.
SpanOutcome = Literal["ok", "error", "unknown"]
SPAN_OUTCOMES: tuple[SpanOutcome, ...] = ("ok", "error", "unknown")

# The signal that grouped a trace's requests into one session. ``none`` is a request
# that named no session, which is a trace of its own: the gateway never infers that
# two requests belong together.
# How much of a request's content a workspace keeps, in increasing order: nothing,
# tool arguments and results, or also the prompt and the model's output.
ContentCapture = Literal["off", "tool_io", "full"]
CONTENT_CAPTURE_LEVELS: tuple[ContentCapture, ...] = ("off", "tool_io", "full")

SessionSource = Literal["client", "harness", "otlp", "none"]
SESSION_SOURCES: tuple[SessionSource, ...] = ("client", "harness", "otlp", "none")


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
    # Allowlisted, typed attributes only (``ports.trace_storage_port.SPAN_ATTRIBUTES``).
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class TraceSpanContent(Base):
    """Where a span's captured content is kept: a reference into the object store.

    The content itself is sealed with its session's data key and stored through
    ``FileStoragePort`` at ``storage_ref``, so no content is held in the database.
    """

    __tablename__ = "trace_span_content"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "trace_id", "span_id"],
            ["trace_spans.workspace_id", "trace_spans.trace_id", "trace_spans.span_id"],
            ondelete="CASCADE",
            name="fk_trace_span_content_span",
        ),
        Index("ix_trace_span_content_created", "created_at"),
    )

    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH), primary_key=True)
    span_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH), primary_key=True)
    storage_ref: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class TraceContentKey(Base):
    """The data key that seals one session's content, stored only wrapped.

    No foreign key to the session: the key is minted when its first content is
    sealed, before the trace writer has stored the session itself.
    """

    __tablename__ = "trace_content_keys"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH), primary_key=True)
    # Which key-encryption backend wrapped it (``secret_box``, ``aws_kms:<key id>``).
    key_ref: Mapped[str] = mapped_column(String)
    wrapped: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now(), index=True
    )


# Who read a span's content: the session's own user, an organization admin the
# workspace lets read it, or a platform operator breaking glass.
ContentReaderKind = Literal["owner", "admin", "break_glass"]
CONTENT_READER_KINDS: tuple[ContentReaderKind, ...] = ("owner", "admin", "break_glass")


class TraceContentAccess(Base):
    """One read of captured content, kept as the record of who read what and why.

    No foreign keys: the record outlives the session, the workspace and the
    reader it names, because it exists to answer for them afterwards.
    """

    __tablename__ = "trace_content_access"
    __table_args__ = (
        CheckConstraint("reader_kind IN ('owner', 'admin', 'break_glass')", name="ck_trace_content_access_reader_kind"),
        Index("ix_trace_content_access_workspace_at", "workspace_id", "accessed_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    span_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    reader_kind: Mapped[str] = mapped_column(String(16))
    # ``user:<dashboard identity>`` or ``master_key``.
    reader: Mapped[str] = mapped_column(String(64))
    # Required for a break-glass read, absent otherwise.
    reason: Mapped[str | None] = mapped_column(String(500))
    accessed_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class WorkspaceTraceSettings(Base):
    """How much of its requests' content a workspace keeps. A workspace with no row keeps none."""

    __tablename__ = "workspace_trace_settings"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    content_capture: Mapped[str] = mapped_column(String, default="off", server_default="off")
    # Whether the organization's owners and admins may read this workspace's content.
    # Off by default: content is its session's own user's, and every admin read is recorded.
    admin_content_access: Mapped[bool] = mapped_column(default=False, server_default=false())
    # The dashboard identity that last changed it, kept as a record of who opted in.
    updated_by_user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
