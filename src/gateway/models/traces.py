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

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    Uuid,
    false,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdCost

# The vocabulary lives with the records, so the records import nothing from here.
from gateway.types.traces import (
    BREAK_GLASS_REASON_MAX_LENGTH,
    BREAK_GLASS_REFERENCE_MAX_LENGTH,
    SHARE_NOTE_MAX_LENGTH,
    TRACE_ID_MAX_LENGTH,
)


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
        # Search matches a prefix of the trace id or of the name. A pattern-ops index
        # serves a prefix ``LIKE`` whatever the database's collation, and indexing both
        # columns lets PostgreSQL answer the search's ``OR`` from the two together.
        Index(
            "ix_traces_workspace_trace_id_pattern",
            "workspace_id",
            "trace_id",
            postgresql_ops={"trace_id": "text_pattern_ops"},
        ),
        Index("ix_traces_workspace_name", "workspace_id", "name", postgresql_ops={"name": "text_pattern_ops"}),
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
    # The ones among them that fail the session, as the trace service decides.
    failing_error_count: Mapped[int] = mapped_column(default=0, server_default="0")
    # A long session's sum outgrows int32; one span's count does not.
    input_tokens: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    output_tokens: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    cost_snapshot: Mapped[Decimal] = mapped_column(UsdCost(), default=Decimal(0), server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class TraceSpan(Base):
    """One span of a trace. Written once, and changed only by its content being deleted."""

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
        # Finds the session a request from the usage log was traced into.
        Index("ix_trace_spans_request_id", "request_id"),
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
    cache_read_tokens: Mapped[int | None] = mapped_column()
    cache_write_tokens: Mapped[int | None] = mapped_column()
    reasoning_tokens: Mapped[int | None] = mapped_column()
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
    # What the span was about, sealed under a per-session data key by ``services.traces._content``
    # (or, from before data keys, a bare OTARI_SECRET_KEY token); kept only where both the
    # deployment and the workspace keep content.
    content_sealed: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class TraceGuardrailDefinition(Base):
    """One version of an agent guardrail a coding agent's hook checked: its gates, and what each says.

    A gate span names the version it was checked against by its fingerprint, so a
    read can say what a gate is for without each check carrying it. The fingerprint
    is recomputed from the gates when one is stored, and a stored one never changes.
    """

    __tablename__ = "trace_guardrail_definitions"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    digest: Mapped[str] = mapped_column(String(64), primary_key=True)
    gates: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class TraceContentShare(Base):
    """A session's owner letting others read what it was about, until it is ended.

    Without one, a session's content opens only for its owner. The audience is a
    role, never a person: the organization's owners and admins, or the deployment's
    operator. Deleted with the session; the record of each read through it stays.
    """

    __tablename__ = "trace_content_shares"
    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "trace_id"],
            ["traces.workspace_id", "traces.trace_id"],
            ondelete="CASCADE",
            name="fk_trace_content_shares_trace",
        ),
        CheckConstraint(
            "audience IN ('organization_admins', 'deployment_operator')", name="check_trace_content_shares_audience"
        ),
        Index("ix_trace_content_shares_trace", "workspace_id", "trace_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    audience: Mapped[str] = mapped_column(String(32))
    # Why the owner shared it, in their words: shown to whoever reads through it.
    note: Mapped[str | None] = mapped_column(String(SHARE_NOTE_MAX_LENGTH))
    shared_by_user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("user.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
    # When it ended: its owner ended it, or lost the workspace.
    revoked_at: Mapped[datetime | None] = mapped_column(UtcDateTime())


class TraceContentRead(Base):
    """One read of a session's content through a share, which the session's owner can see.

    Names the session itself, with no foreign key to it, so the record outlives the
    session and its share: an access record lasts longer than what it describes. An
    erasure clears whose session it was.
    """

    __tablename__ = "trace_content_reads"
    __table_args__ = (
        # The newest reads of a share, and whether a reader has one this hour.
        Index("ix_trace_content_reads_share_read_at", "share_id", "read_at"),
        Index("ix_trace_content_reads_session_user_id", "session_user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Cleared when the share goes with its session; the columns below still say what was read.
    share_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("trace_content_shares.id", ondelete="SET NULL"), index=True
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    session_user_id: Mapped[str | None] = mapped_column(String)
    audience: Mapped[str] = mapped_column(String(32))
    reader_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="SET NULL"), index=True
    )
    # Kept as it was at the read, so the record still says who read once they are gone.
    reader_email: Mapped[str] = mapped_column(String(320))
    read_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )


class TraceContentBarrier(Base):
    """Which generation of a workspace's content capture is current.

    Turning content off and deleting it both move the generation on, and a
    write stores content only when it was captured on the current one. The
    write and the move lock this row, so neither can land in the middle of the
    other on any replica.
    """

    __tablename__ = "trace_content_barriers"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="CASCADE"), primary_key=True
    )
    generation: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")


class TraceUserBarrier(Base):
    """When a user's traces were last erased, so no replica stores spans of theirs taken in before it.

    Erasure sets ``erased_at`` and deletes in one transaction, and a write locks
    the row of each user it stores for, so a batch another replica still held
    from before the erasure is dropped rather than stored after it.
    """

    __tablename__ = "trace_user_barriers"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    erased_at: Mapped[datetime | None] = mapped_column(UtcDateTime())


class TraceBreakGlassRead(Base):
    """One opening of a session's content without its owner's share, and why.

    Written before the content is opened, and changed only by an erasure clearing
    whose session it was. Kept apart from the session and the organization, with no
    foreign key to either, so the record outlives both: an organization cannot
    delete the record of the deployment's access to it.
    """

    __tablename__ = "trace_break_glass_reads"
    __table_args__ = (
        CheckConstraint("kind IN ('organization', 'deployment')", name="check_trace_break_glass_reads_kind"),
        Index("ix_trace_break_glass_reads_organization_read_at", "organization_id", "read_at"),
        Index("ix_trace_break_glass_reads_read_at", "read_at"),
        Index("ix_trace_break_glass_reads_session_user_id", "session_user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # ``organization`` for an organization owner, ``deployment`` for the deployment's operator.
    kind: Mapped[str] = mapped_column(String(16))
    organization_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    workspace_id: Mapped[uuid.UUID] = mapped_column(Uuid)
    trace_id: Mapped[str] = mapped_column(String(TRACE_ID_MAX_LENGTH))
    session_user_id: Mapped[str | None] = mapped_column(String)
    reader_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("user.id", ondelete="SET NULL"), index=True
    )
    reader_email: Mapped[str] = mapped_column(String(320))
    reason: Mapped[str] = mapped_column(String(BREAK_GLASS_REASON_MAX_LENGTH))
    reference: Mapped[str | None] = mapped_column(String(BREAK_GLASS_REFERENCE_MAX_LENGTH))
    read_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=lambda: datetime.now(UTC), server_default=func.now()
    )
