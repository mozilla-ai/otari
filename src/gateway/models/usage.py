"""ORM tables for usage rows and coding-agent telemetry."""

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base
from gateway.models.money import UsdCost


class UsageLog(Base):
    """Usage log model for tracking API requests."""

    __tablename__ = "usage_logs"
    __table_args__ = (
        Index("ix_usage_logs_user_id_timestamp", "user_id", "timestamp"),
        # Supports the activity-log viewer's primary "show errors, newest-first"
        # query. status is low-cardinality; model is high-cardinality and left
        # unindexed on purpose.
        Index("ix_usage_logs_status_timestamp", "status", "timestamp"),
        # Supports the setup guide's two questions about one workspace: has any
        # request in it ever succeeded (oldest first), and what did the last one
        # do (newest first). Both filter a workspace, a source and a status and
        # then order by time, which the workspace-only and status-first indexes
        # above can each answer only halfway: on a deployment with real traffic
        # the guide would otherwise scan the workspace's rows on every dashboard
        # load, and where usage is imported as well most of those rows are the
        # wrong source anyway. Equality columns first, the ordering column last.
        Index(
            "ix_usage_logs_workspace_source_status_timestamp",
            "workspace_id",
            "source",
            "status",
            "timestamp",
        ),
        # Idempotency for imported usage: re-submitting the same (source,
        # source_event_id) must not create a second row. Gateway-originated rows
        # keep source_event_id NULL, and SQL treats NULLs as distinct on both
        # SQLite and Postgres, so many (gateway, NULL) rows coexist freely.
        UniqueConstraint("source", "source_event_id", name="uq_usage_logs_source_event"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    # The workspace this row belongs to; see `APIKey.workspace_id` for why.
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("workspace.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)

    model: Mapped[str] = mapped_column()
    provider: Mapped[str | None] = mapped_column()
    endpoint: Mapped[str] = mapped_column()

    # Provenance. "gateway" for requests Otari served itself; a source slug (e.g.
    # "claude_code") for usage imported through POST /v1/usage/external-events. A row
    # backfilled from hosted history keeps its origin's slug behind a legacy prefix
    # ("otari-ai:gateway", "otari-ai:claude_code"), so asking whether this deployment
    # served a row means asking about the slug behind that prefix: core/usage_source.
    # source_event_id is the upstream event id used for idempotent import (NULL for
    # gateway rows); source_label carries optional session/project attribution.
    source: Mapped[str] = mapped_column(default="gateway", index=True)
    source_event_id: Mapped[str | None] = mapped_column()
    source_label: Mapped[str | None] = mapped_column()
    # Whether this row's cost participates in budget enforcement. True for normal
    # gateway rows; false for imported usage and for rows from keys flagged
    # exclude_from_budget. False rows are recorded (and appear in cost analytics)
    # but their cost is never written to User.spend.
    counts_toward_budget: Mapped[bool] = mapped_column(default=True)

    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    cache_read_tokens: Mapped[int | None] = mapped_column()
    cache_write_tokens: Mapped[int | None] = mapped_column()
    cache_write_1h_tokens: Mapped[int | None] = mapped_column()
    # Which cached-token convention the counts above were reported under: True
    # when the cache buckets are already inside ``prompt_tokens`` (OpenAI shape),
    # False when they are additive to it (Anthropic / Claude Code shape). Written
    # by settlement from ``GatewayUsage.cache_tokens_in_prompt`` and by the
    # external-usage ingest from the value the submitter sent, so a row can be
    # repriced under the convention it was recorded with rather than one inferred
    # from the numbers, which cannot tell the two apart.
    #
    # Nullable, and deliberately not defaulted: "not recorded" and "inclusive" are
    # different answers. Rows written before this column existed are NULL, and
    # repricing falls back to recovering the convention from ``billing_meters``
    # for exactly those (see ``usage_admin_service._row_cache_tokens_included``).
    # A default would make every historical row claim a convention nothing
    # checked, and mis-price the half that were the other one.
    cache_tokens_in_prompt: Mapped[bool | None] = mapped_column()
    billing_meters: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    pricing_breakdown: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    # The settled amount, and the accounting truth for this row
    # (mozilla-ai/otari-ai#1751). Exact to the micro-dollar; see
    # ``models/money.py`` for what that costs on each engine.
    cost: Mapped[Decimal | None] = mapped_column(UsdCost())

    # Why ``cost`` is the amount it is, which the row cannot re-derive on its own:
    # ``pricing_source`` names the price list that settled it ("organization",
    # "managed", "genai_prices"), ``pricing_reference`` identifies the entry in it
    # (a pricing row's id, or a ``provider:model`` key), ``pricing_effective_at``
    # is when that rate took effect, and ``pricing_version`` pins the revision of
    # the list. ``calculated_at`` is when the amount was priced, which is not
    # ``timestamp`` (when the request ran): usage settled or repriced later moves
    # the two apart.
    #
    # All nullable with no backfill. The gateway's own settlement does not record
    # provenance, so these are written by the hosted-usage backfill
    # (mozilla-ai/otari-ai#1798) from the platform's ``gateway_usage_settlement``
    # row, and null reads correctly as "not recorded". The lengths mirror that
    # table's columns rather than this file's usual unbounded strings, so a value
    # copied across always fits.
    #
    # ``pricing_source`` speaks the platform's settlement vocabulary, the values
    # ``_platform.SettledCost.pricing_source`` already carries on the hybrid wire
    # (echoed to callers as ``usage.pricing_source``). It is not the same field as
    # the one on a listed model in ``api/routes/models.py`` ("configured",
    # "default", "dynamic", "none"), which says where a price list entry came from
    # in this deployment rather than what settled one row's amount.
    pricing_source: Mapped[str | None] = mapped_column(String(32))
    pricing_reference: Mapped[str | None] = mapped_column(String(511))
    pricing_effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pricing_version: Mapped[str | None] = mapped_column(String(255))
    calculated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # "success", "error", or "absorbed". ``absorbed`` is a failed attempt that a
    # routing policy recovered from by trying the next candidate: the request
    # itself succeeded (or failed on a later attempt), so counting it as an error
    # would make a working fallback chain look like an outage. Every error metric
    # in the product counts ``status == "error"`` exactly, and ``request_count``
    # excludes absorbed rows, because a request that took two attempts is still one
    # request.
    status: Mapped[str] = mapped_column()
    error_message: Mapped[str | None] = mapped_column()

    # Routing attribution. All nullable: a request that named a plain model was not
    # routed through a policy, and null reads correctly as exactly that.
    #
    # `policy_name` is the name the caller sent. `selection_reason` says why this
    # candidate was chosen ("default", "condition:<keys>", "on_failure",
    # "router:<name>"). `attempt_position` and `attempt_count` locate the row in
    # the plan, so "served on attempt 2 of 3" is a query rather than a log grep.
    # `request_group_id` ties a request's rows together, which is what makes the
    # absorbed attempts findable from the row that served.
    policy_name: Mapped[str | None] = mapped_column(index=True)
    selection_reason: Mapped[str | None] = mapped_column()
    attempt_position: Mapped[int | None] = mapped_column()
    attempt_count: Mapped[int | None] = mapped_column()
    request_group_id: Mapped[str | None] = mapped_column(index=True)

    # HTTP status that classifies a failure, so failures can be grouped with a
    # GROUP BY instead of substring-matching provider-specific error prose. It is
    # the status the provider returned when it sent one (an upstream 401 stays
    # visible as a credential fault even though the caller sees the generic 502
    # that keeps gateway config out of the response), otherwise the gateway's own
    # rejection or classification code (402 missing pricing, 422 tool-loop cap,
    # 504 timeout, 502 unreachable). Nullable: historical rows predate the column,
    # a successful request has no failure to classify, and some failures carry no
    # HTTP status at all (e.g. a stream that ended without usage data).
    status_code: Mapped[int | None] = mapped_column()

    # Total server-side wall-clock for the request, in milliseconds. Nullable:
    # historical rows predate the column, and some write paths (batch jobs,
    # provider-never-reached rejections) have no meaningful request duration.
    latency_ms: Mapped[int | None] = mapped_column()

    # Milliseconds from request start to the first streamed chunk. Nullable:
    # non-streaming requests have no first chunk, historical rows predate the
    # column, and a stream that failed before yielding anything never reached one.
    #
    # ``started_at`` is taken in the handler preamble, so on a routing plan the
    # serving row's value also carries every earlier attempt's setup time.
    # Nothing in the column says so; a percentile keyed by the serving model
    # attributes failover time to the model that actually served.
    #
    # Hybrid (platform-fallback) streams never write this column at all: every
    # settlement callback in build_streaming_response returns before reaching
    # log_usage on that path, and run_streaming_with_fallback passes db=None.
    ttft_ms: Mapped[int | None] = mapped_column()

    api_key = relationship("APIKey", back_populates="usage_logs")
    user = relationship("User", back_populates="usage_logs")

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "api_key_id": self.api_key_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "model": self.model,
            "endpoint": self.endpoint,
            "source": self.source,
            "source_label": self.source_label,
            "counts_toward_budget": self.counts_toward_budget,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_write_1h_tokens": self.cache_write_1h_tokens,
            "cache_tokens_in_prompt": self.cache_tokens_in_prompt,
            "billing_meters": self.billing_meters,
            "pricing_breakdown": self.pricing_breakdown,
            "cost": self.cost,
            "status": self.status,
            "error_message": self.error_message,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "policy_name": self.policy_name,
            "selection_reason": self.selection_reason,
            "attempt_position": self.attempt_position,
            "attempt_count": self.attempt_count,
            "request_group_id": self.request_group_id,
        }


class AgentTelemetry(Base):
    """Content-free outcome metrics and behavioral events from coding agents."""

    __tablename__ = "agent_telemetry"
    __table_args__ = (
        UniqueConstraint("source", "dedup_key", name="uq_agent_telemetry_source_dedup"),
        Index("ix_agent_telemetry_user_id_timestamp", "user_id", "timestamp"),
        # Read-time cumulative-to-delta derivation orders one series' points by time.
        Index("ix_agent_telemetry_series_timestamp", "series_key", "timestamp"),
    )

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key_id: Mapped[str | None] = mapped_column(ForeignKey("api_keys.id", ondelete="SET NULL"), index=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id", ondelete="SET NULL"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True)
    name: Mapped[str] = mapped_column()
    tool_name: Mapped[str | None] = mapped_column()
    decision: Mapped[str | None] = mapped_column()
    success: Mapped[bool | None] = mapped_column()
    duration_ms: Mapped[int | None] = mapped_column()
    status_code: Mapped[int | None] = mapped_column()
    prompt_length: Mapped[int | None] = mapped_column()
    source: Mapped[str] = mapped_column(index=True)
    session_label: Mapped[str | None] = mapped_column()
    dedup_key: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Outcome-metric columns. Populated only on a metric row (``kind="metric"``),
    # NULL on a behavioral one, which is the inverse of the allow-list columns
    # above. ``value`` is stored exactly as OTLP reported it (a running total or
    # an increment, per ``temporality``); the read endpoints do the delta
    # arithmetic, so nothing is normalized at ingest. ``series_key`` is the pure
    # OTLP series identity (name plus attributes), which is what makes a
    # dimensioned metric two series rather than one.
    kind: Mapped[str | None] = mapped_column()
    value: Mapped[float | None] = mapped_column()
    temporality: Mapped[str | None] = mapped_column()
    series_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    series_key: Mapped[str | None] = mapped_column()
