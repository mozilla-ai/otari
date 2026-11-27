"""ORM tables for usage rows and coding-agent telemetry."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, UniqueConstraint, Uuid, event, false
from sqlalchemy.orm import Mapped, mapped_column, relationship

from gateway.models.base import Base, UtcDateTime
from gateway.models.money import UsdCost

# Mirrors ``gateway_usage_settlement.pricing_reference`` on the platform.
PRICING_REFERENCE_MAX_LENGTH = 511

# Who ended a stream before it completed.
StreamCutBy = Literal["provider", "client"]

# ``usage_logs.source`` on a row this gateway served itself. Any other value is usage
# imported through the external-events or OTLP ingest, which reserves this one so an
# import cannot pass as traffic served here.
SERVED_HERE_SLUG = "gateway"

# The endpoint label on a usage row the dashboard's own Playground produced. Its
# rows are still served here, so the label, not the source, tells them apart. An
# identifier, not a URL: it keeps its value if the route ever moves, exactly as
# ``chat.USAGE_ENDPOINT`` does.
PLAYGROUND_USAGE_ENDPOINT = "/v1/playground/chat/completions"

# The endpoint label on a usage row for every decisions path.
DECISIONS_ENDPOINT = "/v1/decisions"


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
    # "claude_code") for usage imported through POST /v1/usage/external-events.
    # source_event_id is the upstream event id used for idempotent import (NULL for
    # gateway rows); source_label carries optional session/project attribution:
    # the imported label, or on a gateway row the trace id of the session the
    # request named (see ``services/traces/_identity.py``).
    source: Mapped[str] = mapped_column(default=SERVED_HERE_SLUG, index=True)
    source_event_id: Mapped[str | None] = mapped_column()
    source_label: Mapped[str | None] = mapped_column()
    # Caller-supplied tags from the request's ``metadata`` (``purpose``, ``country``,
    # ...), bounded at intake; see ``api/routes/_request_tags.py``. Null when the
    # request sent none, and on every row written before the column existed.
    tags: Mapped[dict[str, str] | None] = mapped_column(JSON)
    # The end-user identifier the request's provider calls carry (see
    # ``services/inference/_upstream_user.py``), so an identifier a provider reports
    # in an abuse case can be traced back to its workspace, user and key. Indexed
    # for that lookup. Null where the request had none: imported usage, the
    # pass-through endpoints, a request refused before its user was resolved, and
    # every row written before the column existed.
    safety_identifier: Mapped[str | None] = mapped_column(String(64), index=True)
    # Whether this row's cost participates in budget enforcement. True for normal
    # gateway rows; false for imported usage and for rows from keys flagged
    # exclude_from_budget. False rows are recorded (and appear in cost analytics)
    # but their cost is never written to User.spend.
    counts_toward_budget: Mapped[bool] = mapped_column(default=True)
    # True when the request was served on a credential the deployment pays for,
    # one ``ModelProviderPort`` handed out, and so charged through ``BillingPort``
    # where an overlay binds one. False for every other row, and for every row
    # written before the column existed.
    deployment_paid: Mapped[bool] = mapped_column(default=False, server_default=false())
    # Where the hosted provider (or region) that served the attempt declared it
    # keeps request data, read when the row was written: a country code, ``EU``
    # or ``GLOBAL``. Null for a row the deployment did not serve, or one served
    # where nothing was declared.
    data_location: Mapped[str | None] = mapped_column(String(16), default=None)

    prompt_tokens: Mapped[int | None] = mapped_column()
    completion_tokens: Mapped[int | None] = mapped_column()
    total_tokens: Mapped[int | None] = mapped_column()
    cache_read_tokens: Mapped[int | None] = mapped_column()
    cache_write_tokens: Mapped[int | None] = mapped_column()
    cache_write_1h_tokens: Mapped[int | None] = mapped_column()
    reasoning_tokens: Mapped[int | None] = mapped_column()
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
    # The part of ``cost`` that is per-call tool charges (gateway-run tools and
    # searches the model's provider ran), so a caller can split a request's bill
    # into tokens and tools. Zero where no tool was charged; null where ``cost``
    # is, where looking up the tools' prices failed, and on every row written
    # before the column existed.
    tool_cost: Mapped[Decimal | None] = mapped_column(UsdCost())
    # The part of ``cost`` the deployment's account ran, which is what it charges the
    # organization for: the model's cost where ``deployment_paid``, plus each
    # gateway-run tool call it ran whoever paid for the model call. Null where
    # neither part has a cost, on rows written by a path that does not split it, and
    # on every row written before the column existed.
    deployment_paid_cost: Mapped[Decimal | None] = mapped_column(UsdCost())

    # Why ``cost`` is the amount it is, which the row cannot re-derive on its own:
    # ``pricing_source`` names the price list that settled it ("organization",
    # "managed", "defaults"; older rows read "genai_prices"), ``pricing_reference``
    # identifies the entry in it
    # (a pricing row's id, or a ``provider:model`` key), ``pricing_effective_at``
    # is when that rate took effect, and ``pricing_version`` pins the revision of
    # the list. ``calculated_at`` is when the amount was priced, which is not
    # ``timestamp`` (when the request ran): usage settled or repriced later moves
    # the two apart.
    #
    # All nullable with no backfill, so null reads as "not recorded". Rows this
    # gateway settles (``record_usage``) record them; rows the hosted-usage backfill
    # (mozilla-ai/otari-ai#1798) copies from the platform's
    # ``gateway_usage_settlement`` carry the platform's. The lengths mirror that
    # table's columns rather than this file's usual unbounded strings, so a value
    # copied across always fits.
    #
    # ``pricing_source`` therefore speaks one of two vocabularies: this gateway's
    # ``PriceSource`` ("organization", "deployment", "defaults"), the values it
    # already echoes to callers as ``usage.pricing_source``, or the platform's on a
    # backfilled row. Neither is the field on a listed model in
    # ``api/routes/models.py`` ("configured", "default", "dynamic", "none"), which
    # says where a price list entry came from rather than what settled one row's
    # amount. ``pricing_version`` is left null here: no price list this gateway
    # reads is versioned.
    pricing_source: Mapped[str | None] = mapped_column(String(32))
    pricing_reference: Mapped[str | None] = mapped_column(String(PRICING_REFERENCE_MAX_LENGTH))
    pricing_effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pricing_version: Mapped[str | None] = mapped_column(String(255))
    calculated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # "success", "error", "incomplete" (a stream the client abandoned, billed on an
    # estimate), or "absorbed". ``absorbed`` is a failed attempt that a
    # routing policy recovered from by trying the next candidate: the request
    # itself succeeded (or failed on a later attempt), so counting it as an error
    # would make a working fallback chain look like an outage. Every error metric
    # in the product counts ``status == "error"`` exactly, and ``request_count``
    # excludes absorbed rows, because a request that took two attempts is still one
    # request.
    status: Mapped[str] = mapped_column()
    error_message: Mapped[str | None] = mapped_column()
    # Who ended a stream before it completed: "provider" (the upstream dropped it or
    # failed after the first chunk) or "client" (the caller disconnected). Null for every other
    # row, and for rows written before the column existed: null reads as "not cut".
    # It lets a stream cut by the provider be listed for a refund or a report without
    # substring-matching ``error_message``. A provider-cut row is an "error" and may carry
    # estimated tokens and cost, billed under ``stream_missing_usage_policy`` "estimate".
    stream_cut_by: Mapped[StreamCutBy | None] = mapped_column(String(16))

    # Routing attribution. All nullable: a request that named a plain model was not
    # routed through a policy, and null reads correctly as exactly that.
    #
    # `policy_name` is the name the caller sent. `selection_reason` says why this
    # candidate was chosen ("default", "condition:<keys>", "on_failure",
    # "router:<name>", "catalog"). `attempt_position` and `attempt_count` locate the row in
    # the plan, so "served on attempt 2 of 3" is a query rather than a log grep.
    # `request_group_id` ties a request's rows together, which is what makes the
    # absorbed attempts findable from the row that served. The gateway writes the
    # `Otari-Request-ID` it sent the caller here on every row, routed or not, so a
    # caller can look a request's cost up by that id.
    policy_name: Mapped[str | None] = mapped_column(index=True)
    # The model name the caller sent, before an alias or a policy resolved it into
    # ``model``. Null on rows written before the column, and on rows that bill
    # something other than a model the caller named (a vision side-call, imported
    # usage).
    requested_model: Mapped[str | None] = mapped_column()
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

    # Provider-reported server-side compute time, in milliseconds, best-effort
    # (otari#337): prompt + generation time, excluding model load time where the
    # provider reports load separately (see _PROVIDER_LATENCY_FIELDS in
    # gateway.core.usage). Nullable: most providers report nothing here, and an
    # extraction failure must never block a write.
    provider_latency_ms: Mapped[int | None] = mapped_column()

    # Milliseconds from request start to the first streamed chunk. Nullable:
    # non-streaming requests have no first chunk, historical rows predate the
    # column, and a stream that failed before yielding anything never reached one.
    #
    # ``started_at`` is taken in the handler preamble, so on a routing plan the
    # serving row's value also carries every earlier attempt's setup time.
    # Nothing in the column says so; a percentile keyed by the serving model
    # attributes failover time to the model that actually served.
    ttft_ms: Mapped[int | None] = mapped_column()

    api_key = relationship("APIKey", back_populates="usage_logs")
    user = relationship("User", back_populates="usage_logs")


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


class AgentTelemetryUserBarrier(Base):
    """When a user's telemetry was last erased, so no export admitted before it is stored after it.

    Erasure sets ``erased_at`` and deletes in one transaction, and each write locks
    the row of the user it stores for, so an export another replica admitted before
    the erasure is refused rather than stored after it. The row goes with the user's
    own, after which the foreign key on ``agent_telemetry`` refuses their writes.
    """

    __tablename__ = "agent_telemetry_user_barriers"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    erased_at: Mapped[datetime | None] = mapped_column(UtcDateTime())


# Where a hosted provider (or region) name declares it keeps request data,
# registered by the composition root: the provider domain owns that answer.
_data_location_of: Callable[[str], str | None] | None = None


def configure_usage_data_location(resolver: Callable[[str], str | None] | None) -> None:
    """Register the lookup that stamps each deployment-served row with where it was served from."""
    global _data_location_of  # noqa: PLW0603
    _data_location_of = resolver


@event.listens_for(UsageLog, "init")
def _stamp_data_location(_row: UsageLog, _args: object, kwargs: dict[str, Any]) -> None:
    """Fill in where a deployment-served row was served from, as the row is built.

    Built in the request that served it, so the registered lookup answers what
    the serving credential declared then, and a later change to a declaration
    never reaches the row. A row that names its location keeps it.
    """
    if kwargs.get("deployment_paid") and kwargs.get("provider") and "data_location" not in kwargs:
        if _data_location_of is not None:
            kwargs["data_location"] = _data_location_of(kwargs["provider"])
