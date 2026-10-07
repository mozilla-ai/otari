"""Where agent traces are stored and read back.

A trace is one agent session; its spans are what the gateway observed while
serving it (each request, LLM round, routing attempt, guardrail check, MCP
connection and tool call) and what an instrumented agent sent to the OTLP
receiver. Otari's own storage is two tables in its database, enough for a
deployment reading its own traffic; a deployment holding many tenants' traces
wants a scale-out store instead, which is the second implementation this port
exists for (``ARCHITECTURE.md``, rule 7).

The contract carries the privacy rules, so no adapter can weaken them:

- **Projected documents only.** A span is a typed record. Its free-form part,
  ``attributes``, takes only the keys in :data:`SPAN_ATTRIBUTES`, each with a
  scalar value, and a string must be identifier-shaped, so text a client wrote
  cannot ride in on a span. There is no field for a raw OTLP body.
- **Tenant first.** Every key starts with the workspace, and every read and
  purge takes a :class:`TraceScope` the caller derived from the authenticated
  principal, so an id a client chose never reaches another tenant's rows.
- **Retention is an obligation.** :meth:`TraceStoragePort.expire` and
  :meth:`TraceStoragePort.purge_user` are part of the port, not adapter policy.

``usage_logs`` is not behind this port. Usage rows are the money path and stay
in the database in every build; the cost on a span is a display snapshot.

Writes settle before they return, as :class:`TelemetryStoragePort`'s do: an
adapter commits or flushes its own work, so a caller cannot roll one back and
never holds a transaction open across one. A write is idempotent on
``(workspace_id, trace_id, span_id)``: a span repeated is counted as a
duplicate, and a stored span is never changed.

Stability: this interface is not frozen while Otari is pre-1.0. Overlay authors
should pin a released tag and expect the shape to move.
"""

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol

from gateway.models.traces import (
    SESSION_SOURCES,
    SPAN_KINDS,
    SPAN_ORIGINS,
    SPAN_OUTCOMES,
    TRACE_ID_MAX_LENGTH,
)

# The shape every string a client could have chosen must have before it is stored:
# an identifier, never prose. 128 characters is room for any tool, model or
# provider name and for a namespaced identifier, and too little for a sentence to
# matter if one did fit the character set.
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,128}$")

# The attributes a span may carry beyond its typed fields, with the scalar type each
# takes. Anything a span needs to be queried by is a typed field instead.
SPAN_ATTRIBUTES: Mapping[str, type] = {
    "gen_ai.response.finish_reason": str,
    "gen_ai.request.max_tokens": int,
    "error.type": str,
    "http.response.status_code": int,
    "otari.routing.policy": str,
    "otari.routing.selection_reason": str,
    "otari.routing.attempt_position": int,
    "otari.guardrail.profile": str,
    "otari.guardrail.mode": str,
    "otari.mcp.server": str,
    "otari.mcp.tool_count": int,
    "otari.tool_loop.round": int,
    "otari.harness.version": str,
}


class InvalidSpanError(ValueError):
    """A span broke the document contract: an unknown vocabulary value, an id too long,
    an attribute off the allowlist, or a string that is not identifier-shaped.

    Raised when the record is built, so nothing invalid reaches an adapter. The
    message names the field, never its value.
    """


def is_identifier(value: str) -> bool:
    """Whether a client-supplied string is safe to store as it is."""
    return IDENTIFIER_PATTERN.fullmatch(value) is not None


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise InvalidSpanError(f"{name} must be one of {', '.join(choices)}")


def _check_id(name: str, value: str | None) -> None:
    if value is not None and not 0 < len(value) <= TRACE_ID_MAX_LENGTH:
        raise InvalidSpanError(f"{name} must be 1 to {TRACE_ID_MAX_LENGTH} characters")
    _check_identifier(name, value)


def _check_identifier(name: str, value: str | None) -> None:
    if value is not None and not is_identifier(value):
        raise InvalidSpanError(f"{name} must be identifier-shaped")


@dataclass(frozen=True)
class SpanRecord:
    """One span, projected and ready to store.

    Times are UTC-aware. ``start_time`` is None only for a client-run tool, whose
    start is the end of the step that requested it.
    """

    span_id: str
    kind: str
    origin: str
    name: str
    outcome: str
    parent_span_id: str | None = None
    operation: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: int | None = None
    recovered: bool = False
    error_class: str | None = None
    opens_turn: bool = False
    model: str | None = None
    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_snapshot: Decimal | None = None
    tool_name: str | None = None
    tool_type: str | None = None
    tool_call_id: str | None = None
    request_id: str | None = None
    otel_trace_id: str | None = None
    otel_span_id: str | None = None
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_choice("kind", self.kind, SPAN_KINDS)
        _check_choice("origin", self.origin, SPAN_ORIGINS)
        _check_choice("outcome", self.outcome, SPAN_OUTCOMES)
        for name in ("span_id", "parent_span_id", "request_id", "otel_trace_id", "otel_span_id"):
            _check_id(name, getattr(self, name))
        for name in (
            "name",
            "operation",
            "error_class",
            "model",
            "provider",
            "tool_name",
            "tool_type",
            "tool_call_id",
        ):
            _check_identifier(name, getattr(self, name))
        # Copied before it is checked and frozen after, so the mapping a caller still
        # holds cannot put an unchecked key into the record.
        attributes = dict(self.attributes)
        for key, value in attributes.items():
            expected = SPAN_ATTRIBUTES.get(key)
            if expected is None:
                raise InvalidSpanError("attributes may only hold allowlisted keys")
            if type(value) is not expected:
                raise InvalidSpanError(f"attribute {key} must be a {expected.__name__}")
            if isinstance(value, str) and not is_identifier(value):
                raise InvalidSpanError(f"attribute {key} must be identifier-shaped")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))


@dataclass(frozen=True)
class TraceWrite:
    """A batch of one trace's spans, with who they belong to.

    ``workspace_id``, ``user_id`` and ``api_key_id`` come from the authenticated
    principal, never from the request. The trace-level fields set the trace's
    summary when this write creates it, and are ignored when it already exists.
    """

    workspace_id: uuid.UUID
    trace_id: str
    user_id: str | None
    api_key_id: str | None
    session_source: str
    spans: tuple[SpanRecord, ...]
    harness: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _check_id("trace_id", self.trace_id)
        _check_choice("session_source", self.session_source, SESSION_SOURCES)
        _check_identifier("harness", self.harness)
        _check_identifier("name", self.name)


@dataclass(frozen=True)
class WriteResult:
    """How one write's spans were accounted for.

    ``duplicate`` is a success: a span already stored is not an error to resend.
    ``rejected`` counts spans refused because their trace belongs to another user.
    """

    accepted: int = 0
    duplicate: int = 0
    rejected: int = 0


@dataclass(frozen=True)
class TraceScope:
    """The tenants a read or a purge may touch, derived from the authenticated caller.

    Built through :meth:`deployment` or :meth:`workspaces`, never directly, so
    "every workspace" is always a deliberate choice: an empty workspace set
    matches nothing.
    """

    workspace_ids: frozenset[uuid.UUID]
    deployment_wide: bool = False

    @classmethod
    def deployment(cls) -> "TraceScope":
        """Every workspace: only for a deployment operator."""
        return cls(workspace_ids=frozenset(), deployment_wide=True)

    @classmethod
    def workspaces(cls, workspace_ids: frozenset[uuid.UUID]) -> "TraceScope":
        """Only these workspaces."""
        return cls(workspace_ids=workspace_ids)


@dataclass(frozen=True)
class TraceFilter:
    """What narrows a read or a purge, inside its scope.

    Every field narrows; an unset field does not filter. ``start`` is inclusive
    and ``end`` exclusive, and both apply to a trace's last activity. A naive
    bound means UTC. Several values in a tuple match any of them. ``workspace_ids``
    narrows inside the scope like every other field: a workspace outside the scope
    matches nothing, never widens it.
    """

    workspace_ids: tuple[uuid.UUID, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    user_ids: tuple[str, ...] = ()
    api_key_ids: tuple[str, ...] = ()
    harnesses: tuple[str, ...] = ()
    session_sources: tuple[str, ...] = ()
    has_error: bool | None = None
    trace_id_prefix: str | None = None


@dataclass(frozen=True)
class TraceSummary:
    """A trace's totals, as the session list shows them."""

    workspace_id: uuid.UUID
    trace_id: str
    user_id: str | None
    api_key_id: str | None
    session_source: str
    harness: str | None
    name: str | None
    started_at: datetime
    last_activity_at: datetime
    step_count: int
    span_count: int
    error_count: int
    input_tokens: int
    output_tokens: int
    cost_snapshot: Decimal


@dataclass(frozen=True)
class TracePage:
    """One page of traces, newest activity first."""

    items: tuple[TraceSummary, ...]
    has_more: bool


@dataclass(frozen=True)
class TraceDetail:
    """A trace and its spans, in start order.

    ``truncated`` says the trace holds more spans than the read was allowed to
    return, so a caller never mistakes a cut-off trace for a short one.
    """

    summary: TraceSummary
    spans: tuple[SpanRecord, ...]
    truncated: bool


class TraceStoragePort(Protocol):
    """Store agent traces and read them back, inside a tenant scope."""

    async def write(self, traces: tuple[TraceWrite, ...]) -> WriteResult:
        """Store these traces' spans, creating each trace on its first span.

        Settles before returning. Idempotent per span. A span whose trace already
        belongs to another user is rejected, never merged.
        """
        ...

    async def search(self, scope: TraceScope, filters: TraceFilter, *, limit: int, offset: int) -> TracePage:
        """Return one page of the scope's traces, newest activity first."""
        ...

    async def count(self, scope: TraceScope, filters: TraceFilter) -> int:
        """Return how many of the scope's traces match, sized exactly as a purge would be."""
        ...

    async def get(self, scope: TraceScope, trace_id: str, *, span_limit: int) -> TraceDetail | None:
        """Return one trace with up to ``span_limit`` spans, or None when the scope holds no such trace.

        A trace in another tenant's workspace reads as absent, never as forbidden.
        """
        ...

    async def purge(self, scope: TraceScope, filters: TraceFilter) -> int:
        """Delete the scope's matching traces and their spans. Settles; returns the traces deleted."""
        ...

    async def purge_user(self, user_id: str) -> int:
        """Delete every trace a user owns, in every workspace, for erasure. Settles."""
        ...

    async def expire(self, before: datetime) -> int:
        """Delete every trace whose last activity is before ``before``. Settles; returns the traces deleted."""
        ...
