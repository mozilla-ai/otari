"""Agent traces as records: what is stored for a session, and what narrows a read.

A trace is one agent session; its spans are what the gateway observed while
serving it (each request, LLM round, routing attempt, guardrail check, MCP
connection and tool call) and what an instrumented agent sent to the OTLP
receiver.

The records carry the privacy rules, so nothing downstream can weaken them:

- **Projected documents only.** A span is a typed record. Its free-form part,
  ``attributes``, takes only the keys in :data:`SPAN_ATTRIBUTES`, each with a
  scalar value. An id must be id-shaped, and every other string a short
  identifier that does not look like an email address, a file path or a
  credential, so prose and the obvious kinds of personal data are refused. There
  is no field for a raw OTLP body.
- **Tenant first.** Every key starts with the workspace, and every read takes
  a :class:`TraceScope` the caller derived from the authenticated principal, so
  an id a client chose never reaches another tenant's rows.

``usage_logs`` stays the money path; the cost on a span is a display snapshot.
"""

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

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
SessionSource = Literal["client", "harness", "otlp", "none"]
SESSION_SOURCES: tuple[SessionSource, ...] = ("client", "harness", "otlp", "none")

# The shape every name a client could have chosen must have before it is stored:
# an identifier, never prose. 128 characters is room for any tool, model or
# provider name and for a namespaced identifier, and too little for a sentence to
# matter if one did fit the character set.
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,128}$")
# An id is narrower still: a hash, a UUID, OTel hex, or one of those with a prefix.
ID_PATTERN = re.compile(rf"^[A-Za-z0-9_.:-]{{1,{TRACE_ID_MAX_LENGTH}}}$")

# Values that fit the character set and are still a client's data rather than a
# name: an email address, a file path, or something shaped like a credential.
# A heuristic, not a guarantee; it errs toward refusing, and a refused name is
# simply not stored.
_NOT_A_NAME = (
    re.compile(r"[^@]+@[^@]+\.[A-Za-z]{2,}$"),
    re.compile(r"^[/~.]|//"),
    re.compile(r"/[^/]*\.[A-Za-z][A-Za-z0-9]{0,4}$"),
    re.compile(r"^(sk|pk|rk|tk|ak)[-_]|^(ghp|gho|ghs|ghu|github_pat|glpat|xox[abposr])[-_]|^AKIA[0-9A-Z]{8}|^AIza"),
    re.compile(r"(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{24,}"),
)

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
    """Whether a client-supplied name is safe to store as it is."""
    return IDENTIFIER_PATTERN.fullmatch(value) is not None and not any(pattern.search(value) for pattern in _NOT_A_NAME)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise InvalidSpanError(f"{name} must be one of {', '.join(choices)}")


def _check_id(name: str, value: str | None) -> None:
    if value is not None and ID_PATTERN.fullmatch(value) is None:
        raise InvalidSpanError(f"{name} must be an id of 1 to {TRACE_ID_MAX_LENGTH} characters")


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
    """The tenants a read may touch, derived from the authenticated caller.

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
    """What narrows a read, inside its scope.

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


# The grain a series is bucketed by. Bucket starts cross the seam as canonical UTC
# strings (``YYYY-MM-DDTHH:00:00Z``), so two adapters cannot disagree about a bucket.
TraceBucketGrain = Literal["hour", "day"]


@dataclass(frozen=True)
class TraceBucket:
    """How many traces started in one bucket, split by whether any of their spans failed unrecovered."""

    bucket: str
    succeeded: int
    failed: int


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
