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
  credential, so prose and the obvious kinds of personal data are refused. A
  gateway step's name, the endpoint label the gateway chose, has a shape of its
  own. There is no field for a raw OTLP body.
- **Tenant first.** Every key starts with the workspace, and every read takes
  a :class:`TraceScope` the caller derived from the authenticated principal, so
  an id a client chose never reaches another tenant's rows.

``usage_logs`` stays the money path; the cost on a span is a display snapshot.
"""

import re
import uuid
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Literal, Protocol

from otari_agent.domain.definitions import GateDefinition
from otari_agent.domain.policy import MAX_GATE_ID_LENGTH

# The longest id the tables hold: a gateway request id (a UUID, 36), an OTel
# trace id (32 hex) or span id (16 hex), and the hashes derived from them.
TRACE_ID_MAX_LENGTH = 64

# A trace's key: its workspace and its id.
TraceKey = tuple[uuid.UUID, str]

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
# ``blocked`` is work a guardrail stopped on purpose: the guardrail doing its job,
# so it is never counted as an error.
SpanOutcome = Literal["ok", "error", "unknown", "blocked"]
SPAN_OUTCOMES: tuple[SpanOutcome, ...] = ("ok", "error", "unknown", "blocked")

# The signal that grouped a trace's requests into one session. ``none`` is a request
# that named no session, which is a trace of its own: the gateway never infers that
# two requests belong together.
SessionSource = Literal["client", "harness", "otlp", "none"]
SESSION_SOURCES: tuple[SessionSource, ...] = ("client", "harness", "otlp", "none")

# The shape every name a client could have chosen must have before it is stored:
# an identifier, never prose. 128 characters is room for any tool, model or
# provider name and for a namespaced identifier, and too little for a sentence to
# matter if one did fit the character set.
IDENTIFIER_MAX_LENGTH = 128
IDENTIFIER_PATTERN = re.compile(rf"^[A-Za-z0-9_.:/@+-]{{1,{IDENTIFIER_MAX_LENGTH}}}$")
# An id is narrower still: a hash, a UUID, OTel hex, or one of those with a prefix.
ID_PATTERN = re.compile(rf"^[A-Za-z0-9_.:-]{{1,{TRACE_ID_MAX_LENGTH}}}$")
# A gateway step is named for the endpoint label that served it. The gateway chose
# that value, so it is held to this shape instead of the identifier rule, which
# refuses a leading slash as a client's file path.
ENDPOINT_PATTERN = re.compile(r"^/v[0-9]+(?:/[a-z_]+)+$")

# Values that fit the character set and are still a client's data rather than a
# name: an email address, a file path, or something shaped like a credential,
# including one with a line number or other suffix after it.
# A heuristic, not a guarantee; it errs toward refusing, and a refused name is
# simply not stored.
_NOT_A_NAME = (
    re.compile(r"[^@]+@[^@]+\.[A-Za-z]{2,}"),
    re.compile(r"^[/~.]|//"),
    re.compile(r"/[^/]*\.[A-Za-z][A-Za-z0-9]{0,4}(?=$|[:@])"),
    re.compile(
        r"^(?i:sk|pk|rk|tk|ak|api[-_]?key)[-_]|^(?i:ghp|gho|ghs|ghu|github_pat|glpat|xox[abposr])[-_]|^A[KS]IA[0-9A-Z]{8}|^AIza"
    ),
    re.compile(r"(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])(?=[A-Za-z0-9]*[0-9])[A-Za-z0-9]{24,}"),
)

# The attributes a span may carry beyond its typed fields, with the scalar type each
# takes. Anything a span needs to be queried by is a typed field instead.
SPAN_ATTRIBUTES: Mapping[str, type] = {
    "gen_ai.response.finish_reasons": str,
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
    "otari.llm.ttft_ms": int,
    # USD per million tokens a priced model request paid on each meter, as a decimal.
    "otari.llm.rate.input": str,
    "otari.llm.rate.output": str,
    "otari.llm.rate.cache_read": str,
    "otari.llm.rate.cache_write": str,
    # Characters per kind of input in a model request, and how many tool definitions and
    # messages it held. Sizes only, never text; the dashboard scales them to the exact input tokens.
    "otari.context.system_chars": int,
    "otari.context.tools_chars": int,
    "otari.context.user_chars": int,
    "otari.context.assistant_chars": int,
    "otari.context.tool_result_chars": int,
    "otari.context.file_chars": int,
    "otari.context.tool_count": int,
    "otari.context.message_count": int,
    "otari.tool.decision": str,
    "otari.tool.decision_source": str,
    "otari.tool.succeeded": bool,
    "otari.gate.id": str,
    "otari.gate.type": str,
    "otari.gate.origin": str,
    "otari.gate.outcome": str,
    "otari.gate.moment": str,
    "otari.gate.blocked": bool,
    "otari.policy.digest": str,
}
# The allowlisted attributes OTel defines as an array of strings. Each is stored as
# its elements joined by LIST_SEPARATOR, so every stored attribute stays a scalar,
# and each element is held to the identifier rule on its own.
LIST_ATTRIBUTES = frozenset({"gen_ai.response.finish_reasons"})
LIST_SEPARATOR = ","


# What a span may say it was about, when a deployment keeps content
# (``trace_content_enabled``). Prose, so never a typed field or an attribute,
# and never searched: shown on the span, and only there.
SPAN_CONTENT_KEYS = frozenset({"prompt", "command", "file_path", "gate.evidence", "gate.detail"})
SPAN_CONTENT_MAX_LENGTH = 4_000
# How long the note on a share of a session's content may be.
SHARE_NOTE_MAX_LENGTH = 500
# What opening a session's content without its owner's share has to say for itself.
BREAK_GLASS_REASON_MIN_LENGTH = 10
BREAK_GLASS_REASON_MAX_LENGTH = 1_000
BREAK_GLASS_REFERENCE_MAX_LENGTH = 200


class InvalidSpanError(ValueError):
    """A span broke the document contract: an unknown vocabulary value, an id too long,
    an attribute off the allowlist, or a string that is not identifier-shaped.

    Raised when the record is built, so nothing invalid reaches an adapter. The
    message names the field, never its value.
    """


def is_identifier(value: str) -> bool:
    """Whether a client-supplied name is safe to store as it is."""
    return IDENTIFIER_PATTERN.fullmatch(value) is not None and not any(pattern.search(value) for pattern in _NOT_A_NAME)


def is_gate_id(value: object) -> bool:
    """Whether a value is a gate id a guardrail could declare, which is held to that rule, not the identifier one."""
    return isinstance(value, str) and 0 < len(value) <= MAX_GATE_ID_LENGTH


def is_identifier_list(value: str) -> bool:
    """Whether a list attribute, joined for storage, is short and each of its elements an identifier."""
    return len(value) <= IDENTIFIER_MAX_LENGTH and all(is_identifier(part) for part in value.split(LIST_SEPARATOR))


def is_endpoint(value: str) -> bool:
    """Whether a name the gateway chose is shaped like one of its endpoint labels."""
    return len(value) <= IDENTIFIER_MAX_LENGTH and ENDPOINT_PATTERN.fullmatch(value) is not None


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise InvalidSpanError(f"{name} must be one of {', '.join(choices)}")


def _check_id(name: str, value: str | None) -> None:
    if value is not None and ID_PATTERN.fullmatch(value) is None:
        raise InvalidSpanError(f"{name} must be an id of 1 to {TRACE_ID_MAX_LENGTH} characters")


def _check_identifier(name: str, value: str | None) -> None:
    if value is not None and not is_identifier(value):
        raise InvalidSpanError(f"{name} must be identifier-shaped")


# The operation of a span that carries only what a model request cost, priced by
# the usage ledger after the request: a read folds it onto that request.
USAGE_OPERATION = "usage"
# A judge's further model is priced under ``<gate request id><separator><span id>``,
# so a read adds that cost to its gate's.
GATE_USAGE_SEPARATOR = ":"
# The meters a model request is priced on, each a rate attribute ``otari.llm.rate.<meter>``.
RATE_METERS = ("input", "output", "cache_read", "cache_write")


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
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost_snapshot: Decimal | None = None
    tool_name: str | None = None
    tool_type: str | None = None
    tool_call_id: str | None = None
    request_id: str | None = None
    otel_trace_id: str | None = None
    otel_span_id: str | None = None
    attributes: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    content: Mapping[str, str] = field(default_factory=dict)
    # The span holds content the reader may not open: it is private to the session's owner.
    content_private: bool = False

    def __post_init__(self) -> None:
        _check_choice("kind", self.kind, SPAN_KINDS)
        _check_choice("origin", self.origin, SPAN_ORIGINS)
        _check_choice("outcome", self.outcome, SPAN_OUTCOMES)
        for name in ("span_id", "parent_span_id", "request_id", "otel_trace_id", "otel_span_id", "tool_call_id"):
            _check_id(name, getattr(self, name))
        if not (self.origin == "gateway" and self.kind == "step" and is_endpoint(self.name)):
            _check_identifier("name", self.name)
        for name in (
            "operation",
            "error_class",
            "model",
            "provider",
            "tool_name",
            "tool_type",
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
            if key == "otari.gate.id":
                if not is_gate_id(value):
                    raise InvalidSpanError(f"attribute {key} must be a gate id")
            elif isinstance(value, str) and not (
                is_identifier_list(value) if key in LIST_ATTRIBUTES else is_identifier(value)
            ):
                raise InvalidSpanError(f"attribute {key} must be identifier-shaped")
        object.__setattr__(self, "attributes", MappingProxyType(attributes))
        content = dict(self.content)
        for key, value in content.items():
            if key not in SPAN_CONTENT_KEYS:
                raise InvalidSpanError("content may only hold its known keys")
            if not isinstance(value, str) or len(value) > SPAN_CONTENT_MAX_LENGTH:
                raise InvalidSpanError(
                    f"content {key} must be a string of at most {SPAN_CONTENT_MAX_LENGTH} characters"
                )
        object.__setattr__(self, "content", MappingProxyType(content))


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
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    """When the gateway took these spans in, which an erasure of their user since then outranks."""
    content_generation: int | None = None
    """The workspace's content generation when its spans' content was captured.

    Content is stored only while the workspace is still on that generation, so
    content queued before the workspace turned content off, or deleted it, is
    dropped however late it is written. None keeps no content.
    """

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


ShareAudience = Literal["organization_admins", "deployment_operator"]
SHARE_AUDIENCES: tuple[ShareAudience, ...] = ("organization_admins", "deployment_operator")
# How a read reached a session's content: as its owner, through a share, by
# break-glass (an organization owner or the deployment's operator, recorded), or not at all.
ContentAccess = Literal["owner", "share", "break_glass", "none"]
BreakGlassKind = Literal["organization", "deployment"]


@dataclass(frozen=True)
class ContentReader:
    """Who is reading a session's content, and which shares they may read it through."""

    user_id: uuid.UUID
    email: str
    audiences: frozenset[ShareAudience] = frozenset()
    """The shares this reader may read through: none for a member who manages nothing.

    More than one where a reader is both, such as an operator who also owns or
    administers the organization.
    """


class SessionOwners(Protocol):
    """Who still belongs where a session was kept, asked before content opens for anyone but its owner."""

    async def is_member(self, workspace_id: uuid.UUID, user_id: str | None) -> bool:
        """Whether ``user_id`` names a person who currently sees ``workspace_id``."""
        ...

    async def members(self, pairs: Collection[tuple[uuid.UUID, str | None]]) -> set[tuple[uuid.UUID, str | None]]:
        """Which of these (workspace, user) pairs :meth:`is_member` holds for, asked at once."""
        ...


@dataclass(frozen=True)
class TraceScope:
    """The tenants a read may touch, derived from the authenticated caller.

    Built through :meth:`deployment` or :meth:`workspaces`, never directly, so
    "every workspace" is always a deliberate choice: an empty workspace set
    matches nothing.

    A session's content (prompts, commands, file paths) opens for the user it
    belongs to and nobody else, managers and operators included, unless its owner
    shared it with them while still a member. Content is kept only for a session
    a current member owns, so no session's content belongs to nobody.
    ``all_content`` is for the deployment's own reads, never a person's.
    """

    workspace_ids: frozenset[uuid.UUID]
    deployment_wide: bool = False
    all_content: bool = False
    reader: ContentReader | None = None

    @classmethod
    def deployment(cls) -> "TraceScope":
        """Every workspace, with every session's content: for the deployment's own reads, never a person's."""
        return cls(workspace_ids=frozenset(), deployment_wide=True, all_content=True)

    @classmethod
    def workspaces(cls, workspace_ids: frozenset[uuid.UUID], *, reader: ContentReader | None = None) -> "TraceScope":
        """Only these workspaces, read by ``reader``; without one, no session's content opens."""
        return cls(workspace_ids=workspace_ids, reader=reader)

    def owns(self, user_id: str | None) -> bool:
        """Whether the reader is the user a session belongs to."""
        return self.reader is not None and user_id is not None and user_id == str(self.reader.user_id)

    def reads_content_of(self, user_id: str | None) -> bool:
        """Whether a read in this scope opens the content of a session ``user_id`` owns, without a share."""
        return self.all_content or self.owns(user_id)


@dataclass(frozen=True)
class TraceFilter:
    """What narrows a read, inside its scope.

    Every field narrows; an unset field does not filter. ``start`` and ``end``
    keep the traces active in ``[start, end)``: last active at or after ``start``
    and started before ``end``, so a session still running past ``end`` stays in.
    A naive bound means UTC. Several values in a tuple match any of them. ``workspace_ids``
    narrows inside the scope like every other field: a workspace outside the scope
    matches nothing, never widens it.
    """

    workspace_ids: tuple[uuid.UUID, ...] = ()
    start: datetime | None = None
    end: datetime | None = None
    user_ids: tuple[str, ...] = ()
    api_key_ids: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    harnesses: tuple[str, ...] = ()
    session_sources: tuple[str, ...] = ()
    has_error: bool | None = None
    # Matches any part of a trace's id, name, harness or user id, in any case.
    search: str | None = None


# What the session list can be ordered by. Latency is a session's span from its
# start to its last activity, and tokens are input and output together.
TraceSortKey = Literal["last_activity", "latency", "tokens", "cost", "errors"]


@dataclass(frozen=True)
class TraceSort:
    """The order a page of traces is read in. Ties fall back to newest activity first."""

    key: TraceSortKey = "last_activity"
    descending: bool = True


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
# strings (``YYYY-MM-DDTHH:MM:SSZ``), so two adapters cannot disagree about a bucket.
TraceBucketGrain = Literal["1m", "5m", "30m", "hour", "6h", "day"]
TRACE_BUCKET_DURATION: Mapping[str, timedelta] = MappingProxyType(
    {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "30m": timedelta(minutes=30),
        "hour": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "day": timedelta(days=1),
    }
)


@dataclass(frozen=True)
class TraceBucket:
    """How many sessions were active in one bucket, split by whether any of their spans failed unrecovered.

    A session is active in every bucket between its start and its last activity, so
    a long one counts in each bucket it ran through, not only in the one it began in.
    """

    bucket: str
    succeeded: int
    failed: int


@dataclass(frozen=True)
class TracePage:
    """One page of traces, in the order the read asked for."""

    items: tuple[TraceSummary, ...]
    has_more: bool


@dataclass(frozen=True)
class GuardrailDefinition:
    """One version of an agent guardrail, by the fingerprint its gate spans name."""

    digest: str
    gates: tuple[GateDefinition, ...]


@dataclass(frozen=True)
class TraceDetail:
    """A trace and its spans, in start order.

    ``truncated`` says the trace holds more spans than the read was allowed to
    return, so a caller never mistakes a cut-off trace for a short one.
    ``definitions`` are the guardrail versions its gate spans name, those stored.
    """

    summary: TraceSummary
    spans: tuple[SpanRecord, ...]
    truncated: bool
    definitions: tuple[GuardrailDefinition, ...] = ()
    content_access: ContentAccess = "none"
    content_share: "ContentShare | None" = None
    """The share the content was read through, where it was."""


@dataclass(frozen=True)
class BreakGlass:
    """Why a session's content is being opened without its owner's share, and by whom."""

    kind: BreakGlassKind
    organization_id: uuid.UUID
    reader_user_id: uuid.UUID | None
    reader_email: str
    reason: str
    reference: str | None = None


@dataclass(frozen=True)
class BreakGlassRead:
    """One recorded break-glass opening of a session's content."""

    id: uuid.UUID
    kind: BreakGlassKind
    organization_id: uuid.UUID
    workspace_id: uuid.UUID
    trace_id: str
    session_user_id: str | None
    reader_user_id: uuid.UUID | None
    reader_email: str
    reason: str
    reference: str | None
    read_at: datetime


@dataclass(frozen=True)
class ContentRead:
    """One read of a session's content through a share."""

    reader_user_id: uuid.UUID | None
    reader_email: str
    read_at: datetime


@dataclass(frozen=True)
class ContentShare:
    """A session's owner letting an audience read what it was about, and the reads made through it."""

    id: uuid.UUID
    audience: ShareAudience
    note: str | None
    created_at: datetime
    revoked_at: datetime | None
    reads: tuple[ContentRead, ...] = ()
    """Its newest reads; ``read_count`` says how many there are in all."""
    read_count: int = 0

    @property
    def active(self) -> bool:
        """Whether it still lets its audience read."""
        return self.revoked_at is None
