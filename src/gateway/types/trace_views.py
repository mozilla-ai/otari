"""What a trace read returns: a session with its turns, as the traces service derives it.

A leaf type, so the traces service can build it and the response schemas can map
it without either importing the other.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from gateway.types.traces import SpanRecord, TraceSummary

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
    cache_read_tokens: int
    cost: Decimal
    # True for the steps a session started with before any prompt was seen.
    continued: bool = False


@dataclass(frozen=True)
class GateSummary:
    """How often one agent guardrail gate ran in a session, by what it decided.

    ``span_id`` is the check to open first: the earliest that did anything but
    pass, and None for a gate the guardrail declares that never ran. The type,
    enforcement and message come from the guardrail's definition, when stored.
    """

    gate_id: str
    passed: int
    blocked: int
    flagged: int
    refused: int
    undecided: int
    span_id: str | None
    gate_type: str | None = None
    enforcement: str | None = None
    message: str | None = None
    runs: tuple[str, ...] = ()
    origin: str | None = None


@dataclass(frozen=True)
class ToolCost:
    """An estimate of what one tool call cost, from the model requests either side of it.

    ``subagent`` is exact: what the requests of a subagent the call ran cost.
    """

    span_id: str
    calling: Decimal
    writing: Decimal
    rereading: Decimal
    result_tokens: int
    later_requests: int
    calls_sharing_request: int
    calls_sharing_result: int
    subagent: Decimal | None = None

    @property
    def total(self) -> Decimal:
        return self.calling + self.writing + self.rereading + (self.subagent or Decimal(0))


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
    gates: tuple[GateSummary, ...] = ()
    tool_costs: tuple[ToolCost, ...] = ()
