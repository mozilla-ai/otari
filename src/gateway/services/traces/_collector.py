"""The spans one request records while the gateway serves it.

One collector belongs to one request, as the tool tally does: built in the
preamble, handed to whatever runs part of the request, and read once when the
response has been sent. Every client-supplied string is projected before it is
kept (an identifier stays, anything else is dropped or replaced by a fixed
name), because the storage port refuses prose and a request must never fail
because its trace could not be recorded.
"""

import functools
import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.log_config import logger
from gateway.types.traces import SpanRecord, TraceWrite, is_identifier

# The settled status of an attempt a later candidate made up for, and of one that served.
_ABSORBED = "absorbed"
_SUCCESS = "success"


@dataclass(frozen=True)
class LlmCall:
    """One provider attempt as billing settled it: what the usage writer hands a trace.

    Plain values, so the trace never depends on how usage stores its rows.
    """

    attempt_id: str
    ended: datetime
    latency_ms: int | None
    status: str
    status_code: int | None
    model: str | None
    provider: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost: Decimal | None
    policy_name: str | None = None
    selection_reason: str | None = None
    attempt_position: int | None = None


def identifier_or_none(value: object) -> str | None:
    """A client-supplied value as it may be stored, or None when it is not an identifier-shaped string."""
    return value if isinstance(value, str) and is_identifier(value) else None


def client_tool_span_id(call_id: str) -> str:
    """The span id of a tool the client ran: derived from its call id, never the call id itself.

    A call id is whatever the client sent, so it is hashed rather than stored;
    the same call answered again (a retried request) still lands on one span.
    """
    return "call-" + hashlib.sha256(call_id.encode()).hexdigest()[:32]


def _never_raises[**P](record: Callable[P, None]) -> Callable[P, None]:
    """Keep a recording call from ever failing the request it records: a span it cannot build is dropped and counted."""

    @functools.wraps(record)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            record(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            trace = args[0]
            if isinstance(trace, RequestTrace):
                trace.dropped += 1
            logger.debug("Trace span not recorded (%s)", type(exc).__name__)

    return guarded


def _span_id() -> str:
    return uuid.uuid4().hex


def _status_class(status_code: int | None) -> str | None:
    return f"status_{status_code}" if status_code is not None else None


@dataclass
class RequestTrace:
    """Collects one request's spans, under the step span the request itself is.

    The step's id is the request's ``Otari-Request-ID``, so it is unique per
    request: two retries with the same history are two steps, each with its cost.
    Until a request names its session, it is a trace of its own.
    """

    request_id: str
    workspace_id: uuid.UUID
    user_id: str | None
    api_key_id: str | None
    endpoint: str
    started_at: datetime
    max_spans: int
    spans: list[SpanRecord] = field(default_factory=list)
    dropped: int = 0

    def _add(self, span: SpanRecord) -> None:
        if len(self.spans) >= self.max_spans:
            self.dropped += 1
            return
        self.spans.append(span)

    @_never_raises
    def record_llm_call(self, call: LlmCall) -> None:
        """One provider attempt, from what settlement recorded for it.

        Billing is the authority on what the attempt cost, so the span carries
        exactly that: an absorbed attempt is an error a later candidate
        recovered, and a request's final attempt is ok or failed.
        """
        ended = call.ended
        started = ended - timedelta(milliseconds=call.latency_ms) if call.latency_ms is not None else None
        attributes: dict[str, str | int | float | bool] = {}
        if (policy := identifier_or_none(call.policy_name)) is not None:
            attributes["otari.routing.policy"] = policy
        if (reason := identifier_or_none(call.selection_reason)) is not None:
            attributes["otari.routing.selection_reason"] = reason
        if call.attempt_position is not None:
            attributes["otari.routing.attempt_position"] = call.attempt_position
        if call.status_code is not None:
            attributes["http.response.status_code"] = call.status_code
        self._add(
            SpanRecord(
                span_id=call.attempt_id[:64],
                parent_span_id=self.request_id,
                kind="llm",
                origin="gateway",
                name="chat",
                operation="chat",
                outcome="ok" if call.status == _SUCCESS else "error",
                recovered=call.status == _ABSORBED,
                error_class=None if call.status == _SUCCESS else (_status_class(call.status_code) or "provider_error"),
                start_time=started,
                end_time=ended,
                duration_ms=call.latency_ms,
                model=identifier_or_none(call.model),
                provider=identifier_or_none(call.provider),
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                cost_snapshot=call.cost,
                request_id=self.request_id,
                attributes=attributes,
            )
        )

    @_never_raises
    def record_tool_call(self, *, tool_name: str, tool_type: str, started: datetime, ok: bool) -> None:
        """A tool the gateway ran itself (web search, web fetch, code execution, an MCP tool)."""
        ended = datetime.now(UTC)
        name = identifier_or_none(tool_name)
        self._add(
            SpanRecord(
                span_id=_span_id(),
                parent_span_id=self.request_id,
                kind="tool",
                origin="gateway",
                name=f"execute_tool:{name}" if name and is_identifier(f"execute_tool:{name}") else "execute_tool",
                operation="execute_tool",
                outcome="ok" if ok else "error",
                error_class=None if ok else "tool_error",
                start_time=started,
                end_time=ended,
                duration_ms=_elapsed_ms(started, ended),
                tool_name=name,
                tool_type=identifier_or_none(tool_type),
                request_id=self.request_id,
            )
        )

    @_never_raises
    def record_guardrail(
        self, *, profile: str, mode: str, valid: bool | None, started: datetime, ended: datetime
    ) -> None:
        """One input guardrail check. ``valid`` is None when the check could not be evaluated."""
        attributes: dict[str, str | int | float | bool] = {}
        if (safe_profile := identifier_or_none(profile)) is not None:
            attributes["otari.guardrail.profile"] = safe_profile
        if (safe_mode := identifier_or_none(mode)) is not None:
            attributes["otari.guardrail.mode"] = safe_mode
        self._add(
            SpanRecord(
                span_id=_span_id(),
                parent_span_id=self.request_id,
                kind="guardrail",
                origin="gateway",
                name="guardrail",
                outcome="ok" if valid else "error",
                error_class=None if valid else ("guardrail_unavailable" if valid is None else "guardrail_flagged"),
                start_time=started,
                end_time=ended,
                duration_ms=_elapsed_ms(started, ended),
                request_id=self.request_id,
                attributes=attributes,
            )
        )

    @_never_raises
    def record_mcp_connect(self, *, server: str, tool_count: int | None, started: datetime, ok: bool) -> None:
        """Connecting to one MCP server and listing its tools."""
        ended = datetime.now(UTC)
        attributes: dict[str, str | int | float | bool] = {}
        if (safe_server := identifier_or_none(server)) is not None:
            attributes["otari.mcp.server"] = safe_server
        if tool_count is not None:
            attributes["otari.mcp.tool_count"] = tool_count
        self._add(
            SpanRecord(
                span_id=_span_id(),
                parent_span_id=self.request_id,
                kind="mcp_connect",
                origin="gateway",
                name="mcp_connect",
                outcome="ok" if ok else "error",
                error_class=None if ok else "mcp_unreachable",
                start_time=started,
                end_time=ended,
                duration_ms=_elapsed_ms(started, ended),
                request_id=self.request_id,
                attributes=attributes,
            )
        )

    def finish(self, *, status_code: int | None, completed: bool) -> TraceWrite:
        """Close the request's step span and return everything it recorded, as one write.

        ``completed`` is False when the response was not sent in full (the client
        went away, or the stream ended early), which is an error the client saw
        whatever status line it got.
        """
        ended = datetime.now(UTC)
        # A stream fails after its 200 is sent, so the status line alone can read
        # as success; the request's own provider calls say how it settled.
        failed_call = next(
            (span for span in self.spans if span.kind == "llm" and span.outcome == "error" and not span.recovered),
            None,
        )
        ok = completed and status_code is not None and status_code < 400 and failed_call is None
        if ok:
            error_class = None
        elif not completed:
            error_class = "incomplete"
        elif failed_call is not None and status_code is not None and status_code < 400:
            error_class = failed_call.error_class or "provider_error"
        else:
            error_class = _status_class(status_code) or "unknown"
        step = SpanRecord(
            span_id=self.request_id,
            kind="step",
            origin="gateway",
            name=identifier_or_none(self.endpoint) or "request",
            outcome="ok" if ok else "error",
            error_class=error_class,
            start_time=self.started_at,
            end_time=ended,
            duration_ms=_elapsed_ms(self.started_at, ended),
            request_id=self.request_id,
            attributes={"http.response.status_code": status_code} if status_code is not None else {},
        )
        return TraceWrite(
            workspace_id=self.workspace_id,
            trace_id=self.request_id,
            user_id=self.user_id,
            api_key_id=self.api_key_id,
            session_source="none",
            spans=(step, *self.spans),
        )


def _elapsed_ms(started: datetime, ended: datetime) -> int:
    return max(0, round((ended - started).total_seconds() * 1000))
