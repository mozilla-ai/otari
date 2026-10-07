"""The spans one request records while the gateway serves it.

One collector belongs to one request, as the tool tally does: built in the
preamble, handed to whatever runs part of the request, and read once when the
response has been sent. Every client-supplied string is projected before it is
kept (an identifier stays, anything else is dropped or replaced by a fixed
name), because the storage port refuses prose and a request must never fail
because its trace could not be recorded.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gateway.models.usage import UsageLog
from gateway.ports.trace_storage_port import SpanRecord, TraceWrite, is_identifier
from gateway.services.traces._content_reading import RequestContent, cap
from gateway.services.traces._identity import SessionRef
from gateway.services.traces._output_reading import MAX_OUTPUT_BYTES, dialect_of, read_output
from gateway.services.traces._turns import TurnFacts

# The ``usage_logs.status`` of an attempt a later candidate made up for.
_ABSORBED = "absorbed"
_SUCCESS = "success"


def identifier_or_none(value: object) -> str | None:
    """A client-supplied value as it may be stored, or None when it is not an identifier-shaped string."""
    return value if isinstance(value, str) and is_identifier(value) else None


def client_tool_span_id(call_id: str) -> str:
    """The span id of a tool the client ran: derived from its call id, never the call id itself.

    A call id is whatever the client sent, so it is hashed rather than stored;
    the same call answered again (a retried request) still lands on one span.
    """
    return "call-" + hashlib.sha256(call_id.encode()).hexdigest()[:32]


def _span_id() -> str:
    return uuid.uuid4().hex


def _status_class(status_code: int | None) -> str | None:
    return f"status_{status_code}" if status_code is not None else None


@dataclass
class RequestTrace:
    """Collects one request's spans, under the step span the request itself is.

    The step's id is the request's ``Otari-Request-ID``, so it is unique per
    request: two retries with the same history are two steps, each with its cost.
    A request that names its session joins that session's trace; one that names
    none is a trace of its own.
    """

    request_id: str
    workspace_id: uuid.UUID
    user_id: str | None
    api_key_id: str | None
    endpoint: str
    started_at: datetime
    max_spans: int
    session: SessionRef | None = None
    harness: str | None = None
    turn: TurnFacts | None = None
    spans: list[SpanRecord] = field(default_factory=list)
    dropped: int = 0
    # How much content this request's workspace keeps, and what this request
    # carried. Plaintext lives here, in memory, only until the trace is sealed.
    content_level: str = "off"
    request_content: RequestContent | None = field(default=None, repr=False)
    response_output: str = field(default="", repr=False)
    _response_body: bytearray = field(default_factory=bytearray, repr=False)
    content: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)
    # Model rounds a tool loop ran, waiting for the usage row that bills them all.
    rounds: list[tuple[datetime, datetime]] = field(default_factory=list)

    @property
    def keeps_tool_io(self) -> bool:
        return self.content_level in ("tool_io", "full")

    def _add(self, span: SpanRecord) -> None:
        if len(self.spans) >= self.max_spans:
            self.dropped += 1
            return
        self.spans.append(span)

    def record_llm_call(self, row: UsageLog) -> None:
        """One provider attempt, from the usage row settlement wrote for it.

        The row is the authority on what the attempt cost, so the span carries
        exactly what billing recorded: an absorbed attempt is an error a later
        candidate recovered, and a request's final attempt is ok or failed.
        """
        ended = row.timestamp
        started = ended - timedelta(milliseconds=row.latency_ms) if row.latency_ms is not None else None
        attributes: dict[str, str | int | float | bool] = {}
        if (policy := identifier_or_none(row.policy_name)) is not None:
            attributes["otari.routing.policy"] = policy
        if (reason := identifier_or_none(row.selection_reason)) is not None:
            attributes["otari.routing.selection_reason"] = reason
        if row.attempt_position is not None:
            attributes["otari.routing.attempt_position"] = row.attempt_position
        if row.status_code is not None:
            attributes["http.response.status_code"] = row.status_code
        self._add(
            SpanRecord(
                span_id=row.id[:64],
                parent_span_id=self.request_id,
                kind="llm",
                origin="gateway",
                name="chat",
                operation="chat",
                outcome="ok" if row.status == _SUCCESS else "error",
                recovered=row.status == _ABSORBED,
                error_class=None if row.status == _SUCCESS else (_status_class(row.status_code) or "provider_error"),
                start_time=started,
                end_time=ended,
                duration_ms=row.latency_ms,
                model=identifier_or_none(row.model),
                provider=identifier_or_none(row.provider),
                input_tokens=row.prompt_tokens,
                output_tokens=row.completion_tokens,
                cost_snapshot=row.cost if isinstance(row.cost, Decimal) else None,
                request_id=self.request_id,
                attributes=attributes,
            )
        )
        self._flush_rounds(row.id[:64])

    def keep_response_bytes(self, chunk: bytes) -> None:
        """Keep the response as it is sent, bounded, where the workspace keeps everything."""
        room = MAX_OUTPUT_BYTES - len(self._response_body)
        if self.content_level == "full" and room > 0:
            self._response_body.extend(chunk[:room])

    def read_response(self, *, is_stream: bool) -> None:
        """Turn the kept response into the step's output, and let the raw bytes go."""
        if self._response_body:
            self.response_output = read_output(
                dialect_of(self.endpoint), bytes(self._response_body), is_stream=is_stream
            )
            self._response_body.clear()

    def discard_content(self) -> None:
        """Drop every plaintext this trace held, once it is sealed or abandoned."""
        self.content.clear()
        self.response_output = ""
        self._response_body.clear()

    def record_model_round(self, *, started: datetime, ended: datetime) -> None:
        """One model call of a gateway tool loop, which bills all its rounds on one usage row."""
        self.rounds.append((started, ended))

    def _flush_rounds(self, billed_span_id: str) -> None:
        """Hang a tool loop's rounds under the span of the row that billed them.

        A loop of one round is that span already. A round carries no tokens or cost
        of its own, so totals still come from the billing row alone.
        """
        rounds, self.rounds = self.rounds, []
        if len(rounds) < 2:
            return
        for index, (started, ended) in enumerate(rounds, start=1):
            self._add(
                SpanRecord(
                    span_id=_span_id(),
                    parent_span_id=billed_span_id,
                    kind="llm",
                    origin="gateway",
                    name="chat",
                    operation="chat",
                    outcome="ok",
                    start_time=started,
                    end_time=ended,
                    duration_ms=_elapsed_ms(started, ended),
                    request_id=self.request_id,
                    attributes={"otari.tool_loop.round": index},
                )
            )

    def record_tool_call(
        self,
        *,
        tool_name: str,
        tool_type: str,
        started: datetime,
        ok: bool,
        arguments: str | None = None,
        result: str | None = None,
    ) -> None:
        """A tool the gateway ran itself (web search, web fetch, code execution, an MCP tool).

        ``arguments`` and ``result`` are kept only where the workspace captures tool content.
        """
        ended = datetime.now(UTC)
        name = identifier_or_none(tool_name)
        span_id = _span_id()
        if self.keeps_tool_io and (arguments or result):
            self.content[span_id] = {"arguments": cap(arguments or ""), "result": cap(result or "")}
        self._add(
            SpanRecord(
                span_id=span_id,
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
            opens_turn=self.turn.opens_turn if self.turn is not None else False,
            outcome="ok" if ok else "error",
            error_class=error_class,
            start_time=self.started_at,
            end_time=ended,
            duration_ms=_elapsed_ms(self.started_at, ended),
            request_id=self.request_id,
            attributes={"http.response.status_code": status_code} if status_code is not None else {},
        )
        if self.content_level == "full" and self.request_content is not None:
            step_content = {
                "input": self.request_content.input,
                "output": self.response_output,
                "prior_output": self.request_content.prior_output,
            }
            if any(step_content.values()):
                self.content[self.request_id] = step_content
        return TraceWrite(
            workspace_id=self.workspace_id,
            trace_id=self.session.trace_id(self.workspace_id, self.user_id, self.api_key_id)
            if self.session
            else self.request_id,
            user_id=self.user_id,
            api_key_id=self.api_key_id,
            session_source=self.session.source if self.session else "none",
            harness=self.harness,
            spans=(step, *self._client_tool_spans(), *self.spans),
        )

    def _client_tool_spans(self) -> list[SpanRecord]:
        """The tools the client ran since its previous request, closed by the results this one carries.

        Their start is the end of the step that asked for them, which is not this
        request, so it is left for the read model to fill. A span's id derives from
        its call id, so a retried request answering the same calls adds nothing
        twice. They count against the request's span budget like any other span.
        """
        if self.turn is None:
            return []
        room = max(0, self.max_spans - len(self.spans))
        answered = self.turn.answered[:room]
        self.dropped += len(self.turn.answered) - len(answered)
        tool_io = self.request_content.tool_calls if self.request_content and self.keeps_tool_io else {}
        spans: list[SpanRecord] = []
        for call in answered:
            name = identifier_or_none(call.name)
            span_id = client_tool_span_id(call.call_id)
            io = tool_io.get(call.call_id)
            if io is not None:
                self.content[span_id] = {"arguments": io.arguments, "result": io.result}
            spans.append(
                SpanRecord(
                    span_id=span_id,
                    kind="tool",
                    origin="gateway",
                    name=f"execute_tool:{name}" if name and is_identifier(f"execute_tool:{name}") else "execute_tool",
                    operation="execute_tool",
                    outcome="error" if call.is_error else "ok",
                    error_class="tool_error" if call.is_error else None,
                    end_time=self.started_at,
                    tool_name=name,
                    tool_type="client",
                    tool_call_id=identifier_or_none(call.call_id),
                    request_id=self.request_id,
                )
            )
        return spans


def _elapsed_ms(started: datetime, ended: datetime) -> int:
    return max(0, round((ended - started).total_seconds() * 1000))
