"""What each tool call an agent made cost, estimated from the model requests around it.

A provider bills model requests, never tool calls, so a tool call's cost is read
off the requests either side of it, in the context it ran in:

- calling it: its share of the output of the request that asked for it, which
  also paid for any text or thinking written alongside;
- writing its result: the result enters the context on the next request, which
  is how much that request's context grew beyond the previous request's context
  and output, priced at what that request paid for new input (usually a cache
  write);
- re-reading it: every later request in the same context reads the result
  again, at its cache-read rate, or at its cache-write rate when its cache had
  expired, until the context is compacted.

Calls made by one request share its output equally, and calls whose results one
request carried share that growth equally: a request says how big its context
grew, not how much of it each result was. A subagent has a context of its own,
and the Agent call that ran it also carries what the subagent's own requests
cost, which is exact.

Only requests the usage ledger priced by the token carry the rates this needs,
so a session recorded before they were kept has no estimates.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from gateway.types.trace_views import ToolCost
from gateway.types.traces import SpanRecord

_MAIN = "main"
# A context that shrinks below this share of its largest size was compacted, so a
# result read before it is not read after.
_COMPACTED = Decimal("0.5")
_PER_MILLION = Decimal(1_000_000)
_EXECUTION = "tool.execution"


@dataclass(frozen=True)
class _Request:
    span: SpanRecord
    start: datetime
    rates: Mapping[str, Decimal]
    fresh: int
    output: int
    read: int
    written: int

    @property
    def context(self) -> int:
        return self.fresh + self.read + self.written

    def price(self, tokens: Decimal, meter: str) -> Decimal:
        return tokens * self.rates[meter] / _PER_MILLION

    def new_input_rate(self) -> Decimal:
        """What this request paid per new token of context: mostly a cache write."""
        new = self.fresh + self.written
        if new == 0:
            return self.rates["cache_write"]
        return (self.fresh * self.rates["input"] + self.written * self.rates["cache_write"]) / new

    def reread_rate(self) -> Decimal:
        """What this request paid per token it already had: a cache read, or a write again once the cache expired."""
        return self.rates["cache_read"] if self.read else self.rates["cache_write"]


def _rates(span: SpanRecord) -> dict[str, Decimal] | None:
    rates: dict[str, Decimal] = {}
    for meter in ("input", "output", "cache_read", "cache_write"):
        value = span.attributes.get(f"otari.llm.rate.{meter}")
        if not isinstance(value, str):
            return None
        try:
            rates[meter] = Decimal(value)
        except InvalidOperation:
            return None
    return rates


def _context_of(span: SpanRecord, by_id: Mapping[str, SpanRecord]) -> str:
    """The context a span ran in: a subagent's, under the run of the call that started it, or the main one."""
    parent = by_id.get(span.parent_span_id or "")
    return parent.span_id if parent is not None and parent.operation == _EXECUTION else _MAIN


def _requests(spans: Sequence[SpanRecord], by_id: Mapping[str, SpanRecord]) -> dict[str, list[_Request]]:
    """Each context's priced requests in order, on the model most of them used.

    A context's side requests (a title, a summary) run on another model with a
    context of their own, and would read as the context shrinking.
    """
    by_context: dict[str, list[_Request]] = defaultdict(list)
    for span in spans:
        rates = _rates(span) if span.kind == "llm" else None
        if rates is None or span.start_time is None:
            continue
        read, written = span.cache_read_tokens or 0, span.cache_write_tokens or 0
        by_context[_context_of(span, by_id)].append(
            _Request(
                span=span,
                start=span.start_time,
                rates=rates,
                # A span's input tokens count the cache in; fresh input is the rest.
                fresh=max(0, (span.input_tokens or 0) - read - written),
                output=span.output_tokens or 0,
                read=read,
                written=written,
            )
        )
    kept: dict[str, list[_Request]] = {}
    for context, requests in by_context.items():
        models: dict[str | None, int] = defaultdict(int)
        for request in requests:
            models[request.span.model] += 1
        model = max(models, key=lambda name: models[name])
        kept[context] = sorted((r for r in requests if r.span.model == model), key=lambda r: r.start)
    return kept


def _subagent_costs(spans: Sequence[SpanRecord], by_id: Mapping[str, SpanRecord]) -> dict[str, Decimal]:
    """What the requests under each call's run cost, by the call: a subagent's exact cost."""
    costs: dict[str, Decimal] = defaultdict(Decimal)
    for span in spans:
        if span.kind != "llm" or span.cost_snapshot is None:
            continue
        run = by_id.get(span.parent_span_id or "")
        if run is not None and run.operation == _EXECUTION and run.parent_span_id:
            costs[run.parent_span_id] += span.cost_snapshot
    return costs


def _rereads(requests: Sequence[_Request], after: int, tokens: Decimal) -> tuple[Decimal, int]:
    cost = Decimal(0)
    count = 0
    largest = requests[after].context
    for request in requests[after + 1 :]:
        if request.context < largest * _COMPACTED:
            break
        largest = max(largest, request.context)
        cost += tokens * request.reread_rate() / _PER_MILLION
        count += 1
    return cost, count


def tool_costs(spans: Sequence[SpanRecord]) -> tuple[ToolCost, ...]:
    """An estimate for each tool call between two priced requests, in the order the calls were made."""
    by_id = {span.span_id: span for span in spans}
    contexts = _requests(spans, by_id)
    subagents = _subagent_costs(spans, by_id)
    # Each call by the request that asked for it (the last to start before it) and
    # the one that read its result (the first to start after it ended).
    starts = {context: [request.start for request in requests] for context, requests in contexts.items()}
    placed: list[tuple[SpanRecord, list[_Request], int, int]] = []
    for span in sorted((s for s in spans if s.kind == "tool" and s.start_time is not None), key=_start):
        context = _context_of(span, by_id)
        requests, started = contexts.get(context, []), starts.get(context, [])
        assert span.start_time is not None
        caller = bisect_right(started, span.start_time) - 1
        reader = bisect_left(started, span.end_time or span.start_time)
        if caller >= 0 and reader < len(requests) and reader > caller:
            placed.append((span, requests, caller, reader))
    calls_by_caller: dict[tuple[int, int], int] = defaultdict(int)
    calls_by_reader: dict[tuple[int, int], int] = defaultdict(int)
    for _, requests, caller, reader in placed:
        calls_by_caller[(id(requests), caller)] += 1
        calls_by_reader[(id(requests), reader)] += 1
    costs: list[ToolCost] = []
    for span, requests, caller, reader in placed:
        asked = requests[caller]
        read = requests[reader]
        sharing = calls_by_reader[(id(requests), reader)]
        previous = requests[reader - 1]
        grew = max(0, read.context - previous.context - previous.output)
        result_tokens = Decimal(grew) / sharing
        rereading, later = _rereads(requests, reader, result_tokens)
        costs.append(
            ToolCost(
                span_id=span.span_id,
                calling=asked.price(Decimal(asked.output), "output") / calls_by_caller[(id(requests), caller)],
                writing=result_tokens * read.new_input_rate() / _PER_MILLION,
                rereading=rereading,
                result_tokens=round(result_tokens),
                later_requests=later,
                calls_sharing_request=calls_by_caller[(id(requests), caller)],
                calls_sharing_result=sharing,
                subagent=subagents.get(span.span_id),
            )
        )
    return tuple(costs)


def _start(span: SpanRecord) -> datetime:
    assert span.start_time is not None
    return span.start_time
