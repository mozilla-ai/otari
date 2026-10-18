"""Spans an instrumented agent sends to the OTLP receiver, projected into trace documents.

Every span is kept, not only the ones that carry token usage: an agent's own
spans (``invoke_agent``, ``execute_tool``) are what the gateway cannot see for
itself. A span's session is the Claude session or the conversation it names
(``anthropic.session.id``, ``gen_ai.conversation.id``), or its OTel trace when it names none.

Claude Code's own spans follow neither the GenAI conventions nor the Agent SDK's
names: each says what it is in ``span.type`` and names its session in
``session.id``. Every interaction is the root of an OTel trace of its own, so
without that session a conversation would read as one trace per turn.

Nothing is passed through verbatim. Each name and value is projected the way the
gateway's own spans are, so text an agent put in a span name or attribute never
reaches the store, and the original OTLP body is never kept. Numbers and times
are clamped to what the store holds, and a span that cannot be projected is
dropped and counted, never allowed to fail the export that carried it.
"""

import hashlib
import itertools
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from gateway.log_config import logger
from gateway.services.traces._collector import call_id_digest, identifier_or_none
from gateway.services.traces._identity import SessionRef
from gateway.services.traces._writer import SPANS_DROPPED, TraceWriter
from gateway.types.traces import LIST_SEPARATOR, SpanRecord, TraceWrite, is_identifier, is_identifier_list

# ``gen_ai.operation.name`` values, by the span kind they are.
_LLM_OPERATIONS = frozenset({"chat", "text_completion", "generate_content", "embeddings"})
_AGENT_OPERATIONS = frozenset({"invoke_agent", "create_agent", "invoke_workflow", "plan"})
_TOOL_OPERATIONS = frozenset({"execute_tool"})

_CLAUDE_CODE_SCOPE_PREFIX = "com.anthropic.claude_code"
_CLAUDE_CODE_HARNESS = "claude-code"
_CLAUDE_CODE_KINDS = {"interaction": "step", "llm_request": "llm", "tool": "tool"}
# The child of a tool span that records whether it was allowed to run, and by
# whom: ``source`` is ``hook`` when a hook refused it.
_CLAUDE_CODE_DECISION = "tool.blocked_on_user"
_CLAUDE_CODE_EXECUTION = "tool.execution"

# Token counts and durations are stored as 32-bit integers.
_INT32_MAX = 2_147_483_647
# Digits a numeric string attribute may have: anything longer is past the clamp anyway.
_MAX_INT_DIGITS = 19
# How far ahead of the gateway's clock an exporter's clock may run.
_FUTURE_SKEW = timedelta(minutes=5)

# Spans of one export that are kept. Well under the writer's queue, so one export
# can never fill it and starve every other request's trace.
MAX_SPANS_PER_EXPORT = 1_000

# Sessions one export may write. Each is a trace row of its own, so the span cap
# alone does not bound how many traces an export creates; kept well under it, so it
# binds.
MAX_TRACES_PER_EXPORT = 100


@dataclass(frozen=True)
class OtlpSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time: datetime | None
    end_time: datetime | None
    failed: bool
    attributes: Mapping[str, Any]
    scope: str = ""

    @property
    def claude_code_type(self) -> str | None:
        """The ``span.type`` of a span Claude Code sent, or None for any other exporter's."""
        if not self.scope.startswith(_CLAUDE_CODE_SCOPE_PREFIX):
            return None
        return _str(self.attributes, "span.type")


@dataclass(frozen=True)
class _ToolFacts:
    """What a Claude Code tool span's children say about it: who decided, and whether it ran."""

    decision: str | None = None
    source: str | None = None
    failed: bool = False


@dataclass(frozen=True)
class OtlpProjection:
    writes: tuple[TraceWrite, ...]
    dropped: int = 0


@dataclass(frozen=True)
class _Window:
    floor: datetime
    ceiling: datetime

    def clamp(self, moment: datetime | None) -> datetime | None:
        if moment is None:
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return min(max(moment, self.floor), self.ceiling)


def _str(attributes: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _int(attributes: Mapping[str, Any], key: str) -> int | None:
    """A non-negative count, clamped to what the store holds, or None when the value is not one."""
    value = attributes.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        # ``isdigit`` alone accepts non-ASCII digits such as "²", which ``int`` refuses.
        if not (value.isascii() and value.isdigit() and len(value) <= _MAX_INT_DIGITS):
            return None
        try:
            value = int(value)
        except ValueError:
            return None
    if not isinstance(value, int) or value < 0:
        return None
    return min(value, _INT32_MAX)


def _joined(attributes: Mapping[str, Any], key: str) -> str | None:
    """An array-of-strings attribute joined for storage, or None unless every element is an identifier.

    A bare string is read as an array of one.
    """
    value = attributes.get(key)
    items = (value,) if isinstance(value, str) else value
    if not isinstance(items, list | tuple) or not items or not all(isinstance(item, str) for item in items):
        return None
    stripped = [item.strip() for item in items]
    # Each element is checked before the join, which would split one that held the separator.
    if not all(is_identifier(item) for item in stripped):
        return None
    joined = LIST_SEPARATOR.join(stripped)
    return joined if is_identifier_list(joined) else None


def _kind(operation: str | None) -> str:
    if operation in _LLM_OPERATIONS:
        return "llm"
    if operation in _TOOL_OPERATIONS:
        return "tool"
    if operation in _AGENT_OPERATIONS:
        return "agent"
    return "other"


def _false(attributes: Mapping[str, Any], key: str) -> bool:
    """Whether a Claude Code ``success`` attribute says the work failed; it sends ``"True"``/``"False"``."""
    value = attributes.get(key)
    return value is False or (isinstance(value, str) and value.strip().lower() == "false")


def _session(span: OtlpSpan) -> SessionRef:
    # A Claude Managed Agents session names each thread a conversation of its own.
    keys: tuple[str, ...] = ("anthropic.session.id", "gen_ai.conversation.id", "otari.session_label")
    if span.claude_code_type is not None:
        keys = ("session.id", *keys)
    return SessionRef(source="otlp", key=_str(span.attributes, *keys) or span.trace_id)


def _tool_facts(spans: Iterable[OtlpSpan]) -> dict[tuple[str, str], _ToolFacts]:
    """Each Claude Code tool span's decision and execution outcome, keyed by its OTel trace and span id.

    Only children in the same export are seen. One the batch exporter sent apart
    from its tool is still stored with its own decision; the tool then reads as run.
    """
    facts: dict[tuple[str, str], _ToolFacts] = {}
    for span in spans:
        kind = span.claude_code_type
        if kind not in (_CLAUDE_CODE_DECISION, _CLAUDE_CODE_EXECUTION) or not span.parent_span_id:
            continue
        key = (span.trace_id, span.parent_span_id)
        known = facts.get(key, _ToolFacts())
        if kind == _CLAUDE_CODE_DECISION:
            facts[key] = replace(
                known,
                decision=identifier_or_none(_str(span.attributes, "decision")),
                source=identifier_or_none(_str(span.attributes, "source")),
            )
        else:
            facts[key] = replace(known, failed=span.failed or _false(span.attributes, "success"))
    return facts


def _span_id(otel_trace_id: str, otel_span_id: str) -> str:
    """A received span's id: unique per OTel trace, since spans of several share one session.

    Hashed rather than concatenated, so it stays identifier-shaped and within the
    id length whatever the exporter sent, and apart from the gateway's own ids.
    """
    digest = hashlib.sha256(f"{otel_trace_id}\x00{otel_span_id}".encode()).hexdigest()[:32]
    return f"otlp-{digest}"


def _first_int(attributes: Mapping[str, Any], *keys: str) -> int | None:
    """The first of ``keys`` that holds a count, keeping a reported zero."""
    return next((count for key in keys if (count := _int(attributes, key)) is not None), None)


def _record(span: OtlpSpan, window: _Window, workspace_id: uuid.UUID, tool: _ToolFacts | None = None) -> SpanRecord:
    record = _projected(span, window, tool)
    call_id = call_id_digest(workspace_id, _str(span.attributes, "gen_ai.tool.call.id", "tool_use_id"))
    return replace(record, tool_call_id=call_id) if call_id else record


def _projected(span: OtlpSpan, window: _Window, tool: _ToolFacts | None) -> SpanRecord:
    attributes = span.attributes
    claude_code = span.claude_code_type
    if claude_code is not None:
        return _claude_code_record(span, claude_code, window, tool or _ToolFacts())
    operation = identifier_or_none(_str(attributes, "gen_ai.operation.name"))
    tool_name = identifier_or_none(_str(attributes, "gen_ai.tool.name"))
    kind = _kind(operation)
    if is_identifier(span.name):
        name = span.name
    elif kind == "tool" and tool_name and is_identifier(f"execute_tool:{tool_name}"):
        name = f"execute_tool:{tool_name}"
    else:
        name = operation or "span"
    error_type = identifier_or_none(_str(attributes, "error.type"))
    start, end, duration = _times(span, window)
    projected: dict[str, str | int | float | bool] = {}
    if span.failed and error_type:
        projected["error.type"] = error_type
    if (finish_reasons := _joined(attributes, "gen_ai.response.finish_reasons")) is not None:
        projected["gen_ai.response.finish_reasons"] = finish_reasons
    if (max_tokens := _int(attributes, "gen_ai.request.max_tokens")) is not None:
        projected["gen_ai.request.max_tokens"] = max_tokens
    input_tokens = _int(attributes, "gen_ai.usage.input_tokens")
    cache_read = _first_int(
        attributes,
        "gen_ai.usage.cache_read_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.cached_tokens",
    )
    cache_write = _first_int(attributes, "gen_ai.usage.cache_creation.input_tokens", "gen_ai.usage.cache_write_tokens")
    cached = (cache_read or 0) + (cache_write or 0)
    # The GenAI conventions count cached tokens inside ``input_tokens``, so cached
    # tokens that outnumber it were reported on top of it instead.
    if input_tokens is not None and cached > input_tokens:
        input_tokens = min(input_tokens + cached, _INT32_MAX)
    return SpanRecord(
        span_id=_span_id(span.trace_id, span.span_id),
        parent_span_id=_span_id(span.trace_id, span.parent_span_id) if span.parent_span_id else None,
        kind=kind,
        origin="otlp",
        name=name,
        operation=operation,
        outcome="error" if span.failed else "ok",
        error_class=(error_type or "error") if span.failed else None,
        start_time=start,
        end_time=end,
        duration_ms=duration,
        model=identifier_or_none(_str(attributes, "gen_ai.response.model", "gen_ai.request.model")),
        provider=identifier_or_none(_str(attributes, "gen_ai.provider.name", "gen_ai.system")),
        input_tokens=input_tokens,
        output_tokens=_int(attributes, "gen_ai.usage.output_tokens"),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        reasoning_tokens=_int(attributes, "gen_ai.usage.reasoning.output_tokens"),
        tool_name=tool_name,
        tool_type=identifier_or_none(_str(attributes, "gen_ai.tool.type")),
        otel_trace_id=span.trace_id[:64],
        otel_span_id=span.span_id[:64],
        attributes=projected,
    )


def _times(span: OtlpSpan, window: _Window) -> tuple[datetime | None, datetime | None, int | None]:
    start = window.clamp(span.start_time)
    end = window.clamp(span.end_time)
    if start is not None and end is not None and end < start:
        end = start
    duration = (
        min(round((end - start).total_seconds() * 1000), _INT32_MAX) if start is not None and end is not None else None
    )
    return start, end, duration


def _claude_code_prompt(attributes: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
    """A Claude Code request's whole prompt and the part of it the cache read and wrote.

    Claude Code counts cache reads and writes outside ``input_tokens``, where a
    span's input tokens count them in.
    """
    fresh = _int(attributes, "input_tokens")
    cache_read = _int(attributes, "cache_read_tokens")
    cache_write = _int(attributes, "cache_creation_tokens")
    if fresh is None:
        return None, cache_read, cache_write
    return min(fresh + (cache_read or 0) + (cache_write or 0), _INT32_MAX), cache_read, cache_write


def _claude_code_record(span: OtlpSpan, span_type: str, window: _Window, tool: _ToolFacts) -> SpanRecord:
    """Project one of Claude Code's own spans.

    An interaction is the user's turn, so it opens one as a step, unless it is
    nested under another span (a subagent's). A tool something refused (a hook,
    the user, a permission rule) was blocked, not failed: it never ran. A tool
    that returned an error, such as a command exiting non-zero, ran too: the
    agent reads the error like any other result, so it is not a session error.
    The decision and the run under a tool are facts about it, not outcomes of
    their own.
    """
    attributes = span.attributes
    kind = _CLAUDE_CODE_KINDS.get(span_type, "other")
    nested = span.parent_span_id is not None
    if kind == "step" and nested:
        kind = "agent"
    rejected = kind == "tool" and tool.decision == "reject"
    detail_of_tool = span_type in (_CLAUDE_CODE_DECISION, _CLAUDE_CODE_EXECUTION)
    span_failed = span.failed or _false(attributes, "success")
    returned_error = kind == "tool" and not rejected and (span_failed or tool.failed)
    failed = not rejected and not detail_of_tool and kind != "tool" and span_failed
    error_class = (identifier_or_none(_str(attributes, "error.type", "error")) or "error") if failed else None
    decided = (
        tool
        if kind == "tool"
        else _ToolFacts(
            decision=identifier_or_none(_str(attributes, "decision")),
            source=identifier_or_none(_str(attributes, "source")),
        )
    )
    extra: dict[str, str | bool] = {}
    if span_type == _CLAUDE_CODE_EXECUTION:
        extra["otari.tool.succeeded"] = not span_failed
    elif returned_error:
        extra["otari.tool.succeeded"] = False
    if span_type in ("tool", _CLAUDE_CODE_DECISION) and decided.decision:
        extra["otari.tool.decision"] = decided.decision
        if decided.source:
            extra["otari.tool.decision_source"] = decided.source
    start, end, duration = _times(span, window)
    llm = kind == "llm"
    prompt, cache_read, cache_write = _claude_code_prompt(attributes) if llm else (None, None, None)
    return SpanRecord(
        span_id=_span_id(span.trace_id, span.span_id),
        parent_span_id=_span_id(span.trace_id, span.parent_span_id) if span.parent_span_id else None,
        kind=kind,
        origin="otlp",
        name=span.name if is_identifier(span.name) else f"claude_code.{kind}",
        operation=identifier_or_none(span_type),
        outcome="blocked" if rejected else "error" if failed else "ok",
        error_class=error_class,
        opens_turn=kind == "step",
        start_time=start,
        end_time=end,
        duration_ms=duration,
        model=identifier_or_none(_str(attributes, "gen_ai.request.model", "model")) if llm else None,
        provider=identifier_or_none(_str(attributes, "gen_ai.system")) if llm else None,
        input_tokens=prompt,
        output_tokens=_int(attributes, "output_tokens") if llm else None,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        tool_name=identifier_or_none(_str(attributes, "tool_name")) if kind == "tool" else None,
        tool_type="client" if kind == "tool" else None,
        otel_trace_id=span.trace_id[:64],
        otel_span_id=span.span_id[:64],
        attributes=extra,
    )


def _project(
    spans: list[OtlpSpan],
    *,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
    window: _Window,
    max_traces: int,
) -> OtlpProjection:
    by_session: dict[SessionRef, list[SpanRecord]] = {}
    harness: dict[SessionRef, str | None] = {}
    invalid = 0
    over_cap = 0
    tools = _tool_facts(spans)
    for span in spans:
        if not span.trace_id or not span.span_id:
            continue
        try:
            session = _session(span)
            if session not in by_session and len(by_session) >= max_traces:
                over_cap += 1
                continue
            record = _record(span, window, workspace_id, tools.get((span.trace_id, span.span_id)))
        except Exception as exc:  # noqa: BLE001 - one bad span is dropped, never the export
            # The class only: an exception's message may quote what the exporter sent.
            logger.debug("OTLP span not recorded: %s", type(exc).__name__)
            invalid += 1
            continue
        by_session.setdefault(session, []).append(record)
        named = identifier_or_none(_str(span.attributes, "otari.client_name"))
        harness.setdefault(session, named or (_CLAUDE_CODE_HARNESS if span.claude_code_type is not None else None))
    if invalid:
        SPANS_DROPPED.labels(reason="invalid").inc(invalid)
    if over_cap:
        SPANS_DROPPED.labels(reason="truncated").inc(over_cap)
    writes = tuple(
        TraceWrite(
            workspace_id=workspace_id,
            trace_id=session.trace_id(workspace_id, user_id, api_key_id),
            user_id=user_id,
            api_key_id=api_key_id,
            session_source="otlp",
            harness=harness[session],
            spans=tuple(records),
        )
        for session, records in by_session.items()
    )
    return OtlpProjection(writes=writes, dropped=invalid + over_cap)


def project_otlp_spans(
    spans: list[OtlpSpan],
    *,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
    retention: timedelta,
    now: datetime | None = None,
    max_traces: int = MAX_TRACES_PER_EXPORT,
) -> OtlpProjection:
    """Group received spans by session and project each one, attributed to the exporting key.

    Total: a span that cannot be projected is dropped and counted, and so is every
    span past the export's ``max_traces``-th session. Times are clamped to the
    window a trace can live in, from ``retention`` ago to a few minutes from ``now``.
    """
    now = now or datetime.now(UTC)
    window = _Window(floor=now - retention, ceiling=now + _FUTURE_SKEW)
    try:
        return _project(
            spans,
            workspace_id=workspace_id,
            user_id=user_id,
            api_key_id=api_key_id,
            window=window,
            max_traces=max_traces,
        )
    except Exception as exc:  # noqa: BLE001 - recording spans never fails the export
        logger.warning("OTLP spans not recorded: %s", type(exc).__name__)
        SPANS_DROPPED.labels(reason="invalid").inc(len(spans))
        return OtlpProjection(writes=(), dropped=len(spans))


def record_otlp_export(
    writer: TraceWriter,
    spans: Iterable[OtlpSpan],
    *,
    total: int,
    budget_exempt: bool,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
    retention: timedelta,
) -> None:
    """Hand one export's spans to the trace writer, attributed to the exporting key.

    Keeps the first :data:`MAX_SPANS_PER_EXPORT` of the export's ``total`` spans and
    counts the rest as dropped. Never raises: the export has already been answered for.

    Only a key that may import is traced (``budget_exempt``), the rule the usage
    import holds it to, whatever the export carries: an export of agent spans
    alone never reaches that import's own check.
    """
    if not budget_exempt:
        return
    try:
        kept = list(itertools.islice(spans, MAX_SPANS_PER_EXPORT))
        projection = project_otlp_spans(
            kept, workspace_id=workspace_id, user_id=user_id, api_key_id=api_key_id, retention=retention
        )
        truncated = max(0, total - len(kept))
        for index, write in enumerate(projection.writes):
            writer.submit(write, truncated=truncated if index == 0 else 0)
        if truncated and not projection.writes:
            SPANS_DROPPED.labels(reason="truncated").inc(truncated)
    except Exception as exc:  # noqa: BLE001 - recording spans never fails the export
        logger.warning("OTLP spans not recorded: %s", type(exc).__name__)
        return
    if projection.dropped:
        logger.info("OTLP spans: %d of %d not recorded", projection.dropped, total)
