"""Spans an instrumented agent sends to the OTLP receiver, projected into trace documents.

Every span is kept, not only the ones that carry token usage: an agent's own
spans (``invoke_agent``, ``execute_tool``) are what the gateway cannot see for
itself. A span's session is the Claude session or the conversation it names
(``anthropic.session.id``, ``gen_ai.conversation.id``), or its OTel trace when it names none.

Claude Code's own spans follow neither the GenAI conventions nor the Agent SDK's
names: each says what it is in ``span.type`` and names its session in
``session.id``. Every interaction is the root of an OTel trace of its own, so
without that session a conversation would read as one trace per turn.

``otari hook`` reports each gate it checked the same way, on the session the
harness named, as a guardrail span: required gates block, advisory ones only
flag. A gate span names the tool call it checked, and a read attaches it there.
A gate that fired is the guardrail working, so neither it nor the tool call it
stopped is an error: only a gate that could not decide is. A judge whose calls
ran on a second model reports that model's tokens on a child usage span, which
the usage ledger prices and a trace leaves out.

What a Claude Code model request cost is not on its span: the usage ledger
prices it from the ``api_request`` event on the logs signal. That ingest hands
each priced request back as a usage span, keyed by the request's id so a retried
export counts once, and a read folds its cost onto the request it priced.

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
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from gateway.log_config import logger
from gateway.services.secret_box import secret_box_configured
from gateway.services.traces._collector import call_id_digest, identifier_or_none
from gateway.services.traces._identity import SessionRef
from gateway.services.traces._writer import SPANS_DROPPED, TraceWriter
from gateway.types.traces import (
    GATE_USAGE_SEPARATOR,
    ID_PATTERN,
    LIST_SEPARATOR,
    RATE_METERS,
    SPAN_CONTENT_MAX_LENGTH,
    USAGE_OPERATION,
    SpanRecord,
    TraceWrite,
    is_gate_id,
    is_identifier,
    is_identifier_list,
)

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

HOOK_SCOPE = "otari.hook"
_HOOK_USAGE_SPAN = "otari.gate.usage"
# The usage sources a session's cost is imported from: Claude Code's model requests and a judge gate's.
CLAUDE_CODE_USAGE_SOURCE = "claude_code"
HOOK_USAGE_SOURCE = "otari-hook"
_SESSION_USAGE_SOURCES = frozenset({CLAUDE_CODE_USAGE_SOURCE, HOOK_USAGE_SOURCE})
# A gate that could not decide, which reads like a guardrail that could not run.
_UNDECIDED_GATE = frozenset({"unknown", "error", "not_run"})
_GATE_ATTRIBUTES = (
    "otari.gate.type",
    "otari.gate.origin",
    "otari.gate.outcome",
    "otari.gate.moment",
    "otari.policy.digest",
)

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

    @property
    def names_harness_session(self) -> bool:
        """Whether ``session.id`` is the harness's session: on Claude Code's spans and ``otari hook``'s."""
        return self.scope == HOOK_SCOPE or self.claude_code_type is not None


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


def _request_id(attributes: Mapping[str, Any]) -> str | None:
    """A model request's provider id, which the usage ledger prices it by.

    It is an id, so it is held to the id pattern rather than a name's: a name
    refuses anything shaped like a credential, which a request id often is.
    """
    value = _str(attributes, "request_id")
    return value if value is not None and ID_PATTERN.fullmatch(value) else None


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
    if span.names_harness_session:
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


# Where content lives on the spans that carry it, by the key it is kept under.
# Claude Code sends its keys only with OTEL_LOG_TOOL_DETAILS / OTEL_LOG_USER_PROMPTS,
# and otari hook its own only with OTARI_HOOK_REPORT_CONTENT.
_CLAUDE_CODE_CONTENT = {"prompt": "user_prompt", "command": "full_command", "file_path": "file_path"}
_HOOK_CONTENT = {"gate.evidence": "otari.gate.evidence", "gate.detail": "otari.gate.detail"}
_REDACTED = "<REDACTED>"


def _content(span: OtlpSpan) -> dict[str, str]:
    """What the span says it was about, cut to what the store keeps."""
    if span.scope == HOOK_SCOPE:
        sources = _HOOK_CONTENT
    elif span.claude_code_type is not None:
        sources = _CLAUDE_CODE_CONTENT
    else:
        return {}
    content: dict[str, str] = {}
    for key, source in sources.items():
        value = span.attributes.get(source)
        if isinstance(value, str) and value.strip() and value != _REDACTED:
            content[key] = value[:SPAN_CONTENT_MAX_LENGTH]
    return content


def _record(
    span: OtlpSpan,
    window: _Window,
    workspace_id: uuid.UUID,
    tool: _ToolFacts | None = None,
    *,
    keep_content: bool = False,
) -> SpanRecord:
    record = _projected(span, window, tool)
    call_id = call_id_digest(workspace_id, _str(span.attributes, "gen_ai.tool.call.id", "tool_use_id"))
    content = _content(span) if keep_content else {}
    return replace(record, tool_call_id=call_id or record.tool_call_id, content=content or record.content)


def _projected(span: OtlpSpan, window: _Window, tool: _ToolFacts | None) -> SpanRecord:
    attributes = span.attributes
    if span.scope == HOOK_SCOPE:
        return _gate_record(span, window)
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


def _llm_attributes(attributes: Mapping[str, Any]) -> dict[str, str | int]:
    """What a Claude Code model request says about itself beyond its tokens: why it stopped and how fast."""
    kept: dict[str, str | int] = {}
    if reason := identifier_or_none(_str(attributes, "stop_reason")):
        kept["gen_ai.response.finish_reasons"] = reason
    if (ttft := _int(attributes, "ttft_ms")) is not None:
        kept["otari.llm.ttft_ms"] = ttft
    return kept


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
    llm = kind == "llm"
    extra: dict[str, str | int | bool] = {}
    if llm:
        extra.update(_llm_attributes(attributes))
    if span_type == _CLAUDE_CODE_EXECUTION:
        extra["otari.tool.succeeded"] = not span_failed
    elif returned_error:
        extra["otari.tool.succeeded"] = False
    if span_type in ("tool", _CLAUDE_CODE_DECISION) and decided.decision:
        extra["otari.tool.decision"] = decided.decision
        if decided.source:
            extra["otari.tool.decision_source"] = decided.source
    start, end, duration = _times(span, window)
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
        request_id=_request_id(attributes) if llm else None,
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


def _judge_prompt(attributes: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
    """A judge's whole prompt and the part of it the cache read and wrote.

    ``otari hook`` reports its judge's tokens the way Claude counts them, the
    cache outside ``gen_ai.usage.input_tokens``.
    """
    fresh = _int(attributes, "gen_ai.usage.input_tokens")
    cache_read = _int(attributes, "gen_ai.usage.cache_read_tokens")
    cache_write = _int(attributes, "gen_ai.usage.cache_write_tokens")
    if fresh is None:
        return None, cache_read, cache_write
    return min(fresh + (cache_read or 0) + (cache_write or 0), _INT32_MAX), cache_read, cache_write


def _gate_record(span: OtlpSpan, window: _Window) -> SpanRecord:
    """Project one gate ``otari hook`` checked: a guardrail span, labeled by the gate's id.

    A judge gate also carries the model its calls ran on and the tokens they used.
    """
    attributes = span.attributes
    # Held to the rule a guardrail declares its ids by, so a gate such as
    # ``api-key-in-diff`` keeps the id its definition and counts are keyed on.
    sent = _str(attributes, "otari.gate.id")
    gate_id = sent if is_gate_id(sent) else None
    undecided = _str(attributes, "otari.gate.outcome") in _UNDECIDED_GATE
    required = _str(attributes, "otari.gate.enforcement") == "required"
    extra: dict[str, str | bool] = {
        key: value for key in _GATE_ATTRIBUTES if (value := identifier_or_none(_str(attributes, key))) is not None
    }
    if gate_id:
        extra["otari.gate.id"] = gate_id
        if is_identifier(gate_id):
            extra["otari.guardrail.profile"] = gate_id
    extra["otari.guardrail.mode"] = "block" if required else "monitor"
    blocked = attributes.get("otari.gate.blocked") is True
    extra["otari.gate.blocked"] = blocked
    model = identifier_or_none(_str(attributes, "gen_ai.request.model"))
    prompt, cache_read, cache_write = _judge_prompt(attributes) if model else (None, None, None)
    start, end, duration = _times(span, window)
    return SpanRecord(
        span_id=_span_id(span.trace_id, span.span_id),
        parent_span_id=_span_id(span.trace_id, span.parent_span_id) if span.parent_span_id else None,
        kind="guardrail",
        origin="otlp",
        name=f"gate:{gate_id}" if gate_id and is_identifier(f"gate:{gate_id}") else "otari.gate",
        operation=identifier_or_none(_str(attributes, "otari.gate.moment")),
        outcome="error" if undecided else "blocked" if blocked else "ok",
        error_class="guardrail_unavailable" if undecided else None,
        start_time=start,
        end_time=end,
        duration_ms=duration,
        model=model,
        provider=identifier_or_none(_str(attributes, "gen_ai.provider.name")) if model else None,
        # The id its model calls were priced under, so a read can fold their cost onto it.
        request_id=span.span_id if model and ID_PATTERN.fullmatch(span.span_id) else None,
        input_tokens=prompt,
        output_tokens=_int(attributes, "gen_ai.usage.output_tokens") if model else None,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        tool_name=identifier_or_none(_str(attributes, "gen_ai.tool.name")),
        otel_trace_id=span.trace_id[:64],
        otel_span_id=span.span_id[:64],
        attributes=extra,
    )


def _harness(span: OtlpSpan) -> str | None:
    named = identifier_or_none(_str(span.attributes, "otari.client_name"))
    if named:
        return named
    if span.scope == HOOK_SCOPE:
        return identifier_or_none(_str(span.attributes, "otari.harness"))
    return _CLAUDE_CODE_HARNESS if span.claude_code_type is not None else None


def _project(
    spans: list[OtlpSpan],
    *,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
    window: _Window,
    max_traces: int,
    keep_content: bool,
) -> OtlpProjection:
    by_session: dict[SessionRef, list[SpanRecord]] = {}
    harness: dict[SessionRef, str | None] = {}
    invalid = 0
    over_cap = 0
    tools = _tool_facts(spans)
    for span in spans:
        if not span.trace_id or not span.span_id:
            continue
        if span.scope == HOOK_SCOPE and span.name == _HOOK_USAGE_SPAN:
            continue
        try:
            session = _session(span)
            if session not in by_session and len(by_session) >= max_traces:
                over_cap += 1
                continue
            record = _record(
                span, window, workspace_id, tools.get((span.trace_id, span.span_id)), keep_content=keep_content
            )
        except Exception as exc:  # noqa: BLE001 - one bad span is dropped, never the export
            # The class only: an exception's message may quote what the exporter sent.
            logger.debug("OTLP span not recorded: %s", type(exc).__name__)
            invalid += 1
            continue
        by_session.setdefault(session, []).append(record)
        harness.setdefault(session, _harness(span))
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
    keep_content: bool = False,
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
            keep_content=keep_content,
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
    content_enabled: bool = False,
) -> None:
    """Hand one export's spans to the trace writer, attributed to the exporting key.

    Keeps the first :data:`MAX_SPANS_PER_EXPORT` of the export's ``total`` spans and
    counts the rest as dropped. Never raises: the export has already been answered for.

    Only a key that may import is traced (``budget_exempt``), the rule the usage
    import holds it to, whatever the export carries: an export of agent spans
    alone never reaches that import's own check. What a span was about is kept
    only where the deployment enabled it and can store it encrypted.
    """
    if not budget_exempt:
        return
    try:
        kept = list(itertools.islice(spans, MAX_SPANS_PER_EXPORT))
        projection = project_otlp_spans(
            kept,
            workspace_id=workspace_id,
            user_id=user_id,
            api_key_id=api_key_id,
            retention=retention,
            keep_content=content_enabled and secret_box_configured(),
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


@dataclass(frozen=True)
class PricedRequest:
    """A model request the usage ledger priced: its id, the session it names and what it cost."""

    request_id: str
    session_key: str
    at: datetime
    cost: Decimal
    source: str
    # USD per million tokens on each of :data:`RATE_METERS`, where the request was priced by the token.
    rates: Mapping[str, Decimal] = field(default_factory=dict)


def gate_usage_event_id(scope: str, name: str, span_id: str, parent_span_id: str) -> str | None:
    """The ledger id a judge's further-model usage span is priced under, or None for any other span."""
    if scope != HOOK_SCOPE or name != _HOOK_USAGE_SPAN or not parent_span_id:
        return None
    return f"{parent_span_id}{GATE_USAGE_SEPARATOR}{span_id}"


def _usage_span(request: PricedRequest, window: _Window) -> SpanRecord:
    at = window.clamp(request.at)
    return SpanRecord(
        span_id="usage-" + hashlib.sha256(request.request_id.encode()).hexdigest()[:40],
        kind="llm",
        origin="otlp",
        name="claude_code.usage",
        operation=USAGE_OPERATION,
        outcome="ok",
        start_time=at,
        end_time=at,
        request_id=request.request_id,
        cost_snapshot=request.cost,
        attributes={
            f"otari.llm.rate.{meter}": format(request.rates[meter].normalize(), "f")
            for meter in RATE_METERS
            if meter in request.rates
        },
    )


def record_priced_requests(
    writer: TraceWriter,
    requests: Iterable[PricedRequest],
    *,
    budget_exempt: bool,
    workspace_id: uuid.UUID,
    user_id: str | None,
    api_key_id: str | None,
    retention: timedelta,
) -> None:
    """Hand what each priced model request, Claude Code's or a judge gate's, cost to its session. Never raises.

    Only for a key that may import (``budget_exempt``), as :func:`record_otlp_export` holds its spans to,
    and only usage from a source whose sessions the trace store has.
    """
    if not budget_exempt:
        return
    now = datetime.now(UTC)
    window = _Window(floor=now - retention, ceiling=now + _FUTURE_SKEW)
    try:
        by_session: dict[SessionRef, list[SpanRecord]] = {}
        for request in itertools.islice(requests, MAX_SPANS_PER_EXPORT):
            # A ledger id a span cannot carry could never fold onto one, and would fail the whole export.
            if request.source not in _SESSION_USAGE_SOURCES or not ID_PATTERN.fullmatch(request.request_id):
                continue
            session = SessionRef(source="otlp", key=request.session_key)
            if session not in by_session and len(by_session) >= MAX_TRACES_PER_EXPORT:
                continue
            by_session.setdefault(session, []).append(_usage_span(request, window))
        for session, records in by_session.items():
            writer.submit(
                TraceWrite(
                    workspace_id=workspace_id,
                    trace_id=session.trace_id(workspace_id, user_id, api_key_id),
                    user_id=user_id,
                    api_key_id=api_key_id,
                    session_source="otlp",
                    harness=_CLAUDE_CODE_HARNESS,
                    spans=tuple(records),
                )
            )
    except Exception as exc:  # noqa: BLE001 - a session's cost never fails the export that priced it
        logger.warning("OTLP request costs not recorded: %s", type(exc).__name__)
