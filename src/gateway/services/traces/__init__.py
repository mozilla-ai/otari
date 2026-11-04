"""The traces domain: what the gateway records about each agent session it serves.

A request's spans are collected in memory and handed to a bounded writer once
its response is sent; ``TraceService`` stores them in this deployment's database.
"""

from gateway.services.traces._capture import RequestTraces
from gateway.services.traces._collector import LlmCall, RequestTrace, identifier_or_none
from gateway.services.traces._content import SpanContent
from gateway.services.traces._context import ContextWeights, context_of, read_context
from gateway.services.traces._identity import SessionRef, harness_of, is_subagent, resolve_session
from gateway.services.traces._membership import TraceMembershipListener
from gateway.services.traces._otlp import (
    CLAUDE_CODE_USAGE_SOURCE,
    HOOK_SCOPE,
    HOOK_USAGE_SOURCE,
    MAX_SPANS_PER_EXPORT,
    OtlpSpan,
    PricedRequest,
    gate_usage_event_id,
    project_otlp_spans,
    record_otlp_export,
    record_priced_requests,
)
from gateway.services.traces._retention import run_trace_retention
from gateway.services.traces._service import TraceService
from gateway.services.traces._turns import AnsweredCall, TurnFacts, read_turn
from gateway.services.traces._writer import TraceWriter

__all__ = [
    "ContextWeights",
    "AnsweredCall",
    "CLAUDE_CODE_USAGE_SOURCE",
    "HOOK_SCOPE",
    "HOOK_USAGE_SOURCE",
    "MAX_SPANS_PER_EXPORT",
    "LlmCall",
    "OtlpSpan",
    "PricedRequest",
    "RequestTrace",
    "RequestTraces",
    "SpanContent",
    "SessionRef",
    "TraceMembershipListener",
    "TraceService",
    "TraceWriter",
    "TurnFacts",
    "harness_of",
    "identifier_or_none",
    "is_subagent",
    "context_of",
    "gate_usage_event_id",
    "project_otlp_spans",
    "read_context",
    "read_turn",
    "record_otlp_export",
    "record_priced_requests",
    "resolve_session",
    "run_trace_retention",
]
