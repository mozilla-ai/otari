"""The traces domain: what the gateway records about each agent session it serves.

Recording happens on every plane and needs no database: a request's spans are
collected in memory and handed to a bounded writer once its response is sent.
Storage is behind ``TraceStoragePort``.
"""

from gateway.services.traces._capture import RequestTraces, TracingLogWriter
from gateway.services.traces._collector import RequestTrace, identifier_or_none
from gateway.services.traces._identity import SessionRef, harness_of, resolve_session
from gateway.services.traces._otlp import OtlpSpan, project_otlp_spans
from gateway.services.traces._reading import TraceService
from gateway.services.traces._retention import run_trace_retention
from gateway.services.traces._turns import AnsweredCall, TurnFacts, read_turn
from gateway.services.traces._writer import TraceWriter

__all__ = [
    "AnsweredCall",
    "OtlpSpan",
    "RequestTrace",
    "RequestTraces",
    "SessionRef",
    "TraceService",
    "TraceWriter",
    "TracingLogWriter",
    "TurnFacts",
    "harness_of",
    "identifier_or_none",
    "project_otlp_spans",
    "read_turn",
    "resolve_session",
    "run_trace_retention",
]
