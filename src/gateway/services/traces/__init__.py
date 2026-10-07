"""The traces domain: what the gateway records about each agent session it serves.

Recording happens on every plane and needs no database: a request's spans are
collected in memory and handed to a bounded writer once its response is sent.
Storage is behind ``TraceStoragePort``.

What is exported here is the domain's surface: recording a request
(``RequestTrace``, ``RequestTraces``, the readers of its body, the writer),
ingesting OTLP spans, reading sessions back, and the content and retention
services. Everything else stays in its private module.
"""

from gateway.services.traces._capture import RequestTraces, TracingLogWriter
from gateway.services.traces._collector import RequestTrace
from gateway.services.traces._content_access import ContentAccessService
from gateway.services.traces._content_keys import ContentKeys
from gateway.services.traces._content_reading import RequestContent, read_content
from gateway.services.traces._identity import harness_of, resolve_session
from gateway.services.traces._otlp import OtlpSpan, project_otlp_spans
from gateway.services.traces._reading import TraceService
from gateway.services.traces._retention import run_trace_retention
from gateway.services.traces._settings import ContentCapturePolicy, TraceSettingsService
from gateway.services.traces._turns import TurnFacts, read_turn
from gateway.services.traces._writer import TraceWriter

__all__ = [
    "ContentCapturePolicy",
    "ContentAccessService",
    "ContentKeys",
    "OtlpSpan",
    "RequestContent",
    "RequestTrace",
    "RequestTraces",
    "TraceService",
    "TraceSettingsService",
    "TraceWriter",
    "TracingLogWriter",
    "TurnFacts",
    "harness_of",
    "project_otlp_spans",
    "read_content",
    "read_turn",
    "resolve_session",
    "run_trace_retention",
]
