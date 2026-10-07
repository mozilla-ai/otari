"""The traces domain: what the gateway records about each agent session it serves.

Recording happens on every plane and needs no database: a request's spans are
collected in memory and handed to a bounded writer once its response is sent.
Storage is behind ``TraceStoragePort``.
"""

from gateway.services.traces._capture import RequestTraces, TracingLogWriter
from gateway.services.traces._collector import RequestTrace, identifier_or_none
from gateway.services.traces._retention import run_trace_retention
from gateway.services.traces._writer import TraceWriter

__all__ = [
    "RequestTrace",
    "RequestTraces",
    "TraceWriter",
    "TracingLogWriter",
    "identifier_or_none",
    "run_trace_retention",
]
