"""The traces domain: what the gateway records about each agent session it serves.

A request's spans are collected in memory and handed to a bounded writer once
its response is sent; ``TraceService`` stores them in this deployment's database.
"""

from gateway.services.traces._capture import RequestTraces
from gateway.services.traces._collector import LlmCall, RequestTrace, identifier_or_none
from gateway.services.traces._identity import SessionRef, harness_of, resolve_session
from gateway.services.traces._retention import run_trace_retention
from gateway.services.traces._service import TraceService
from gateway.services.traces._turns import AnsweredCall, TurnFacts, read_turn
from gateway.services.traces._writer import TraceWriter, stored_by

__all__ = [
    "AnsweredCall",
    "LlmCall",
    "RequestTrace",
    "RequestTraces",
    "SessionRef",
    "TraceService",
    "TraceWriter",
    "TurnFacts",
    "harness_of",
    "identifier_or_none",
    "read_turn",
    "resolve_session",
    "run_trace_retention",
    "stored_by",
]
