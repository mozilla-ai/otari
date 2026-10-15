"""Errors that the traces domain may raise.

Each carries the HTTP status it renders as.
"""

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError


class TraceNotFoundError(TenancyNotFoundError):
    """No trace with this id in the caller's scope: one in another tenant's workspace reads the same."""

    def __init__(self, trace_id: str):
        super().__init__(f"Trace '{trace_id}' not found")


class RequestTraceNotFoundError(TenancyNotFoundError):
    """No trace in the caller's scope holds this request: one outside it, never traced, or expired, reads the same."""

    def __init__(self, request_id: str):
        super().__init__(f"No trace holds request '{request_id}'")


class TraceAmbiguousError(TenancyConflictError):
    """A trace id that is in more than one workspace of the caller's scope, read without naming one."""

    def __init__(self, trace_id: str):
        super().__init__(f"Trace '{trace_id}' is in more than one workspace; name one with workspace_id")
