"""Errors that the traces domain may raise.

Each carries the HTTP status it renders as.
"""

from fastapi import status

from gateway.exceptions import (
    TenancyConflictError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)


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


class GuardrailDigestMismatchError(TenancyValidationError):
    """A guardrail version sent under a fingerprint its gates do not hash to: storing it would name another's."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    def __init__(self) -> None:
        super().__init__("The digest does not match these gates.")


class GuardrailVersionLimitError(TenancyConflictError):
    """A workspace that already keeps as many guardrail versions as it may, sent a new one."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"This workspace already keeps {limit} guardrail versions; no more are stored.")


class TraceContentShareForbiddenError(TenancyForbiddenError):
    """Only the user a session belongs to shares its content, or sees and ends its shares."""

    def __init__(self) -> None:
        super().__init__("Only the session's own user can share its content")


class TraceContentShareNotFoundError(TenancyNotFoundError):
    """No share with this id on the session."""

    def __init__(self, share_id: str):
        super().__init__(f"Share '{share_id}' not found")


class TraceBreakGlassUnnamedError(TenancyForbiddenError):
    """A break-glass opening is recorded against a named person, which a header master key is not."""

    def __init__(self) -> None:
        super().__init__("Sign in to open a session's content this way: a master key names nobody to record")


class TraceBreakGlassForbiddenError(TenancyForbiddenError):
    """Only an organization's owners open its sessions' content without a share."""

    def __init__(self) -> None:
        super().__init__("Only an owner of the organization can open a session's content this way")
