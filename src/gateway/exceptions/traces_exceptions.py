"""Errors that the traces domain may raise.

Each carries the HTTP status it renders as.
"""

from fastapi import status

from gateway.exceptions import (
    TenancyConflictError,
    TenancyError,
    TenancyForbiddenError,
    TenancyNotFoundError,
    TenancyValidationError,
)


class TraceNotFoundError(TenancyNotFoundError):
    """No trace with this id in the caller's scope: one in another tenant's workspace reads the same."""

    def __init__(self, trace_id: str):
        super().__init__(f"Trace '{trace_id}' not found")


class TraceContentNotFoundError(TenancyNotFoundError):
    """The span has no content the caller may read: none was captured, it expired, or it was purged."""

    def __init__(self, span_id: str):
        super().__init__(f"No content for span '{span_id}'")


class ContentCaptureAboveCeilingError(TenancyValidationError):
    """A workspace asked to keep more content than the deployment permits."""

    def __init__(self, ceiling: str):
        super().__init__(f"This deployment permits content capture up to '{ceiling}'")


class ContentEncryptionNotConfiguredError(TenancyConflictError):
    """Capture was asked for on a deployment whose key backend cannot seal content."""

    def __init__(self) -> None:
        super().__init__("Content encryption is not configured on this deployment")


class TraceContentKeysUnavailableError(TenancyError):
    """The key backend could not open stored content just now. Fixed text: the cause goes to the log."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    def __init__(self) -> None:
        super().__init__("The trace content key backend is unavailable")


class TraceContentNotYoursError(TenancyForbiddenError):
    """The caller may see the session but not what it said: content is its own user's."""

    def __init__(self) -> None:
        super().__init__("Only the person who ran this session can read its content")
