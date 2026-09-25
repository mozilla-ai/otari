"""Errors a deployment raises when its peer control plane cannot answer.

A hybrid gateway asks a control plane elsewhere what a workspace may do and
with which credential. These name the ways that question fails, in this
deployment's terms rather than in the peer's HTTP ones.
"""

from fastapi import status

from gateway.exceptions._base import TenancyError


class ControlPlaneError(TenancyError):
    """The control plane could not answer a question about this request."""


class ControlPlaneNotConfiguredError(ControlPlaneError):
    """Hybrid mode is selected and no control plane address is set.

    The deployment's own fault rather than the caller's, so it renders as a 500.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR


class ControlPlaneRefusedError(ControlPlaneError):
    """The control plane answered, and its answer was a refusal.

    The status travels on the instance because it is the peer's answer rather
    than a property of the condition: one refusal may be a 402 and the next a
    429. ``retry_after`` carries the peer's own hint, which a caller rendering
    this as a response owes the client.
    """

    def __init__(self, message: str, *, status_code: int, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class ControlPlaneUnavailableError(ControlPlaneError):
    """The control plane could not be reached, or answered something unusable."""

    status_code = status.HTTP_502_BAD_GATEWAY
