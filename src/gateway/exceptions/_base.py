"""Error bases that carry the HTTP status their condition maps to.

The status sits on the class rather than at the raise site because it is a
property of the condition, not of the endpoint that hit it.
"""

from fastapi import status


class TenancyError(Exception):
    """Base class for a tenancy operation that cannot be completed."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TenancyNotFoundError(TenancyError):
    """A resource does not exist, or does not exist *for this caller*.

    The two are deliberately one status. A workspace in another organization
    must not be distinguishable from a workspace that was never created, or the
    404 becomes an existence oracle for other tenants' data.
    """

    status_code = status.HTTP_404_NOT_FOUND


class TenancyForbiddenError(TenancyError):
    """The caller is known, the resource is visible, and the action is not theirs."""

    status_code = status.HTTP_403_FORBIDDEN


class TenancyConflictError(TenancyError):
    """The request collides with something that already exists."""

    status_code = status.HTTP_409_CONFLICT


class TenancyValidationError(TenancyError):
    """The request is well-formed but would leave the tenancy model inconsistent."""

    status_code = status.HTTP_400_BAD_REQUEST
