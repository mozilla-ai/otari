"""Why rating a routed request can fail, each with the HTTP status it maps to."""

from fastapi import status


class RoutingFeedbackError(Exception):
    """Base for a rating the gateway could not pass to the router that made the decision."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RoutedRequestNotFoundError(RoutingFeedbackError):
    """No request with this id was served in the caller's workspace.

    One answer for a request that never existed and for one in another
    workspace, so the status is not an oracle for other tenants' request ids.
    """

    status_code = status.HTTP_404_NOT_FOUND


class RequestNotRateableError(RoutingFeedbackError):
    """The request exists, and no router that learns from feedback decided it."""

    status_code = status.HTTP_409_CONFLICT


class RatingAlreadyRecordedError(RoutingFeedbackError):
    """The router already holds a rating for this request and keeps only one."""

    status_code = status.HTTP_409_CONFLICT


class RouterFeedbackFailedError(RoutingFeedbackError):
    """The router that decided the request could not record the rating."""

    status_code = status.HTTP_502_BAD_GATEWAY
