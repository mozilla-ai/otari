"""Errors the routing domain raises about a model recommendation.

Two families. The recommendation errors are plain exceptions: the route
renders each as a generic detail of its own, the way the tools domain's
refusals are rendered, so nothing an upstream said reaches the caller. The
stored-recommender errors are in the tenancy family, so the one handler in
``gateway.main`` renders them with their status and the admin routes stay thin.
"""

from gateway.exceptions._base import TenancyConflictError, TenancyNotFoundError, TenancyValidationError


class CandidatesNotAllowedError(Exception):
    """The request's candidates do not fit its harness. The message says which way."""


class RecommenderNotConfiguredError(Exception):
    """This deployment names no recommender that can answer. The message says what to configure."""


class RecommendationFailedError(Exception):
    """The recommender could not answer.

    ``upstream_status`` is the HTTP status the upstream answered with, where
    one did, so a caller can tell a bad request from a rate limit from an
    outage without reading the message, which may name the upstream.
    """

    def __init__(self, message: str, *, upstream_status: int | None = None) -> None:
        super().__init__(message)
        self.upstream_status = upstream_status


class UnreadableRecommendationError(ValueError):
    """The recommender answered, but named no candidate."""


class InvalidAgentRecommenderError(TenancyValidationError):
    """A stored recommender could not work as configured. The message names what is wrong."""


class AgentRecommenderUnsafeApiBaseError(TenancyValidationError):
    """The API base is an address the gateway refuses to dial."""


class AgentRecommenderNotFoundError(TenancyNotFoundError):
    """No stored recommender has this id."""

    def __init__(self) -> None:
        super().__init__("No subagent router has this id")


class AgentRecommenderSwitchConflictError(TenancyConflictError):
    """Another recommender was switched on at the same moment."""

    def __init__(self) -> None:
        super().__init__("Another subagent router was switched on at the same moment. Reload and try again.")
