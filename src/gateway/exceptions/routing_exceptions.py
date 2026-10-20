"""Errors the routing domain raises about a model recommendation, and about the recommenders an operator adds.

The recommendation's errors are plain exceptions rather than the tenancy
family: the route renders each as a generic detail of its own, the way the
tools domain's refusals are rendered, so nothing an upstream said reaches the
caller. The recommenders' admin errors are the tenancy family, each carrying
its status, and none of their messages carries key material.
"""

from gateway.exceptions import TenancyConflictError, TenancyNotFoundError, TenancyValidationError


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


class PolicyOnHybridGatewayError(Exception):
    """A request named one of this gateway's routing policies, which a hybrid gateway does not apply.

    The message says what to name instead.
    """


class InvalidAgentRecommenderError(TenancyValidationError):
    """The recommender could not work: an unknown provider, a missing key or address, an unsafe address, or no price."""


class AgentRecommenderNotFoundError(TenancyNotFoundError):
    def __init__(self) -> None:
        super().__init__("No subagent recommender has this id")


class AgentRecommenderSwitchConflictError(TenancyConflictError):
    def __init__(self) -> None:
        super().__init__("Another subagent recommender was switched on at the same moment. Reload and try again.")
