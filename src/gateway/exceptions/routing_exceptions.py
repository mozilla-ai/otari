"""Errors the routing domain raises about a model recommendation.

Plain exceptions rather than the tenancy family: the route renders each as a
generic detail of its own, the way the tools domain's refusals are rendered,
so nothing an upstream said reaches the caller.
"""


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
