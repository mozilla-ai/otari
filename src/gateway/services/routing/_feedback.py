import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.routing_exceptions import (
    RequestNotRateableError,
    RoutedRequestNotFoundError,
    RouterFeedbackFailedError,
)
from gateway.ports.routing_port import LearningRouterBackend, RoutingPort
from gateway.repositories.routing import RoutedRequestRepository


class RoutingFeedbackService:
    """Hand a caller's rating of a response to the router backend that chose the model for it."""

    def __init__(self, uow: UnitOfWork, requests: RoutedRequestRepository, routing: RoutingPort) -> None:
        self._uow = uow
        self._requests = requests
        self._routing = routing

    async def rate(self, *, request_id: str, workspace_id: uuid.UUID, score: float) -> None:
        """Record ``score`` (0 to 1) for the request ``request_id`` names in ``workspace_id``.

        Raises:
            RoutedRequestNotFoundError: no request with this id was served in this workspace.
            RequestNotRateableError: the request exists, and no learning router backend decided it.
            RouterFeedbackFailedError: the backend that decided it is unavailable now, or could not
                record the rating.
        """
        async with self._uow:
            routed = await self._requests.find(request_id, workspace_id)
        if routed is None:
            raise RoutedRequestNotFoundError(f"No request '{request_id}' was served in this workspace.")
        if routed.routing_backend is None or routed.routing_decision_id is None:
            raise RequestNotRateableError(
                f"Request '{request_id}' was not routed by a router that learns from feedback, so there is "
                "nothing to rate. Only requests through a routing policy whose router learns from ratings "
                "(router: smart_router) can be rated."
            )
        backend = self._routing.backend(routed.routing_backend)
        if not isinstance(backend, LearningRouterBackend):
            raise RouterFeedbackFailedError(
                f"Request '{request_id}' was routed by '{routed.routing_backend}', which this gateway is not "
                "configured to reach any more, so the rating cannot be delivered."
            )
        await backend.record_feedback(routed.routing_decision_id, score)
