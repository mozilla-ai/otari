"""``POST /api/v1/routing/feedback``: rate a routed response, so the router that chose its model learns from it.

The caller names the response by the ``Otari-Request-ID`` header it came back
with, and authenticates with a workspace API key of the workspace that served
it. The rating is handed to the router backend that decided the request, which
is only possible for a backend that learns from ratings (``smart_router``).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from gateway.api.deps import RoutingFeedbackServiceDep, verify_api_key
from gateway.exceptions.routing_exceptions import RoutingFeedbackError
from gateway.models.api_keys import APIKey
from gateway.schemas.routing import RoutingFeedbackRequest

# Declared on the router as well as taken by the handler, so a route added here
# later starts gated; FastAPI resolves it once per request either way.
router = APIRouter(prefix="/routing", tags=["routing"], dependencies=[Depends(verify_api_key)])


@router.post(
    "/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "No valid workspace API key."},
        status.HTTP_404_NOT_FOUND: {"description": "No request with this id was served in the key's workspace."},
        status.HTTP_409_CONFLICT: {
            "description": "The request was not routed by a router that learns from ratings, or was already rated."
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "The router could not record the rating. Retry."},
    },
)
async def rate_routed_response(
    request: RoutingFeedbackRequest,
    api_key: Annotated[APIKey, Depends(verify_api_key)],
    service: RoutingFeedbackServiceDep,
) -> Response:
    """Rate one response from 0 to 1, by the ``Otari-Request-ID`` it was returned with.

    The key's workspace must be the one that served the request; a request in
    any other workspace is answered as one that does not exist.
    """
    try:
        await service.rate(request_id=request.request_id, workspace_id=api_key.workspace_id, score=request.score)
    except RoutingFeedbackError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
