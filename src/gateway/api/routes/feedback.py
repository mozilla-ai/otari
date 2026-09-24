"""Authenticated, user-initiated product feedback."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from gateway.api.deps import get_feedback_service, get_session_identity, verify_master_key
from gateway.core.feature import CoreFeature
from gateway.exceptions.feedback_exceptions import FeedbackDeliveryError
from gateway.models.tenancy import User
from gateway.rate_limit import RateLimiter
from gateway.schemas.feedback import FeedbackSubmission
from gateway.services.feedback import FeedbackService

router = APIRouter(prefix="/feedback", tags=["feedback"], dependencies=[Depends(verify_master_key)])
MAX_BODY_BYTES = 32 * 1024


async def _read_submission(request: Request) -> FeedbackSubmission:
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Send feedback as application/json.")
    content = bytearray()
    try:
        async for chunk in request.stream():
            if len(content) + len(chunk) > MAX_BODY_BYTES:
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Feedback is too large.")
            content.extend(chunk)
    except ClientDisconnect:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Feedback upload was interrupted.") from None
    try:
        return FeedbackSubmission.model_validate_json(bytes(content))
    except ValidationError:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Check the feedback text."
        ) from None


@router.post(
    "",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        413: {"description": "Feedback exceeds 32 KiB."},
        415: {"description": "Only application/json is accepted."},
        422: {"description": "Invalid feedback fields."},
        429: {"description": "Feedback limit reached. Retry-After gives the wait in seconds."},
        503: {"description": "Delivery could not be confirmed."},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": FeedbackSubmission.model_json_schema()}},
        }
    },
)
async def submit_feedback(
    request: Request,
    service: Annotated[FeedbackService, Depends(get_feedback_service)],
    session_identity: Annotated[User | None, Depends(get_session_identity)],
) -> Response:
    """Send feedback privately to the Otari team."""
    body = await _read_submission(request)
    # After validation, so a body that never leaves the gateway costs no send.
    limiter: RateLimiter | None = getattr(request.app.state, "feedback_rate_limiter", None)
    if limiter is not None:
        try:
            limiter.check(str(session_identity.id) if session_identity else "master")
        except HTTPException as exc:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS, "Please wait before sending more feedback.", headers=exc.headers
            ) from None
    try:
        await service.submit(body)
    except FeedbackDeliveryError as exc:
        detail = {
            413: "Feedback is too large.",
            422: "Check the feedback text.",
            429: "Please wait before sending more feedback.",
        }.get(exc.status_code, "We could not confirm delivery. Please try again.")
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        raise HTTPException(exc.status_code, detail, headers=headers) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


FEATURE = CoreFeature(
    name="feedback", surface=None, enabled=lambda config: config.feedback_enabled, routers=lambda config: (router,)
)
