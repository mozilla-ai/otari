"""Forward explicit feedback without caller credentials or automatic retries."""

import asyncio

import httpx

from gateway.core.config import API_ROOT
from gateway.exceptions.feedback_exceptions import FeedbackDeliveryError
from gateway.log_config import logger
from gateway.rate_limit import RateLimiter
from gateway.schemas.feedback import FeedbackSubmission

FEEDBACK_RECEIVER = f"https://api.otari.ai{API_ROOT}/feedback/submissions"
# Each send is a real POST upstream, and the receiver's own limit is shared by
# everyone behind this gateway's address, so one caller must not spend it.
SENDS_PER_WINDOW = 5
WINDOW_SEC = 10 * 60


def new_feedback_rate_limiter() -> RateLimiter:
    """A fresh per-caller send limit, kept on ``app.state.feedback_rate_limiter``."""
    return RateLimiter(SENDS_PER_WINDOW, window_sec=WINDOW_SEC)


class FeedbackService:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def submit(self, body: FeedbackSubmission) -> None:
        try:
            async with (
                asyncio.timeout(10),
                httpx.AsyncClient(
                    timeout=10, follow_redirects=False, trust_env=False, transport=self._transport
                ) as client,
                client.stream("POST", FEEDBACK_RECEIVER, json=body.model_dump()) as response,
            ):
                if response.status_code == 204:
                    return
                # The status only: the body is the receiver's and may echo the message.
                logger.warning("Feedback receiver answered %s", response.status_code)
                if response.status_code in (413, 422, 429):
                    retry = response.headers.get("Retry-After", "")
                    retry_after = (
                        min(3600, max(1, int(retry)))
                        if retry.isascii() and retry.isdecimal() and len(retry) <= 6
                        else None
                    )
                    raise FeedbackDeliveryError(
                        response.status_code, retry_after if response.status_code == 429 else None
                    )
        except (httpx.HTTPError, TimeoutError):
            logger.warning("Feedback delivery could not be confirmed")
            raise FeedbackDeliveryError from None
        raise FeedbackDeliveryError
