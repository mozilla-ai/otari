"""The core ``FeedbackDeliveryPort`` adapter: an HTTPS post to the otari.ai intake.

Forwards a fresh payload holding the message alone, with no caller credentials,
browser headers or submitter, follows no redirect and does not retry.
"""

import asyncio

import httpx

from gateway.core.config import API_ROOT
from gateway.exceptions.feedback_exceptions import FeedbackDeliveryError
from gateway.log_config import logger

FEEDBACK_RECEIVER = f"https://api.otari.ai{API_ROOT}/feedback/submissions"


class HttpFeedbackDeliveryAdapter:
    """``FeedbackDeliveryPort`` over HTTPS, for every deployment but the intake's own."""

    def __init__(self, *, user_agent: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # Informational only: which build and which kind of deployment sent it.
        self._headers = {"User-Agent": user_agent}
        self._transport = transport

    async def submit(self, message: str, submitter: str) -> None:
        # Never sent: the receiver learns nothing about who wrote the message.
        del submitter
        try:
            async with (
                asyncio.timeout(10),
                httpx.AsyncClient(
                    timeout=10, follow_redirects=False, trust_env=False, transport=self._transport
                ) as client,
                client.stream("POST", FEEDBACK_RECEIVER, json={"message": message}, headers=self._headers) as response,
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
