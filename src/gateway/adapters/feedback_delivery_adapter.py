"""The core feedback delivery: a forward to the otari.ai intake, without caller credentials or automatic retries."""

import asyncio
import uuid

import httpx

from gateway.core.config import API_ROOT
from gateway.exceptions.feedback_exceptions import FeedbackDeliveryError
from gateway.log_config import logger
from gateway.ports.feedback_delivery_port import FeedbackDeliveryPort

FEEDBACK_RECEIVER = f"https://api.otari.ai{API_ROOT}/feedback/submissions"


class ReceiverFeedbackDeliveryAdapter(FeedbackDeliveryPort):
    """``FeedbackDeliveryPort`` adapter that posts to the otari.ai intake.

    The sender's id stays here: it names an account on this deployment, which
    means nothing to the intake.
    """

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def deliver(self, *, message: str, reply_to: str | None, sender_id: uuid.UUID | None) -> None:
        del sender_id
        # Named field by field: the receiver refuses any field it does not know.
        payload = {"message": message}
        if reply_to is not None:
            payload["reply_to"] = reply_to
        try:
            async with (
                asyncio.timeout(10),
                httpx.AsyncClient(
                    timeout=10, follow_redirects=False, trust_env=False, transport=self._transport
                ) as client,
                client.stream("POST", FEEDBACK_RECEIVER, json=payload) as response,
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
