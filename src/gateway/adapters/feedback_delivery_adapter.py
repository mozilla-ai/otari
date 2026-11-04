"""The core feedback delivery: a forward to the otari.ai intake."""

import uuid

from gateway.ports.feedback_delivery_port import FeedbackDeliveryPort
from gateway.services.feedback import FeedbackService


class ReceiverFeedbackDeliveryAdapter(FeedbackDeliveryPort):
    """``FeedbackDeliveryPort`` adapter that posts to the otari.ai intake.

    The sender's id stays here: it names an account on this deployment, which
    means nothing to the intake.
    """

    def __init__(self, service: FeedbackService | None = None) -> None:
        self._service = service or FeedbackService()

    async def deliver(self, *, message: str, reply_to: str | None, sender_id: uuid.UUID | None) -> None:
        del sender_id
        await self._service.submit(message, reply_to)
