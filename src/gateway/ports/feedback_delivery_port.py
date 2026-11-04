"""Where a message a signed-in person sends the Otari team goes once the gateway accepts it.

The core adapter forwards it to the otari.ai intake, and sends nothing that
identifies who wrote it but the address they agreed to be replied to at. A
build that operates the intake itself binds an adapter that delivers in this
process, where the sender is an account it already knows.

An adapter raises :class:`gateway.exceptions.feedback_exceptions.FeedbackDeliveryError`
when it could not confirm delivery, and the route answers with its status.
"""

import uuid
from typing import Protocol


class FeedbackDeliveryPort(Protocol):
    """What a build does with one accepted message."""

    async def deliver(self, *, message: str, reply_to: str | None, sender_id: uuid.UUID | None) -> None:
        """Deliver ``message``, or raise ``FeedbackDeliveryError``.

        ``reply_to`` is the session's own email, present only when the person
        was shown it. ``sender_id`` is the signed-in user, or ``None`` for the
        master key.
        """
        ...
