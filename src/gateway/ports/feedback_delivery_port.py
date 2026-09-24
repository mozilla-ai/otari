"""Where a person's message to the Otari team goes once the gateway accepts it.

The core adapter forwards it over HTTPS to the otari.ai intake, which is what a
standalone deployment has always done. The seam exists for the one deployment
that is the intake's own neighbor: otari.ai's hosted control plane binds an
in-process adapter, so its users are not all one source address at the
receiver and a post does not leave through the public URL to come back in.

Stability: this interface is not frozen while Otari is pre-1.0.
"""

from typing import Protocol


class FeedbackDeliveryPort(Protocol):
    """Deliver one message, or raise ``FeedbackDeliveryError`` saying why not."""

    async def submit(self, message: str, submitter: str) -> None:
        """Deliver ``message``, already validated and trimmed.

        ``submitter`` is the caller the gateway authenticated, as an opaque key:
        the dashboard session's user id, or ``"master"`` for the master key. It
        is for an adapter that can act on it in-process (a per-person limit, an
        audit trail kept inside the same deployment). An adapter that sends the
        message across a network must not send it: the feedback promise is that
        no account identifier leaves with the message.

        Raises ``gateway.exceptions.feedback_exceptions.FeedbackDeliveryError``
        when delivery is refused or cannot be confirmed; its ``status_code`` and
        ``retry_after`` are what the route answers with.
        """
        ...
