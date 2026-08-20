"""The unit of outgoing mail, and the failure a transport reports.

Split from the transports and the mailer so a template can be rendered, asserted
on, or previewed without anything that opens a socket.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MailMessage:
    """One rendered message, addressed to nobody yet.

    The recipient is the mailer's argument rather than a field here, so one
    rendered message can be sent to several addresses and a template's output
    is comparable in a test without a stand-in address in it.
    """

    subject: str
    html: str
    text: str


@dataclass(frozen=True, slots=True)
class MailEnvelope:
    """A message plus the addresses a transport needs to deliver it."""

    to: str
    sender_email: str
    sender_name: str
    message: MailMessage


@dataclass(frozen=True, slots=True)
class MailDelivery:
    """What became of one send.

    ``delivered`` is what a caller with a fallback branches on (an invitation
    still returns its accept link when this is false). ``reason`` is for the
    operator, not the recipient: it names why a transport refused or failed and
    is surfaced only on the master-key mail-test endpoint, never on a public
    error path.
    """

    delivered: bool
    transport: str
    reason: str | None = None


class MailDeliveryError(Exception):
    """A transport could not deliver a message.

    Raised by a transport and caught by the mailer, which is the one place that
    turns a delivery failure into a :class:`MailDelivery` rather than an
    exception: no caller should have to wrap a send in a ``try``.
    """


class MailNotConfiguredError(Exception):
    """A surface that has no non-mail fallback was reached on a deployment with no mail.

    Carries the settings that are missing so the refusal can name them. The API
    layer renders this as a 503; a surface that *does* have a fallback (an
    invitation's shareable link) never raises it and degrades instead.
    """

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        detail = ", ".join(missing) if missing else "mail_transport"
        super().__init__(f"Outgoing mail is not configured on this deployment (missing: {detail}).")


__all__ = [
    "MailDelivery",
    "MailDeliveryError",
    "MailEnvelope",
    "MailMessage",
    "MailNotConfiguredError",
]
