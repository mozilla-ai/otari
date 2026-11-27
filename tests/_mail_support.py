"""A transport that keeps what a test's deployment sent, and the token reader for it.

Paired with the ``recorded_mail`` fixture, which puts this in place of the one a
configuration selects. A test asserts on the envelope the mailer built rather
than on the text of a log line.
"""

import re
from collections.abc import Callable

import pytest

from gateway.services.mail import MailEnvelope

_TOKEN_IN_LINK = re.compile(r"token=([A-Za-z0-9_.~-]+)")

MAIL_CONFIGURED = {
    "mail_transport": "smtp",
    "smtp_host": "smtp.example.com",
    "mail_from_email": "otari@example.com",
}
"""The transport settings a deployment needs before a send is attempted at all.

Spread into a test's own config. Paired with ``recorded_mail``, which is what keeps
the send off a socket.

NOTE: ``public_base_url`` is not here, and a deployment without one cannot deliver a
link whatever its transport. A test that needs delivery sets it alongside these,
which is also what lets a test set the transport and withhold the address.
"""


class RecordingTransport:
    """Keeps every envelope handed to it, newest last, and delivers nothing.

    Reports itself as SMTP, so a test asserting on the transport a delivery
    names describes what the deployment under test is configured for.
    """

    name = "smtp"

    def __init__(self) -> None:
        self.sent: list[MailEnvelope] = []

    @property
    def last(self) -> MailEnvelope:
        """The newest envelope of the whole test, for a test whose only send is the one under test.

        NOTE: This is not the newest envelope of the last action. An action
        that follows another send reads its own message through
        :meth:`sent_during`.
        """
        assert self.sent, "no mail was sent"
        return self.sent[-1]

    def sent_during[T](self, call: Callable[[], T]) -> tuple[T, MailEnvelope]:
        """Run ``call`` and answer with its result and the one message it sent.

        Fails where it sent none or several, so a token read from the envelope
        belongs to this action and cannot be an earlier send's.
        """
        before = len(self.sent)
        result = call()
        during = self.sent[before:]
        assert len(during) == 1, f"the call sent {len(during)} messages rather than one"
        return result, during[0]

    def deliver(self, envelope: MailEnvelope) -> None:
        self.sent.append(envelope)


def configure_mail(monkeypatch: pytest.MonkeyPatch, config: object) -> None:
    """Point a shared test config at a transport, for a suite that cannot build its own.

    NOTE: The caller must also use the ``recorded_mail`` fixture, or the send is attempted for real,
    and must have a ``public_base_url`` for a link to be deliverable at all.
    """
    for name, value in MAIL_CONFIGURED.items():
        monkeypatch.setattr(config, name, value)


def token_in(envelope: MailEnvelope) -> str:
    """The token carried by the link in a recorded message.

    Read from the plain-text body, which carries the same link as the HTML one.
    Fails the test where this message holds no token, rather than answering
    with an empty string.

    NOTE: Which message this is remains the caller's to establish. Reading a
    token out of the wrong envelope is not something this can detect, so an
    action that follows another send takes its envelope from
    :meth:`RecordingTransport.sent_during`.
    """
    match = _TOKEN_IN_LINK.search(envelope.message.text)
    assert match is not None, f"no token in the message titled {envelope.message.subject!r}"
    return match.group(1)
