"""The seam a test reads a sent message through.

A test that needs the message a flow sent asks the recording transport for it,
rather than reading it out of a log. The envelope it records is the one the
mailer built, so a test asserts on the recipient, the subject and the rendered
body instead of on a log line's text.
"""

import pytest

from _mail_support import RecordingTransport, token_in

from gateway.core.config import GatewayConfig
from gateway.services.mail import MailEnvelope, MailMessage, Mailer

SMTP_CONFIGURED = {"smtp_host": "smtp.example.com", "mail_from_email": "otari@example.com"}

MESSAGE = MailMessage(
    subject="Verify your address",
    html='<a href="https://otari.example.com/verify#token=abc123">Verify</a>',
    text="Verify: https://otari.example.com/verify#token=abc123",
)


def _ready(**overrides: object) -> GatewayConfig:
    """A config that can send a message carrying a link back to itself."""
    return GatewayConfig(**{**SMTP_CONFIGURED, "public_base_url": "https://otari.example.com", **overrides})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_send_is_recorded_with_the_envelope_the_mailer_built(recorded_mail: RecordingTransport) -> None:
    delivery = await Mailer(_ready()).send(to="ada@example.com", message=MESSAGE)

    assert delivery.delivered
    assert [envelope.to for envelope in recorded_mail.sent] == ["ada@example.com"]
    assert recorded_mail.last.message.subject == "Verify your address"
    assert recorded_mail.last.sender_email == "otari@example.com"


@pytest.mark.asyncio
async def test_a_deployment_with_no_mail_records_nothing_and_reports_it(recorded_mail: RecordingTransport) -> None:
    """The fixture answers for the configuration, so a test about no mail still gets no mail.

    NOTE: A recorder that answered unconditionally would make every such test pass for the wrong
    reason, because the mailer would believe mail was configured.
    """
    delivery = await Mailer(GatewayConfig()).send(to="ada@example.com", message=MESSAGE)

    assert not delivery.delivered
    assert delivery.transport == "none"
    assert recorded_mail.sent == []


@pytest.mark.asyncio
async def test_a_recorded_send_reports_the_transport_a_real_one_would(recorded_mail: RecordingTransport) -> None:
    """A test asserting on the transport a delivery names stays truthful through the swap."""
    delivery = await Mailer(_ready()).send(to="ada@example.com", message=MESSAGE)

    assert delivery.transport == "smtp"


def test_the_token_of_a_recorded_message_is_read_from_its_link() -> None:
    envelope = _envelope()

    assert token_in(envelope) == "abc123"


def test_a_message_carrying_no_token_says_so_rather_than_answering_wrongly() -> None:
    envelope = _envelope(text="Your invitation was canceled.", html="<p>Your invitation was canceled.</p>")

    with pytest.raises(AssertionError):
        token_in(envelope)


def _envelope(**overrides: str) -> MailEnvelope:
    message = MailMessage(
        subject=overrides.get("subject", MESSAGE.subject),
        html=overrides.get("html", MESSAGE.html),
        text=overrides.get("text", MESSAGE.text),
    )
    return MailEnvelope(
        to="ada@example.com",
        sender_email="otari@example.com",
        sender_name="Otari",
        message=message,
    )
