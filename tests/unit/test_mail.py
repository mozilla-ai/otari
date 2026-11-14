"""Mail configuration, transport selection, the mailer, and the template renderer.

No real SMTP server is exercised. The transports are covered for the two things
that must never happen (a send that raises into the request that triggered it,
and a deployment discovering at send time that it has no mail), and the console
transport is what proves a *templated message actually sends* without one.
"""

import ast
import logging
import random
import smtplib
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from email import message_from_string, policy
from email.utils import parseaddr
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.core.config import MAIL_TRANSPORT_SETTINGS, GatewayConfig
from gateway.log_config import CredentialMaskingFilter
from gateway.log_config import logger as gateway_logger
from gateway.services import mail as mail_package
from gateway.services.mail import (
    ConsoleTransport,
    MailDelivery,
    MailDeliveryError,
    MailEnvelope,
    Mailer,
    MailMessage,
    MailNotConfiguredError,
    MailTemplateError,
    SmtpTransport,
    normalized_address,
    render_email,
    select_transport,
)
from gateway.services.mail.templates import FONT_FILES, MARK_PATH, _load
from gateway.services.mail.transports import build_mime
from gateway.services.tenancy.invitation_email import render_invitation_email
from gateway.services.tenancy.password_reset_email import render_account_exists_email, render_password_reset_email
from gateway.services.tenancy.verification_email import render_verification_email

MESSAGE = MailMessage(subject="Hi", html="<p>Hi</p>", text="Hi")


@pytest.fixture(autouse=True)
def _mail_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every test here to its own inputs.

    ``GatewayConfig()`` layers ``OTARI_*`` env vars (and a ``.env``) over its
    defaults, so a developer with ``OTARI_SMTP_HOST`` exported would flip the
    assertions about an unconfigured deployment, which are most of this file.
    """
    for name in (
        "OTARI_MAIL_TRANSPORT",
        "OTARI_SMTP_HOST",
        "OTARI_SMTP_PORT",
        "OTARI_SMTP_USER",
        "OTARI_SMTP_PASSWORD",
        "OTARI_MAIL_FROM_EMAIL",
        "OTARI_MAIL_FROM_NAME",
        "OTARI_PUBLIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def gateway_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Route the ``gateway`` logger (which does not propagate) into caplog."""
    gateway_logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="gateway")
    try:
        yield caplog
    finally:
        gateway_logger.removeHandler(caplog.handler)


SMTP_CONFIGURED = {"smtp_host": "smtp.example.com", "mail_from_email": "otari@example.com"}


def _ready(**overrides: object) -> GatewayConfig:
    """A config that can send a message carrying a link back to itself."""
    return GatewayConfig(**{**SMTP_CONFIGURED, "public_base_url": "https://otari.example.com", **overrides})  # type: ignore[arg-type]


# --- Configuration: which transport, and is it ready ------------------------


def test_mail_is_off_with_nothing_configured() -> None:
    """The state of every deployment that never heard of mail, and it is not an error."""
    config = GatewayConfig()
    assert config.effective_mail_transport == "none"
    assert config.mail_enabled is False
    assert config.mail_ready is False


def test_auto_selects_smtp_only_with_both_a_host_and_a_from_address() -> None:
    """A host with no from-address sends mail no recipient could trust; the reverse has nothing to send through."""
    assert GatewayConfig(smtp_host="smtp.example.com").effective_mail_transport == "none"
    assert GatewayConfig(mail_from_email="otari@example.com").effective_mail_transport == "none"
    assert GatewayConfig(**SMTP_CONFIGURED).effective_mail_transport == "smtp"  # type: ignore[arg-type]


def test_mail_ready_also_needs_a_public_base_url() -> None:
    """Every message the control plane sends carries a link back into this deployment."""
    configured = GatewayConfig(**SMTP_CONFIGURED)  # type: ignore[arg-type]
    assert configured.mail_enabled is True
    assert configured.mail_ready is False
    assert _ready().mail_ready is True


def test_none_turns_mail_off_even_where_smtp_is_configured() -> None:
    config = _ready(mail_transport="none")
    assert config.effective_mail_transport == "none"
    assert config.mail_enabled is False
    assert config.missing_mail_settings == ("mail_transport",)


def test_missing_settings_name_what_would_turn_mail_on() -> None:
    """ "Unavailable" is only honest if it says what to set."""
    assert GatewayConfig().missing_mail_settings == ("smtp_host", "mail_from_email", "public_base_url")
    assert GatewayConfig(smtp_host="smtp.example.com").missing_mail_settings == (
        "mail_from_email",
        "public_base_url",
    )
    assert GatewayConfig(**SMTP_CONFIGURED).missing_mail_settings == ("public_base_url",)  # type: ignore[arg-type]
    assert _ready().missing_mail_settings == ()


def test_readiness_never_depends_on_a_validator_having_run() -> None:
    """A config that was never validated must still answer truthfully.

    An explicit ``smtp`` with nothing behind it is refused at startup, but the
    refusal is not what makes the answer here correct: ``effective_mail_transport``
    reports what a send would actually use, so it agrees with
    ``select_transport`` in every state, validated or not. A readiness answer
    that is only truthful because a check ran somewhere else is the shape of bug
    this design exists to rule out.
    """
    half_configured = GatewayConfig(mail_transport="smtp", smtp_host="smtp.example.com")

    assert half_configured.effective_mail_transport == "none"
    assert half_configured.mail_enabled is False
    assert half_configured.mail_ready is False
    assert select_transport(half_configured) is None
    assert Mailer(half_configured).can_send_links is False
    # And it still names what is missing rather than reporting nothing to fix.
    assert half_configured.missing_mail_settings == ("mail_from_email", "public_base_url")


def test_config_and_transport_never_disagree_about_whether_mail_exists() -> None:
    """The two readers of "is mail configured" must not be able to drift apart.

    ``/api/v1/bootstrap`` reads the config property and the invitation path asks the
    mailer; if those could differ, the dashboard would offer an affordance the
    request path refuses. Exhaustive over the settings that decide it, because
    the states that broke this before were the ones nobody thinks to write a
    case for.
    """
    for transport in MAIL_TRANSPORT_SETTINGS:
        for host in (None, "smtp.example.com"):
            for sender in (None, "otari@example.com"):
                for url in (None, "https://otari.example.com"):
                    config = GatewayConfig(
                        mail_transport=transport,
                        smtp_host=host,
                        mail_from_email=sender,
                        public_base_url=url,
                    )
                    state = (transport, host, sender, url)
                    assert config.mail_enabled == (select_transport(config) is not None), state
                    assert config.mail_ready == Mailer(config).can_send_links, state
                    # "Empty exactly when ready" is the contract GET /api/v1/settings/mail
                    # publishes, so it holds for every state and not just the tidy ones.
                    assert (config.missing_mail_settings == ()) == config.mail_ready, state


def test_an_unknown_transport_is_refused_at_load() -> None:
    with pytest.raises(ValueError, match="mail_transport must be one of"):
        GatewayConfig(mail_transport="sendgrid")


def test_asking_for_smtp_without_smtp_settings_fails_at_startup_not_at_send_time() -> None:
    config = GatewayConfig(mail_transport="smtp", smtp_host="smtp.example.com")
    with pytest.raises(ValueError, match="mail_from_email"):
        config.validate_mail_transport()


def test_selecting_console_warns_that_it_writes_tokens_to_the_log(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    """The one sanctioned exception to never-log-a-token, so it says so out loud.

    A console "delivery" writes the message body, and an invitation body carries
    the accept token. The transport is opt-in per deployment and useless without
    the link, so the trade is announced at startup rather than redacted away.
    """
    GatewayConfig(mail_transport="console").validate_mail_transport()

    assert "token-bearing" in gateway_logs.text


def test_a_real_transport_warns_about_nothing(gateway_logs: pytest.LogCaptureFixture) -> None:
    GatewayConfig(**SMTP_CONFIGURED).validate_mail_transport()  # type: ignore[arg-type]
    GatewayConfig().validate_mail_transport()

    assert gateway_logs.text == ""


@pytest.mark.parametrize(
    "sender",
    [
        "not-an-address",
        "<otari@example.com>",
        "Otari <otari@example.com>",
        "otari@example.com(ops)",
        "otari@[127.0.0.1]",
        '"otari"@example.com',
    ],
)
def test_an_explicit_smtp_transport_refuses_a_from_address_that_is_not_a_plain_address(sender: str) -> None:
    config = GatewayConfig(mail_transport="smtp", smtp_host="smtp.example.com", mail_from_email=sender)

    with pytest.raises(ValueError, match="mail_from_email is not a plain email address"):
        config.validate_mail_transport()


def test_the_auto_default_does_not_refuse_an_odd_from_address() -> None:
    """The check rides on the new explicit opt-in, so it cannot break a running deployment.

    ``mail_transport`` is introduced by this change, so nothing out there is
    already set to ``smtp``; a deployment that has been sending under the
    implicit path with an unusual address keeps booting.
    """
    GatewayConfig(smtp_host="smtp.example.com", mail_from_email="otari@localhost").validate_mail_transport()


def test_the_auto_default_validates_clean_with_nothing_configured() -> None:
    """No mail is the ordinary state of a self-hosted deployment, not a misconfiguration."""
    GatewayConfig().validate_mail_transport()


# --- Transport selection ----------------------------------------------------


def test_no_transport_is_an_absent_object_not_a_disabled_one() -> None:
    """A surface asks whether mail exists; it never has to interpret a flag on a sender."""
    assert select_transport(GatewayConfig()) is None
    assert isinstance(select_transport(GatewayConfig(**SMTP_CONFIGURED)), SmtpTransport)  # type: ignore[arg-type]
    assert isinstance(select_transport(GatewayConfig(mail_transport="console")), ConsoleTransport)


def test_console_is_a_transport_without_any_smtp_settings() -> None:
    mailer = Mailer(GatewayConfig(mail_transport="console", public_base_url="https://otari.example.com"))
    assert mailer.transport_name == "console"
    assert mailer.is_configured is True
    assert mailer.can_send_links is True


# --- The mailer -------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_reports_the_no_transport_case_rather_than_raising() -> None:
    delivery = await Mailer(GatewayConfig()).send(to="ada@example.com", message=MESSAGE)
    assert delivery.delivered is False
    assert delivery.transport == "none"
    assert delivery.reason is not None


@pytest.mark.asyncio
async def test_send_reports_a_failure_rather_than_raising() -> None:
    """An unreachable host must not turn a mail failure into a request failure."""
    # Patched rather than dialing a closed port: on a network that black-holes
    # loopback instead of refusing, a real connect would wait out
    # SMTP_TIMEOUT_SECONDS before this assertion could pass.
    config = _ready()
    with patch("gateway.services.mail.transports.smtplib.SMTP", side_effect=OSError("connection refused")):
        delivery = await Mailer(config).send(to="ada@example.com", message=MESSAGE)
    assert delivery.delivered is False
    assert delivery.transport == "smtp"
    assert delivery.reason is not None


@pytest.mark.asyncio
async def test_a_failure_reason_never_carries_the_recipient_into_the_logs(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    config = _ready()
    with patch("gateway.services.mail.transports.smtplib.SMTP", side_effect=OSError("connection refused")):
        await Mailer(config).send(to="ada@example.com", message=MESSAGE)
    assert "ada@example.com" not in gateway_logs.text
    assert "a***@example.com" in gateway_logs.text


def _smtp_client_failing_with(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.sendmail.side_effect = exc
    return client


@pytest.mark.asyncio
async def test_a_refused_recipient_is_reported_without_the_address(gateway_logs: pytest.LogCaptureFixture) -> None:
    recipient = "secret.invitee@elsewhere.test"
    refusal = smtplib.SMTPRecipientsRefused({recipient: (550, b"5.1.1 no such user")})
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = _smtp_client_failing_with(refusal)
        delivery = await Mailer(_ready()).send(to=recipient, message=MESSAGE)

    assert delivery.delivered is False
    assert delivery.reason == "The email server refused the recipient (550)."
    assert all(recipient not in record.getMessage() for record in gateway_logs.records)
    assert "s***@elsewhere.test" in gateway_logs.text


_SERVER_REPLY = b"5.7.1 <secret.invitee@elsewhere.test> is not accepted"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (
            smtplib.SMTPAuthenticationError(535, _SERVER_REPLY),
            "The email server refused the sign-in (535).",
        ),
        (
            smtplib.SMTPConnectError(421, _SERVER_REPLY),
            "The email server refused the connection (421).",
        ),
        (
            smtplib.SMTPDataError(554, _SERVER_REPLY),
            "The email server refused the message (554).",
        ),
        (
            smtplib.SMTPHeloError(501, _SERVER_REPLY),
            "The email server refused the greeting (501).",
        ),
        (
            smtplib.SMTPNotSupportedError("STARTTLS extension not supported by server."),
            "The email server lacks a feature this send needs: STARTTLS extension not supported by server.",
        ),
        (
            smtplib.SMTPResponseException(451, _SERVER_REPLY),
            "The email server refused the request (451).",
        ),
        (
            smtplib.SMTPSenderRefused(553, _SERVER_REPLY, "otari@example.com"),
            "The email server refused the sender (553).",
        ),
        (
            smtplib.SMTPServerDisconnected("Connection unexpectedly closed"),
            "The email server closed the connection.",
        ),
        (
            smtplib.SMTPException(_SERVER_REPLY.decode()),
            "The email server could not complete the send.",
        ),
        (
            TimeoutError("timed out"),
            "The connection to the email server failed: timed out",
        ),
        (
            TimeoutError(),
            "The email server did not answer.",
        ),
        (
            ValueError(_SERVER_REPLY.decode()),
            "The message could not be prepared for sending.",
        ),
    ],
)
def test_a_failed_send_names_what_failed_and_no_address(failure: Exception, reason: str) -> None:
    envelope = MailEnvelope(
        to="secret.invitee@elsewhere.test", sender_email="otari@example.com", sender_name="Otari", message=MESSAGE
    )
    transport = SmtpTransport(host="smtp.example.com", port=587, use_tls=False, user=None, password=None)
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = _smtp_client_failing_with(failure)
        with pytest.raises(MailDeliveryError) as raised:
            transport.deliver(envelope)

    assert str(raised.value) == reason


@pytest.mark.asyncio
async def test_send_never_raises_even_on_a_malformed_header() -> None:
    """Defense in depth: message serialization runs before any socket I/O.

    ``as_string()`` is evaluated as a plain argument expression before
    ``client.sendmail(...)`` is even called, so this raises the same
    ``HeaderParseError`` a real server would, with no network needed to prove
    the mailer still reports a failure instead of propagating it.
    """
    message = MailMessage(subject="Broken\r\nheader: injected", html="<p>Hi</p>", text="Hi")
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        # Bypasses build_mime's own sanitization by handing the header a value
        # that only email.header rejects once assembled.
        with patch("gateway.services.mail.transports.sanitize_header_value", side_effect=lambda value: value):
            delivery = await Mailer(_ready()).send(to="ada@example.com", message=message)
    assert delivery.delivered is False


_NOT_PLAIN = [
    "grace@example.test(x.y)",
    "x<grace@example.test>",
    "<grace@example.test>",
    "grace@example.test, ada@example.test",
    "grace@example.test\r\nRCPT TO:<ada@example.test>",
    "grace@example.test\n",
    " grace@example.test",
    "grace hopper@example.test",
    '"grace\nhopper"@example.test',
    "grace@example.test\x00",
    "grace\u200b@example.test",
    "grace@example.test\x1f",
    "\x1cgrace@example.test",
    ":" * 2000,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("recipient", ["", *_NOT_PLAIN])
async def test_a_recipient_that_is_not_a_plain_address_is_never_handed_to_smtp(recipient: str) -> None:
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        client = MagicMock()
        mock_smtp.return_value.__enter__.return_value = client
        delivery = await Mailer(_ready()).send(to=recipient, message=MESSAGE)

    assert delivery == MailDelivery(
        delivered=False, transport="smtp", reason="The recipient is not a plain email address."
    )
    mock_smtp.assert_not_called()
    client.sendmail.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("recipient", _NOT_PLAIN)
async def test_a_recipient_that_is_not_a_plain_address_is_not_in_the_log(
    recipient: str, gateway_logs: pytest.LogCaptureFixture
) -> None:
    await Mailer(_ready()).send(to=recipient, message=MESSAGE)

    assert "Failed to send mail to *** via smtp" in gateway_logs.text
    assert "example.test" not in gateway_logs.text


@pytest.mark.asyncio
@pytest.mark.parametrize("recipient", ["grace", "grace@localhost", '"grace"@example.test', "grace\\x@example.test"])
async def test_a_recipient_that_is_not_an_email_address_is_never_handed_to_smtp(recipient: str) -> None:
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        delivery = await Mailer(_ready()).send(to=recipient, message=MESSAGE)

    assert delivery.delivered is False
    mock_smtp.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_to", _NOT_PLAIN)
async def test_a_reply_address_that_is_not_a_plain_address_is_never_handed_to_smtp(reply_to: str) -> None:
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        delivery = await Mailer(_ready()).send(to="support@example.com", message=MESSAGE, reply_to=reply_to)

    assert delivery == MailDelivery(
        delivered=False, transport="smtp", reason="The reply address is not a plain email address."
    )
    mock_smtp.assert_not_called()


@pytest.mark.asyncio
async def test_the_envelope_recipient_is_exactly_the_address_given() -> None:
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        client = MagicMock()
        mock_smtp.return_value.__enter__.return_value = client
        delivery = await Mailer(_ready()).send(to="Grace+Otari@Mail.Example.test", message=MESSAGE)

    assert delivery.delivered is True
    _sender, recipients, body = client.sendmail.call_args.args
    assert recipients == ["Grace+Otari@Mail.Example.test"]
    assert "\nTo: Grace+Otari@Mail.Example.test\n" in body


def test_build_mime_puts_no_newline_in_any_header() -> None:
    """The header half of the same guarantee, asserted on the assembled message."""
    envelope = MailEnvelope(
        to="ada@example.com>\r\nBcc: attacker@evil.example",
        sender_email="otari@example.com\r\nBcc: attacker@evil.example",
        sender_name="Otari\r\nBcc: attacker@evil.example",
        message=MESSAGE,
        reply_to="grace@example.com\r\nBcc: attacker@evil.example",
    )

    assembled = build_mime(envelope)

    for header in ("From", "To", "Subject", "Reply-To"):
        assert "\r" not in assembled[header], header
        assert "\n" not in assembled[header], header
    # And it serializes, which is what a raw newline would have prevented.
    assert assembled.as_string()


def test_reply_to_is_a_header_only_when_given() -> None:
    envelope = MailEnvelope(
        to="support@example.com", sender_email="otari@example.com", sender_name="Otari", message=MESSAGE
    )

    assert build_mime(envelope)["Reply-To"] is None
    assert build_mime(replace(envelope, reply_to="grace@example.com"))["Reply-To"] == "grace@example.com"


@pytest.mark.parametrize(
    "name",
    [
        "grace@example.com via Otari",
        "Zoë via Otari",
        'Otari "Support"',
        "=?utf-8?q?Otari_Security?=@evil.example via Otari",
    ],
)
def test_a_mail_client_reads_the_sender_name_as_a_name(name: str) -> None:
    """Unquoted, a name holding an address is read as the From address; unencoded, "=?" as other text."""
    envelope = MailEnvelope(
        to="support@example.com", sender_email="otari@example.com", sender_name=name, message=MESSAGE
    )

    wire = build_mime(envelope).as_string()

    wire.encode("ascii")
    (sender,) = message_from_string(wire, policy=policy.default)["From"].addresses
    assert (sender.display_name, sender.addr_spec) == (name, "otari@example.com")


@pytest.mark.asyncio
async def test_send_names_the_sender_for_one_message_only() -> None:
    mailer = Mailer(_ready(mail_from_name="Otari"))
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        client = MagicMock()
        mock_smtp.return_value.__enter__.return_value = client
        await mailer.send(to="support@example.com", message=MESSAGE, sender_name="grace@example.com via Otari")
        relayed = client.sendmail.call_args.args
        await mailer.send(to="support@example.com", message=MESSAGE)
        ordinary = client.sendmail.call_args.args

    from_address = SMTP_CONFIGURED["mail_from_email"]
    assert relayed[0] == ordinary[0] == from_address
    assert f'\nFrom: "grace@example.com via Otari" <{from_address}>\n' in relayed[2]
    assert f"\nFrom: Otari <{from_address}>\n" in ordinary[2]


@pytest.mark.asyncio
async def test_send_puts_the_reply_address_on_the_wire() -> None:
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        client = MagicMock()
        mock_smtp.return_value.__enter__.return_value = client
        delivery = await Mailer(_ready()).send(to="support@example.com", message=MESSAGE, reply_to="grace@example.com")

    assert delivery.delivered is True
    _sender, recipients, body = client.sendmail.call_args.args
    assert recipients == ["support@example.com"]
    assert "\nReply-To: grace@example.com\n" in body


@pytest.mark.asyncio
async def test_send_survives_a_crlf_in_the_configured_from_name() -> None:
    """mail_from_name is operator config, not caller input the subject path already sanitizes.

    Without stripping it, a stray CR/LF reaches ``as_string()`` unescaped and
    raises ``HeaderParseError`` on every send, which the mailer would report as
    a silent undelivered with nothing pointing at the From name as the cause.
    """
    config = _ready(mail_from_name="Otari\r\nBcc: attacker@example.com")
    with patch("gateway.services.mail.transports.smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = MagicMock()
        delivery = await Mailer(config).send(to="ada@example.com", message=MESSAGE)
    assert delivery.delivered is True


@pytest.mark.asyncio
async def test_a_templated_message_sends_over_a_configured_transport(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    """The definition of done, end to end, with no SMTP server to stand up."""
    mailer = Mailer(GatewayConfig(mail_transport="console", public_base_url="https://otari.example.com"))
    delivery = await mailer.send_template(
        "mail_test",
        to="ada@example.com",
        subject="Otari test message",
        values={"UI_BASE_URL": "https://otari.example.com", "TRANSPORT": "console"},
    )
    assert delivery.delivered is True
    assert delivery.transport == "console"
    assert "Your Otari mail settings work" in gateway_logs.text
    # The recipient is redacted here too: these lines land in the same stream.
    assert "ada@example.com" not in gateway_logs.text


def test_the_console_transport_keeps_its_link_past_the_credential_mask(
    gateway_logs: pytest.LogCaptureFixture,
) -> None:
    """The handler masks credentials in every record; this record's link is the point of it."""
    gateway_logs.handler.addFilter(mask := CredentialMaskingFilter())
    try:
        ConsoleTransport().deliver(
            MailEnvelope(
                to="ada@example.com",
                sender_email="otari@example.com",
                sender_name="Otari",
                message=MailMessage(
                    subject="Join",
                    html="<p>Join</p>",
                    text="Accept: https://otari.example.com/#/accept-invitation?token=abc123",
                ),
            )
        )
    finally:
        gateway_logs.handler.removeFilter(mask)

    assert "accept-invitation?token=abc123" in gateway_logs.text


def test_require_ready_refuses_and_names_what_is_missing() -> None:
    """What a surface with no non-mail fallback (password reset) raises."""
    with pytest.raises(MailNotConfiguredError) as excinfo:
        Mailer(GatewayConfig()).require_ready()
    assert excinfo.value.missing == ("smtp_host", "mail_from_email", "public_base_url")
    assert "smtp_host" in str(excinfo.value)


def test_require_ready_passes_once_a_linked_message_can_be_sent() -> None:
    Mailer(_ready()).require_ready()


def test_a_configured_transport_without_a_public_url_still_cannot_send_links() -> None:
    mailer = Mailer(GatewayConfig(**SMTP_CONFIGURED))  # type: ignore[arg-type]
    assert mailer.is_configured is True
    assert mailer.can_send_links is False
    with pytest.raises(MailNotConfiguredError):
        mailer.require_ready()


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        pytest.param({"mail_transport": "none"}, False, id="no transport"),
        pytest.param({"mail_transport": "console"}, False, id="console transport"),
        pytest.param({"public_base_url": None}, False, id="SMTP with no base URL"),
        pytest.param({"public_base_url": "   "}, False, id="SMTP with a whitespace-only base URL"),
        pytest.param({"public_base_url": "otari.example.com"}, False, id="SMTP with a base URL that has no scheme"),
        pytest.param(
            {"public_base_url": None, "ui_base_url": "https://app.example.com"},
            False,
            id="SMTP with an interface address and no base URL",
        ),
        pytest.param({"public_base_url": "https:///path"}, False, id="SMTP with a base URL that has no host"),
        pytest.param({"public_base_url": "https://[broken"}, False, id="SMTP with a base URL that cannot be read"),
        pytest.param({}, True, id="SMTP with a base URL"),
        pytest.param({"public_base_url": "HTTP://otari.example.com"}, True, id="SMTP with an uppercase scheme"),
    ],
)
def test_links_are_delivered_only_through_a_transport_that_delivers_and_with_an_absolute_address(
    settings: dict[str, object], expected: bool
) -> None:
    assert Mailer(_ready(**settings)).can_deliver_links is expected


def test_each_transport_states_whether_it_delivers_to_recipients() -> None:
    smtp = select_transport(_ready())
    console = select_transport(_ready(mail_transport="console"))

    assert smtp is not None
    assert console is not None
    assert (smtp.delivers_to_recipients, console.delivers_to_recipients) == (True, False)


def test_link_is_absolute_when_the_deployment_knows_its_address_and_relative_otherwise() -> None:
    assert Mailer(_ready()).link("/#/accept-invitation?token=abc") == (
        "https://otari.example.com/#/accept-invitation?token=abc"
    )
    assert Mailer(GatewayConfig()).link("/#/accept-invitation?token=abc") == "/#/accept-invitation?token=abc"


def test_link_names_the_interface_when_an_edge_serves_it_elsewhere() -> None:
    # A link in an email is followed by a browser, so it has to name wherever
    # the interface is reached, not the address this process answers on.
    mailer = Mailer(_ready(ui_base_url="https://app.example.com/ui"))

    assert mailer.link("/#/verify-email?token=abc") == "https://app.example.com/ui/#/verify-email?token=abc"


def test_whatever_can_send_links_permits_is_followable_from_an_inbox() -> None:
    # ``can_send_links`` gates the send, so it has to still imply an absolute
    # link now that links are built from the interface address rather than from
    # the one that flag reads.
    mailer = Mailer(_ready(ui_base_url="https://app.example.com/ui"))

    assert mailer.can_send_links is True
    assert mailer.link("/#/verify-email?token=t") == "https://app.example.com/ui/#/verify-email?token=t"


def test_a_query_on_the_interface_address_travels_ahead_of_the_hash_route() -> None:
    # An edge serving one interface for several deployments needs each link to
    # say which one built it. Placed before the hash so the page's own location
    # carries it, where the route's query would only reach the route.
    mailer = Mailer(_ready(ui_base_url="https://app.example.com/ui/?edge=eu"))

    assert mailer.link("/#/verify-email?token=abc") == "https://app.example.com/ui/?edge=eu#/verify-email?token=abc"
    assert mailer.link("/x") == "https://app.example.com/ui/x?edge=eu"
    assert mailer.link("/x?y=1") == "https://app.example.com/ui/x?y=1&edge=eu"


def test_a_trailing_slash_inside_a_query_value_survives_the_link() -> None:
    mailer = Mailer(_ready(ui_base_url="https://app.example.com/ui/?edge=team/"))

    assert mailer.link("/#/verify-email?token=abc") == "https://app.example.com/ui/?edge=team/#/verify-email?token=abc"


def test_asset_root_is_the_interface_address_without_its_query() -> None:
    config = _ready(ui_base_url="https://otari.example.com/dashboard/?region=eu")
    assert Mailer(config).asset_root == "https://otari.example.com/dashboard"
    assert Mailer(GatewayConfig()).asset_root is None


def test_link_does_not_double_a_trailing_slash() -> None:
    assert Mailer(_ready(public_base_url="https://otari.example.com/")).link("/x") == "https://otari.example.com/x"


# --- The template renderer --------------------------------------------------


def test_render_wraps_a_body_in_the_shared_layout() -> None:
    message = render_email("mail_test", subject="Otari test message", values={"UI_BASE_URL": "u", "TRANSPORT": "t"})
    assert message.html.startswith("<!doctype html>")
    assert "<title>Otari test message</title>" in message.html
    assert "Your Otari mail settings work" in message.html
    assert message.text.rstrip().endswith("Otari")


def test_the_mark_and_fonts_load_from_the_asset_root_and_are_left_out_without_one() -> None:
    values = {"UI_BASE_URL": "u", "TRANSPORT": "t"}
    with_root = render_email("mail_test", subject="x", values=values, asset_root="https://otari.example.com/ui")
    assert f'src="https://otari.example.com/ui/{MARK_PATH}"' in with_root.html
    for _, path, _ in FONT_FILES:
        assert f"url('https://otari.example.com/ui/{path}')" in with_root.html

    without = render_email("mail_test", subject="x", values=values)
    assert "<img" not in without.html
    assert "@font-face" not in without.html


def test_every_body_template_is_a_run_of_card_rows() -> None:
    """The layout places a body between table rows, where a browser hoists anything else out of the card."""
    package = Path(mail_package.__file__).parents[2] / "templates" / "email"
    bodies = sorted(path for path in package.glob("*.html") if not path.name.startswith("_"))
    assert bodies, "no body templates found; this test would pass vacuously"
    for path in bodies:
        body = path.read_text(encoding="utf-8").strip()
        assert body.startswith("<tr>") and body.endswith("</tr>"), path.name


@pytest.mark.parametrize(
    "render",
    [
        lambda root: render_invitation_email(
            organization_name="Acme",
            inviter_name="Ada",
            inviter_email="ada@example.com",
            role="member",
            accept_link="https://otari.example.com/#/accept-invitation?token=t",
            expiry_hours=168,
            asset_root=root,
        ),
        lambda root: render_verification_email(
            verify_link="https://otari.example.com/v", expiry_hours=24, asset_root=root
        ),
        lambda root: render_password_reset_email(
            reset_link="https://otari.example.com/r", expiry_hours=2, asset_root=root
        ),
        lambda root: render_account_exists_email(
            sign_in_link="https://otari.example.com/", has_password=True, provider=None, asset_root=root
        ),
    ],
    ids=["invitation", "verify_email", "password_reset", "account_exists"],
)
def test_every_message_loads_its_mark_and_fonts_from_the_asset_root(render: Callable[[str], MailMessage]) -> None:
    html = render("https://otari.example.com/ui").html
    assert f'src="https://otari.example.com/ui/{MARK_PATH}"' in html
    assert "@font-face" in html


@pytest.mark.asyncio
async def test_send_template_loads_the_mark_from_the_deployment() -> None:
    mailer = Mailer(_ready(ui_base_url="https://otari.example.com/ui/?region=eu"))
    with patch.object(Mailer, "send", autospec=True) as send:
        await mailer.send_template(
            "mail_test", to="ada@example.com", subject="x", values={"UI_BASE_URL": "u", "TRANSPORT": "t"}
        )
    message = send.call_args.kwargs["message"]
    assert f'src="https://otari.example.com/ui/{MARK_PATH}"' in message.html


def test_every_asset_a_message_loads_is_a_file_the_dashboard_serves() -> None:
    """A moved file would break the mark or the fonts in every message without failing anything else."""
    public = Path(__file__).parents[2] / "web" / "public"
    for path in (MARK_PATH, *(path for _, path, _ in FONT_FILES)):
        assert (public / path).is_file(), path


def test_the_preheader_is_filled_escaped_and_hidden() -> None:
    message = render_email(
        "mail_test",
        subject="x",
        values={"UI_BASE_URL": "u", "TRANSPORT": "<b>smtp</b>"},
        preheader="Sent over {{TRANSPORT}}.",
    )
    assert "Sent over &lt;b&gt;smtp&lt;/b&gt;." in message.html
    assert 'style="display:none;' in message.html
    assert "Sent over" not in message.text

    assert (
        "display:none" not in render_email("mail_test", subject="x", values={"UI_BASE_URL": "u", "TRANSPORT": "t"}).html
    )


def test_a_placeholder_with_no_value_fails_here_rather_than_in_an_inbox() -> None:
    with pytest.raises(MailTemplateError, match="TRANSPORT"):
        render_email("mail_test", subject="Otari test message", values={"UI_BASE_URL": "u"})


def test_every_shipped_template_loads() -> None:
    """The name constraint below must not reject a template this package ships.

    ``_layout.html`` starts with an underscore, so a stricter pattern would have
    broken every message while each body template still looked fine on its own.
    """
    package = Path(mail_package.__file__).parents[2] / "templates" / "email"
    shipped = sorted(path.name for path in package.iterdir() if path.suffix in {".html", ".txt"})
    assert shipped, "no templates found; this test would pass vacuously"
    for name in shipped:
        assert _load(name)


def test_a_template_name_that_is_not_ours_is_refused() -> None:
    """``name`` is interpolated into a resource path and the result is cached forever."""
    for name in ("../../../etc/passwd", "..%2fsecrets.html", "/etc/passwd", "nope.exe", ""):
        with pytest.raises(MailTemplateError, match="valid email template name"):
            _load(name)


def test_a_missing_template_is_named() -> None:
    with pytest.raises(MailTemplateError, match="password_reset"):
        render_email("password_reset", subject="x", values={})


def test_a_reserved_value_name_is_refused_rather_than_silently_overwritten() -> None:
    with pytest.raises(MailTemplateError, match="SUBJECT"):
        render_email("mail_test", subject="x", values={"SUBJECT": "mine", "UI_BASE_URL": "u", "TRANSPORT": "t"})
    with pytest.raises(MailTemplateError, match="MARK"):
        render_email("mail_test", subject="x", values={"MARK": "<img>", "UI_BASE_URL": "u", "TRANSPORT": "t"})


def test_a_value_is_never_rescanned_as_a_placeholder() -> None:
    """One pass, so a value that looks like a placeholder is emitted verbatim.

    A sequential replace-per-value would substitute into text an earlier pass
    had just inserted, which is how a free-text organization name becomes an
    injection into its own email.
    """
    message = render_email(
        "mail_test",
        subject="{{TRANSPORT}}",
        values={"UI_BASE_URL": "{{TRANSPORT}}", "TRANSPORT": "console"},
    )
    assert message.subject == "console"
    assert "{{TRANSPORT}}" in message.text


def test_the_subject_is_stripped_of_newlines_but_the_body_is_not() -> None:
    message = render_email(
        "mail_test",
        subject="Test {{TRANSPORT}}",
        values={"UI_BASE_URL": "u", "TRANSPORT": "smtp\r\nBcc: attacker@example.com"},
    )
    assert "\r" not in message.subject
    assert "\n" not in message.subject


@pytest.mark.parametrize("subject", ["=?utf-8?b?4oCuZXZpbOKArA==?= invites you", "Join =?utf-8?q?Acme?="])
def test_a_mail_client_reads_the_subject_as_the_text_given(subject: str) -> None:
    envelope = MailEnvelope(
        to="support@example.com",
        sender_email="otari@example.com",
        sender_name="Otari",
        message=replace(MESSAGE, subject=subject),
    )

    received = message_from_string(build_mime(envelope).as_string(), policy=policy.default)

    assert received["Subject"] == subject


# --- The shape of the guard -------------------------------------------------


def test_the_mail_package_gates_nothing_behind_an_assert() -> None:
    """No availability check here may be an ``assert``, at any depth.

    ``python -O`` strips assertions, so one guarding a send has two different
    wrong behaviors depending on how the process was started: a crash at send
    time unoptimized, and a silent no-op optimized. Those are precisely the two
    answers #648 rules out, from one line. otari.ai's ``send_email`` is the live
    example (``assert settings.emails_enabled``), which is why this is asserted
    structurally rather than trusted to review: the mail package's guarantees
    are ``return``s and typed exceptions, and a future edit must not quietly
    swap one for an assertion.
    """
    package = Path(mail_package.__file__).parent
    modules = sorted(package.glob("*.py"))
    assert modules, "no mail modules found; this test would pass vacuously"

    offenders = [
        f"{path.name}:{node.lineno}"
        for path in modules
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Assert)
    ]
    assert offenders == []


# --- Address handling -------------------------------------------------------


def test_normalized_address_lowercases_trims_and_refuses_a_non_address() -> None:
    assert normalized_address("  Ada@Example.COM ") == "ada@example.com"
    assert normalized_address("ada@example") is None
    assert normalized_address("not an address") is None
    assert normalized_address("") is None


def test_normalized_address_refuses_the_edge_shapes() -> None:
    assert normalized_address("a@b.c") == "a@b.c"
    assert normalized_address("a@b.c.d") == "a@b.c.d"
    assert normalized_address("@b.c") is None
    assert normalized_address("a@.bc") is None
    assert normalized_address("a@bc.") is None
    assert normalized_address("a@b@c.d") is None
    assert normalized_address("a b@c.d") is None
    assert normalized_address("a@b .c") is None


def test_normalized_address_refuses_an_empty_domain_label() -> None:
    """A mail server refuses these, and an invitation to one used to fail where nobody saw it."""
    assert normalized_address("x@example.test.") is None
    assert normalized_address("x@.example.test") is None
    assert normalized_address("x@example..test") is None
    assert normalized_address("x@sub.example.test") == "x@sub.example.test"


_ACCEPTED_SPELLINGS = {
    "grace@example.test": "grace@example.test",
    "grace+otari@example.test": "grace+otari@example.test",
    "grace.hopper@mail.eu.example.test": "grace.hopper@mail.eu.example.test",
    "  Grace@Example.TEST\t": "grace@example.test",
    "o'brien_x-y=z!#$%&*/?^`{|}~@example.test": "o'brien_x-y=z!#$%&*/?^`{|}~@example.test",
}

_REFUSED_SPELLINGS = [
    "grace@example.test(x.y)",
    "(x.y)grace@example.test",
    "x<grace@example.test>",
    "<grace@example.test>",
    '"grace"@example.test',
    '"grace hopper"@example.test',
    "grace@example.test,ada@example.test",
    "grace@example.test;ada@example.test",
    "team:grace@example.test;",
    "a@[127.0.0.1]",
    "grace\\hopper@example.test",
    "grace\x00@example.test",
    "\x00grace@example.test",
    "grace\thopper@example.test",
    "grace@example.test\x7f",
    "grace@exa\x85mple.test",
    "grace\u200b@example.test",
    "grace@\u202eexample.test",
    "=?utf-8?q?ada=40example.test=2C?=grace@example.test",
    "grace@=?utf-8?q?example?=.test",
]


@pytest.mark.parametrize(("spelling", "address"), _ACCEPTED_SPELLINGS.items())
def test_an_accepted_address_is_the_address_an_email_to_it_goes_to(spelling: str, address: str) -> None:
    assert normalized_address(spelling) == address
    assert parseaddr(address) == ("", address)


@pytest.mark.parametrize("spelling", _REFUSED_SPELLINGS)
def test_text_that_is_more_than_a_plain_address_is_not_an_address(spelling: str) -> None:
    assert normalized_address(spelling) is None


def test_every_address_accepted_from_generated_text_is_the_address_an_email_to_it_goes_to() -> None:
    extras = "ab.@+-_'!=? \t\x00\x7f()<>[],;:\\\"\u00f6"
    generator = random.Random(20240)
    accepted = 0
    for _ in range(20_000):
        characters = list("grace@example.test")
        for _ in range(generator.randint(0, 3)):
            characters.insert(generator.randint(0, len(characters)), generator.choice(extras))
        text = "".join(characters)
        address = normalized_address(text)
        if address is not None:
            accepted += 1
            assert parseaddr(address) == ("", address), text
    assert accepted > 1_000


def test_normalized_address_is_linear_on_a_long_run_of_dots() -> None:
    started = time.perf_counter()
    assert normalized_address("!@!." + "!." * 50_000 + "@") is None
    assert time.perf_counter() - started < 0.5
