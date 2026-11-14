"""How a rendered message reaches (or does not reach) a recipient.

One protocol with two implementations, selected from configuration by
:func:`select_transport`, and a third state that is deliberately not a
transport: ``None``, which is what a deployment with no mail configured gets.
Modeling "no transport" as the absence of one, rather than as a flag checked
inside the sender, is what lets a surface ask whether mail exists *before* it
offers an affordance that depends on it.

The SMTP implementation is otari.ai's ``app/utils.py`` mechanics on the standard
library (``smtplib`` + ``email.mime``) rather than its ``emails`` package: the
platform's own templates are already compiled HTML, so there is nothing here for
a templating engine to render either (see ``mail.templates``), and this edition's
dependency footprint stays as it is.
"""

import smtplib
import ssl
from email.charset import Charset
from email.errors import MessageError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, parseaddr
from typing import Protocol

from gateway.core.addresses import normalized_address
from gateway.core.config import GatewayConfig
from gateway.log_config import CARRIES_CREDENTIAL, logger
from gateway.services.mail.message import MailDeliveryError, MailEnvelope

# Every SMTP step (connect, STARTTLS, login, sendmail) is bounded by this, so a
# black-holed host costs one worker thread this long and not indefinitely.
SMTP_TIMEOUT_SECONDS = 10


def sanitize_header_value(value: str) -> str:
    """Strip CR/LF so a value cannot inject or malform an email header.

    Otari never delivers a literal newline in a header value, so this is a
    hardening measure against unexpected input, not a feature: it changes
    nothing for the values this codebase actually produces. Shared with callers
    that build a subject line (``mail.templates.render_email``) rather than
    duplicated, since the risk (a stray CR/LF reaching ``as_string()``) is the
    same for any header, not specific to one caller.
    """
    return value.replace("\r", " ").replace("\n", " ")


def _is_sent_as_given(address: str) -> bool:
    """Whether ``address`` is an email address that an email to it reaches as exactly that text."""
    return normalized_address(address) == address.lower() and parseaddr(address) == ("", address)


def redact_email(email: str) -> str:
    """An email address safe to put in a log line: first character, then the domain.

    Text that is not a plain email address is shown as ``***``.
    """
    if not _is_sent_as_given(email):
        return "***"
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def _from_header(name: str, address: str) -> str:
    """The From header, with the name always read back as the text it is.

    formataddr quotes a name holding "@" or ".", which unquoted a mail client
    reads as the address, and encodes one outside ASCII. It leaves an ASCII
    name holding "=?" as it is, which a client would decode as an encoded word
    into other text, so that name is encoded here.
    """
    if "=?" in name:
        # Raises for an address outside ASCII, as formataddr does; the SMTP
        # transport reports that UnicodeError as a delivery failure.
        address.encode("ascii")
        return f"{Charset('utf-8').header_encode(name)} <{address}>"
    return formataddr((name, address))


def build_mime(envelope: MailEnvelope) -> MIMEMultipart:
    """Assemble the multipart/alternative message a transport puts on the wire.

    Shared rather than per-transport: what a message *is* does not depend on how
    it travels, and the console transport rendering a different shape from the
    one SMTP would send is precisely the bug a local preview exists to catch.
    """
    message = MIMEMultipart("alternative")
    subject = sanitize_header_value(envelope.message.subject)
    # A subject that holds "=?" is encoded, because a mail client would decode it as an encoded word.
    message["Subject"] = Charset("utf-8").header_encode(subject) if "=?" in subject else subject
    # sender_name is not a value a caller would already have sanitized (unlike
    # the subject): a stray CR/LF in it would raise HeaderParseError on every
    # send, silently and from then on, with one warning line that never points
    # at the From name as the cause.
    message["From"] = _from_header(
        sanitize_header_value(envelope.sender_name), sanitize_header_value(envelope.sender_email)
    )
    message["To"] = sanitize_header_value(envelope.to)
    if envelope.reply_to:
        message["Reply-To"] = sanitize_header_value(envelope.reply_to)
    # Plain text first, then HTML: per RFC 2046, a mail client renders the last
    # alternative it understands, so this is the order that prefers HTML.
    message.attach(MIMEText(envelope.message.text, "plain"))
    message.attach(MIMEText(envelope.message.html, "html"))
    return message


_REFUSED_STEPS: tuple[tuple[type[smtplib.SMTPResponseException], str], ...] = (
    (smtplib.SMTPAuthenticationError, "the sign-in"),
    (smtplib.SMTPConnectError, "the connection"),
    (smtplib.SMTPDataError, "the message"),
    (smtplib.SMTPHeloError, "the greeting"),
    (smtplib.SMTPSenderRefused, "the sender"),
)


def _failure_reason(exc: Exception) -> str:
    """Say why a send failed, without an email address and without the email server's reply text."""
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        codes = sorted({code for code, _reply in exc.recipients.values()})
        return f"The email server refused the recipient ({', '.join(map(str, codes))})."
    if isinstance(exc, smtplib.SMTPResponseException):
        step = next((step for kind, step in _REFUSED_STEPS if isinstance(exc, kind)), "the request")
        return f"The email server refused {step} ({exc.smtp_code})."
    if isinstance(exc, smtplib.SMTPNotSupportedError):
        return f"The email server lacks a feature this send needs: {exc}"
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return "The email server closed the connection."
    if isinstance(exc, smtplib.SMTPException):
        return "The email server could not complete the send."
    if isinstance(exc, OSError):
        detail = str(exc)
        return f"The connection to the email server failed: {detail}" if detail else "The email server did not answer."
    return "The message could not be prepared for sending."


class MailTransport(Protocol):
    """One way of delivering a message.

    ``deliver`` is synchronous and may block on the network: the mailer is what
    off-loads it to a thread, so a transport implementation stays as simple as
    the library it wraps. It raises :class:`MailDeliveryError` when it cannot
    deliver; returning a status would put the "did it work" decision in two
    places.
    """

    name: str
    # False for a transport that shows an email to the deployment operator and delivers it to nobody.
    delivers_to_recipients: bool

    def deliver(self, envelope: MailEnvelope) -> None: ...


class SmtpTransport:
    """Delivers through an SMTP server."""

    name = "smtp"
    delivers_to_recipients = True

    def __init__(self, *, host: str, port: int, use_tls: bool, user: str | None, password: str | None) -> None:
        self._host = host
        self._port = port
        self._use_tls = use_tls
        self._user = user
        self._password = password

    def deliver(self, envelope: MailEnvelope) -> None:
        if not _is_sent_as_given(envelope.to):
            raise MailDeliveryError("The recipient is not a plain email address.")
        if envelope.reply_to and not _is_sent_as_given(envelope.reply_to):
            raise MailDeliveryError("The reply address is not a plain email address.")
        try:
            with smtplib.SMTP(self._host, self._port, timeout=SMTP_TIMEOUT_SECONDS) as client:
                if self._use_tls:
                    # smtplib.starttls() with no context builds one that skips
                    # both certificate and hostname verification (CPython's
                    # documented default, not a bug there), which makes STARTTLS
                    # insecure against a network-level MITM unless the caller
                    # passes its own context. ssl.create_default_context() is
                    # that context: it verifies against the system trust store.
                    client.starttls(context=ssl.create_default_context())
                if self._user and self._password:
                    client.login(self._user, self._password)
                # as_string() is where a malformed header would raise (building
                # the message above only stages the values); MessageError is in
                # the except clause as defense in depth, since build_mime
                # already sanitizes every header this codebase produces.
                # The sender is sanitized here as well as in the header,
                # because smtplib raises ValueError for a newline in MAIL FROM.
                client.sendmail(
                    sanitize_header_value(envelope.sender_email),
                    [envelope.to],
                    build_mime(envelope).as_string(),
                )
        # NOTE: This method raises MailDeliveryError or nothing.
        # ValueError is how smtplib refuses a newline in an address.
        except (OSError, smtplib.SMTPException, MessageError, UnicodeError, ValueError) as exc:
            raise MailDeliveryError(_failure_reason(exc)) from exc


class ConsoleTransport:
    """Logs each message instead of delivering it, for local development.

    A real transport rather than a mock, so every mail-dependent surface is
    available and every message is inspectable without standing up an SMTP
    server. Logs the plain-text alternative only: it is the variant a human
    reads in a terminal, and both are rendered from the same values.

    **The logged body carries a bearer credential** (an invitation's accept
    token), which is why this is a development transport. That trade, and why
    it is not redacted, is in ``src/gateway/AGENTS.md``.
    """

    name = "console"
    delivers_to_recipients = False

    def deliver(self, envelope: MailEnvelope) -> None:
        logger.info(
            "[mail:console] to=%s from=%s subject=%s\n%s",
            redact_email(envelope.to),
            envelope.sender_email,
            envelope.message.subject,
            envelope.message.text,
            extra=CARRIES_CREDENTIAL,
        )


def select_transport(config: GatewayConfig) -> MailTransport | None:
    """Build the transport this configuration selects, or ``None`` for no mail.

    ``None`` is the honest answer for a deployment that never configured mail,
    and is what every mail-dependent surface gates on. ``config.mail_transport``
    is validated at load, so an explicit ``smtp`` cannot reach here without the
    settings it needs.
    """
    transport = config.effective_mail_transport
    if transport == "console":
        return ConsoleTransport()
    if transport == "smtp" and config.smtp_host and config.mail_from_email:
        return SmtpTransport(
            host=config.smtp_host,
            port=config.smtp_port,
            use_tls=config.smtp_tls,
            user=config.smtp_user,
            password=config.smtp_password,
        )
    return None


__all__ = [
    "SMTP_TIMEOUT_SECONDS",
    "ConsoleTransport",
    "MailTransport",
    "SmtpTransport",
    "build_mime",
    "normalized_address",
    "redact_email",
    "sanitize_header_value",
    "select_transport",
]
