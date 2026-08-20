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

import re
import smtplib
import ssl
from email.errors import MessageError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Protocol

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
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


# What this codebase treats as an address it could deliver to. Deliberately
# permissive (there is no useful regex for RFC 5322, and the SMTP server is the
# real authority): it rejects the shapes that are certainly not addresses, which
# is what an operator typing into a form needs, and nothing more. Shared with
# tenancy's member/invitation addresses rather than kept per caller, so "an
# address Otari will accept" has one answer.
_ADDRESS_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalized_address(value: str) -> str | None:
    """Lower-case and trim an address, or return ``None`` if it cannot be one."""
    candidate = value.strip().lower()
    return candidate if _ADDRESS_PATTERN.match(candidate) else None


def redact_email(email: str) -> str:
    """An email address safe to put in a log line: first character, then the domain."""
    local, _, domain = email.partition("@")
    if not domain:
        return "***"
    return f"{local[:1]}***@{domain}"


def build_mime(envelope: MailEnvelope) -> MIMEMultipart:
    """Assemble the multipart/alternative message a transport puts on the wire.

    Shared rather than per-transport: what a message *is* does not depend on how
    it travels, and the console transport rendering a different shape from the
    one SMTP would send is precisely the bug a local preview exists to catch.
    """
    message = MIMEMultipart("alternative")
    message["Subject"] = sanitize_header_value(envelope.message.subject)
    # sender_name is operator config, not a caller-supplied value a caller would
    # already have sanitized (unlike the subject): a stray CR/LF in it would
    # raise HeaderParseError on every send, silently and from then on, with one
    # warning line that never points at the From name as the cause.
    message["From"] = f"{sanitize_header_value(envelope.sender_name)} <{envelope.sender_email}>"
    message["To"] = sanitize_header_value(envelope.to)
    # Plain text first, then HTML: per RFC 2046, a mail client renders the last
    # alternative it understands, so this is the order that prefers HTML.
    message.attach(MIMEText(envelope.message.text, "plain"))
    message.attach(MIMEText(envelope.message.html, "html"))
    return message


class MailTransport(Protocol):
    """One way of delivering a message.

    ``deliver`` is synchronous and may block on the network: the mailer is what
    off-loads it to a thread, so a transport implementation stays as simple as
    the library it wraps. It raises :class:`MailDeliveryError` when it cannot
    deliver; returning a status would put the "did it work" decision in two
    places.
    """

    name: str

    def deliver(self, envelope: MailEnvelope) -> None: ...


class SmtpTransport:
    """Delivers through an SMTP server."""

    name = "smtp"

    def __init__(self, *, host: str, port: int, use_tls: bool, user: str | None, password: str | None) -> None:
        self._host = host
        self._port = port
        self._use_tls = use_tls
        self._user = user
        self._password = password

    def deliver(self, envelope: MailEnvelope) -> None:
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
                client.sendmail(envelope.sender_email, [envelope.to], build_mime(envelope).as_string())
        except (OSError, smtplib.SMTPException, MessageError, UnicodeError) as exc:
            raise MailDeliveryError(f"{type(exc).__name__}: {exc}") from exc


class ConsoleTransport:
    """Logs each message instead of delivering it, for local development.

    A real transport rather than a mock: an operator who sets
    ``mail_transport: console`` gets a deployment where every mail-dependent
    surface is *available* and every message is inspectable, which is what makes
    a template reviewable without standing up an SMTP server. It is deliberately
    not the default, because a deployment that silently logged the mail it was
    asked to send would be the accepting-and-dropping this design rules out.

    Logs the rendered plain-text alternative and not the HTML one: the text
    variant is the one a human reads in a terminal, and the pair is rendered
    from the same values, so a truncated HTML dump would add noise without
    adding a check. The recipient is redacted for the same reason it is
    everywhere else here, since these lines end up in the same log stream.

    **It writes the message body, and a control-plane message body carries a
    bearer credential**: an invitation's accept token today, a password-reset
    token once #650 lands. That is the point (a developer needs to follow the
    link) and it is also why this is a development transport and not one to
    select on a deployment whose logs are shipped or shared. Redacting the link
    would make the transport useless for the one job it has, so the trade is
    stated instead: the config field says it, the docs say it, and
    ``validate_mail_transport`` warns once at startup when it is selected. It is
    the one sanctioned exception to the never-log-a-token rule, and it is opt-in
    per deployment rather than reachable by default.
    """

    name = "console"

    def deliver(self, envelope: MailEnvelope) -> None:
        logger.info(
            "[mail:console] to=%s from=%s subject=%s\n%s",
            redact_email(envelope.to),
            envelope.sender_email,
            envelope.message.subject,
            envelope.message.text,
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
