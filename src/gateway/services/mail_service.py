"""Outgoing mail: a stdlib SMTP sender for the templated messages this edition sends.

Ported from otari.ai's ``app/utils.py`` (SMTP config, ``send_email_safe``), but
without its dependencies: the platform's *authoritative* templates are already
its compiled ``build/*.html``/``*.txt`` output (its own comment on the source
``.mjml`` says those are hand-tuned, not round-tripped), so there is nothing
here for a templating engine to render, and ``smtplib``/``email.mime`` from the
standard library cover the send. Keeps this edition's dependency footprint the
way it already is (see ``root_page.py`` for the same "no templating engine"
choice for the tutorial page).

Every send goes through :func:`send_mail`, which never raises: a mail failure
must not fail the request that triggered it (creating an invitation still
succeeds; only the notification did not go out), so an error is logged, with
the recipient redacted, and reported back as ``False``.
"""

import smtplib
import ssl
from email.errors import MessageError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from gateway.core.config import GatewayConfig
from gateway.log_config import logger


def _redact_email(email: str) -> str:
    """An email address safe to put in a log line: first character, then the domain."""
    local, _, domain = email.partition("@")
    if not domain:
        return "***"
    return f"{local[:1]}***@{domain}"


def sanitize_header_value(value: str) -> str:
    """Strip CR/LF so a value cannot inject or malform an email header.

    Otari never delivers a literal newline in a header value, so this is a
    hardening measure against unexpected input, not a feature: it changes
    nothing for the values this codebase actually produces. Shared with
    callers that build a subject line (``invitation_email.render_invitation_email``)
    rather than duplicated, since the risk (a stray CR/LF reaching
    ``as_string()``) is the same for any header, not specific to one caller.
    """
    return value.replace("\r", " ").replace("\n", " ")


def send_mail(config: GatewayConfig, *, to: str, subject: str, html: str, text: str) -> bool:
    """Send one message, or report why it could not be sent.

    Returns ``False`` without raising when mail is not configured
    (``config.mail_enabled``) or the send itself fails; a caller that needs to
    tell an operator "share this link yourself" reads the return value, and
    nothing upstream of this function needs a try/except.
    """
    # ``mail_enabled`` is ``bool(smtp_host and mail_from_email)``, but the
    # property alone doesn't narrow either field's type for mypy, so both are
    # re-checked here to get the plain ``str`` smtplib expects.
    if not config.mail_enabled or config.smtp_host is None or config.mail_from_email is None:
        logger.info("Mail is not configured (no smtp_host/mail_from_email); skipping send to %s", _redact_email(to))
        return False

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    # mail_from_name is operator config, not a caller-supplied value a caller
    # would already have sanitized (unlike the subject): a stray CR/LF in it
    # would raise HeaderParseError below on every send, silently and from
    # then on, with the except clause turning that into mail_sent=False and
    # one warning line that never points at the From name as the cause.
    message["From"] = f"{sanitize_header_value(config.mail_from_name)} <{config.mail_from_email}>"
    message["To"] = to
    # Plain text first, then HTML: per RFC 2046, a mail client renders the last
    # alternative it understands, so this is the order that prefers HTML.
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=10) as client:
            if config.smtp_tls:
                # smtplib.starttls() with no context builds one that skips
                # both certificate and hostname verification (CPython's
                # documented default, not a bug there), which makes STARTTLS
                # insecure against a network-level MITM unless the caller
                # passes its own context. ssl.create_default_context() is
                # that context: it verifies against the system trust store.
                client.starttls(context=ssl.create_default_context())
            if config.smtp_user and config.smtp_password:
                client.login(config.smtp_user, config.smtp_password)
            # ``as_string()`` is where a malformed header would raise (message
            # construction above only stages the values), and both this
            # function's own From header and the subject a caller builds (see
            # invitation_email.render_invitation_email) already go through
            # sanitize_header_value above; ``MessageError`` is caught here too
            # as defense in depth, so this function keeps its "never raises"
            # promise even for a header this codebase does not currently
            # produce.
            client.sendmail(config.mail_from_email, [to], message.as_string())
    except (OSError, smtplib.SMTPException, MessageError, UnicodeError):
        logger.warning("Failed to send mail to %s", _redact_email(to), exc_info=True)
        return False
    return True


__all__ = ["sanitize_header_value", "send_mail"]
