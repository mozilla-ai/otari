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
    message["From"] = f"{config.mail_from_name} <{config.mail_from_email}>"
    message["To"] = to
    # Plain text first, then HTML: per RFC 2046, a mail client renders the last
    # alternative it understands, so this is the order that prefers HTML.
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=10) as client:
            if config.smtp_tls:
                client.starttls()
            if config.smtp_user and config.smtp_password:
                client.login(config.smtp_user, config.smtp_password)
            client.sendmail(config.mail_from_email, [to], message.as_string())
    except (OSError, smtplib.SMTPException):
        logger.warning("Failed to send mail to %s", _redact_email(to), exc_info=True)
        return False
    return True


__all__ = ["send_mail"]
