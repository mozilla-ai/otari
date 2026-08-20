"""Outgoing mail: one interface, a selectable transport, and an honest no-transport mode.

Import from here, not from the modules below: ``Mailer`` is the whole of what a
mail-dependent feature needs, and keeping the transport and template mechanics
behind it is what lets a caller (an invitation today, verification and password
reset next) be written without knowing how a message travels.
"""

from gateway.services.mail.mailer import Mailer
from gateway.services.mail.message import (
    MailDelivery,
    MailDeliveryError,
    MailEnvelope,
    MailMessage,
    MailNotConfiguredError,
)
from gateway.services.mail.templates import MailTemplateError, render_email
from gateway.services.mail.transports import (
    ConsoleTransport,
    MailTransport,
    SmtpTransport,
    normalized_address,
    sanitize_header_value,
    select_transport,
)

__all__ = [
    "ConsoleTransport",
    "MailDelivery",
    "MailDeliveryError",
    "MailEnvelope",
    "MailMessage",
    "MailNotConfiguredError",
    "MailTemplateError",
    "MailTransport",
    "Mailer",
    "SmtpTransport",
    "normalized_address",
    "render_email",
    "sanitize_header_value",
    "select_transport",
]
