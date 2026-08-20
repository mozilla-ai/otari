"""The one interface a mail-dependent feature calls.

A caller asks two questions and never a third: *can this deployment send mail
that links back to itself* (:attr:`Mailer.can_send_links`), and *did this
message go out* (:meth:`Mailer.send`). Which shape a given surface takes, and
why, is in ``src/gateway/AGENTS.md``.
"""

import asyncio

from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.services.mail.message import (
    MailDelivery,
    MailDeliveryError,
    MailEnvelope,
    MailMessage,
    MailNotConfiguredError,
)
from gateway.services.mail.templates import render_email
from gateway.services.mail.transports import redact_email, select_transport

# What a console-only deployment sends from when no address is configured. Never
# reached by SMTP, which requires mail_from_email before its transport is built.
_FALLBACK_SENDER = "otari@localhost"


class Mailer:
    """Sends the templated messages this edition produces, or explains that it cannot.

    Constructed per use from the live config rather than held as a singleton, so
    a transport is selected from the settings in force at send time and nothing
    has to be invalidated when they change. Selection is a couple of comparisons
    and a dataclass; no connection is opened until a message is sent.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        self._transport = select_transport(config)

    @property
    def transport_name(self) -> str:
        """``smtp``, ``console``, or ``none`` when this deployment sends no mail."""
        return self._transport.name if self._transport else "none"

    @property
    def is_configured(self) -> bool:
        """Whether a transport exists at all.

        Delegated rather than recomputed from ``self._transport``. The dashboard
        reads the config property through ``/v1/bootstrap`` while the invitation
        path asks the mailer, and two readers deriving one answer separately can
        drift; one delegating to the other cannot. ``select_transport`` is still
        held to the same answer by a test, since it is the third place the
        question could be decided.
        """
        return self._config.mail_enabled

    @property
    def can_send_links(self) -> bool:
        """Whether a message carrying a link back to this deployment can be sent.

        The readiness every control-plane message actually needs: each one puts
        an absolute URL into someone's inbox, and a relative link there means
        nothing. This is what a mail-dependent surface gates on.
        """
        return self._config.mail_ready

    @property
    def missing_settings(self) -> tuple[str, ...]:
        """Which settings stand between this deployment and a delivered link."""
        return self._config.missing_mail_settings

    def require_ready(self) -> None:
        """Raise :class:`MailNotConfiguredError` unless a linked message can be sent.

        For the surfaces with no fallback. A surface that can degrade (an
        invitation) must not call this: refusing there would take away a flow
        that works perfectly well without mail.
        """
        if not self.can_send_links:
            raise MailNotConfiguredError(self.missing_settings)

    def link(self, path: str) -> str:
        """Build an absolute link into this deployment, or a relative one if it cannot.

        A relative link is still a valid thing to hand an operator to share; it
        is only worthless *in an email*, which is why ``can_send_links`` and not
        this is what gates a send.
        """
        base = self._config.public_base_url
        return f"{base.rstrip('/')}{path}" if base else path

    async def send(self, *, to: str, message: MailMessage) -> MailDelivery:
        """Deliver one rendered message. Never raises.

        A mail failure must not fail the request that triggered it: creating an
        invitation still succeeds when only the notification did not go out. The
        reason is logged with the recipient redacted and returned for an
        operator-facing surface to show; a caller that only needs the yes/no
        reads ``delivered``.
        """
        transport = self._transport
        if transport is None:
            logger.info("Mail is not configured; skipping send to %s", redact_email(to))
            return MailDelivery(delivered=False, transport="none", reason="Mail is not configured on this deployment.")

        envelope = MailEnvelope(
            to=to,
            sender_email=self._config.mail_from_email or _FALLBACK_SENDER,
            sender_name=self._config.mail_from_name,
            message=message,
        )
        try:
            # deliver() is synchronous socket I/O, and each SMTP step carries its
            # own timeout, so calling it directly would block this worker's whole
            # event loop for tens of seconds against a slow or unreachable host,
            # stalling every other request it is serving. Off-loaded so only this
            # coroutine waits.
            await asyncio.to_thread(transport.deliver, envelope)
        except MailDeliveryError as exc:
            logger.warning("Failed to send mail to %s via %s: %s", redact_email(to), transport.name, exc)
            return MailDelivery(delivered=False, transport=transport.name, reason=str(exc))
        return MailDelivery(delivered=True, transport=transport.name)

    async def send_template(self, template: str, *, to: str, subject: str, values: dict[str, str]) -> MailDelivery:
        """Render a template and send it, the one call a simple message needs.

        A template that does not exist, or one whose placeholders the caller did
        not supply, raises :class:`~gateway.services.mail.templates.MailTemplateError`
        rather than being reported as a delivery failure: that is a bug in this
        codebase, not a property of the deployment's mail configuration, and
        collapsing the two would hide it behind "mail didn't send".
        """
        return await self.send(to=to, message=render_email(template, subject=subject, values=values))


__all__ = ["Mailer"]
