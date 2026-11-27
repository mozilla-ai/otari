"""The one interface a mail-dependent feature calls.

A caller asks one question before a send and one after it.
Before: *does an email carrying a link back to this deployment reach its recipient*
(:attr:`Mailer.can_deliver_links`).
After: *did this message go out* (:meth:`Mailer.send`).
Which shape a given surface takes, and why, is in ``src/gateway/AGENTS.md``.
"""

import asyncio
from enum import StrEnum

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


class MailReadiness(StrEnum):
    """Whether this deployment is configured to put a link into somebody's inbox.

    The members are exclusive and cover every deployment.
    ``none`` means no message is sent at all.
    ``delivers`` means a message is sent to the address it names, which is what the
    configuration permits rather than what a mail server went on to do with it.
    Whether a send succeeds is :meth:`Mailer.send`'s answer, once.

    NOTE: An operator confirms a real end-to-end delivery through the mail test
    surface. This value never stands for a send that has been tried.
    """

    NONE = "none"
    DELIVERS = "delivers"


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
        """``smtp``, or ``none`` when this deployment sends no mail."""
        return self._transport.name if self._transport else "none"

    @property
    def is_configured(self) -> bool:
        """Whether a transport exists at all.

        Whether a transport exists, and nothing about whether a link reaches
        anybody. ``can_deliver_links`` is what a surface gates on.

        Delegated rather than recomputed from ``self._transport``, because two
        readers deriving one answer separately can drift and one delegating to the
        other cannot. ``select_transport`` is held to the same answer by a test,
        since it is the third place the question could be decided.
        """
        return self._config.mail_enabled

    @property
    def can_deliver_links(self) -> bool:
        """Whether an email carrying a link back to this deployment reaches its recipient.

        The readiness every control-plane message actually needs: each one puts
        an absolute URL into someone's inbox, and a relative link there means
        nothing. This is what a mail-dependent surface gates on.
        """
        return self._config.mail_ready

    @property
    def readiness(self) -> MailReadiness:
        """Whether an email carrying a link back to this deployment reaches its recipient.

        Delegates to the predicate above rather than deciding again, so the
        published answer and the one a surface gates on cannot drift.
        """
        return MailReadiness.DELIVERS if self.can_deliver_links else MailReadiness.NONE

    @property
    def missing_settings(self) -> tuple[str, ...]:
        """Which settings stand between this deployment and a delivered link."""
        return self._config.missing_mail_settings

    def require_can_deliver(self) -> None:
        """Raise :class:`MailNotConfiguredError` unless a linked message reaches its recipient.

        For the surfaces with no fallback. Creating an invitation must not call
        this, because that flow hands its link back instead and works without
        mail. Signup does call it, ahead of reading the address, so that its
        refusal cannot differ by address.
        """
        if not self.can_deliver_links:
            raise MailNotConfiguredError(self.missing_settings)

    def link(self, path: str) -> str:
        """Build an absolute link to this deployment's interface, or a relative one if it cannot.

        Absolute against ``effective_ui_base_url``, not this process's address:
        what follows a link is a browser.

        A relative link is still a valid thing to hand an operator to share; it
        is only worthless *in an email*, which is why ``can_deliver_links`` and
        not this is what gates a send.
        """
        return self._config.ui_link(path)

    @property
    def asset_root(self) -> str | None:
        """Where this deployment's dashboard serves its static files, or ``None`` if its address is unknown."""
        return self._config.effective_ui_base_url.partition("?")[0].rstrip("/") or None

    async def send(
        self, *, to: str, message: MailMessage, reply_to: str | None = None, sender_name: str | None = None
    ) -> MailDelivery:
        """Deliver one rendered message, with ``reply_to`` as its Reply-To when given. Never raises.

        ``sender_name`` replaces the configured From name for this message only, as
        for one relayed on someone's behalf. The From address stays this
        deployment's own: a mail service sends only from addresses it verified.

        A mail failure must not fail the request that triggered it: creating an
        invitation still succeeds when only the notification did not go out. The
        reason is logged with the recipient redacted and returned for an
        operator-facing surface to show; a caller that only needs the yes/no
        reads ``delivered``.
        """
        transport = self._transport
        # Tested together because a transport is only built where the sender is set,
        # so one of the two being absent is not a state this can be reached in.
        sender_email = self._config.mail_from_email
        if transport is None or sender_email is None:
            logger.info("Mail is not configured; skipping send to %s", redact_email(to))
            return MailDelivery(delivered=False, transport="none", reason="Mail is not configured on this deployment.")

        envelope = MailEnvelope(
            to=to,
            sender_email=sender_email,
            sender_name=sender_name or self._config.mail_from_name,
            message=message,
            reply_to=reply_to,
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

    async def send_template(
        self, template: str, *, to: str, subject: str, values: dict[str, str], preheader: str = ""
    ) -> MailDelivery:
        """Render a template and send it, the one call a simple message needs.

        A template that does not exist, or one whose placeholders the caller did
        not supply, raises :class:`~gateway.services.mail.templates.MailTemplateError`
        rather than being reported as a delivery failure: that is a bug in this
        codebase, not a property of the deployment's mail configuration, and
        collapsing the two would hide it behind "mail didn't send".
        """
        message = render_email(
            template, subject=subject, values=values, preheader=preheader, asset_root=self.asset_root
        )
        return await self.send(to=to, message=message)


__all__ = ["Mailer"]
