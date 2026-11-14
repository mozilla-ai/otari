"""Sending the invitation emails that are due."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.tokens import generate_token, hash_token
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.log_config import logger
from gateway.repositories.organizations import DueInvitation, OrganizationRepositories
from gateway.services.mail import Mailer
from gateway.services.organizations._delivery_rule import DeliveryRule
from gateway.services.organizations._invitation_email import render_invitation_email
from gateway.services.organizations._invitation_facts import InvitationFacts, InvitationFactsReader

# One more attempt follows each wait, in order.
RETRY_WAITS = (timedelta(minutes=5), timedelta(minutes=10))
MAX_ATTEMPTS = len(RETRY_WAITS) + 1
# NOTE: This must be longer than one send can take, or a second email is sent while the first is on its way.
CLAIM_HOLD = timedelta(minutes=10)
POLL_SECONDS = 30.0


def _accept_path(token: str) -> str:
    return f"/#/accept-invitation?token={token}"


class InvitationDelivery:
    """Sends each invitation email that is due, and tries again later when an email is not delivered."""

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: OrganizationRepositories,
        *,
        rule: DeliveryRule,
        facts: InvitationFactsReader,
        mailer: Mailer,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._invitations = repositories.invitations
        self._rule = rule
        self._facts = facts
        self._mailer = mailer
        self._clock = clock

    async def send_due(self) -> bool:
        """Send the invitation email that has been due the longest, and return whether one was due.

        Each email carries a new accept link, so the link in an earlier email of the same invitation stops working.
        An email that is not delivered is due again after a wait, for ``MAX_ATTEMPTS`` attempts in all.
        No email is sent for an invitation that has expired or was declined, or that the delivery rule does not allow.
        No send is due for such an invitation afterwards.
        Nothing is sent while the deployment cannot deliver an email that carries a link.

        NOTE: Callers must call this outside a Unit of Work block, because the email must not be sent in a transaction.
        """
        if not self._mailer.can_deliver_links:
            return False
        now = self._clock()
        token = generate_token()
        token_hash = hash_token(token)
        async with self._uow:
            due = await self._invitations.claim_due_send(
                now=now, held_until=now + CLAIM_HOLD, max_attempts=MAX_ATTEMPTS, token_hash=token_hash
            )
        if due is None:
            return False
        if due.attempt > MAX_ATTEMPTS:
            # NOTE: An invitation that is made due again without a reset of its attempts gets no email.
            return True

        async with self._uow:
            facts = await self._facts_to_send(due, now)
            if facts is None:
                await self._invitations.finish_send(due.id, token_hash=token_hash)
                return True

        message = render_invitation_email(
            facts,
            accept_link=self._mailer.link(_accept_path(token)),
            expires_at=due.expires_at,
            asset_root=self._mailer.asset_root,
        )
        delivery = await self._mailer.send(to=due.email, message=message)

        if due.attempt == MAX_ATTEMPTS:
            # The claim of the last attempt left no email due, so there is no result to record.
            if not delivery.delivered:
                logger.warning(
                    "Invitation email not delivered after %s attempts (invitation %s, organization %s)",
                    MAX_ATTEMPTS,
                    due.id,
                    due.organization_id,
                )
            return True
        async with self._uow:
            if delivery.delivered:
                await self._invitations.finish_send(due.id, token_hash=token_hash)
            else:
                await self._invitations.retry_send(
                    due.id, token_hash=token_hash, next_send_at=now + RETRY_WAITS[due.attempt - 1]
                )
        return True

    async def _facts_to_send(self, due: DueInvitation, now: datetime) -> InvitationFacts | None:
        """Return the facts for the email of ``due``, or None when no email is to be sent for it."""
        if due.expires_at <= now or due.declined_at is not None:
            return None
        if not await self._rule.allows(due.email, due.organization_id, now=now):
            return None
        return await self._facts.read(due, now=now)

    @staticmethod
    async def poll(build: Callable[[UnitOfWork], "InvitationDelivery"], interval: float = POLL_SECONDS) -> None:
        """Send the due invitation emails every ``interval`` seconds, until the task is canceled.

        ``build`` receives a new Unit of Work for each invitation.
        An error is logged, and the next poll tries again.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                sent = True
                while sent:
                    async with create_unit_of_work() as uow:
                        sent = await build(uow).send_due()
            except DATABASE_ERRORS as exc:
                logger.warning(
                    "Invitation email poll failed: %s; retrying in %ss", describe_database_error(exc), interval
                )
            except Exception as exc:
                # NOTE: Only the class is logged, because the message of an error can hold an address or a link.
                logger.warning("Invitation email poll failed (%s); retrying in %ss", type(exc).__name__, interval)
