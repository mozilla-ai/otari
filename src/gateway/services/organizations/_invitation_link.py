"""What an invitation link still opens, and whether it allows an account."""

from collections.abc import Callable
from datetime import UTC, datetime

from gateway.core.addresses import same_address
from gateway.core.tokens import hash_token, is_shared_token
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.organizations import OrganizationInvitation
from gateway.ports.signup_invite_port import SignupInvitePort
from gateway.repositories.organizations import OrganizationRepositories
from gateway.services.mail import Mailer
from gateway.services.organizations._blocked_addresses import BlockedAddresses
from gateway.services.organizations._delivery_rule import DeliveryRule


class InvitationLink(SignupInvitePort):
    """Answers what an invitation link still opens, for everybody who holds one.

    One reader for the whole question, so a link the deployment refuses cannot go on allowing an
    account somewhere else.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: OrganizationRepositories,
        *,
        blocked: BlockedAddresses,
        delivery: DeliveryRule,
        mailer: Mailer,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._invitations = repositories.invitations
        self._organizations = repositories.organizations
        self._blocked = blocked
        self._delivery = delivery
        self._mailer = mailer
        self._clock = clock

    async def open(self, token: str, *, lock: bool = False) -> OrganizationInvitation | None:
        """The invitation this token still opens, or None where none does.

        The only answer to that question. An invitation that has expired or been declined opens
        nothing, and neither does one whose address this deployment refuses.

        ``lock`` holds the invitation until the caller's transaction ends, so a cancellation or a
        decline of it waits.

        NOTE: Callers must call this inside an open Unit of Work block.
        """
        invitation = await self._invitations.get_by_token_hash(hash_token(token), lock=lock)
        # The clock is read after the lookup, which waits when another transaction holds the row.
        now = self._clock()
        if invitation is None or now >= invitation.expires_at or invitation.declined_at is not None:
            return None
        if await self._refuses_the_address(invitation, hand_carried=is_shared_token(token), now=now):
            return None
        return invitation

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        """Whether this token allows an account for ``claimed_address``.

        A link carried by hand reached no mailbox, so it allows none. Nor does a link whose
        address this deployment refuses, nor one naming an organization whose members arrive
        through its own connection.
        """
        if is_shared_token(token):
            return False
        async with self._uow:
            invitation = await self.open(token)
            if invitation is None or not same_address(invitation.email, claimed_address):
                return False
            # The lock comes before the read, because turning the requirement on takes the same lock.
            await self._organizations.lock_settings(invitation.organization_id)
            settings = await self._organizations.sso_settings(invitation.organization_id)
            # The second read takes a row lock the caller holds until it commits, so a cancellation
            # cannot be honored against a link this already allowed. The organization lock comes
            # first, because every writer of these rows takes them in that order.
            if await self.open(token, lock=True) is None:
                return False
        # Only the requirement itself can apply here. The account does not exist yet, so it holds
        # no link to the connection and claims no exemption. Acceptance asks the whole rule again.
        return settings is None or not settings.required

    async def _refuses_the_address(
        self, invitation: OrganizationInvitation, *, hand_carried: bool, now: datetime
    ) -> bool:
        """Whether the deployment refuses this address its invitation.

        The blocklist is the deployment operator's own refusal, so it is asked wherever an invitation
        is opened. Whether another organization has proven the address's domain is that organization's
        data, so it is asked only of a link an admin is carrying by hand. A link is minted in that
        kind only where nobody could deliver it, so the kind carries that fact however the
        deployment is configured later. The delivery rule asks the blocklist itself.
        """
        if hand_carried:
            return not await self._delivery.allows(invitation.email, invitation.organization_id, now=now)
        return bool(await self._blocked.blocked_among([invitation.email]))
