"""What an invitation link still opens, and whether it entitles an account."""

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

    One reader for the whole question, so a link the deployment refuses cannot go on proving an
    address somewhere else.
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

    async def open(self, token: str) -> OrganizationInvitation | None:
        """The invitation this token still opens, or None where none does.

        The only answer to that question. An invitation that has expired or been declined opens
        nothing, and neither does one whose address this deployment refuses.

        NOTE: Callers must call this inside an open Unit of Work block.
        """
        now = self._clock()
        invitation = await self._invitations.get_by_token_hash(hash_token(token))
        if invitation is None or now >= invitation.expires_at or invitation.declined_at is not None:
            return None
        if await self._refuses_the_address(invitation, hand_carried=is_shared_token(token), now=now):
            return None
        return invitation

    async def entitles_an_account(self, token: str, *, claimed_address: str) -> bool:
        """Whether this token entitles an account for ``claimed_address``.

        A link carried by hand reached no mailbox, so it entitles nothing and an account on that
        address is still reached by verifying it. A link this deployment would refuse entitles
        nothing either, and neither does one naming an organization whose members arrive through
        its own connection, which is what keeps a signup from minting an account that joining
        would then turn away.
        """
        if is_shared_token(token):
            return False
        async with self._uow:
            invitation = await self.open(token)
            if invitation is None or not same_address(invitation.email, claimed_address):
                return False
            settings = await self._organizations.sso_settings(invitation.organization_id)
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
