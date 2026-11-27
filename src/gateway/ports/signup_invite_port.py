"""Whether an invitation allows an account for the address somebody claims."""

from typing import Protocol


class SignupInvitePort(Protocol):
    """Whether an invitation token allows an account for the address somebody claims.

    NOTE: One implementation answers for the whole deployment. A build that sends invitations from
    more than one place recognizes all of them here, because binding replaces rather than adds.
    """

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        """Whether an invitation named by ``token`` allows an account for ``claimed_address``.

        NOTE: An implementation must answer true only for a link that reached the claimed
        mailbox. A true answer is recorded as that mailbox having answered, which is the
        strongest thing an address can carry, and no caller can check the claim.

        NOTE: An implementation must read through the request's own Unit of Work, and must hold
        the invitation it answers true for until that transaction ends. The answer is used in the
        transaction that writes the account, and an unlocked read lets a cancellation commit
        before the write it admitted.

        NOTE: An implementation must compare addresses by ``gateway.core.addresses.same_address``
        and must not normalize further. A looser comparison allows one address and records
        another.
        """
        ...
