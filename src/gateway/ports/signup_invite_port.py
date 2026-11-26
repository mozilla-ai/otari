"""Whether an invitation a deployment sent allows an account for the address it names."""

from typing import Protocol


class SignupInvitePort(Protocol):
    """Whether an invitation token allows an account for the address somebody claims."""

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        """Whether an invitation named by ``token`` allows an account for ``claimed_address``.

        An implementation must answer true only for a link that reached the claimed mailbox,
        because the answer is taken as proof of that mailbox.

        NOTE: An implementation must read through the request's own Unit of Work. The answer is
        used in the transaction that writes the account, and one read through a session of its
        own can name an invitation that is gone by the time the account is written.

        NOTE: An implementation must compare addresses by ``gateway.core.addresses.same_address``
        and must not normalize further. A looser comparison allows one address and records
        another.
        """
        ...
