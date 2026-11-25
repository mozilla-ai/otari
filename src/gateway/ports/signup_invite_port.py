"""Invitations a deployment sends that prove which address they reached.

A link emailed to somebody already proves the mailbox answers, so signup can take it instead of
sending a verification email of its own. A link shared with an admin proves less, and says so.
The vocabulary for how much is the identity domain's ``AddressAuthority``, not this port's own.
"""

from typing import Protocol

from gateway.models.identity import AddressAuthority


class SignupInvitePort(Protocol):
    """What an invitation token proves about the address somebody claims."""

    async def proof_for(self, token: str, *, claimed_address: str) -> AddressAuthority | None:
        """Which proof of ``claimed_address`` this token carries, or None where it carries none.

        NOTE: An implementation must compare addresses by ``gateway.core.addresses.same_address``
        and must not normalize further. Signup picks the account by that same rule, so a looser
        comparison proves one address and claims another.
        """
        ...
