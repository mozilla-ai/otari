"""Invitations a deployment sends that prove which address they reached.

A link emailed to somebody already proves the mailbox answers, so signup can take it instead of
sending a verification email of its own. A link shared with an admin proves less, and says so.
"""

from enum import StrEnum
from typing import Protocol


class AddressProof(StrEnum):
    """How a signup proved the address it claims, where it proved one at all."""

    MAILBOX = "mailbox"
    """An invitation emailed to the address, so its holder receives mail there."""

    SHARED = "shared"
    """An invitation shared with an admin as a link, so only that organization records the address.

    Answering ``MAILBOX`` for such a link lets domain auto-join take the account, which is the cost
    of getting this answer wrong.
    """


class SignupInvitePort(Protocol):
    """What an invitation token proves about the address somebody claims."""

    async def proof_for(self, token: str, *, claimed_address: str) -> AddressProof | None:
        """Which proof of ``claimed_address`` this token carries, or None where it carries none.

        NOTE: An implementation must compare addresses by ``gateway.core.addresses.same_address``
        and must not normalize further. Signup picks the account by that same rule, so a looser
        comparison proves one address and claims another.
        """
        ...
