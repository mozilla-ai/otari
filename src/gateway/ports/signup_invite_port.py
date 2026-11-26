"""Invitations a deployment sends, and whether one entitles an account.

A link that reached a mailbox proves the address answers, so signup can take it instead of
sending a verification email of its own. Whether a token entitles an account is one question
with one answer, and the implementation is the only thing that decides it.
"""

from typing import Protocol


class SignupInvitePort(Protocol):
    """Whether an invitation token entitles an account for the address somebody claims."""

    async def entitles_an_account(self, token: str, *, claimed_address: str) -> bool:
        """Whether this token entitles an account for ``claimed_address``.

        An entitled signup is verified as it is created and sends no verification email, so an
        implementation must answer true only for a link that reached the claimed mailbox.

        NOTE: An implementation must compare addresses by ``gateway.core.addresses.same_address``
        and must not normalize further. Signup picks the account by that same rule, so a looser
        comparison entitles one address and claims another.
        """
        ...
