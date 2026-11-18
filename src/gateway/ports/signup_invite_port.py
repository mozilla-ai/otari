"""Invitations a build sends that prove which address they reached.

A build that emails someone a link (an invite to a plan, say) already knows the
link reached that mailbox. Sign-up can take that as proof of the address, so
the person sets a password and is signed in without a second, verification
email. The core sends no such invitations, so its adapter knows none.
"""

from typing import Protocol


class SignupInvitePort(Protocol):
    """Which address an invitation token was sent to."""

    async def invited_email(self, token: str) -> str | None:
        """The address ``token`` was sent to, while it can still be used; ``None`` for any other token."""
        ...
