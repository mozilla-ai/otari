"""Null Object signup-invite adapter, for a build that sends no invitation a signup can present.

Satisfies :class:`gateway.ports.signup_invite_port.SignupInvitePort` with the honest answer for
such a build: every token is unknown, so an address is only ever recorded once its own mailbox
has answered.
"""

from gateway.ports.signup_invite_port import SignupInvitePort


class NullSignupInviteAdapter(SignupInvitePort):
    """Knows no invitation token."""

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        del token, claimed_address
        return False
