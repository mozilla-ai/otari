"""Null Object signup-invite adapter, for a build whose invitations allow no account.

Satisfies :class:`gateway.ports.signup_invite_port.SignupInvitePort` with the honest answer
for a deployment that sends no invitation a signup can present: every token is unknown, so
no address is recorded on an invitation's word.
"""

from gateway.ports.signup_invite_port import SignupInvitePort


class NullSignupInviteAdapter(SignupInvitePort):
    """``SignupInvitePort`` adapter for a build whose invitations allow no account."""

    async def allows_an_account(self, token: str, *, claimed_address: str) -> bool:
        del token, claimed_address
        return False
