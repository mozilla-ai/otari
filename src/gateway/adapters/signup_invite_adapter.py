"""The adapter for a deployment whose invitations entitle no account."""

from gateway.ports.signup_invite_port import SignupInvitePort


class NullSignupInviteAdapter(SignupInvitePort):
    """Knows no invitation token."""

    async def entitles_an_account(self, token: str, *, claimed_address: str) -> bool:
        """No token entitles an account here."""
        del token, claimed_address
        return False
