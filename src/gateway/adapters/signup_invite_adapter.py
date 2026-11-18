"""The core ``SignupInvitePort``: the core sends no invitation that proves an address."""

from gateway.ports.signup_invite_port import SignupInvitePort


class NullSignupInviteAdapter(SignupInvitePort):
    """Knows no invitation token."""

    async def invited_email(self, token: str) -> str | None:
        """No token names an address here."""
        del token
        return None
