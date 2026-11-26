"""The adapter for a deployment whose invitations prove nothing about a signup's address."""

from gateway.models.identity import AddressAuthority
from gateway.ports.signup_invite_port import SignupInvitePort


class NullSignupInviteAdapter(SignupInvitePort):
    """Knows no invitation token."""

    async def proof_for(self, token: str, *, claimed_address: str) -> AddressAuthority | None:
        """No token proves an address here."""
        del token, claimed_address
        return None
