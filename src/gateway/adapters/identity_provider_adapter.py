"""Identity adapter enforcing the base build's roster policy on an OAuth sign-in.

Satisfies :class:`gateway.ports.identity_provider_port.IdentityProviderPort` with
the policy Otari's base build already applies to every other way in: an account
exists here because an operator put it here. A social identity signs in as an
account already on the roster and never creates one, so enabling Google or GitHub
sign-in widens *how* a member authenticates, never *who* may.

This is a real implementation and not a Null Object, per ``ARCHITECTURE.md``'s
cardinal property. There is a live decision behind the port (link, refuse, and
whether the provider's assertion is enough to lift the local verification gate),
and it is the decision an overlay is most likely to want to replace: a hosted
edition provisions on first sight, and an enterprise edition maps a directory
connection onto an organization. Both bind here without editing this tree.
"""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.identity_exceptions import (
    InvalidEmailError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
)
from gateway.models.tenancy import User
from gateway.repositories.tenancy import UserRepository
from gateway.services.tenancy.email_address import validated_email


class RosterIdentityProviderAdapter:
    """Resolves an OAuth identity onto an account an operator already added.

    The adapter stages its writes on the request's session and does not commit them.
    The session is ``None`` where the deployment has no database, and ``resolve`` needs one.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        self._session = session

    async def resolve(
        self,
        *,
        provider: str,
        email: str | None,
        full_name: str | None,
        email_verified: bool,
    ) -> User:
        """Return the roster identity this OAuth identity signs in as.

        The address must be one the provider verified.
        An unverified or missing address is refused whether or not it is on the roster.

        A successful call stages these changes on the identity and commits none of them:

        - It records ``provider`` if the identity names none.
          An identity that already names a provider keeps it.
        - It marks an unverified address verified, which lets a deployment with no mail admit a member.
          Verifying the address also removes its password and its pending email verification token.
          The provider confirms who owns the address, not who set a password on it while it was unverified.
          A password on an address that is already verified is kept.
        - It sets ``full_name`` if the identity has none.

        NOTE: Concurrent sign-ins on one identity are serialized on PostgreSQL only.
        On SQLite, the later of two concurrent sign-ins can overwrite the provider that the first recorded.

        Raises:
            OAuthEmailNotVerifiedError: If the provider returned no address, or one it did not verify.
            OAuthIdentityUnknownError: If no active identity here holds that address.

        """
        assert self._session is not None, "resolving an identity needs a database session"
        if not email_verified or not email:
            raise OAuthEmailNotVerifiedError(provider)
        # An address this gateway would never store names no account, so it is refused as unknown.
        try:
            address = validated_email(email)
        except InvalidEmailError as error:
            raise OAuthIdentityUnknownError(provider) from error

        users = UserRepository(self._session)
        identity = await users.get_by_email(address)
        # A deactivated identity is refused as unknown, so the answer does not confirm that the account exists.
        if identity is None or not identity.is_active:
            raise OAuthIdentityUnknownError(provider)

        # The row is locked and re-read because a concurrent sign-in may have changed it since the first read.
        await users.lock(identity.id)
        await self._session.refresh(identity)

        if identity.oauth_provider is None:
            identity.oauth_provider = provider
        if identity.email_verified_at is None:
            identity.email_verified_at = datetime.now(UTC)
            # The address was unverified when these were set, so they may not be this person's.
            identity.hashed_password = None
            identity.email_verification_token_hash = None
            identity.email_verification_token_expires_at = None
        if not identity.full_name and full_name:
            identity.full_name = full_name
        self._session.add(identity)
        return identity


__all__ = ["RosterIdentityProviderAdapter"]
