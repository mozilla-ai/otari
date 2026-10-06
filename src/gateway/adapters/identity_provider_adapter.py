"""Core adapter for ``IdentityProviderPort``."""

from datetime import UTC, datetime

from gateway.exceptions.identity_exceptions import (
    InvalidEmailError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
)
from gateway.models.tenancy import User
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.repositories.tenancy import UserRepository
from gateway.services.tenancy.email_address import validated_email
from gateway.services.tenancy.organization_service import OrganizationService


class DeploymentIdentityProviderAdapter(IdentityProviderPort):
    """Applies this deployment's ``open_signup`` setting to an OAuth sign-in.

    The adapter leaves its changes for the caller to commit.
    """

    def __init__(self, users: UserRepository, organizations: OrganizationService, *, open_signup: bool) -> None:
        self._users = users
        self._organizations = organizations
        self._open_signup = open_signup

    async def resolve(
        self,
        *,
        provider: str,
        email: str | None,
        full_name: str | None,
        email_verified: bool,
    ) -> User:
        """Return the account for this OAuth sign-in.

        The provider must have verified the email address.
        If the provider sent no address, or did not verify it, the sign-in is refused and no account is created.

        If ``open_signup`` is enabled and there is no existing account for the address, a new account is created.
        The new account has its own organization and workspace.
        No verification email is sent, because the provider has already verified the address.

        A successful call stages these changes on the account and commits none of them:

        - It records ``provider`` if the account names none.
          An account that already names a provider keeps it.
        - It marks an unverified address verified, which lets a deployment with no mail admit a member.
          Verifying the address also removes its password and its pending email verification token.
          The provider confirms who owns the address, not who set a password on it while it was unverified.
          A password on an address that is already verified is kept.
        - It sets ``full_name`` if the account has none.

        NOTE: Concurrent sign-ins on one account are serialized on PostgreSQL only.
        On SQLite, the later of two concurrent sign-ins can overwrite the provider that the first recorded.

        Raises:
            OAuthEmailNotVerifiedError: If the provider returned no address, or one it did not verify.
            OAuthIdentityUnknownError: If the account for the address is deactivated or deleted,
                or if no account exists and ``open_signup`` is disabled.

        """
        if not email_verified or not email:
            raise OAuthEmailNotVerifiedError(provider)
        # An address this gateway would never store names no account, so it is refused as unknown.
        try:
            address = validated_email(email)
        except InvalidEmailError as error:
            raise OAuthIdentityUnknownError(provider) from error

        identity = await self._users.get_by_email(address)
        if identity is None and self._open_signup:
            registration = await self._organizations.provision_signup_tenancy(email=address, full_name=full_name)
            identity = registration.identity
        if identity is None:
            raise OAuthIdentityUnknownError(provider)

        # The account is read again under a lock, so a change since the first read is seen before anything is written.
        identity = await self._users.get_locked(identity.id)
        # A deactivated account is refused as unknown, so the response does not confirm that the account exists.
        if identity is None or not identity.is_active:
            raise OAuthIdentityUnknownError(provider)

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
        self._users.stage(identity)
        return identity


__all__ = ["DeploymentIdentityProviderAdapter"]
