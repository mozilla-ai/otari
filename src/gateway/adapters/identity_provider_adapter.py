"""Core adapter for ``IdentityProviderPort``."""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from gateway.exceptions.identity_exceptions import (
    InvalidEmailError,
    OAuthEmailMissingError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
)
from gateway.models.identity import EmailAssertion, ProviderIdentity
from gateway.ports.identity_provider_port import IdentityProviderPort, IdentityResolution
from gateway.repositories.tenancy import UserRepository
from gateway.services.oauth_service import provider_label, trusts_email_claim
from gateway.services.tenancy import OrganizationService, unverified_identity_is_claimable
from gateway.services.tenancy.email_address import validated_email


class DeploymentIdentityProviderAdapter(IdentityProviderPort):
    """Applies this deployment's ``open_signup`` setting to an OAuth sign-in.

    The adapter leaves its changes for the caller's Unit of Work block to commit.
    """

    def __init__(
        self,
        users: UserRepository,
        organizations: OrganizationService,
        forget_unproven_access: Callable[[uuid.UUID], Awaitable[None]],
        *,
        open_signup: bool,
    ) -> None:
        """``forget_unproven_access`` stages the end of one identity's sessions and passkeys."""
        self._users = users
        self._organizations = organizations
        self._forget_unproven_access = forget_unproven_access
        self._open_signup = open_signup

    async def resolve(self, external: ProviderIdentity) -> IdentityResolution:
        """Return the account that holds the address of an OAuth identity with no link.

        Only a provider whose email claim no tenant can set signs in here.
        The provider must have verified the email address.
        If the provider sent no address, or did not verify it, the sign-in is refused and no account is created.

        If ``open_signup`` is enabled and there is no existing account for the address, a new account is created.
        The new account has its own organization and workspace.
        No verification email is sent, because the provider has already verified the address.

        A successful call then stages these changes on the account:

        - It records ``provider`` if the account names none.
          An account that already names a provider keeps it.
        - It marks an unverified address verified, which lets a deployment with no mail admit a member.
          Verifying the address also removes its password, its pending email verification and reset tokens,
          its sessions and its passkeys, and ``password_removed`` says whether there was a password.
          The provider confirms who owns the address, not who set a password on it while it was unverified.
          A password on an address that is already verified is kept.
        - It sets ``full_name`` if the account has none, and on an unverified address replaces it,
          since whoever signed up chose that name rather than the provider's verified owner.

        NOTE: Concurrent sign-ins on one account are serialized on PostgreSQL only.
        On SQLite, the later of two concurrent sign-ins can overwrite the provider that the first recorded.

        An account nobody has verified yet is signed in only where ``unverified_identity_is_claimable`` allows,
        the rule the signup form and a password reset follow.

        Settling the claimed account's tenancy commits a block, which releases the lock, so the account
        is read again under a lock before this sign-in changes it.

        ``new_account`` is true when this sign-in created the account, or is the first way in to one an
        organization added and nobody had claimed.

        Raises:
            OAuthEmailMissingError: If the provider returned no address.
            OAuthEmailNotVerifiedError: If the provider did not verify the address.
            OAuthIdentityUnknownError: If a tenant can set the provider's email claim,
                or no active account holds the address and ``open_signup`` does not register a new one.

        """
        label = provider_label(external.provider)
        if not trusts_email_claim(external.provider):
            raise OAuthIdentityUnknownError(label)
        if not external.email:
            raise OAuthEmailMissingError(label)
        if external.email_verified is not EmailAssertion.VERIFIED:
            raise OAuthEmailNotVerifiedError(label)
        # An address this gateway would never store names no account, so it is refused as unknown.
        try:
            address = validated_email(external.email)
        except InvalidEmailError as error:
            raise OAuthIdentityUnknownError(label) from error

        created = False
        identity = await self._users.get_by_email(address)
        if identity is None and self._open_signup:
            registration = await self._organizations.provision_signup_tenancy(
                email=address, full_name=external.full_name
            )
            identity = registration.identity
            created = registration.created
        if identity is None:
            raise OAuthIdentityUnknownError(label)

        # The account is read again under a lock, so a change since the first read is seen before anything is written.
        identity = await self._users.get_locked(identity.id)
        # A deactivated account is refused as unknown, so the response does not confirm that the account exists.
        if identity is None or not identity.is_active:
            raise OAuthIdentityUnknownError(label)
        # The address is looked up again rather than compared here, so both lookups treat letter case alike.
        holder = await self._users.get_by_email(address)
        if holder is None or holder.id != identity.id:
            raise OAuthIdentityUnknownError(label)
        if not await unverified_identity_is_claimable(self._organizations, identity, open_signup=self._open_signup):
            raise OAuthIdentityUnknownError(label)

        first_claim = identity.hashed_password is None and identity.email_verified_at is None
        if not created and await self._organizations.provision_tenancy_for_claim(
            identity,
            open_signup=self._open_signup,
            full_name=external.full_name,
            replaces_unproven=True,
        ):
            # Provisioning may have pointed the account at a new organization, so it is read again.
            identity = await self._users.get_locked(identity.id)
            if identity is None or not identity.is_active:
                raise OAuthIdentityUnknownError(label)
        if identity.oauth_provider is None:
            identity.oauth_provider = external.provider
        password_removed = False
        if identity.email_verified_at is None or identity.email_vouched_at is not None:
            password_removed = identity.hashed_password is not None
            identity.email_verified_at = datetime.now(UTC)
            identity.email_vouched_at = None
            # The address was unverified, or only vouched for, when these were set, so they may not be this person's.
            identity.hashed_password = None
            identity.email_verification_token_hash = None
            identity.email_verification_token_expires_at = None
            identity.password_reset_token_hash = None
            identity.password_reset_token_expires_at = None
            await self._forget_unproven_access(identity.id)
        if not identity.full_name and external.full_name:
            identity.full_name = external.full_name
        self._users.stage(identity)
        return IdentityResolution(
            identity=identity, new_account=created or first_claim, password_removed=password_removed
        )


__all__ = ["DeploymentIdentityProviderAdapter"]
