"""Core adapter for ``IdentityProviderPort``."""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from gateway.exceptions.identity_exceptions import (
    InvalidEmailError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
)
from gateway.ports.identity_provider_port import IdentityProviderPort, IdentityResolution
from gateway.repositories.tenancy import UserRepository
from gateway.services.oauth_service import provider_label
from gateway.services.tenancy import OrganizationService, unverified_identity_is_claimable
from gateway.services.tenancy.email_address import validated_email


class DeploymentIdentityProviderAdapter(IdentityProviderPort):
    """Applies this deployment's ``open_signup`` setting to an OAuth sign-in.

    The adapter leaves its changes to the account for the caller to commit. The one exception is the
    organization and workspace a new or reclaimed account gets, which ``OrganizationService`` commits in
    its own Unit of Work block before the account itself is changed.
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

    async def resolve(
        self,
        *,
        provider: str,
        email: str | None,
        full_name: str | None,
        email_verified: bool,
    ) -> IdentityResolution:
        """Return the account for this OAuth sign-in.

        The provider must have verified the email address.
        If the provider sent no address, or did not verify it, the sign-in is refused and no account is created.

        If ``open_signup`` is enabled and there is no existing account for the address, a new account is created
        and committed. The new account has its own organization and workspace.
        No verification email is sent, because the provider has already verified the address.

        A successful call then stages these changes on the account and commits none of them:

        - It records ``provider`` if the account names none.
          An account that already names a provider keeps it.
        - It marks an unverified address verified, which lets a deployment with no mail admit a member.
          Verifying the address also removes its password, its pending email verification and reset tokens,
          its sessions and its passkeys.
          The provider confirms who owns the address, not who set a password on it while it was unverified.
          A password on an address that is already verified is kept.
        - It sets ``full_name`` if the account has none.

        NOTE: Concurrent sign-ins on one account are serialized on PostgreSQL only.
        On SQLite, the later of two concurrent sign-ins can overwrite the provider that the first recorded.

        An account nobody has verified yet is signed in only where ``unverified_identity_is_claimable`` allows,
        the rule the signup form and a password reset follow.

        Under ``open_signup``, claiming an unverified account that no organization still wants (its
        invitation revoked, declined or lapsed, or the member removed) gives it an organization and
        workspace of its own and points it at them, the way a new account is created. That tenancy is
        committed first, and the account is read again under a lock before this sign-in changes it.

        ``new_account`` is true when this sign-in created the account, or is the first way in to one an
        organization added and nobody had claimed.

        Raises:
            OAuthEmailNotVerifiedError: If the provider returned no address, or one it did not verify.
            OAuthIdentityUnknownError: If no active account holds the address,
                and ``open_signup`` does not register a new one.

        """
        label = provider_label(provider)
        if not email_verified or not email:
            raise OAuthEmailNotVerifiedError(label)
        # An address this gateway would never store names no account, so it is refused as unknown.
        try:
            address = validated_email(email)
        except InvalidEmailError as error:
            raise OAuthIdentityUnknownError(label) from error

        created = False
        identity = await self._users.get_by_email(address)
        if identity is None and self._open_signup:
            registration = await self._organizations.provision_signup_tenancy(email=address, full_name=full_name)
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
        if not created and self._open_signup and identity.email_verified_at is None:
            await self._organizations.provision_tenancy_for_claim(identity, full_name=full_name)
            # That block committed, which released the lock taken above.
            identity = await self._users.get_locked(identity.id)
            if identity is None or not identity.is_active:
                raise OAuthIdentityUnknownError(label)
        if identity.oauth_provider is None:
            identity.oauth_provider = provider
        if identity.email_verified_at is None or identity.email_vouched_at is not None:
            identity.email_verified_at = datetime.now(UTC)
            identity.email_vouched_at = None
            # The address was unverified, or only vouched for, when these were set, so they may not be this person's.
            identity.hashed_password = None
            identity.email_verification_token_hash = None
            identity.email_verification_token_expires_at = None
            identity.password_reset_token_hash = None
            identity.password_reset_token_expires_at = None
            await self._forget_unproven_access(identity.id)
        if not identity.full_name and full_name:
            identity.full_name = full_name
        self._users.stage(identity)
        return IdentityResolution(identity=identity, new_account=created or first_claim)


__all__ = ["DeploymentIdentityProviderAdapter"]
