"""Identity adapter applying this deployment's own signup posture to an OAuth sign-in.

Satisfies :class:`gateway.ports.identity_provider_port.IdentityProviderPort` with
the answer ``POST /api/v1/auth/signup`` already gives the same address, which is
``open_signup``. Closed, the default and the single-tenant posture, an account
exists here because an operator put it here: a social identity signs in as
somebody already on the roster and never creates one, so enabling Google or
GitHub widens *how* a member authenticates, never *who* may. Open, the posture a
control plane serving many tenants runs, an address nobody has added is
registered with an organization and workspace of its own.

One setting for both doors on purpose. A deployment that registers a stranger who
types a password, and refuses the same stranger who proves the same address
through Google, is answering one question two ways depending on which door was
knocked on.

This is a real implementation and not a Null Object, per ``ARCHITECTURE.md``'s
cardinal property. There is a live decision behind the port (register, link,
refuse, and whether the provider's assertion is enough to lift the local
verification gate). An overlay still replaces it for a policy no setting here
expresses: an enterprise edition maps a directory connection onto an
organization, binding without editing this tree.
"""

from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.identity_exceptions import (
    InvalidEmailError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
)
from gateway.models.tenancy import User
from gateway.ports.identity_provider_port import IdentityProviderPort
from gateway.repositories.tenancy import UserRepository
from gateway.services.budgets import WorkspaceBudgetDefaultService
from gateway.services.tenancy.email_address import validated_email
from gateway.services.tenancy.organization_service import OrganizationService


class DeploymentIdentityProviderAdapter(IdentityProviderPort):
    """Resolves an OAuth identity onto an account here, registering one where signup is open.

    The adapter stages its writes on the request's session and does not commit them.
    The session is ``None`` where the deployment has no database, and ``resolve`` needs one.
    """

    def __init__(self, session: AsyncSession | None, *, open_signup: bool) -> None:
        self._session = session
        self._open_signup = open_signup

    async def resolve(
        self,
        *,
        provider: str,
        email: str | None,
        full_name: str | None,
        email_verified: bool,
    ) -> User:
        """Return the identity this OAuth identity signs in as, registering one where signup is open.

        The address must be one the provider verified.
        An unverified or missing address is refused whether or not it is on the roster,
        and registers nobody: otherwise anyone who can make a provider echo a string
        could take an account on somebody else's mailbox.

        Where ``open_signup`` is on, an address no identity holds is registered with an
        organization and workspace of its own, through the same
        ``OrganizationService.provision_signup_tenancy`` the password form uses, so an
        account that arrived through a provider is not a second kind of member.
        Unlike that form this needs no mail: the provider already proved the address,
        so there is no verification link to send and nothing to strand the caller.

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
            OAuthIdentityUnknownError: If the address belongs to a deactivated identity, or to no
                identity on a deployment that keeps signup closed.

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
        if identity is None and self._open_signup:
            identity = await self._register(address, full_name=full_name)
        # A deactivated identity is refused as unknown, so the answer does not confirm that the account
        # exists, and it is never registered afresh: that would undo the deactivation.
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

    async def _register(self, address: str, *, full_name: str | None) -> User | None:
        """Register an address nobody holds, or return the identity that just took it.

        The savepoint keeps a lost race from poisoning the transaction the session
        row is also written in, the reason
        ``OrganizationDomainService.auto_join_for_user`` takes one: two first
        sign-ins on one address would otherwise cost the loser a failed sign-in
        rather than a sign-in to the winner's account.

        The loser is found by re-reading the address rather than by matching the
        unique index by name, because this unit of work can violate other
        constraints too and only a row that now exists proves which one it hit.

        The new identity is left unverified and nameless for ``resolve`` to stamp,
        so one block records the provider and lifts the verification gate for a
        registered identity and a rostered one alike.
        """
        assert self._session is not None
        try:
            async with self._session.begin_nested():
                return await OrganizationService(
                    self._session,
                    membership_listener=WorkspaceBudgetDefaultService(self._session),
                ).provision_signup_tenancy(email=address, full_name=full_name)
        except IntegrityError:
            identity = await UserRepository(self._session).get_by_email(address)
            if identity is None:
                raise
            return identity


__all__ = ["DeploymentIdentityProviderAdapter"]
