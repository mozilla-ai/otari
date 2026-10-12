import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import (
    OAuthAccountLinkedElsewhereError,
    OAuthEmailNotVerifiedError,
    OAuthIdentityUnknownError,
)
from gateway.log_config import logger
from gateway.models.identity import EmailAssertion, ProviderIdentity
from gateway.models.tenancy import User
from gateway.ports.identity_provider_port import IdentityProviderPort, IdentityResolution
from gateway.repositories.identity import IdentityRepositories
from gateway.services.oauth_service import provider_label
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.organization_domain_service import OrganizationDomainService

# Stages a dashboard session for an account and returns its token and expiry.
SessionStarter = Callable[[uuid.UUID], Awaitable[tuple[str, datetime]]]


@dataclass(frozen=True)
class SignIn:
    """An account that signed in, and the dashboard session the sign-in started.

    ``active_organization_id`` is the organization the session acts in, or ``None`` for an account with no
    live membership. ``new_account`` is true when the sign-in created the account, or is the first way in to
    one nobody had claimed.
    """

    account: User
    active_organization_id: uuid.UUID | None
    new_account: bool
    session_token: str
    session_expires_at: datetime


class IdentityService:
    """Decide which account a person signs in as, and start their dashboard session.

    Each method runs in a Unit of Work block, so what it and its collaborators stage commits together.
    """

    def __init__(
        self,
        uow: UnitOfWork,
        repositories: IdentityRepositories,
        *,
        identity_provider: IdentityProviderPort,
        organization_domains: OrganizationDomainService,
        organizations: OrganizationService,
        start_session: SessionStarter,
    ) -> None:
        self._uow = uow
        self._links = repositories.links
        self._users = repositories.users
        self._identity_provider = identity_provider
        self._organization_domains = organization_domains
        self._organizations = organizations
        self._start_session = start_session

    async def sign_in_with_oauth(self, external: ProviderIdentity) -> SignIn:
        """Sign in the account that an identity a provider has proven belongs to.

        An identity linked to an account signs in as that account, and its password and verification stay as they are.
        An identity with no link signs in as the account the identity provider policy returns, and is linked to it.
        An account holds at most one link for each provider.
        The account's changes, any membership it joins by its address's domain, its active organization,
        and the session commit together.

        Raises:
            OAuthEmailNotVerifiedError: If the provider says the address is not verified.
            OAuthAccountLinkedElsewhereError: If the account the policy returns is linked to another identity
                at the same provider.
            OAuthIdentityUnknownError: If the account is deactivated or gone.
            OAuthIdentityLinkConflictError: If a concurrent sign-in linked the identity to another account first.
            TenancyError: If the identity provider policy refuses the identity.

        """
        label = provider_label(external.provider)
        if external.email_verified is EmailAssertion.UNVERIFIED:
            raise OAuthEmailNotVerifiedError(label)
        async with self._uow:
            resolution = await self._account_for(external, label)
            account = resolution.identity
            await self._organization_domains.auto_join_for_user(account)
            active_organization_id = await self._organizations.active_organization_id_for_sign_in(account)
            token, expires_at = await self._start_session(account.id)
        return SignIn(
            account=account,
            active_organization_id=active_organization_id,
            new_account=resolution.new_account,
            session_token=token,
            session_expires_at=expires_at,
        )

    async def _account_for(self, external: ProviderIdentity, label: str) -> IdentityResolution:
        link = await self._links.get_by_identity(provider=external.provider, subject=external.subject)
        if link is not None:
            return IdentityResolution(identity=await self._active_account(link.user_id, label), new_account=False)

        resolution = await self._identity_provider.resolve(external)
        account = await self._active_account(resolution.identity.id, label)
        # A concurrent first sign-in can link this identity, or this account at the provider, after that lookup.
        existing = await self._links.get_for_user(provider=external.provider, user_id=account.id)
        if existing is None:
            await self._links.add(external, user_id=account.id)
        elif existing.subject != external.subject:
            logger.warning(
                "Refused a %s sign-in to account %s, which is linked to another identity at that provider",
                external.provider,
                account.id,
            )
            raise OAuthAccountLinkedElsewhereError(label)
        return IdentityResolution(identity=account, new_account=resolution.new_account)

    async def _active_account(self, user_id: uuid.UUID, label: str) -> User:
        """Return the account read under a lock, so a concurrent deactivation is seen before the sign-in completes."""
        account = await self._users.get_locked(user_id)
        if account is None or not account.is_active:
            raise OAuthIdentityUnknownError(label)
        return account
