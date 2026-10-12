import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import OAuthEmailNotVerifiedError, OAuthIdentityUnknownError
from gateway.log_config import logger
from gateway.models.identity import EmailAssertion, ProviderIdentity
from gateway.models.tenancy import User
from gateway.ports.identity_provider_port import IdentityProviderPort, IdentityResolution
from gateway.repositories.identity import IdentityRepositories
from gateway.services.oauth_service import provider_label


class IdentityService:
    """Decide which account a person signs in as.

    Each method runs in a Unit of Work block, so what it and the ports it calls stage commits together.
    """

    def __init__(
        self, uow: UnitOfWork, repositories: IdentityRepositories, *, identity_provider: IdentityProviderPort
    ) -> None:
        self._uow = uow
        self._links = repositories.links
        self._users = repositories.users
        self._identity_provider = identity_provider

    async def sign_in_with_oauth(self, external: ProviderIdentity) -> IdentityResolution:
        """Return the account that an identity a provider has proven signs in as.

        An identity linked to an account signs in as that account, and the sign-in changes nothing on it.
        An identity with no link signs in as the account the identity provider policy returns, and is linked to it.
        An account holds at most one link for each provider.

        Raises:
            OAuthEmailNotVerifiedError: If the provider says the address is not verified.
            OAuthIdentityUnknownError: If the account is deactivated or gone,
                or the account the policy returns is linked to another identity at the same provider.
            OAuthIdentityLinkConflictError: If a concurrent sign-in linked the identity to another account first.
            TenancyError: If the identity provider policy refuses the identity.

        """
        label = provider_label(external.provider)
        if external.email_verified is EmailAssertion.UNVERIFIED:
            raise OAuthEmailNotVerifiedError(label)
        async with self._uow:
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
                raise OAuthIdentityUnknownError(label)
            return IdentityResolution(identity=account, new_account=resolution.new_account)

    async def _active_account(self, user_id: uuid.UUID, label: str) -> User:
        """Return the account read under a lock, so a concurrent deactivation is seen before the sign-in completes."""
        account = await self._users.get_locked(user_id)
        if account is None or not account.is_active:
            raise OAuthIdentityUnknownError(label)
        return account
