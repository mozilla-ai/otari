from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.identity_provider_port import EmailAssertion, IdentityProviderPort, IdentityResolution
from gateway.services.oauth_service import OAuthIdentity


class IdentityService:
    """Decide which account a person signs in as.

    Each method runs in a Unit of Work block, so what it and the ports it calls stage commits together.
    """

    def __init__(self, uow: UnitOfWork, *, identity_provider: IdentityProviderPort) -> None:
        self._uow = uow
        self._identity_provider = identity_provider

    async def sign_in_with_oauth(self, external: OAuthIdentity) -> IdentityResolution:
        """Return the account that an identity a provider has proven signs in as.

        Raises:
            TenancyError: If this identity may not sign in to this deployment.

        """
        async with self._uow:
            return await self._identity_provider.resolve(
                provider=external.provider,
                email=external.email,
                full_name=external.full_name,
                email_verified=external.email_verified is EmailAssertion.VERIFIED,
            )
