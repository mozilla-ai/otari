"""The runtime path: what the deployment advertises, and the credential that serves a candidate."""

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.providers import HostedProviderModelRepository, HostedProviderRepository
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider, credential_of, lookup_name


class HostedProviderRuntime:
    """Answer the two questions the dispatch path and the catalog ask."""

    def __init__(
        self, uow: UnitOfWork, *, providers: HostedProviderRepository, models: HostedProviderModelRepository
    ) -> None:
        self.uow = uow
        self.providers = providers
        self.models = models

    async def serveable_models(self) -> dict[str, frozenset[str]]:
        """Each serveable provider with the models switched on under it.

        The same refusals as :meth:`resolve`, so the listing and the resolve
        agree: a provider switched off and one whose ciphertext no configured
        key reads are both absent. A model with no row is served but not
        advertised, because the offered list is discovery's best effort.
        """
        async with self.uow:
            serveable = [
                row.provider
                for row in await self.providers.list_all()
                if row.enabled and credential_of(row, quiet=True) is not None
            ]
            served = await self.models.enabled_models_for_providers(serveable)
        return {provider: frozenset(served.get(provider, set())) for provider in serveable}

    async def resolve(self, provider: str, model: str | None = None) -> ResolvedHostedProvider | None:
        """A usable credential for ``provider``, or None.

        None covers every case the caller is not asked to tell apart: no row, a
        provider switched off, a model switched off, and a ciphertext no
        configured key reads. The model gate is one-sided: a row that exists
        and is switched off refuses, while a model with no row serves.
        """
        async with self.uow:
            row = await self.providers.get_by_provider(lookup_name(provider))
            if row is None or not row.enabled:
                return None
            if model is not None:
                offered = await self.models.get_by_model(row.provider, model)
                if offered is not None and not offered.enabled:
                    return None
            return credential_of(row)
