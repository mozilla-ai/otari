"""The runtime path: what the deployment advertises, and the credential that serves a candidate."""

import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.providers import HostedProvider
from gateway.repositories.providers import HostedProviderModelRepository, HostedProviderRepository
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider, credential_of, lookup_name


def _withheld_from(row: HostedProvider, organization_id: uuid.UUID | None) -> bool:
    """Whether ``organization_id`` is served as if ``row`` did not exist.

    ``None`` is the deployment's own view, which sees every provider.
    """
    return (
        organization_id is not None
        and row.restricted_to_organization_id is not None
        and row.restricted_to_organization_id != organization_id
    )


class HostedProviderRuntime:
    """Answer the two questions the dispatch path and the catalog ask."""

    def __init__(
        self, uow: UnitOfWork, *, providers: HostedProviderRepository, models: HostedProviderModelRepository
    ) -> None:
        self.uow = uow
        self.providers = providers
        self.models = models

    async def serveable_models(self, *, organization_id: uuid.UUID | None) -> dict[str, frozenset[str]]:
        """Each provider serveable to ``organization_id``, with the models switched on under it.

        The same refusals as :meth:`resolve`, so the listing and the resolve
        agree: a provider switched off, one restricted to another organization
        and one whose ciphertext no configured key reads are all absent. A model
        with no row is served but not advertised, because the offered list is
        discovery's best effort.
        """
        async with self.uow:
            serveable = [
                row.provider
                for row in await self.providers.list_all()
                if row.enabled
                and not _withheld_from(row, organization_id)
                and credential_of(row, quiet=True) is not None
            ]
            served = await self.models.enabled_models_for_providers(serveable)
        return {provider: frozenset(served.get(provider, set())) for provider in serveable}

    async def resolve(
        self, provider: str, model: str | None = None, *, organization_id: uuid.UUID | None
    ) -> ResolvedHostedProvider | None:
        """A usable credential for ``provider`` on behalf of ``organization_id``, or None.

        None covers every case the caller is not asked to tell apart: no row, a
        provider switched off or restricted to another organization, a model
        switched off, and a ciphertext no configured key reads. The model gate
        is one-sided: a row that exists and is switched off refuses, while a
        model with no row serves.
        """
        async with self.uow:
            row = await self.providers.get_by_provider(lookup_name(provider))
            if row is None or not row.enabled or _withheld_from(row, organization_id):
                return None
            if model is not None:
                offered = await self.models.get_by_model(row.provider, model)
                if offered is not None and not offered.enabled:
                    return None
            return credential_of(row)
