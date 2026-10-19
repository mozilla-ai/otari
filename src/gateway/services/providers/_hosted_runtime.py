"""The runtime path: what the deployment advertises, and the credential that serves a candidate."""

import uuid

from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.model_provider_port import NO_ORGANIZATION, HostedCredential, HostedModels, ModelProviderPort
from gateway.repositories.providers import HostedProviderModelRepository
from gateway.services.providers._hosted_credentials import ResolvedHostedProvider
from gateway.services.providers._hosted_sources import HostedSources


def organization_to_serve(organization_id: uuid.UUID | None) -> uuid.UUID:
    """The organization the hosted rung asks the port about for a request.

    A request with no organization (its workspace belongs to none) is served on
    what is restricted to no organization, so it is asked as one no restriction
    names; billing then has no organization to charge, and charges none.
    """
    return organization_id if organization_id is not None else NO_ORGANIZATION


async def hosted_models_for(port: ModelProviderPort, *, organization_id: uuid.UUID | None) -> HostedModels:
    """What the port advertises to a request's organization, a request with no organization included."""
    return await port.get_hosted_models(organization_id=organization_to_serve(organization_id))


async def hosted_credential_for(
    port: ModelProviderPort,
    *,
    organization_id: uuid.UUID | None,
    workspace_id: uuid.UUID | None,
    provider: str,
    model: str | None,
) -> HostedCredential | None:
    """The deployment-owned credential the port resolves for a request, a request with no organization included."""
    return await port.resolve_hosted_credential(
        organization_id=organization_to_serve(organization_id),
        workspace_id=workspace_id,
        provider=provider,
        model=model,
    )


class HostedProviderRuntime:
    """Answer the two questions the dispatch path and the catalog ask."""

    def __init__(self, uow: UnitOfWork, *, sources: HostedSources, models: HostedProviderModelRepository) -> None:
        self.uow = uow
        self.sources = sources
        self.models = models

    async def serveable_models(self, *, organization_id: uuid.UUID | None) -> dict[str, frozenset[str]]:
        """Each provider serveable to ``organization_id``, with the models switched on under it.

        The same refusals as :meth:`resolve`, so the listing and the resolve
        agree: a provider switched off, one restricted to another organization
        and one whose ciphertext no configured key reads are all absent.
        """
        async with self.uow:
            serveable = [
                source.name
                for source in await self.sources.all()
                if source.enabled
                and not source.withheld_from(organization_id)
                and source.credential(quiet=True) is not None
            ]
            served = await self.models.enabled_models_for_providers(serveable)
        return {name: frozenset(served.get(name, set())) for name in serveable}

    async def resolve(
        self, name: str, model: str | None = None, *, organization_id: uuid.UUID | None
    ) -> ResolvedHostedProvider | None:
        """A usable credential for the hosted provider ``name`` on behalf of ``organization_id``, or None.

        None covers every case the caller is not asked to tell apart: no
        provider, one switched off or restricted to another organization, a
        model the provider does not offer switched on, and a ciphertext no
        configured key reads. A model with no row is refused, so the operator's
        roster is the whole of what the deployment's credential serves.
        """
        async with self.uow:
            source = await self.sources.get(name)
            if source is None or not source.enabled or source.withheld_from(organization_id):
                return None
            if model is not None:
                offered = await self.models.get_by_model(source.name, model)
                if offered is None or not offered.enabled:
                    return None
            return source.credential()
