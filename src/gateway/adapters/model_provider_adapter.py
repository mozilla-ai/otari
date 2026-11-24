"""Core adapter for ``ModelProviderPort``: the deployment's hosted providers.

Satisfies :class:`gateway.ports.model_provider_port.ModelProviderPort` from the
hosted providers an operator configures (``services/providers``), so a
candidate that reached the end of the credential ladder with no BYO key and no
configured instance is served on the deployment's own credential. The port is
asked after every rung upstream of it has missed, so a row here never displaces
a credential the caller supplied.

Built with the request's Unit of Work, which the composition root hands in.
Without one (the boot-time shape check) the adapter is built with no service
and answers nothing.

The adapter hands nothing out over the wire: its one consumer on the request
path puts the key in an any-llm client in this process and dials the upstream.
"""

import uuid
from dataclasses import replace

from gateway.core.config import region_instance
from gateway.ports.model_provider_port import HostedCredential, HostedModels, ModelProviderPort
from gateway.services.providers import HostedProviderService, HostedServing, ResolvedHostedProvider


class HostedProviderModelProviderAdapter(ModelProviderPort):
    """Serve a candidate on one of the deployment's hosted providers."""

    def __init__(self, service: HostedProviderService | None) -> None:
        """Bind the hosted-providers service, or nothing where the deployment has no database.

        Built per request, so every candidate of a request is resolved on one
        read of the hosted providers rather than on a read each.
        """
        self._service = service
        self._serving: HostedServing | None = None

    async def resolve_hosted_credential(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        provider: str,
        model: str | None,
    ) -> HostedCredential | None:
        """Resolve the deployment's own credential for ``provider``.

        ``model`` selects no region but it can refuse: only a model the operator switched on in this provider's
        roster is served, so one switched off is refused, and so is one never
        offered unless the provider serves unlisted models.

        ``organization_id`` decides one thing: a provider restricted to another
        organization answers ``None``, as if it were not configured, so a tenant
        learns nothing about a credential kept from it. ``workspace_id`` is
        attribution only and unused.

        ``provider`` is the caller's *instance* name: the any-llm
        implementation for a bare selector, or a hosted provider's own name. A ``config.yml`` instance of its own
        name is credentialed upstream of this port and never reaches it. An
        accepted alias (``openai-compatible``) is served by the provider it
        folds to, and is answered under that provider's name.

        A provider with regions answers its first enabled region, carrying the
        rest as ``alternates`` in order; a name that pins one
        (``tokenhub@sg``) answers that region alone.
        """
        del workspace_id
        if self._service is None:
            return None
        if self._serving is None:
            self._serving = self._service.serving()
        regions = await self._serving.resolve(provider, model, organization_id=organization_id)
        if not regions:
            return None
        credentials = [_credential(resolved, provider) for resolved in regions]
        return replace(credentials[0], alternates=tuple(credentials[1:]))

    async def get_hosted_models(self, *, organization_id: uuid.UUID | None) -> HostedModels:
        """The providers this deployment serves ``organization_id`` hosted inference on, with their advertised models.

        A provider restricted to another organization is absent, as
        ``resolve_hosted_credential`` refuses it; ``None`` is the deployment's
        own view and sees every provider. The service answers the same refusals
        for the listing as for the resolve, which is what the port requires.
        """
        if self._service is None:
            return {}
        return await self._service.serveable_models(organization_id=organization_id)


def _credential(resolved: ResolvedHostedProvider, provider: str) -> HostedCredential:
    """One region's connection as the port hands it out.

    Named by the stored name, so an alias settles under the key the provider is
    priced and switched under rather than under the alias, which nothing
    prices; a region's name pins it (``tokenhub@sg``), so its usage and its
    own price are kept apart from the provider's other regions.
    """
    name = resolved.name or provider
    return HostedCredential(
        api_key=resolved.api_key,
        api_base=resolved.api_base,
        response_provider=region_instance(name, resolved.region),
        client_args=resolved.client_args,
        implementation=resolved.implementation,
        supported_countries=resolved.supported_countries,
        data_location=resolved.data_location,
        routing_locality=resolved.routing_locality,
    )
