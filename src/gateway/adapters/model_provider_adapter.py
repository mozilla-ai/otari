"""Core adapter for ``ModelProviderPort``: the deployment's hosted providers.

Satisfies :class:`gateway.ports.model_provider_port.ModelProviderPort` from the
hosted providers an operator configures (``services/providers``), so a
candidate that reached the end of the credential ladder with no BYO key and no
configured instance is served on the deployment's own credential. The port is
asked after every rung upstream of it has missed, so a row here never displaces
a credential the caller supplied.

Built with the request's Unit of Work, which the composition root hands in. In
hybrid mode there is no database and no Unit of Work, so the adapter is built
with no service and answers nothing: a hybrid data plane resolves credentials
over the platform protocol, never through this port.

The adapter hands nothing out over the wire: its one consumer on the request
path puts the key in an any-llm client in this process and dials the upstream.
"""

import uuid

from gateway.ports.model_provider_port import HostedCredential, HostedModels, ModelProviderPort
from gateway.services.providers import HostedProviderService


class HostedProviderModelProviderAdapter(ModelProviderPort):
    """Serve a candidate on one of the deployment's hosted providers."""

    def __init__(self, service: HostedProviderService | None) -> None:
        """Bind the hosted-providers service, or nothing where the deployment has no database."""
        self._service = service

    async def resolve_hosted_credential(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        provider: str,
        model: str | None,
    ) -> HostedCredential | None:
        """Resolve the deployment's own credential for ``provider``.

        ``model`` selects nothing (a hosted provider is one credential) but it
        can refuse: only a model the operator switched on in this provider's
        roster is served, so one switched off or never offered is refused.

        ``organization_id`` decides one thing: a provider restricted to another
        organization answers ``None``, as if it were not configured, so a tenant
        learns nothing about a credential kept from it. ``workspace_id`` is
        attribution only and unused.

        ``provider`` is the caller's *instance* name: the any-llm
        implementation for a bare selector, or a hosted provider's own name. A ``config.yml`` instance of its own
        name is credentialed upstream of this port and never reaches it. An
        accepted alias (``openai-compatible``) is served by the provider it
        folds to, and is answered under that provider's name.
        """
        del workspace_id
        if self._service is None:
            return None
        resolved = await self._service.resolve(provider, model, organization_id=organization_id)
        if resolved is None:
            return None
        return HostedCredential(
            api_key=resolved.api_key,
            api_base=resolved.api_base,
            # The stored name, so an alias settles under the key the provider is
            # priced and switched under rather than under the alias, which
            # nothing prices. Equal to ``provider`` for anything but an alias.
            response_provider=resolved.name or provider,
            client_args=resolved.client_args,
            implementation=resolved.implementation,
        )

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
