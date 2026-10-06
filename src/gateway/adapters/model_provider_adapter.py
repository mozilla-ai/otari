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
        can refuse: a model the operator switched off on this provider is not
        served, which is what makes that switch more than a stored boolean.

        ``organization_id`` and ``workspace_id`` are unused: whether this
        deployment holds a credential for a provider is a deployment-wide fact.
        An adapter that decides per organization raises
        ``HostedAccessDeniedError`` from here rather than changing what a
        credential is.

        ``provider`` is the caller's *instance* name, which equals the any-llm
        implementation for a bare selector. A ``config.yml`` instance of its own
        name is credentialed upstream of this port and never reaches it.
        """
        del organization_id, workspace_id
        if self._service is None:
            return None
        resolved = await self._service.resolve(provider, model)
        if resolved is None:
            return None
        return HostedCredential(
            api_key=resolved.api_key,
            api_base=resolved.api_base,
            # Echoed back unchanged, so what serves a candidate is what was
            # asked for and the pricing key settlement reprices on does not move.
            response_provider=provider,
            client_args=resolved.client_args,
        )

    async def get_hosted_models(self, *, organization_id: uuid.UUID | None) -> HostedModels:
        """The providers this deployment serves hosted inference on, each with the models it advertises.

        ``organization_id`` is unused for the reason ``resolve_hosted_credential``
        ignores it. The service answers the same refusals for the listing as
        for the resolve, which is what the port requires of the two.
        """
        del organization_id
        if self._service is None:
            return {}
        return await self._service.serveable_models()
