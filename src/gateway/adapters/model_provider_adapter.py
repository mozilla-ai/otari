"""Core adapter for ``ModelProviderPort``: no hosted-inference fleet.

Satisfies :class:`gateway.ports.model_provider_port.ModelProviderPort` with
Otari's own answer: there is no fleet here to proxy to. Every candidate that
reaches this adapter has already failed the caller's BYO-credential lookup, and
a deployment pointing at its own backends resolves a credential the ordinary
way well upstream of this port, so what remains is genuinely unserved and this
adapter's whole contract is saying so. An overlay binds a hosted-inference
adapter behind the same port.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.ports.model_provider_port import HostedCredential


class SelfHostedModelProviderAdapter:
    """Core adapter: a self-hosted deployment has no hosted account to consult.

    Session-agnostic: the answer is a deployment-wide constant, not something
    that depends on the request's data.
    """

    def __init__(self, session: AsyncSession | None) -> None:
        # Accepted to match the container's per-request factory; unused, because
        # the core answer never depends on the session.
        del session

    async def resolve_hosted_credential(
        self,
        *,
        organization_id: uuid.UUID,
        workspace_id: uuid.UUID | None,
        provider: str,
        model: str | None,
    ) -> HostedCredential | None:
        return None
