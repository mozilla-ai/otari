"""A hosted model provider that a test binds in place of the core adapter."""

import uuid
from typing import Any

from fastapi.testclient import TestClient

from gateway.ports.model_provider_port import HostedCredential, ModelProviderPort


class HostedModelProvider:
    """Serves the named providers on a deployment-owned credential."""

    def __init__(self, *providers: str, error: Exception | None = None) -> None:
        self.providers = frozenset(providers)
        self.error = error
        self.asked_for: list[uuid.UUID] = []

    async def resolve_hosted_credential(
        self, *, organization_id: uuid.UUID, workspace_id: uuid.UUID | None, provider: str, model: str | None
    ) -> HostedCredential | None:
        del organization_id, workspace_id, model
        if provider in self.providers:
            return HostedCredential(api_key="fleet", api_base=None, response_provider=provider)
        return None

    async def get_hosted_providers(self, *, organization_id: uuid.UUID) -> frozenset[str]:
        self.asked_for.append(organization_id)
        if self.error is not None:
            raise self.error
        return self.providers


def bind_model_provider(client: TestClient, model_provider: HostedModelProvider) -> None:
    """Binds ``model_provider`` on the app under test."""
    container: Any = client.app.state.container  # type: ignore[attr-defined]
    container.bind(ModelProviderPort, lambda session: model_provider)
