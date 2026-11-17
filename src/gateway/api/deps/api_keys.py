"""How a request reaches its API key service, the key format this build mints, and token exchange."""

import uuid
from typing import Annotated

from fastapi import Depends, Request

from gateway.api.deps._base import (
    ContainerDep,
    PortSessionDep,
    get_config,
    get_unit_of_work,
    get_user_access,
)
from gateway.api.deps.auth import federated_token_verifier
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.repositories.api_keys import ApiKeyRepositories
from gateway.repositories.tenancy import WorkspaceRepository
from gateway.services.api_keys import (
    ApiKeyService,
    FederatedTokenVerifier,
    FederationTokenExchange,
)
from gateway.services.secret_box import configured_secret_keys


def get_api_key_format_port(db: PortSessionDep, container: ContainerDep) -> ApiKeyFormatPort:
    """Resolve the key-format adapter this build bound at startup."""
    return container.resolve(ApiKeyFormatPort, db)


ApiKeyFormatPortDep = Annotated[ApiKeyFormatPort, Depends(get_api_key_format_port)]


def get_api_key_service(uow: Annotated[UnitOfWork, Depends(get_unit_of_work)]) -> ApiKeyService:
    """Build the request's API key service on the request's Unit of Work; key owners' lists come from organizations."""
    return ApiKeyService(uow, ApiKeyRepositories.on(uow), users=get_user_access(uow))


ApiKeyServiceDep = Annotated[ApiKeyService, Depends(get_api_key_service)]


def build_federation_token_exchange(
    uow: UnitOfWork, config: GatewayConfig, verifier: FederatedTokenVerifier
) -> FederationTokenExchange:
    """Build the token exchange on a Unit of Work; the key's organization is read through the same one."""
    workspaces = WorkspaceRepository(uow)

    async def organization_of_workspace(workspace_id: uuid.UUID) -> uuid.UUID | None:
        workspace = await workspaces.get(workspace_id)
        return None if workspace is None else workspace.organization_id

    return FederationTokenExchange(
        uow,
        ApiKeyRepositories.on(uow),
        verifier=verifier,
        organization_of_workspace=organization_of_workspace,
        ttl_seconds=config.federation_access_token_ttl_sec,
        secrets=configured_secret_keys(),
    )


def get_federation_token_exchange(
    request: Request,
    uow: Annotated[UnitOfWork, Depends(get_unit_of_work)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> FederationTokenExchange:
    """The request's token exchange."""
    return build_federation_token_exchange(uow, config, federated_token_verifier(request, config))


FederationTokenExchangeDep = Annotated[FederationTokenExchange, Depends(get_federation_token_exchange)]
