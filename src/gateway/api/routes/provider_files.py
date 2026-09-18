"""Contributed control-plane Files router with deployment-owned gateway authentication."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import SecretStr

from gateway.api.deps import get_config, get_unit_of_work
from gateway.api.routes.hybrid_files import FilesRoute
from gateway.core.config import GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.services.provider_files.accounts import FileAccountResolver
from gateway.services.provider_files.capabilities import require_file_operation
from gateway.services.provider_files.cleanup import ProviderFileCleanup
from gateway.services.provider_files.contracts import (
    FILES_PROTOCOL_VERSION,
    AbandonUpload,
    CleanupClaim,
    CleanupResult,
    FileAccount,
    FileListRequest,
    FileScope,
    FilesError,
    FinalizeUpload,
    LeaseResult,
    OutputPrepare,
    OutputRegister,
    PrepareUpload,
    References,
    ResolveFile,
    WireModel,
)
from gateway.services.provider_files.lifecycle import ProviderFileService
from gateway.services.provider_files.outputs import ProviderFileOutputs

# Callbacks share the request's open Unit of Work and reach it through repositories.
ForegroundAuthenticator = Callable[[Request, UnitOfWork], Awaitable[FileScope]]
GatewayAuthenticator = Callable[[Request, UnitOfWork], Awaitable[FileScope]]
AttemptAuthorizer = Callable[[FileScope, OutputPrepare, UnitOfWork], Awaitable[FileAccount]]
AccountResolver = Callable[[FileScope, str, uuid.UUID | None, bool, UnitOfWork], Awaitable[FileAccount]]


def _wire(value: Any) -> Any:
    """Secrets are exposed only at the authenticated, no-store internal response boundary."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, WireModel):
        result = {key: _wire(getattr(value, key)) for key in type(value).model_fields if key in value.model_fields_set}
        result.update({key: _wire(item) for key, item in (value.model_extra or {}).items()})
        return result
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, dict):
        return {key: _wire(item) for key, item in value.items()}
    return value


def create_provider_files_router(
    *,
    authenticate: ForegroundAuthenticator,
    authenticate_gateway: GatewayAuthenticator,
    authorize_attempt: AttemptAuthorizer,
    resolve_hosted: AccountResolver | None = None,
) -> APIRouter:
    """Contribute behind attached-gateway capability; authentication is never optional."""
    router = APIRouter(prefix="/gateway/files", tags=["provider-files"], route_class=FilesRoute)
    Uow = Annotated[UnitOfWork, Depends(get_unit_of_work)]
    Config = Annotated[GatewayConfig, Depends(get_config)]

    async def principal(request: Request, uow: Uow, response: Response) -> FileScope:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Otari-Files-Protocol"] = FILES_PROTOCOL_VERSION
        scope = await FileAccountResolver(uow).authenticate(lambda: authenticate(request, uow))
        if request.headers.get("X-Otari-Files-Protocol") != FILES_PROTOCOL_VERSION:
            raise FilesError(409, "Unsupported Files protocol")
        return scope

    async def gateway(request: Request, uow: Uow, response: Response) -> FileScope:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Otari-Files-Protocol"] = FILES_PROTOCOL_VERSION
        scope = await FileAccountResolver(uow).authenticate(lambda: authenticate_gateway(request, uow))
        if request.headers.get("X-Otari-Files-Protocol") != FILES_PROTOCOL_VERSION:
            raise FilesError(409, "Unsupported Files protocol")
        return scope

    def service(uow: Uow, config: Config) -> ProviderFileService:
        if not config.files_provider_native_enabled:
            raise FilesError(404, "Provider-native Files are not enabled")
        if config.files_max_count is None or config.files_max_outstanding_bytes is None:
            raise FilesError(503, "Provider file quotas are not configured")
        return ProviderFileService(
            uow,
            max_bytes=config.files_max_bytes,
            max_files=config.files_max_count,
            max_outstanding_bytes=config.files_max_outstanding_bytes,
            rate_limit_rpm=config.files_rate_limit_rpm,
            retention_seconds=(config.files_retention_hours or 168) * 3600,
            operation_seconds=config.files_operation_timeout_seconds,
            diagnostic_seconds=config.files_diagnostic_retention_days * 86400,
        )

    async def account(
        scope: FileScope,
        uow: UnitOfWork,
        generation_id: uuid.UUID | None = None,
        *,
        provider: str | None = None,
        cleanup: bool = False,
    ) -> FileAccount:
        return await FileAccountResolver(uow).resolve(
            scope, generation_id, provider=provider, cleanup=cleanup, resolve_hosted=resolve_hosted
        )

    Principal = Annotated[FileScope, Depends(principal)]
    Gateway = Annotated[FileScope, Depends(gateway)]
    Service = Annotated[ProviderFileService, Depends(service)]

    @router.post("/uploads/prepare")
    async def prepare(body: PrepareUpload, scope: Principal, uow: Uow, lifecycle: Service) -> Any:
        require_file_operation(body.provider, "upload")
        require_file_operation(body.provider, "delete")
        selected = await account(scope, uow, provider=body.provider)
        return _wire(await lifecycle.prepare(scope, selected, body))

    @router.post("/uploads/{binding_id}/finalize")
    async def finalize(binding_id: uuid.UUID, body: FinalizeUpload, scope: Principal, lifecycle: Service) -> Any:
        return _wire(await lifecycle.finalize(scope, binding_id, body.metadata, body.expires_in_seconds))

    @router.post("/uploads/{binding_id}/abandon")
    async def abandon(
        binding_id: uuid.UUID, body: AbandonUpload, scope: Gateway, lifecycle: Service
    ) -> dict[str, bool]:
        await lifecycle.abandon(binding_id, scope.gateway_id, body)
        return {"ok": True}

    @router.post("/list")
    async def list_files(body: FileListRequest, scope: Principal, lifecycle: Service) -> Any:
        return _wire(await lifecycle.list_files(scope, body))

    @router.post("/references/resolve")
    async def references(body: References, scope: Principal, uow: Uow, lifecycle: Service) -> Any:
        generation = await lifecycle.references(scope, body.ids, provider=body.provider)
        return _wire(await account(scope, uow, generation, provider=body.provider))

    @router.post("/outputs/prepare")
    async def prepare_output(body: OutputPrepare, scope: Principal, uow: Uow, lifecycle: Service) -> Any:
        # The authorizer intersects the original model plan, prices, and workspace tool policy.
        selected = await FileAccountResolver(uow).authorize_attempt(
            scope, body, lambda: authorize_attempt(scope, body, uow)
        )
        return _wire(await ProviderFileOutputs(lifecycle).prepare(scope, selected, body))

    @router.post("/outputs/register")
    async def register_output(body: OutputRegister, scope: Principal, lifecycle: Service) -> Any:
        return _wire(await ProviderFileOutputs(lifecycle).register(scope, body.operation_id, body.metadata))

    @router.post("/outputs/{operation_id}/abandon")
    async def abandon_output(operation_id: uuid.UUID, body: AbandonUpload, scope: Gateway, lifecycle: Service) -> Any:
        return _wire(
            await ProviderFileOutputs(lifecycle).abandon(
                operation_id, scope.gateway_id, body.cleanup_token.get_secret_value(), body.metadata
            )
        )

    @router.post("/outputs/{operation_id}/complete")
    async def complete_output(
        operation_id: uuid.UUID, body: CleanupResult, scope: Gateway, lifecycle: Service
    ) -> dict[str, bool]:
        await ProviderFileOutputs(lifecycle).complete(
            operation_id, scope.gateway_id, body.cleanup_token.get_secret_value()
        )
        return {"ok": True}

    @router.get("/status")
    async def backlog(scope: Gateway, lifecycle: Service) -> dict[str, int]:
        return await lifecycle.backlog(scope.organization_id)

    @router.post("/cleanup/claim")
    async def claim(body: CleanupClaim, scope: Gateway, uow: Uow, lifecycle: Service) -> Any:
        async def credential(generation_id: uuid.UUID) -> FileAccount:
            return await account(scope, uow, generation_id, cleanup=True)

        lease = await ProviderFileCleanup(lifecycle).claim(
            scope.organization_id,
            scope.gateway_id,
            body.limit,
            include_managed=scope.default_gateway,
            resolve_account=credential,
        )
        return {"lease": _wire(lease)}

    @router.post("/cleanup/{lease_id}/result")
    async def complete_lease(
        lease_id: uuid.UUID, body: LeaseResult, scope: Gateway, lifecycle: Service
    ) -> dict[str, bool]:
        await ProviderFileCleanup(lifecycle).complete(scope.organization_id, scope.gateway_id, lease_id, body)
        return {"ok": True}

    @router.post("/{file_id}/resolve")
    async def resolve(file_id: str, body: ResolveFile, scope: Principal, uow: Uow, lifecycle: Service) -> Any:
        selected = None
        if body.operation != "metadata":
            generation = await lifecycle.references(scope, [file_id], provider=body.provider)
            selected = await account(scope, uow, generation, provider=body.provider)
        return _wire(await lifecycle.resolve(scope, file_id, body.operation, selected, provider=body.provider))

    @router.post("/{binding_id}/cleanup-result")
    async def cleanup_result(
        binding_id: uuid.UUID, body: CleanupResult, scope: Gateway, lifecycle: Service
    ) -> dict[str, bool]:
        await lifecycle.cleanup_result(
            binding_id, scope.gateway_id, body.cleanup_token.get_secret_value(), body.deleted
        )
        return {"ok": True}

    return router
