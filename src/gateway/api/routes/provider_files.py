"""Contributed control-plane Files router with deployment-owned gateway authentication."""

import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db
from gateway.api.routes.hybrid_files import FilesRoute
from gateway.core.config import GatewayConfig
from gateway.services.provider_files.accounts import FileAccountResolver
from gateway.services.provider_files.cleanup import ProviderFileCleanup
from gateway.services.provider_files.contracts import (
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

# Authenticators must resolve registered gateway identity; foreground also requires a workspace API key.
ForegroundAuthenticator = Callable[[Request, AsyncSession], Awaitable[FileScope]]
GatewayAuthenticator = Callable[[Request, AsyncSession], Awaitable[FileScope]]
AttemptAuthorizer = Callable[[FileScope, OutputPrepare, AsyncSession], Awaitable[FileAccount]]
AccountResolver = Callable[[FileScope, uuid.UUID | None, bool, AsyncSession], Awaitable[FileAccount]]


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
    Db = Annotated[AsyncSession, Depends(get_db)]
    Config = Annotated[GatewayConfig, Depends(get_config)]

    async def principal(request: Request, db: Db, response: Response) -> FileScope:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Otari-Files-Protocol"] = "1"
        scope = await authenticate(request, db)
        await FileAccountResolver(db).repo.lock_user(scope.user_id)
        return scope

    async def gateway(request: Request, db: Db, response: Response) -> FileScope:
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Otari-Files-Protocol"] = "1"
        return await authenticate_gateway(request, db)

    def service(db: AsyncSession, config: GatewayConfig) -> ProviderFileService:
        if not config.files_provider_native_enabled:
            raise FilesError(404, "Provider-native Files are not enabled")
        if config.files_max_count is None or config.files_max_outstanding_bytes is None:
            raise FilesError(503, "Provider file quotas are not configured")
        return ProviderFileService(
            db,
            max_bytes=config.files_max_bytes,
            max_files=config.files_max_count,
            max_outstanding_bytes=config.files_max_outstanding_bytes,
            rate_limit_rpm=config.files_rate_limit_rpm,
            retention_seconds=(config.files_retention_hours or 168) * 3600,
            operation_seconds=config.files_operation_timeout_seconds,
            diagnostic_seconds=config.files_diagnostic_retention_days * 86400,
        )

    async def account(
        scope: FileScope, db: AsyncSession, generation_id: uuid.UUID | None = None, *, cleanup: bool = False
    ) -> FileAccount:
        resolver = FileAccountResolver(db)
        if generation_id is None:
            selected = await resolver.select_byo(scope)
            if selected is not None:
                return selected
        else:
            row = await resolver.repo.account(generation_id)
            if row is None or row.organization_id != scope.organization_id:
                raise FilesError(404, "Provider account unavailable")
            if row.credential_source == "organization_key":
                return await resolver.resolve_byo(generation_id, scope.organization_id, cleanup=cleanup)
        if resolve_hosted is None:
            raise FilesError(404, "Anthropic provider account unavailable")
        if not scope.default_gateway:
            raise FilesError(403, "Managed provider files require the default gateway")
        return await resolve_hosted(scope, generation_id, cleanup, db)

    Principal = Annotated[FileScope, Depends(principal)]
    Gateway = Annotated[FileScope, Depends(gateway)]

    @router.post("/uploads/prepare")
    async def prepare(body: PrepareUpload, scope: Principal, db: Db, config: Config) -> Any:
        return _wire(await service(db, config).prepare(scope, await account(scope, db), body))

    @router.post("/uploads/{binding_id}/finalize")
    async def finalize(binding_id: uuid.UUID, body: FinalizeUpload, scope: Principal, db: Db, config: Config) -> Any:
        return _wire(await service(db, config).finalize(scope, binding_id, body.metadata, body.expires_in_seconds))

    @router.post("/uploads/{binding_id}/abandon")
    async def abandon(
        binding_id: uuid.UUID, body: AbandonUpload, scope: Gateway, db: Db, config: Config
    ) -> dict[str, bool]:
        await service(db, config).abandon(binding_id, scope.gateway_id, body)
        return {"ok": True}

    @router.post("/list")
    async def list_files(body: FileListRequest, scope: Principal, db: Db, config: Config) -> Any:
        return _wire(await service(db, config).list_files(scope, body))

    @router.post("/references/resolve")
    async def references(body: References, scope: Principal, db: Db, config: Config) -> Any:
        generation = await service(db, config).references(scope, body.ids)
        return _wire(await account(scope, db, generation))

    @router.post("/outputs/prepare")
    async def prepare_output(body: OutputPrepare, scope: Principal, db: Db, config: Config) -> Any:
        # The authorizer intersects the original model plan, prices, and workspace tool policy.
        generation = await FileAccountResolver(db).repo.account(body.generation_id)
        if generation is None or generation.organization_id != scope.organization_id:
            raise FilesError(404, "Provider account unavailable")
        if generation.credential_source == "hosted_backend" and not scope.default_gateway:
            raise FilesError(403, "Managed provider files require the default gateway")
        selected = await authorize_attempt(scope, body, db)
        return _wire(await ProviderFileOutputs(service(db, config)).prepare(scope, selected, body))

    @router.post("/outputs/register")
    async def register_output(body: OutputRegister, scope: Principal, db: Db, config: Config) -> Any:
        return _wire(await ProviderFileOutputs(service(db, config)).register(scope, body.operation_id, body.metadata))

    @router.post("/outputs/{operation_id}/abandon")
    async def abandon_output(
        operation_id: uuid.UUID,
        body: AbandonUpload,
        scope: Gateway,
        db: Db,
        config: Config,
    ) -> Any:
        return _wire(
            await ProviderFileOutputs(service(db, config)).abandon(
                operation_id,
                scope.gateway_id,
                body.cleanup_token.get_secret_value(),
                body.metadata,
            )
        )

    @router.post("/outputs/{operation_id}/complete")
    async def complete_output(
        operation_id: uuid.UUID, body: CleanupResult, scope: Gateway, db: Db, config: Config
    ) -> dict[str, bool]:
        await ProviderFileOutputs(service(db, config)).complete(
            operation_id, scope.gateway_id, body.cleanup_token.get_secret_value()
        )
        return {"ok": True}

    @router.get("/status")
    async def backlog(scope: Gateway, db: Db, config: Config) -> dict[str, int]:
        return await service(db, config).repo.backlog(scope.organization_id)

    @router.post("/cleanup/claim")
    async def claim(body: CleanupClaim, scope: Gateway, db: Db, config: Config) -> Any:
        async def credential(generation_id: uuid.UUID) -> FileAccount:
            return await account(scope, db, generation_id, cleanup=True)

        lease = await ProviderFileCleanup(service(db, config)).claim(
            scope.organization_id,
            scope.gateway_id,
            body.limit,
            include_managed=scope.default_gateway,
            resolve_account=credential,
        )
        return {"lease": _wire(lease)}

    @router.post("/cleanup/{lease_id}/result")
    async def complete_lease(
        lease_id: uuid.UUID, body: LeaseResult, scope: Gateway, db: Db, config: Config
    ) -> dict[str, bool]:
        await ProviderFileCleanup(service(db, config)).complete(scope.organization_id, scope.gateway_id, lease_id, body)
        return {"ok": True}

    @router.post("/{file_id}/resolve")
    async def resolve(file_id: str, body: ResolveFile, scope: Principal, db: Db, config: Config) -> Any:
        lifecycle = service(db, config)
        selected = None
        if body.operation != "metadata":
            generation = await lifecycle.references(scope, [file_id])
            selected = await account(scope, db, generation)
        return _wire(await lifecycle.resolve(scope, file_id, body.operation, selected))

    @router.post("/{binding_id}/cleanup-result")
    async def cleanup_result(
        binding_id: uuid.UUID, body: CleanupResult, scope: Gateway, db: Db, config: Config
    ) -> dict[str, bool]:
        await service(db, config).cleanup_result(
            binding_id, scope.gateway_id, body.cleanup_token.get_secret_value(), body.deleted
        )
        return {"ok": True}

    return router
