"""Reserve output capacity before dispatch and commit generated IDs before exposure."""

import json
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr

from gateway.models.provider_files import ProviderFileBinding, ProviderFileOutputOperation
from gateway.services.provider_files.contracts import (
    FileAccount,
    FileMetadata,
    FileScope,
    FilesError,
    Operation,
    OutputCleanup,
    OutputPrepare,
)
from gateway.services.provider_files.lifecycle import ProviderFileService
from gateway.services.secret_box import decrypt_secret, encrypt_secret


class ProviderFileOutputs:
    def __init__(self, service: ProviderFileService) -> None:
        self.service = service
        self.uow = service.uow
        self.repo = service.repo

    @staticmethod
    def _token(row: ProviderFileOutputOperation) -> str:
        return encrypt_secret(
            json.dumps({"output": str(row.id), "gateway": row.initiating_gateway_id, "nonce": row.cleanup_token_hash})
        )

    async def prepare(self, scope: FileScope, account: FileAccount, request: OutputPrepare) -> Operation:
        """The gateway authority must verify request_id/attempt_id before calling this method."""
        async with self.uow:
            return await self._prepare(scope, account, request)

    async def _prepare(self, scope: FileScope, account: FileAccount, request: OutputPrepare) -> Operation:
        now = datetime.now(UTC)
        await self.service._lock_scope(scope)
        await self.service._account(scope, account)
        if request.generation_id != account.generation_id:
            raise FilesError(409, "Inference account conflict")
        row = await self.repo.output_operation(request.operation_id)
        if row is not None:
            if (
                (
                    row.organization_id,
                    row.workspace_id,
                    row.user_id,
                    row.initiating_gateway_id,
                    row.request_id,
                    row.attempt_id,
                    row.provider_account_generation_id,
                )
                != (
                    scope.organization_id,
                    scope.workspace_id,
                    scope.user_id,
                    scope.gateway_id,
                    request.request_id,
                    request.attempt_id,
                    request.generation_id,
                )
                or row.state != "active"
                or row.deadline <= now
            ):
                raise FilesError(409, "Output operation conflict")
        else:
            count, size = await self.repo.capacity(scope.workspace_id, scope.user_id, now)
            reserved = min(20, self.service.max_files - count)
            available = min(reserved * self.service.max_bytes, self.service.max_outstanding_bytes - size)
            if reserved <= 0 or available <= 0:
                raise FilesError(429, "File capacity exceeded")
            row = ProviderFileOutputOperation(
                id=request.operation_id,
                organization_id=scope.organization_id,
                workspace_id=scope.workspace_id,
                user_id=scope.user_id,
                provider_account_generation_id=account.generation_id,
                initiating_gateway_id=scope.gateway_id,
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                cleanup_token_hash=secrets.token_hex(32),
                deadline=now + timedelta(seconds=self.service.operation_seconds),
                reserved_files=reserved,
                reserved_bytes=available,
            )
            await self.repo.save(row)
        return Operation(
            id=row.id,
            cleanup_token=SecretStr(self._token(row)),
            deadline=row.deadline,
            account=account,
            max_bytes=self.service.max_bytes,
            expires_in_seconds=self.service.retention_seconds,
        )

    async def register(self, scope: FileScope, operation_id: uuid.UUID, metadata: FileMetadata) -> FileMetadata:
        async with self.uow:
            result = await self._register(scope, operation_id, metadata)
        if result is None:
            raise FilesError(409, "Output operation has been revoked")
        return result

    async def _register(self, scope: FileScope, operation_id: uuid.UUID, metadata: FileMetadata) -> FileMetadata | None:
        now = datetime.now(UTC)
        await self.repo.lock_user(scope.user_id)
        await self.repo.lock_organization(scope.organization_id)
        row = await self.repo.output_operation(operation_id)
        if row is None or (row.organization_id, row.workspace_id, row.user_id, row.initiating_gateway_id) != (
            scope.organization_id,
            scope.workspace_id,
            scope.user_id,
            scope.gateway_id,
        ):
            raise FilesError(404, "Output operation unavailable")
        existing = await self.repo.by_provider_id(row.provider_account_generation_id, metadata.id)
        if existing is not None:
            if not self.service._owns(existing, scope):
                raise FilesError(409, "Provider file ownership conflict")
            if existing.state != "active" or existing.expires_at <= now:
                raise FilesError(409, "Provider file is no longer active")
            return self.service._metadata(existing)
        account = await self.repo.account(row.provider_account_generation_id)
        size = metadata.size_bytes if metadata.size_bytes is not None else self.service.max_bytes
        active = (
            row.state == "active"
            and row.deadline > now
            and account is not None
            and account.status == "active"
            and row.reserved_files > 0
            and row.reserved_bytes >= size
            and size <= self.service.max_bytes
            and await self.repo.active_user(scope.user_id)
            and await self.repo.workspace_exists(scope.workspace_id, scope.organization_id)
        )
        expires = now + timedelta(seconds=self.service.retention_seconds)
        if metadata.expires_at is not None:
            expires = min(expires, metadata.expires_at)
        active = active and expires > now
        binding = ProviderFileBinding(
            output_operation_id=row.id,
            organization_id=row.organization_id,
            workspace_id=row.workspace_id,
            user_id=row.user_id,
            provider_account_generation_id=row.provider_account_generation_id,
            provider_file_id=metadata.id,
            encrypted_metadata=encrypt_secret(metadata.model_dump_json(exclude_unset=True)),
            size_bytes=size,
            purpose=metadata.purpose,
            provider_created_at=metadata.created_at,
            downloadable=metadata.downloadable is True,
            expires_at=expires,
            provider_expires_at=metadata.expires_at,
            operation_deadline=row.deadline,
            initiating_gateway_id=row.initiating_gateway_id,
            cleanup_token_hash=secrets.token_hex(32),
            state="active" if active else "pending_cleanup",
            cleanup_reason=None if active else "revoked_output",
            cleanup_after=None if active else now,
        )
        await self.repo.save(binding)
        row.reserved_files = max(0, row.reserved_files - 1)
        row.reserved_bytes = max(0, row.reserved_bytes - size)
        return metadata if active else None

    async def complete(self, operation_id: uuid.UUID, gateway_id: str, token: str) -> None:
        async with self.uow:
            await self._complete(operation_id, gateway_id, token)

    async def _complete(self, operation_id: uuid.UUID, gateway_id: str, token: str) -> None:
        row = await self.repo.output_operation(operation_id)
        if row is None:
            raise FilesError(404, "Output operation unavailable")
        await self.repo.lock_organization(row.organization_id)
        await self.repo.refresh(row)
        try:
            payload = json.loads(decrypt_secret(token))
        except (ValueError, TypeError):
            raise FilesError(403, "Invalid cleanup authority") from None
        if payload != {"output": str(row.id), "gateway": gateway_id, "nonce": row.cleanup_token_hash}:
            raise FilesError(403, "Invalid cleanup authority")
        row.state, row.reserved_bytes, row.reserved_files = "completed", 0, 0

    async def abandon(
        self,
        operation_id: uuid.UUID,
        gateway_id: str,
        token: str,
        metadata: FileMetadata | None,
        file_id: str | None = None,
    ) -> OutputCleanup:
        async with self.uow:
            return await self._abandon(operation_id, gateway_id, token, metadata, file_id)

    async def _abandon(
        self,
        operation_id: uuid.UUID,
        gateway_id: str,
        token: str,
        metadata: FileMetadata | None,
        file_id: str | None = None,
    ) -> OutputCleanup:
        row = await self.repo.output_operation(operation_id)
        if row is None:
            raise FilesError(404, "Output operation unavailable")
        await self.repo.lock_organization(row.organization_id)
        await self.repo.refresh(row)
        try:
            payload = json.loads(decrypt_secret(token))
        except (ValueError, TypeError):
            raise FilesError(403, "Invalid cleanup authority") from None
        if payload != {"output": str(row.id), "gateway": gateway_id, "nonce": row.cleanup_token_hash}:
            raise FilesError(403, "Invalid cleanup authority")
        identifier = metadata.id if metadata is not None else file_id
        if not identifier:
            raise FilesError(400, "Provider file ID is required")
        existing = await self.repo.by_provider_id(row.provider_account_generation_id, identifier)
        if existing is not None and existing.output_operation_id != operation_id:
            raise FilesError(409, "Provider file ownership conflict")
        if existing is None:
            existing = ProviderFileBinding(
                output_operation_id=row.id,
                organization_id=row.organization_id,
                workspace_id=row.workspace_id,
                user_id=row.user_id,
                provider_account_generation_id=row.provider_account_generation_id,
                provider_file_id=identifier,
                encrypted_metadata=encrypt_secret(metadata.model_dump_json(exclude_unset=True))
                if metadata is not None
                else None,
                size_bytes=metadata.size_bytes
                if metadata is not None and metadata.size_bytes is not None
                else self.service.max_bytes,
                purpose=metadata.purpose if metadata is not None else None,
                provider_created_at=metadata.created_at if metadata is not None else None,
                downloadable=metadata is not None and metadata.downloadable is True,
                expires_at=datetime.now(UTC),
                operation_deadline=row.deadline,
                initiating_gateway_id=gateway_id,
                cleanup_token_hash=secrets.token_hex(32),
            )
            await self.repo.save(existing)
        existing.state = "pending_cleanup"
        existing.cleanup_reason, existing.cleanup_after = "output_abandoned", datetime.now(UTC)
        return OutputCleanup(operation_id=existing.id, cleanup_token=SecretStr(self.service._token(existing)))
