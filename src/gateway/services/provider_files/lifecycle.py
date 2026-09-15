"""Control-plane authorization and atomic provider-file state transitions."""

import json
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.provider_files import ProviderAccountGeneration, ProviderFileBinding, ProviderFileRateWindow
from gateway.repositories.tenancy.provider_file_repository import ProviderFileRepository
from gateway.services.provider_files.contracts import (
    AbandonUpload,
    FileAccount,
    FileListRequest,
    FileMetadata,
    FilePage,
    FileScope,
    FilesError,
    Operation,
    PrepareUpload,
    ResolvedFile,
)
from gateway.services.secret_box import decrypt_secret, encrypt_secret


class ProviderFileService:
    """Own transactions; callers supply identities derived by their gateway authenticator."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        max_bytes: int,
        max_files: int,
        max_outstanding_bytes: int,
        rate_limit_rpm: int = 60,
        retention_seconds: int = 604800,
        operation_seconds: int = 600,
        diagnostic_seconds: int = 2592000,
    ) -> None:
        if not (max_bytes > 0 and max_files > 0 and max_outstanding_bytes >= max_bytes):
            raise ValueError("Explicit positive file and outstanding-byte quotas are required")
        if not 3600 <= retention_seconds <= 7776000 or operation_seconds <= 0 or rate_limit_rpm <= 0:
            raise ValueError("Invalid provider file limits")
        self.db = db
        self.repo = ProviderFileRepository(db)
        self.max_bytes = max_bytes
        self.max_files = max_files
        self.max_outstanding_bytes = max_outstanding_bytes
        self.rate_limit_rpm = rate_limit_rpm
        self.retention_seconds = retention_seconds
        self.operation_seconds = operation_seconds
        self.diagnostic_seconds = diagnostic_seconds

    async def _lock_scope(self, scope: FileScope) -> None:
        await self.repo.lock_user(scope.user_id)
        if not await self.repo.lock_organization(scope.organization_id):
            raise FilesError(404, "File unavailable")
        if not await self.repo.active_user(scope.user_id):
            raise FilesError(404, "File unavailable")
        if not await self.repo.workspace_exists(scope.workspace_id, scope.organization_id):
            raise FilesError(404, "File unavailable")

    async def _rate_limit(self, scope: FileScope, now: datetime) -> None:
        window = int(now.timestamp()) // 60
        row = await self.db.get(ProviderFileRateWindow, (scope.workspace_id, scope.user_id))
        if row is None:
            row = ProviderFileRateWindow(workspace_id=scope.workspace_id, user_id=scope.user_id, window=window)
            self.db.add(row)
        if row.window != window:
            row.window, row.count = window, 0
        if row.count >= self.rate_limit_rpm:
            raise FilesError(429, "File operation rate limit exceeded")
        row.count += 1
        await self.db.flush()

    async def _account(self, scope: FileScope, account: FileAccount) -> ProviderAccountGeneration:
        row = await self.repo.account(account.generation_id)
        if row is None or row.organization_id != scope.organization_id or row.status != "active":
            raise FilesError(404, "Provider account unavailable")
        if row.credential_source == "hosted_backend" and not scope.default_gateway:
            raise FilesError(403, "Managed provider files require the default gateway")
        return row

    @staticmethod
    def _owns(row: ProviderFileBinding, scope: FileScope) -> bool:
        return (row.organization_id, row.workspace_id, row.user_id) == (
            scope.organization_id,
            scope.workspace_id,
            scope.user_id,
        )

    @staticmethod
    def _metadata(row: ProviderFileBinding) -> FileMetadata:
        if row.encrypted_metadata is None:
            raise FilesError(404, "File unavailable")
        return FileMetadata.model_validate_json(decrypt_secret(row.encrypted_metadata))

    @staticmethod
    def _token(row: ProviderFileBinding) -> str:
        # Authenticated encryption permits idempotent prepare to mint equivalent scoped tokens.
        return encrypt_secret(
            json.dumps({"id": str(row.id), "gateway": row.initiating_gateway_id, "nonce": row.cleanup_token_hash})
        )

    @staticmethod
    def _check_token(row: ProviderFileBinding, gateway_id: str, token: str) -> None:
        try:
            payload = json.loads(decrypt_secret(token))
        except (ValueError, TypeError):
            raise FilesError(403, "Invalid cleanup authority") from None
        expected = {"id": str(row.id), "gateway": gateway_id, "nonce": row.cleanup_token_hash}
        if payload != expected or row.initiating_gateway_id != gateway_id:
            raise FilesError(403, "Invalid cleanup authority")

    def _operation(self, row: ProviderFileBinding, account: FileAccount) -> Operation:
        return Operation(
            id=row.id,
            cleanup_token=SecretStr(self._token(row)),
            deadline=row.operation_deadline,
            account=account,
            max_bytes=row.size_bytes,
            expires_in_seconds=int((row.expires_at - row.created_at).total_seconds()),
        )

    async def prepare(self, scope: FileScope, account: FileAccount, request: PrepareUpload) -> Operation:
        now = datetime.now(UTC)
        await self._lock_scope(scope)
        await self._account(scope, account)
        existing = await self.repo.get(request.operation_id)
        duration = min(request.expires_in_seconds or self.retention_seconds, self.retention_seconds)
        reserved_bytes = min(request.size_bytes, self.max_bytes)
        if existing is not None:
            if (
                not self._owns(existing, scope)
                or existing.initiating_gateway_id != scope.gateway_id
                or existing.provider_account_generation_id != account.generation_id
                or existing.size_bytes != reserved_bytes
                or int((existing.expires_at - existing.created_at).total_seconds()) != duration
            ):
                raise FilesError(409, "Upload operation conflict")
            if existing.state != "pending_upload" or existing.operation_deadline <= now:
                raise FilesError(409, "Upload operation is no longer pending")
            return self._operation(existing, account)
        if request.size_bytes <= 0:
            raise FilesError(413, "File size limit exceeded")
        await self._rate_limit(scope, now)
        count, size = await self.repo.capacity(scope.workspace_id, scope.user_id, now)
        if count >= self.max_files or size + reserved_bytes > self.max_outstanding_bytes:
            raise FilesError(429, "File capacity exceeded")
        row = ProviderFileBinding(
            id=request.operation_id,
            organization_id=scope.organization_id,
            workspace_id=scope.workspace_id,
            user_id=scope.user_id,
            provider_account_generation_id=account.generation_id,
            size_bytes=reserved_bytes,
            created_at=now,
            expires_at=now + timedelta(seconds=duration),
            operation_deadline=now + timedelta(seconds=self.operation_seconds),
            initiating_gateway_id=scope.gateway_id,
            cleanup_token_hash=secrets.token_hex(32),
        )
        self.db.add(row)
        await self.db.commit()
        return self._operation(row, account)

    async def finalize(
        self,
        scope: FileScope,
        binding_id: uuid.UUID,
        metadata: FileMetadata,
        expires_in_seconds: int | None = None,
    ) -> FileMetadata:
        now = datetime.now(UTC)
        await self.repo.lock_user(scope.user_id)
        await self.repo.lock_organization(scope.organization_id)
        row = await self.repo.get(binding_id)
        if row is None or not self._owns(row, scope) or row.initiating_gateway_id != scope.gateway_id:
            raise FilesError(404, "Upload operation unavailable")
        collision = await self.repo.by_provider_id(row.provider_account_generation_id, metadata.id)
        if collision is not None and collision.id != row.id:
            raise FilesError(409, "Provider file ownership conflict")
        if row.provider_file_id is not None:
            if row.provider_file_id != metadata.id or self._metadata(row) != metadata:
                raise FilesError(409, "Upload operation conflict")
            if row.state == "active" and row.expires_at > now:
                return self._metadata(row)
            raise FilesError(409, "Upload operation has been revoked")
        account = await self.repo.account(row.provider_account_generation_id)
        active = (
            row.state == "pending_upload"
            and row.operation_deadline > now
            and account is not None
            and account.status == "active"
            and metadata.size_bytes <= row.size_bytes
            and await self.repo.active_user(scope.user_id)
            and await self.repo.workspace_exists(scope.workspace_id, scope.organization_id)
        )
        row.provider_file_id = metadata.id
        row.encrypted_metadata = encrypt_secret(metadata.model_dump_json(exclude_unset=True))
        row.size_bytes = metadata.size_bytes
        row.downloadable = metadata.downloadable
        if expires_in_seconds is not None:
            if not 3600 <= expires_in_seconds <= 7776000:
                raise FilesError(400, "Invalid file retention")
            row.expires_at = min(row.expires_at, row.created_at + timedelta(seconds=expires_in_seconds))
        row.provider_expires_at = metadata.expires_at
        if metadata.expires_at is not None:
            row.expires_at = min(row.expires_at, metadata.expires_at)
        active = active and row.expires_at > now
        row.state = "active" if active else "pending_cleanup"
        if not active:
            row.cleanup_after, row.cleanup_reason = now, "revoked_operation"
        row.updated_at = now
        await self.db.commit()
        if not active:
            raise FilesError(409, "Upload operation has been revoked")
        return metadata

    async def abandon(self, binding_id: uuid.UUID, gateway_id: str, request: AbandonUpload) -> None:
        row = await self.repo.get(binding_id)
        if row is None:
            raise FilesError(404, "Upload operation unavailable")
        await self.repo.lock_organization(row.organization_id)
        await self.db.refresh(row)
        self._check_token(row, gateway_id, request.cleanup_token.get_secret_value())
        if row.state == "active" and (request.metadata is None or self._metadata(row) != request.metadata):
            raise FilesError(409, "Upload was already finalized")
        if request.metadata is not None:
            collision = await self.repo.by_provider_id(row.provider_account_generation_id, request.metadata.id)
            if collision is not None and collision.id != row.id:
                raise FilesError(409, "Provider file ownership conflict")
            if row.provider_file_id is not None and row.provider_file_id != request.metadata.id:
                raise FilesError(409, "Upload operation conflict")
            row.provider_file_id = request.metadata.id
            row.encrypted_metadata = encrypt_secret(request.metadata.model_dump_json(exclude_unset=True))
        row.provider_outcome_unknown = request.outcome_unknown
        if request.deleted or (row.provider_file_id is None and not request.outcome_unknown):
            row.state, row.deleted_at = "deleted", datetime.now(UTC)
        elif row.provider_file_id is not None:
            row.state, row.cleanup_after = "pending_cleanup", datetime.now(UTC)
            row.cleanup_reason = "upload_abandoned"
        await self.db.commit()

    async def list_files(self, scope: FileScope, request: FileListRequest) -> FilePage:
        now = datetime.now(UTC)
        await self._lock_scope(scope)
        await self._rate_limit(scope, now)
        snapshot, before = now, None
        limit = request.limit or 20
        scope_key = f"{scope.organization_id}:{scope.workspace_id}:{scope.user_id}"
        if request.page:
            try:
                cursor = json.loads(decrypt_secret(request.page))
                if cursor["scope"] != scope_key or cursor["limit"] != limit:
                    raise ValueError
                snapshot = datetime.fromisoformat(cursor["snapshot"])
                before = (datetime.fromisoformat(cursor["created_at"]), uuid.UUID(cursor["id"]))
            except (ValueError, KeyError, TypeError):
                raise FilesError(400, "Invalid file page") from None
        rows = await self.repo.visible(
            scope.organization_id,
            scope.workspace_id,
            scope.user_id,
            now,
            ids=request.ids,
            limit=101 if request.ids is not None else limit + 1,
            before=before,
            snapshot=snapshot,
        )
        next_page = None
        if request.ids is None and len(rows) > limit:
            rows = rows[:limit]
            last = rows[-1]
            next_page = encrypt_secret(
                json.dumps(
                    {
                        "scope": scope_key,
                        "limit": limit,
                        "snapshot": snapshot.isoformat(),
                        "created_at": last.created_at.isoformat(),
                        "id": str(last.id),
                    }
                )
            )
        result = FilePage(data=[self._metadata(row) for row in rows], next_page=next_page)
        await self.db.commit()
        return result

    async def resolve(
        self,
        scope: FileScope,
        file_id: str,
        operation: str,
        account: FileAccount | None = None,
    ) -> ResolvedFile:
        now = datetime.now(UTC)
        await self._lock_scope(scope)
        await self._rate_limit(scope, now)
        rows = await self.repo.visible(scope.organization_id, scope.workspace_id, scope.user_id, now, ids=[file_id])
        if len(rows) != 1:
            raise FilesError(404, "File unavailable")
        row = rows[0]
        if operation == "download" and not row.downloadable:
            raise FilesError(400, "This file is not downloadable")
        if operation != "metadata":
            if account is None or account.generation_id != row.provider_account_generation_id:
                raise FilesError(404, "Provider account unavailable")
            await self._account(scope, account)
        if operation == "delete":
            row.state, row.cleanup_reason, row.cleanup_after = "pending_cleanup", "delete", now
            row.initiating_gateway_id = scope.gateway_id
        result = ResolvedFile(
            metadata=self._metadata(row),
            account=account if operation != "metadata" else None,
            operation_id=row.id if operation == "delete" else None,
            cleanup_token=SecretStr(self._token(row)) if operation == "delete" else None,
        )
        await self.db.commit()
        return result

    async def references(self, scope: FileScope, ids: list[str]) -> uuid.UUID:
        await self._lock_scope(scope)
        if not ids or len(ids) > 100:
            raise FilesError(400, "Invalid file references")
        rows = await self.repo.visible(
            scope.organization_id, scope.workspace_id, scope.user_id, datetime.now(UTC), ids=list(set(ids))
        )
        if len(rows) != len(set(ids)):
            raise FilesError(404, "File unavailable")
        accounts = {row.provider_account_generation_id for row in rows}
        if len(accounts) != 1:
            raise FilesError(400, "Files must belong to one provider account")
        return accounts.pop()

    async def cleanup_result(self, binding_id: uuid.UUID, gateway_id: str, token: str, deleted: bool) -> None:
        row = await self.repo.get(binding_id)
        if row is None:
            raise FilesError(404, "Cleanup operation unavailable")
        await self.repo.lock_organization(row.organization_id)
        await self.db.refresh(row)
        self._check_token(row, gateway_id, token)
        if row.state == "deleted":
            return
        if row.state != "pending_cleanup":
            raise FilesError(409, "File is not awaiting cleanup")
        self.apply_cleanup(row, deleted)
        await self.db.commit()

    @staticmethod
    def apply_cleanup(row: ProviderFileBinding, deleted: bool) -> None:
        now = datetime.now(UTC)
        if deleted:
            row.state, row.deleted_at, row.cleanup_after = "deleted", now, None
        else:
            row.cleanup_attempts += 1
            row.cleanup_after = now + timedelta(seconds=min(21600, 60 * 2 ** min(row.cleanup_attempts - 1, 9)))
        row.updated_at = now
