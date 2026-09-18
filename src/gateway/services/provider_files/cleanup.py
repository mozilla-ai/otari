"""Durable, fenced cleanup leases for stateless gateway executors."""

import hashlib
import hmac
import secrets
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr

from gateway.services.provider_files.accounts import FileAccountResolver
from gateway.services.provider_files.contracts import CleanupItem, CleanupLease, FileAccount, FilesError, LeaseResult
from gateway.services.provider_files.lifecycle import ProviderFileService


class ProviderFileCleanup:
    def __init__(self, service: ProviderFileService) -> None:
        self.service = service
        self.uow = service.uow
        self.repo = service.repo

    async def claim(
        self,
        organization_id: uuid.UUID,
        gateway_id: str,
        limit: int = 20,
        *,
        include_managed: bool = False,
        resolve_account: Callable[[uuid.UUID], Awaitable[FileAccount]] | None = None,
    ) -> CleanupLease | None:
        """Organization scope must come from the registered gateway, never the request body."""
        async with self.uow:
            now = datetime.now(UTC)
            await self.repo.lock_organization(organization_id)
            await self.repo.expire_bindings(organization_id, now, self.service.diagnostic_seconds)
            account = await self.repo.cleanup_account(
                organization_id, now, include_managed=include_managed and resolve_account is not None
            )
            if account is None:
                return None
            credential = (
                await resolve_account(account.id)
                if resolve_account is not None
                else await FileAccountResolver(self.uow).resolve_byo(account.id, organization_id, cleanup=True)
            )
            rows = await self.repo.cleanup_bindings(account.id, now, limit)
            token = secrets.token_urlsafe(32)
            account.lease_id = uuid.uuid4()
            account.lease_token_hash = hashlib.sha256(token.encode()).hexdigest()
            account.lease_gateway_id = gateway_id
            account.lease_deadline = now + timedelta(seconds=300)
            for row in rows:
                row.lease_id = account.lease_id
            return CleanupLease(
                id=account.lease_id,
                token=SecretStr(token),
                deadline=account.lease_deadline,
                account=credential,
                items=[
                    CleanupItem(binding_id=row.id, file_id=row.provider_file_id)
                    for row in rows
                    if row.provider_file_id is not None
                ],
            )

    async def complete(
        self, organization_id: uuid.UUID, gateway_id: str, lease_id: uuid.UUID, result: LeaseResult
    ) -> None:
        async with self.uow:
            await self.repo.lock_organization(organization_id)
            account = await self.repo.leased_account(organization_id, lease_id)
            digest = hashlib.sha256(result.token.get_secret_value().encode()).hexdigest()
            if (
                account is None
                or account.lease_gateway_id != gateway_id
                or account.lease_deadline is None
                or account.lease_deadline <= datetime.now(UTC)
                or not hmac.compare_digest(account.lease_token_hash or "", digest)
            ):
                raise FilesError(409, "Cleanup lease unavailable")
            rows = await self.repo.leased_bindings(account.id, lease_id)
            if set(result.results) - {row.id for row in rows}:
                raise FilesError(409, "Cleanup lease item conflict")
            for row in rows:
                if row.state == "pending_cleanup":
                    self.service.apply_cleanup(row, result.results.get(row.id, False))
                row.lease_id = None
            account.lease_id = account.lease_token_hash = account.lease_gateway_id = account.lease_deadline = None
