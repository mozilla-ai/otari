"""Durable, fenced cleanup leases for stateless gateway executors."""

import hashlib
import hmac
import secrets
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr
from sqlalchemy import or_, select, update
from sqlmodel import col

from gateway.models.provider_files import ProviderAccountGeneration, ProviderFileBinding
from gateway.services.provider_files.accounts import FileAccountResolver
from gateway.services.provider_files.contracts import CleanupItem, CleanupLease, FileAccount, FilesError, LeaseResult
from gateway.services.provider_files.lifecycle import ProviderFileService


class ProviderFileCleanup:
    def __init__(self, service: ProviderFileService) -> None:
        self.service = service
        self.db = service.db
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
        now = datetime.now(UTC)
        await self.repo.lock_organization(organization_id)
        await self.db.execute(
            update(ProviderFileBinding)
            .where(
                col(ProviderFileBinding.organization_id) == organization_id,
                col(ProviderFileBinding.state) == "active",
                col(ProviderFileBinding.expires_at) <= now,
            )
            .values(state="pending_cleanup", cleanup_reason="expiry", cleanup_after=now)
        )
        await self.db.execute(
            update(ProviderFileBinding)
            .where(
                col(ProviderFileBinding.organization_id) == organization_id,
                col(ProviderFileBinding.provider_file_id).is_(None),
                col(ProviderFileBinding.operation_deadline) < now - timedelta(seconds=self.service.diagnostic_seconds),
                col(ProviderFileBinding.state).in_(["pending_upload", "pending_cleanup"]),
            )
            .values(state="deleted", deleted_at=now, provider_outcome_unknown=False)
        )
        due = select(col(ProviderFileBinding.provider_account_generation_id)).where(
            col(ProviderFileBinding.state) == "pending_cleanup",
            col(ProviderFileBinding.provider_file_id).is_not(None),
            col(ProviderFileBinding.cleanup_after) <= now,
        )
        account = (
            await self.db.execute(
                select(ProviderAccountGeneration)
                .where(
                    col(ProviderAccountGeneration.organization_id) == organization_id,
                    col(ProviderAccountGeneration.credential_source).in_(
                        ["organization_key", "hosted_backend"]
                        if include_managed and resolve_account is not None
                        else ["organization_key"]
                    ),
                    col(ProviderAccountGeneration.id).in_(due),
                    or_(
                        col(ProviderAccountGeneration.lease_deadline).is_(None),
                        col(ProviderAccountGeneration.lease_deadline) <= now,
                    ),
                )
                .order_by(col(ProviderAccountGeneration.id))
                .limit(1)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if account is None:
            await self.db.commit()
            return None
        credential = (
            await resolve_account(account.id)
            if resolve_account is not None
            else await FileAccountResolver(self.db).resolve_byo(account.id, organization_id, cleanup=True)
        )
        rows = list(
            (
                await self.db.execute(
                    select(ProviderFileBinding)
                    .where(
                        col(ProviderFileBinding.provider_account_generation_id) == account.id,
                        col(ProviderFileBinding.state) == "pending_cleanup",
                        col(ProviderFileBinding.provider_file_id).is_not(None),
                        col(ProviderFileBinding.cleanup_after) <= now,
                    )
                    .order_by(col(ProviderFileBinding.cleanup_after), col(ProviderFileBinding.id))
                    .limit(min(20, max(1, limit)))
                )
            ).scalars()
        )
        token = secrets.token_urlsafe(32)
        account.lease_id = uuid.uuid4()
        account.lease_token_hash = hashlib.sha256(token.encode()).hexdigest()
        account.lease_gateway_id = gateway_id
        account.lease_deadline = now + timedelta(seconds=300)
        for row in rows:
            row.lease_id = account.lease_id
        result = CleanupLease(
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
        await self.db.commit()
        return result

    async def complete(
        self, organization_id: uuid.UUID, gateway_id: str, lease_id: uuid.UUID, result: LeaseResult
    ) -> None:
        await self.repo.lock_organization(organization_id)
        account = (
            await self.db.execute(
                select(ProviderAccountGeneration)
                .where(
                    col(ProviderAccountGeneration.organization_id) == organization_id,
                    col(ProviderAccountGeneration.lease_id) == lease_id,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        digest = hashlib.sha256(result.token.get_secret_value().encode()).hexdigest()
        if (
            account is None
            or account.lease_gateway_id != gateway_id
            or account.lease_deadline is None
            or account.lease_deadline <= datetime.now(UTC)
            or not hmac.compare_digest(account.lease_token_hash or "", digest)
        ):
            raise FilesError(409, "Cleanup lease unavailable")
        rows = list(
            (
                await self.db.execute(
                    select(ProviderFileBinding).where(
                        col(ProviderFileBinding.provider_account_generation_id) == account.id,
                        col(ProviderFileBinding.lease_id) == lease_id,
                    )
                )
            ).scalars()
        )
        if set(result.results) - {row.id for row in rows}:
            raise FilesError(409, "Cleanup lease item conflict")
        for row in rows:
            if row.state == "pending_cleanup":
                self.service.apply_cleanup(row, result.results.get(row.id, False))
            row.lease_id = None
        account.lease_id = account.lease_token_hash = account.lease_gateway_id = account.lease_deadline = None
        await self.db.commit()
