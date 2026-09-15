"""Conservative account selection and credential retirement."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.provider_files import ProviderAccountGeneration
from gateway.models.provider_keys import OrgProviderKey, WorkspaceProviderKeyOverride
from gateway.repositories.tenancy.org_provider_key_repository import WorkspaceProviderKeyOverrideRepository
from gateway.repositories.tenancy.provider_file_repository import ProviderFileRepository
from gateway.services.provider_files.contracts import FileAccount, FileScope, FilesError
from gateway.services.secret_box import decrypt_secret


def select_file_key(
    candidates: Sequence[tuple[OrgProviderKey, WorkspaceProviderKeyOverride | None]],
) -> OrgProviderKey | None:
    """Select explicit defaults or a unique live key; never choose the oldest key."""
    pinned = [(key, override) for key, override in candidates if override is not None and override.is_default]
    defaults = [(key, override) for key, override in candidates if key.is_org_default]
    selected = pinned or defaults
    if len(selected) > 1:
        raise FilesError(409, "Anthropic provider account is ambiguous")
    if selected:
        key, override = selected[0]
        if key.archived_at is not None or (override is not None and override.disabled):
            raise FilesError(404, "Anthropic provider account unavailable")
        return key
    live = [
        key
        for key, override in candidates
        if key.archived_at is None and not (override is not None and override.disabled)
    ]
    if len(live) > 1:
        raise FilesError(409, "Anthropic provider account is ambiguous")
    return live[0] if live else None


class FileAccountResolver:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = ProviderFileRepository(db)

    async def select_byo(self, scope: FileScope) -> FileAccount | None:
        await self.repo.lock_user(scope.user_id)
        if not await self.repo.lock_organization(scope.organization_id):
            raise FilesError(404, "Anthropic provider account unavailable")
        candidates = await WorkspaceProviderKeyOverrideRepository(self.db).all_candidates(
            organization_id=scope.organization_id, workspace_id=scope.workspace_id
        )
        key = select_file_key([(key, override) for key, override in candidates if key.provider == "anthropic"])
        if key is None:
            return None
        generation = await self.repo.latest_account("organization_key", str(key.id), scope.organization_id)
        if generation is None or generation.status == "retired":
            number = generation.generation + 1 if generation is not None else 1
            generation = ProviderAccountGeneration(
                generation=number,
                credential_source="organization_key",
                credential_ref=str(key.id),
                organization_id=scope.organization_id,
            )
            self.db.add(generation)
            await self.db.flush()
        elif generation.status != "active":
            raise FilesError(409, "Anthropic provider account is retiring")
        return self._credential(key, generation)

    @staticmethod
    def _credential(key: OrgProviderKey, generation: ProviderAccountGeneration) -> FileAccount:
        if not key.encrypted_api_key:
            raise FilesError(404, "Anthropic provider account unavailable")
        # Client args may change account selection or transport. The initial contract accepts none.
        if key.client_args:
            raise FilesError(400, "Provider file account requires a standard Anthropic credential")
        return FileAccount(
            generation_id=generation.id, api_key=SecretStr(decrypt_secret(key.encrypted_api_key)), api_base=key.api_base
        )

    async def resolve_byo(
        self, generation_id: uuid.UUID, organization_id: uuid.UUID, *, cleanup: bool = False
    ) -> FileAccount:
        row = await self.repo.account(generation_id)
        if (
            row is None
            or row.organization_id != organization_id
            or row.credential_source != "organization_key"
            or (not cleanup and row.status != "active")
        ):
            raise FilesError(404, "Anthropic provider account unavailable")
        key = await self.db.get(OrgProviderKey, uuid.UUID(row.credential_ref))
        if key is None or key.organization_id != organization_id or (not cleanup and key.archived_at is not None):
            raise FilesError(404, "Anthropic provider account unavailable")
        return self._credential(key, row)


async def retire_byo_account(db: AsyncSession, key: OrgProviderKey, *, release_secret: bool) -> None:
    """Revoke first, retaining the old secret until all known work settles."""
    repo = ProviderFileRepository(db)
    await repo.lock_organization(key.organization_id)
    await db.refresh(key)
    row = await repo.latest_account("organization_key", str(key.id), key.organization_id)
    if row is None:
        return
    await retire_account_generation(db, row, release_secret=release_secret)


async def retire_account_generation(
    db: AsyncSession,
    row: ProviderAccountGeneration,
    *,
    release_secret: bool,
) -> None:
    """Apply retirement for BYO or an adapter-owned source in the credential transaction."""
    repo = ProviderFileRepository(db)
    await repo.lock_organization(row.organization_id)
    await db.refresh(row)
    now = datetime.now(UTC)
    row.status = "retiring"
    await repo.revoke(now, "credential_retirement", organization_id=row.organization_id, generation_id=row.id)
    if release_secret and await repo.account_busy(row.id, now):
        # Retirement must persist even though replacement is refused.
        await db.commit()
        from gateway.services.tenancy.errors import TenancyConflictError

        raise TenancyConflictError("Provider file cleanup must finish before replacing or deleting this credential")
    if release_secret:
        row.status, row.retired_at = "retired", now
