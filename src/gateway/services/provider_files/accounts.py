"""Conservative account selection and credential retirement."""

import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime

from pydantic import SecretStr

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.provider_files import ProviderAccountGeneration
from gateway.models.provider_keys import OrgProviderKey, WorkspaceProviderKeyOverride
from gateway.repositories.tenancy.provider_file_repository import ProviderFileRepository
from gateway.services.provider_files.capabilities import check_file_account
from gateway.services.provider_files.contracts import FileAccount, FileScope, FilesError, OutputPrepare
from gateway.services.secret_box import decrypt_secret


def select_file_key(
    candidates: Sequence[tuple[OrgProviderKey, WorkspaceProviderKeyOverride | None]],
) -> OrgProviderKey | None:
    """Select explicit defaults or a unique live key; never choose the oldest key."""
    pinned = [(key, override) for key, override in candidates if override is not None and override.is_default]
    defaults = [(key, override) for key, override in candidates if key.is_org_default]
    selected = pinned or defaults
    if len(selected) > 1:
        raise FilesError(409, "Provider account is ambiguous")
    if selected:
        key, override = selected[0]
        if key.archived_at is not None or (override is not None and override.disabled):
            raise FilesError(404, "Provider account unavailable")
        return key
    live = [
        key
        for key, override in candidates
        if key.archived_at is None and not (override is not None and override.disabled)
    ]
    if len(live) > 1:
        raise FilesError(409, "Provider account is ambiguous")
    return live[0] if live else None


class FileAccountResolver:
    def __init__(self, uow: UnitOfWork) -> None:
        self.uow = uow
        self.repo = ProviderFileRepository(uow)

    async def authenticate(self, authenticate: Callable[[], Awaitable[FileScope]]) -> FileScope:
        async with self.uow:
            return await authenticate()

    async def authorize_attempt(
        self, scope: FileScope, body: OutputPrepare, authorize: Callable[[], Awaitable[FileAccount]]
    ) -> FileAccount:
        async with self.uow:
            generation = await self.repo.account(body.generation_id)
            if generation is None or generation.organization_id != scope.organization_id:
                raise FilesError(404, "Provider account unavailable")
            if generation.credential_source == "hosted_backend" and not scope.default_gateway:
                raise FilesError(403, "Managed provider files require the default gateway")
            return await authorize()

    async def resolve(
        self,
        scope: FileScope,
        generation_id: uuid.UUID | None = None,
        *,
        provider: str | None = None,
        cleanup: bool = False,
        resolve_hosted: Callable[[FileScope, str, uuid.UUID | None, bool, UnitOfWork], Awaitable[FileAccount]]
        | None = None,
    ) -> FileAccount:
        async with self.uow:
            if not cleanup:
                await self.repo.lock_user(scope.user_id)
                if not await self.repo.lock_organization(scope.organization_id):
                    raise FilesError(404, "Provider account unavailable")
                if not await self.repo.active_user(scope.user_id) or not await self.repo.workspace_exists(
                    scope.workspace_id, scope.organization_id
                ):
                    raise FilesError(404, "Provider account unavailable")
            if generation_id is None:
                provider = provider or "anthropic"
                selected = await self.select_byo(scope, provider=provider)
                if selected is not None:
                    return selected
            else:
                row = await self.repo.account(generation_id)
                if (
                    row is None
                    or row.organization_id != scope.organization_id
                    or (provider is not None and row.provider != provider)
                ):
                    raise FilesError(404, "Provider account unavailable")
                provider = row.provider
                if row.credential_source == "organization_key":
                    return await self.resolve_byo(generation_id, scope.organization_id, cleanup=cleanup)
            if resolve_hosted is None:
                raise FilesError(404, "Provider account unavailable")
            if not scope.default_gateway:
                raise FilesError(403, "Managed provider files require the default gateway")
            selected = await resolve_hosted(scope, provider, generation_id, cleanup, self.uow)
            check_file_account(selected, provider)
            if generation_id is not None and selected.generation_id != generation_id:
                raise FilesError(502, "Authorization service returned an invalid file account")
            return selected

    async def select_byo(self, scope: FileScope, *, provider: str = "anthropic") -> FileAccount | None:
        async with self.uow:
            return await self._select_byo(scope, provider=provider)

    async def _select_byo(self, scope: FileScope, *, provider: str) -> FileAccount | None:
        await self.repo.lock_user(scope.user_id)
        if not await self.repo.lock_organization(scope.organization_id):
            raise FilesError(404, "Provider account unavailable")
        candidates = await self.repo.key_candidates(scope.organization_id, scope.workspace_id)
        key = select_file_key([(key, override) for key, override in candidates if key.provider == provider])
        if key is None:
            return None
        generation = await self.repo.latest_account("organization_key", str(key.id), scope.organization_id)
        if generation is None or generation.status == "retired":
            number = generation.generation + 1 if generation is not None else 1
            generation = ProviderAccountGeneration(
                provider=provider,
                generation=number,
                credential_source="organization_key",
                credential_ref=str(key.id),
                organization_id=scope.organization_id,
            )
            await self.repo.save(generation)
        elif generation.status != "active":
            raise FilesError(409, "Provider account is retiring")
        return self._credential(key, generation)

    @staticmethod
    def _credential(key: OrgProviderKey, generation: ProviderAccountGeneration) -> FileAccount:
        if not key.encrypted_api_key or key.provider != generation.provider:
            raise FilesError(404, "Provider account unavailable")
        # Client args may change account selection or transport. The initial contract accepts none.
        if key.client_args:
            raise FilesError(400, "Provider file account requires a standard credential")
        return FileAccount(
            generation_id=generation.id,
            provider=generation.provider,
            api_key=SecretStr(decrypt_secret(key.encrypted_api_key)),
            api_base=key.api_base,
        )

    async def resolve_byo(
        self, generation_id: uuid.UUID, organization_id: uuid.UUID, *, cleanup: bool = False
    ) -> FileAccount:
        async with self.uow:
            return await self._resolve_byo(generation_id, organization_id, cleanup=cleanup)

    async def _resolve_byo(self, generation_id: uuid.UUID, organization_id: uuid.UUID, *, cleanup: bool) -> FileAccount:
        row = await self.repo.account(generation_id)
        if (
            row is None
            or row.organization_id != organization_id
            or row.credential_source != "organization_key"
            or (not cleanup and row.status != "active")
        ):
            raise FilesError(404, "Provider account unavailable")
        key = await self.repo.provider_key(uuid.UUID(row.credential_ref))
        if key is None or key.organization_id != organization_id or (not cleanup and key.archived_at is not None):
            raise FilesError(404, "Provider account unavailable")
        return self._credential(key, row)


async def retire_byo_account(uow: UnitOfWork, key: OrgProviderKey, *, release_secret: bool) -> bool:
    """Revoke in the caller's open transaction; return whether cleanup blocks secret release."""
    repo = ProviderFileRepository(uow)
    await repo.lock_organization(key.organization_id)
    await repo.refresh(key)
    row = await repo.latest_account("organization_key", str(key.id), key.organization_id)
    if row is None:
        return False
    return await retire_account_generation(uow, row, release_secret=release_secret)


async def retire_account_generation(
    uow: UnitOfWork,
    row: ProviderAccountGeneration,
    *,
    release_secret: bool,
) -> bool:
    """Return a blocked retirement, so the caller commits revocation before reporting refusal.

    The caller's block must include any secret replacement or deletion. When
    blocked, leave the secret intact and raise only after that block commits.
    """
    repo = ProviderFileRepository(uow)
    await repo.lock_organization(row.organization_id)
    await repo.refresh(row)
    now = datetime.now(UTC)
    row.status = "retiring"
    await repo.revoke(now, "credential_retirement", organization_id=row.organization_id, generation_id=row.id)
    if release_secret and await repo.account_busy(row.id, now):
        return True
    if release_secret:
        row.status, row.retired_at = "retired", now
    return False
