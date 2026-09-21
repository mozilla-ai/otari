"""Provider-file reactions within the tenant mutation's transaction."""

import uuid
from datetime import UTC, datetime

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.provider_keys import OrgProviderKey
from gateway.repositories.tenancy.provider_file_repository import ProviderFileRepository
from gateway.services.provider_files.accounts import retire_byo_account


class ProviderFileRevocations:
    def __init__(self, repo: ProviderFileRepository, uow: UnitOfWork) -> None:
        self.repo = repo
        self.uow = uow

    async def retire_key(self, key: OrgProviderKey, *, release_secret: bool) -> bool:
        return await retire_byo_account(self.uow, key, release_secret=release_secret)

    async def workspace_key_disabled(
        self, organization_id: uuid.UUID, workspace_id: uuid.UUID, key_id: uuid.UUID
    ) -> None:
        generation = await self.repo.latest_account("organization_key", str(key_id), organization_id)
        if generation is not None:
            await self.repo.revoke(
                datetime.now(UTC),
                "workspace_credential_disabled",
                organization_id=organization_id,
                workspace_id=workspace_id,
                generation_id=generation.id,
            )

    async def workspace_deleted(self, organization_id: uuid.UUID, workspace_id: uuid.UUID) -> None:
        await self.repo.revoke(
            datetime.now(UTC), "workspace_deletion", organization_id=organization_id, workspace_id=workspace_id
        )

    async def user_deleted(self, user_id: str) -> None:
        await self.repo.revoke_user(user_id, datetime.now(UTC))
