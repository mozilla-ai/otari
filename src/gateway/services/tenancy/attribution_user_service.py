"""Revoke an attribution user's keys and dependent resources atomically."""

import uuid
from datetime import UTC, datetime

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.tenancy.attribution_user_repository import AttributionUserRepository
from gateway.services.tenancy.errors import TenancyNotFoundError
from gateway.services.tenancy.revocation_listener import RevocationListener


class AttributionUserService:
    def __init__(
        self, repo: AttributionUserRepository, uow: UnitOfWork, revocation_listener: RevocationListener
    ) -> None:
        self.repo = repo
        self.uow = uow
        self.revocation_listener = revocation_listener

    async def delete(self, user_id: str, organization_id: uuid.UUID | None) -> None:
        async with self.uow:
            user = await self.repo.lock_active(user_id, organization_id)
            if user is None:
                raise TenancyNotFoundError("User not found")
            await self.revocation_listener.user_deleted(user_id)
            await self.repo.soft_delete(user, datetime.now(UTC))
