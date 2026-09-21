"""Transaction-local reactions to credential and tenant revocation."""

import uuid
from typing import Protocol

from gateway.models.provider_keys import OrgProviderKey


class RevocationListener(Protocol):
    """The caller owns locks and commit; implementations never commit or roll back."""

    async def retire_key(self, key: OrgProviderKey, *, release_secret: bool) -> bool:
        """Revoke key-backed resources; return whether cleanup blocks secret release."""

    async def workspace_key_disabled(
        self, organization_id: uuid.UUID, workspace_id: uuid.UUID, key_id: uuid.UUID
    ) -> None:
        """Revoke resources using this key in the workspace."""

    async def workspace_deleted(self, organization_id: uuid.UUID, workspace_id: uuid.UUID) -> None:
        """Revoke resources before workspace deletion."""

    async def user_deleted(self, user_id: str) -> None:
        """Revoke resources while the caller holds the attribution user's lock."""
