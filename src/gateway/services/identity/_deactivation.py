"""Participants in the transaction that deactivates an identity."""

import uuid
from typing import Protocol


class SelfServiceKeyDeactivationListener(Protocol):
    """Switch off self-service credentials inside the caller's transaction, without committing."""

    async def deactivate_self_service_keys(self, user_id: uuid.UUID) -> None: ...
