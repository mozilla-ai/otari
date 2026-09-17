"""The interface a domain implements to react to workspace membership changes.

The domain that changes membership calls it and never imports the domain that implements it.
"""

import uuid
from collections.abc import Sequence
from typing import Protocol

from gateway.models.tenancy import WorkspaceMember


class MembershipListener(Protocol):
    """Reacts to a membership change inside the transaction that made it.

    NOTE: an implementation must not commit or roll back.
    The caller owns the transaction, so a refused change takes the listener's writes back with it.
    """

    async def member_joined(self, member: WorkspaceMember) -> None:
        """A member joined a workspace, or an inactive membership became active."""

    async def member_removed(self, member: WorkspaceMember) -> None:
        """A membership is about to be deleted."""

    async def workspace_deleted(self, workspace_id: uuid.UUID, member_ids: Sequence[uuid.UUID]) -> None:
        """A workspace and these memberships are about to be deleted."""


__all__ = ["MembershipListener"]
