"""The interface a domain implements to set up a workspace the moment it is created.

The domain that creates workspaces calls it and never imports the domain that implements it.
"""

import uuid
from typing import Protocol


class WorkspaceListener(Protocol):
    """Reacts to a new workspace inside the transaction that created it.

    NOTE: an implementation must not commit or roll back.
    The caller owns the transaction, so a refused creation takes the listener's writes back with it.
    """

    async def workspace_created(self, workspace_id: uuid.UUID) -> None:
        """A workspace was just created."""


class NullWorkspaceListener:
    """Sets up nothing, for a deployment where a new workspace needs nothing staged."""

    async def workspace_created(self, workspace_id: uuid.UUID) -> None:
        """Do nothing."""


__all__ = ["NullWorkspaceListener", "WorkspaceListener"]
