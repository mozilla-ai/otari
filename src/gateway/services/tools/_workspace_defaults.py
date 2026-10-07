"""What a new workspace starts with for code execution."""

import uuid

from gateway.repositories.tools import WorkspaceCodeExecutionPolicyRepository


class CodeExecutionWorkspaceDefaults:
    """Starts each new workspace with code execution on.

    Bound where the deployment reads no policy as off, which a hosted control
    plane does, so there a workspace nobody configured could never run code. It
    holds a repository and nothing that commits, so a refused creation takes its
    write back with it.
    """

    def __init__(self, policies: WorkspaceCodeExecutionPolicyRepository) -> None:
        self._policies = policies

    async def workspace_created(self, workspace_id: uuid.UUID) -> None:
        """Stage an enabled policy for the new workspace."""
        await self._policies.create_enabled(workspace_id)
