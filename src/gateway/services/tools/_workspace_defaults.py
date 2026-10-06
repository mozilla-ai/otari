"""What a new workspace starts with for code execution."""

import uuid

from gateway.repositories.tools import WorkspaceCodeExecutionPolicyRepository


class CodeExecutionWorkspaceDefaults:
    """Starts each new workspace with code execution on, where the deployment reads no policy as off.

    A hosted control plane does, so there a workspace nobody configured could
    never run code. It holds a repository and nothing that commits, so a refused
    creation takes its write back with it.
    """

    def __init__(self, policies: WorkspaceCodeExecutionPolicyRepository, *, on_by_default: bool) -> None:
        self._policies = policies
        self._on_by_default = on_by_default

    async def workspace_created(self, workspace_id: uuid.UUID) -> None:
        """Stage an enabled policy for the new workspace, where this deployment needs one."""
        if self._on_by_default:
            await self._policies.create_enabled(workspace_id)
