"""Where a workspace's code execution policy comes from.

The policy says who may run code and within which limits, and nothing here runs code.
"""

import uuid
from dataclasses import dataclass
from typing import Protocol

from gateway.exceptions.tools_exceptions import CodeExecutionPolicyResolutionFailedError
from gateway.models.tools import ResolvedCodeExecutionPolicy


@dataclass(frozen=True)
class CodeExecutionPolicyScope:
    """Whose policy to resolve."""

    workspace_id: uuid.UUID | None


class CodeExecutionPolicyPort(Protocol):
    """A workspace's code execution policy."""

    async def resolve(self, scope: CodeExecutionPolicyScope) -> ResolvedCodeExecutionPolicy | None:
        """The policy for this scope, or ``None`` where the workspace has none.

        Raises:
            CodeExecutionPolicyResolutionFailedError: the policy could not be resolved, and ``reason`` says why.
        """
        ...


__all__ = [
    "CodeExecutionPolicyPort",
    "CodeExecutionPolicyResolutionFailedError",
    "CodeExecutionPolicyScope",
]
