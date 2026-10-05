"""Where a workspace's code execution policy comes from.

One deployment holds a workspace's policy itself and another asks a peer that holds it.
Both answer the same question, so a caller reads the policy without knowing which answered.

The policy says who may run code and within which limits, and nothing here runs code.
"""

import uuid
from dataclasses import dataclass
from typing import Protocol

from gateway.exceptions.tools_exceptions import CodeExecutionPolicyResolutionFailedError
from gateway.models.tools import ResolvedCodeExecutionPolicy


@dataclass(frozen=True)
class CodeExecutionPolicyScope:
    """Whose policy to resolve.

    A deployment that holds the policy reads the workspace ID.
    A deployment that asks a peer reads the caller's token.
    Neither field defaults, so a caller cannot drop the one its deployment reads.
    """

    workspace_id: uuid.UUID | None
    user_token: str | None


class CodeExecutionPolicyPort(Protocol):
    """A workspace's code execution policy."""

    async def resolve(self, scope: CodeExecutionPolicyScope) -> ResolvedCodeExecutionPolicy | None:
        """The policy for this scope, or ``None`` where the workspace has none.

        NOTE: only an implementation that reads stored rows may return ``None``.
        One that asks a control plane must return a policy or raise, because the protocol requires ``enabled``.

        Raises:
            CodeExecutionPolicyResolutionFailedError: the policy could not be resolved, and ``reason`` says why.
            ControlPlaneError: a peer that holds the policy refused or could not answer.
        """
        ...


__all__ = [
    "CodeExecutionPolicyPort",
    "CodeExecutionPolicyResolutionFailedError",
    "CodeExecutionPolicyScope",
]
