"""The request path's read of a workspace's code execution policy.

The policy itself, its management service and its repository belong to the tools
domain (``services/tools``, ``repositories/tools``). This read stays here, at the
path the pipeline, the local policy adapter, the Playground and otari-ai import,
because it takes the request's session and a tools service may not hold one.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tools import CodeExecutor, ResolvedCodeExecutionPolicy
from gateway.repositories.tools import WorkspaceCodeExecutionPolicyRepository


async def resolve_workspace_code_execution_policy(
    db: AsyncSession,
    workspace_id: uuid.UUID,
) -> ResolvedCodeExecutionPolicy | None:
    """The workspace's stored policy, or ``None`` when it has none.

    ``None`` and "a row that narrows nothing" are deliberately the same outcome
    for the caller; the distinction only matters to the management surface,
    which reports it as ``configured``.
    """
    policy = await WorkspaceCodeExecutionPolicyRepository(db).get(workspace_id)
    if policy is None:
        return None
    return ResolvedCodeExecutionPolicy(
        enabled=policy.enabled,
        default_purpose_hint=policy.default_purpose_hint,
        max_iterations=policy.max_iterations,
        exec_timeout_s=policy.exec_timeout_s,
        image=policy.image,
        tools=frozenset(policy.tools) if policy.tools is not None else None,
        # NOTE: a stored executor outside the vocabulary reads as no pin, so an old row does not fail every request.
        executor=CodeExecutor.parse(policy.executor),
    )


__all__ = ["resolve_workspace_code_execution_policy"]
