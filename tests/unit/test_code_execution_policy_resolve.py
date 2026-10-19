"""Resolving a workspace's code execution policy from this deployment's own rows."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.adapters.code_execution_policy_adapter import LocalCodeExecutionPolicy
from gateway.exceptions.tools_exceptions import (
    CodeExecutionPolicyResolutionFailedError,
    CodeExecutionPolicyResolutionFailure,
)
from gateway.models.tools import ResolvedCodeExecutionPolicy
from gateway.ports.code_execution_policy_port import CodeExecutionPolicyScope


@pytest.mark.asyncio
async def test_a_local_scope_without_a_workspace_is_a_resolution_failure() -> None:
    with pytest.raises(CodeExecutionPolicyResolutionFailedError) as ei:
        await LocalCodeExecutionPolicy(MagicMock(), on_without_policy=True).resolve(
            CodeExecutionPolicyScope(workspace_id=None)
        )

    assert ei.value.reason is CodeExecutionPolicyResolutionFailure.NO_WORKSPACE


@pytest.mark.asyncio
async def test_a_local_scope_reads_the_workspace_row() -> None:
    workspace_id = uuid.uuid4()
    stored = ResolvedCodeExecutionPolicy(
        enabled=False, default_purpose_hint=None, max_iterations=None, exec_timeout_s=None, image=None, tools=None
    )
    policies = MagicMock()
    policies.resolve = AsyncMock(return_value=stored)

    policy = await LocalCodeExecutionPolicy(policies, on_without_policy=True).resolve(
        CodeExecutionPolicyScope(workspace_id=workspace_id)
    )

    assert policy is stored
    policies.resolve.assert_awaited_once_with(workspace_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("on_without_policy", [True, False])
async def test_no_row_follows_the_deployments_default(on_without_policy: bool) -> None:
    policies = MagicMock()
    policies.resolve = AsyncMock(return_value=None)

    policy = await LocalCodeExecutionPolicy(policies, on_without_policy=on_without_policy).resolve(
        CodeExecutionPolicyScope(workspace_id=uuid.uuid4())
    )

    if on_without_policy:
        assert policy is None
    else:
        assert policy is not None
        assert policy.enabled is False
