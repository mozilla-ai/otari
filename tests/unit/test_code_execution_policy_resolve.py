"""Resolving the code execution policy a workspace's requests run under."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.models.tools import ResolvedCodeExecutionPolicy
from gateway.services.tools import WorkspaceCodeExecutionPolicies


def _policies(stored: ResolvedCodeExecutionPolicy | None, *, on_without_policy: bool) -> WorkspaceCodeExecutionPolicies:
    policies = WorkspaceCodeExecutionPolicies(MagicMock(), on_without_policy=on_without_policy)
    policies.resolve = AsyncMock(return_value=stored)  # type: ignore[method-assign]
    return policies


@pytest.mark.asyncio
async def test_a_request_reads_the_workspace_row() -> None:
    workspace_id = uuid.uuid4()
    stored = ResolvedCodeExecutionPolicy(
        enabled=False, default_purpose_hint=None, max_iterations=None, exec_timeout_s=None, image=None, tools=None
    )
    policies = _policies(stored, on_without_policy=True)

    policy = await policies.for_request(workspace_id)

    assert policy is stored
    policies.resolve.assert_awaited_once_with(workspace_id)  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("on_without_policy", [True, False])
async def test_no_row_follows_the_deployments_default(on_without_policy: bool) -> None:
    policy = await _policies(None, on_without_policy=on_without_policy).for_request(uuid.uuid4())

    if on_without_policy:
        assert policy is None
    else:
        assert policy is not None
        assert policy.enabled is False
