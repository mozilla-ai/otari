"""The two places a workspace's code execution policy comes from.

``LocalCodeExecutionPolicy`` reads the row this deployment holds.
``RemoteCodeExecutionPolicy`` asks a peer, speaking `docs/hybrid-mode-protocol.md`.
"""

from __future__ import annotations

from gateway.core.config import GatewayConfig
from gateway.exceptions.tools_exceptions import (
    CodeExecutionPolicyResolutionFailedError,
    CodeExecutionPolicyResolutionFailure,
)
from gateway.models.tools import ResolvedCodeExecutionPolicy
from gateway.ports.code_execution_policy_port import CodeExecutionPolicyPort, CodeExecutionPolicyScope
from gateway.services.control_plane import ResolveEndpoint, resolve
from gateway.services.tools import WorkspaceCodeExecutionPolicies, read_code_execution_policy

# The policy of a workspace that holds no row where this deployment makes code execution opt-in.
_NOT_TURNED_ON = ResolvedCodeExecutionPolicy(
    enabled=False,
    default_purpose_hint=None,
    max_iterations=None,
    exec_timeout_s=None,
    image=None,
    tools=None,
    executor=None,
)


class LocalCodeExecutionPolicy(CodeExecutionPolicyPort):
    """The policy stored in this deployment's own database."""

    def __init__(self, policies: WorkspaceCodeExecutionPolicies, *, on_without_policy: bool) -> None:
        self._policies = policies
        self._on_without_policy = on_without_policy

    async def resolve(self, scope: CodeExecutionPolicyScope) -> ResolvedCodeExecutionPolicy | None:
        # NOTE: a missing workspace fails closed, because the policy is a veto and no answer must not read as no limit.
        if scope.workspace_id is None:
            raise CodeExecutionPolicyResolutionFailedError(CodeExecutionPolicyResolutionFailure.NO_WORKSPACE)
        policy = await self._policies.resolve(scope.workspace_id)
        if policy is None and not self._on_without_policy:
            return _NOT_TURNED_ON
        return policy


class RemoteCodeExecutionPolicy(CodeExecutionPolicyPort):
    """The policy the control plane holds for this deployment's workspaces."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    async def resolve(self, scope: CodeExecutionPolicyScope) -> ResolvedCodeExecutionPolicy | None:
        if not scope.user_token:
            raise CodeExecutionPolicyResolutionFailedError(CodeExecutionPolicyResolutionFailure.NO_CALLER_CREDENTIAL)
        answer = await resolve(
            self._config,
            user_token=scope.user_token,
            endpoint=ResolveEndpoint.CODE_EXECUTION,
            body={},
        )
        if not isinstance(answer, dict):
            raise CodeExecutionPolicyResolutionFailedError(CodeExecutionPolicyResolutionFailure.ANSWER_UNREADABLE)
        try:
            return read_code_execution_policy(answer)
        except ValueError:
            raise CodeExecutionPolicyResolutionFailedError(
                CodeExecutionPolicyResolutionFailure.ANSWER_UNREADABLE
            ) from None
