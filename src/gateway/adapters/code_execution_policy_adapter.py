"""The two places a workspace's code execution policy comes from.

``LocalCodeExecutionPolicy`` reads the row this deployment holds.
``RemoteCodeExecutionPolicy`` asks a peer, speaking `docs/hybrid-mode-protocol.md`.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.deployment import Plane, deployment_for
from gateway.exceptions.tools_exceptions import (
    CodeExecutionPolicyResolutionFailedError,
    CodeExecutionPolicyResolutionFailure,
)
from gateway.models.tools import ResolvedCodeExecutionPolicy
from gateway.ports.code_execution_policy_port import CodeExecutionPolicyPort, CodeExecutionPolicyScope
from gateway.services.control_plane import ResolveEndpoint, resolve
from gateway.services.tenancy.workspace_code_execution_policy_service import (
    read_code_execution_policy,
    resolve_workspace_code_execution_policy,
)


class LocalCodeExecutionPolicy(CodeExecutionPolicyPort):
    """The policy stored in this deployment's own database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve(self, scope: CodeExecutionPolicyScope) -> ResolvedCodeExecutionPolicy | None:
        # NOTE: a missing workspace fails closed, because the policy is a veto and no answer must not read as no limit.
        if scope.workspace_id is None:
            raise CodeExecutionPolicyResolutionFailedError(CodeExecutionPolicyResolutionFailure.NO_WORKSPACE)
        return await resolve_workspace_code_execution_policy(self._session, scope.workspace_id)


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


def build_code_execution_policy_port(config: GatewayConfig, session: AsyncSession | None) -> CodeExecutionPolicyPort:
    """The implementation for the planes this deployment serves.

    A deployment serving a control plane holds the policy and needs a session.
    One that does not has a peer holding it, and needs none.
    """
    if deployment_for(config).supports(Plane.CONTROL):
        if session is None:
            msg = "a session is required where this deployment holds the code execution policy"
            raise ValueError(msg)
        return LocalCodeExecutionPolicy(session)
    return RemoteCodeExecutionPolicy(config)
