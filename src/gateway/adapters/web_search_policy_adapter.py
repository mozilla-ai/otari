"""The two places a workspace's web search policy comes from.

``LocalWebSearchPolicy`` reads the row this deployment holds.
``RemoteWebSearchPolicy`` asks a peer, speaking `docs/hybrid-mode-protocol.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.core.deployment import Plane, deployment_for
from gateway.exceptions.tools_exceptions import WebSearchPolicyResolutionFailedError, WebSearchPolicyResolutionFailure
from gateway.models.tools import ResolvedWebSearchConfig
from gateway.ports.web_search_policy_port import WebSearchPolicyPort, WebSearchPolicyScope
from gateway.services.control_plane import ResolveEndpoint, resolve
from gateway.services.tenancy.workspace_web_search_service import (
    InvalidStoredWebSearchDomainError,
    read_web_search_policy,
    resolve_workspace_web_search_config,
)
from gateway.services.web_retrieval_backend import WEB_SEARCH_TOOL_NAME


class LocalWebSearchPolicy(WebSearchPolicyPort):
    """The policy stored in this deployment's own database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[str]
    ) -> ResolvedWebSearchConfig | None:
        # A stored row authorizes no tool by name, so the requested tools do not change the answer.
        del requested_tools
        if scope.workspace_id is None:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.NO_WORKSPACE)
        try:
            return await resolve_workspace_web_search_config(self._session, scope.workspace_id)
        except InvalidStoredWebSearchDomainError:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.STORED_POLICY_INVALID) from None


class RemoteWebSearchPolicy(WebSearchPolicyPort):
    """The policy the control plane holds for this deployment's workspaces."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[str]
    ) -> ResolvedWebSearchConfig | None:
        if not scope.user_token:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.NO_CALLER_CREDENTIAL)
        answer = await resolve(
            self._config,
            user_token=scope.user_token,
            endpoint=ResolveEndpoint.WEB_SEARCH,
            body={"requested_tools": list(requested_tools)},
        )
        if not isinstance(answer, dict):
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE)
        try:
            policy = read_web_search_policy(answer)
        except ValueError:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE) from None
        return replace(policy, authorized_tools=_authorized_tools(answer))


def _authorized_tools(answer: dict[str, Any]) -> frozenset[str]:
    """The tool names the peer authorizes, read strictly so a malformed answer fails closed."""
    # An answer without the field predates per-tool authorization and authorizes Search alone.
    authorized = answer.get("authorized_tools", [WEB_SEARCH_TOOL_NAME])
    if not isinstance(authorized, list) or any(not isinstance(tool, str) for tool in authorized):
        raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE)
    return frozenset(authorized)


def build_web_search_policy_port(config: GatewayConfig, session: AsyncSession | None) -> WebSearchPolicyPort:
    """The implementation for the planes this deployment serves.

    A deployment serving a control plane holds the policy and needs a session.
    One that does not has a peer holding it, and needs none.
    """
    if deployment_for(config).supports(Plane.CONTROL):
        if session is None:
            msg = "a session is required where this deployment holds the web search policy"
            raise ValueError(msg)
        return LocalWebSearchPolicy(session)
    return RemoteWebSearchPolicy(config)
