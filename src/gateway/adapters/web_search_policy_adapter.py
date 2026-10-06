"""The two places a workspace's web search policy comes from.

``LocalWebSearchPolicy`` reads the rows this deployment holds.
``RemoteWebSearchPolicy`` asks a peer, speaking `docs/hybrid-mode-protocol.md`.
Either answer carries the workspace's own search key, where its organization has one it may use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import WEB_SEARCH_PROVIDERS, GatewayConfig
from gateway.exceptions.tools_exceptions import WebSearchPolicyResolutionFailedError, WebSearchPolicyResolutionFailure
from gateway.log_config import logger
from gateway.models.tools import ResolvedWebSearchConfig, WebSearchCredential, WebTool
from gateway.ports.web_search_policy_port import WebSearchPolicyPort, WebSearchPolicyScope
from gateway.repositories.tenancy import WorkspaceRepository
from gateway.repositories.tools import WorkspaceWebSearchKeyOverrideRepository
from gateway.services.control_plane import ResolveEndpoint, resolve
from gateway.services.tenancy.workspace_web_search_service import (
    InvalidStoredWebSearchDomainError,
    read_web_search_policy,
    resolve_workspace_web_search_config,
)
from gateway.services.tools import workspace_search_credential

# The policy of a workspace that holds no row, carrying only its search key.
_NO_NARROWING = ResolvedWebSearchConfig(
    enabled=True,
    max_results=None,
    purpose_hint=None,
    allowed_domains=None,
    blocked_domains=None,
    provider_options=None,
    authorized_tools=None,
)


class LocalWebSearchPolicy(WebSearchPolicyPort):
    """The policy stored in this deployment's own database."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[WebTool]
    ) -> ResolvedWebSearchConfig | None:
        # A stored row authorizes no tool by name, so the requested tools decide only whether a key is read.
        if scope.workspace_id is None:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.NO_WORKSPACE)
        try:
            policy = await resolve_workspace_web_search_config(self._session, scope.workspace_id)
        except InvalidStoredWebSearchDomainError:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.STORED_POLICY_INVALID) from None
        if WebTool.SEARCH not in requested_tools:
            return policy
        credential = await workspace_search_credential(
            WorkspaceRepository(self._session),
            WorkspaceWebSearchKeyOverrideRepository(self._session),
            scope.workspace_id,
        )
        if credential is None:
            return policy
        return replace(policy or _NO_NARROWING, credential=credential)


class RemoteWebSearchPolicy(WebSearchPolicyPort):
    """The policy the control plane holds for this deployment's workspaces."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[WebTool]
    ) -> ResolvedWebSearchConfig | None:
        if not scope.user_token:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.NO_CALLER_CREDENTIAL)
        answer = await resolve(
            self._config,
            user_token=scope.user_token,
            endpoint=ResolveEndpoint.WEB_SEARCH,
            body={"requested_tools": [tool.value for tool in requested_tools]},
        )
        if not isinstance(answer, dict):
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE)
        try:
            policy = read_web_search_policy(answer)
        except ValueError:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE) from None
        return replace(policy, authorized_tools=_authorized_tools(answer), credential=_credential(answer))


def _credential(answer: dict[str, Any]) -> WebSearchCredential | None:
    """The workspace's own search key, read strictly so a malformed answer fails closed.

    A provider this deployment cannot call is ignored rather than refused, as an unknown
    value from a peer is, so the workspace searches with the deployment's search instead.
    """
    raw = answer.get("credential")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE)
    provider, api_key = raw.get("provider"), raw.get("api_key")
    if not isinstance(provider, str) or not isinstance(api_key, str) or not api_key:
        raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE)
    if provider not in WEB_SEARCH_PROVIDERS:
        logger.warning("Ignoring a workspace web search key for a provider this gateway cannot call: %s", provider)
        return None
    return WebSearchCredential(provider=provider, api_key=api_key)


def _authorized_tools(answer: dict[str, Any]) -> frozenset[str]:
    """The tool names the peer authorizes, read strictly so a malformed answer fails closed.

    Strings rather than :class:`WebTool`: a peer may name a tool this deployment does not know, which is ignored.
    """
    # An answer without the field predates per-tool authorization and authorizes Search alone.
    authorized = answer.get("authorized_tools", [WebTool.SEARCH.value])
    if not isinstance(authorized, list) or any(not isinstance(tool, str) for tool in authorized):
        raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.ANSWER_UNREADABLE)
    return frozenset(authorized)
