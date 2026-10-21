"""Where a workspace's web search policy comes from: the rows this deployment holds.

The answer carries the workspace's own search key, where its organization has one it may use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncSession

from gateway.exceptions.tools_exceptions import WebSearchPolicyResolutionFailedError, WebSearchPolicyResolutionFailure
from gateway.models.tools import ResolvedWebSearchConfig, WebTool
from gateway.ports.web_search_policy_port import WebSearchPolicyPort, WebSearchPolicyScope
from gateway.services.tenancy.workspace_web_search_service import (
    InvalidStoredWebSearchDomainError,
    resolve_workspace_web_search_config,
)
from gateway.services.tools import WorkspaceSearchKeys

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
# The policy of a workspace that holds no row where this deployment makes web access opt-in.
_NOT_TURNED_ON = replace(_NO_NARROWING, enabled=False)


class LocalWebSearchPolicy(WebSearchPolicyPort):
    """The policy stored in this deployment's own database."""

    def __init__(self, session: AsyncSession, *, search_keys: WorkspaceSearchKeys, on_without_policy: bool) -> None:
        self._session = session
        self._search_keys = search_keys
        self._on_without_policy = on_without_policy

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[WebTool]
    ) -> ResolvedWebSearchConfig | None:
        # A stored row authorizes no tool by name, so the requested tools decide only which keys are read:
        # the search key for a search, the fetch key for either tool, since it also enriches search hits.
        if scope.workspace_id is None:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.NO_WORKSPACE)
        try:
            policy = await resolve_workspace_web_search_config(self._session, scope.workspace_id)
        except InvalidStoredWebSearchDomainError:
            raise WebSearchPolicyResolutionFailedError(WebSearchPolicyResolutionFailure.STORED_POLICY_INVALID) from None
        if policy is None and not self._on_without_policy:
            return _NOT_TURNED_ON
        search = WebTool.SEARCH in requested_tools
        keys = await self._search_keys.keys_for(
            scope.workspace_id, search=search, fetch=search or WebTool.FETCH in requested_tools
        )
        if keys.search is None and keys.fetch is None:
            return policy
        return replace(policy or _NO_NARROWING, credential=keys.search, fetch_credential=keys.fetch)
