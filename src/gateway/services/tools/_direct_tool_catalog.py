"""Non-billable discovery of the default direct search and fetch calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from gateway.exceptions import TenancyError
from gateway.exceptions.tools_exceptions import SearchToolSelectionError
from gateway.models.tools import WebTool
from gateway.services.tools._web_search import (
    organization_key_search_instance,
    pick_direct_fetch_instance,
    pick_direct_search_instance,
    resolve_fetch,
    resolve_search,
)
from gateway.services.tools._web_search._resolve import resolve_search_on_key

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.core.settings.tools import ToolInstance, ToolKind
    from gateway.models.api_keys import APIKey
    from gateway.models.tools import ResolvedWebSearchConfig


class ToolPermissions(Protocol):
    """The API-key domain's permission check, without its execution dependencies."""

    async def tool_refusal(
        self, config: GatewayConfig, api_key: APIKey | None, *, kind: ToolKind, provider: str, tool: str
    ) -> str | None: ...


class DirectToolCatalog:
    """Discover direct defaults using the workspace policy the catalog already read."""

    def __init__(self, *, config: GatewayConfig, permissions: ToolPermissions) -> None:
        self._config = config
        self._permissions = permissions

    async def available_defaults(
        self, api_key: APIKey | None, *, policy: ResolvedWebSearchConfig | None, workspace_allows: bool
    ) -> frozenset[WebTool]:
        """Resolve configuration and grants, without executing a tool or reading its policy twice.

        Keyless catalog callers receive deployment defaults, as a direct master-key
        call does. A dashboard session discovers these but cannot execute a direct call.
        """
        if not workspace_allows:
            return frozenset()
        available: set[WebTool] = set()
        kinds: tuple[tuple[WebTool, ToolKind], ...] = ((WebTool.SEARCH, "search"), (WebTool.FETCH, "fetch"))
        for web_tool, kind in kinds:
            try:
                tool = self._search(policy) if web_tool is WebTool.SEARCH else self._fetch(policy)
            except TenancyError:
                continue
            if (
                await self._permissions.tool_refusal(
                    self._config, api_key, kind=kind, provider=tool.provider, tool=tool.name
                )
                is None
            ):
                available.add(web_tool)
        return frozenset(available)

    def _search(self, policy: ResolvedWebSearchConfig | None) -> ToolInstance:
        """Resolve the same default or organization-key fallback as a direct search."""
        credential = policy.credential if policy is not None else None
        try:
            tool = pick_direct_search_instance(self._config, None)
        except SearchToolSelectionError:
            standing = organization_key_search_instance(policy)
            if standing is None or credential is None:
                raise
            resolve_search_on_key(credential, name=standing.name, request_options=None)
            return standing
        resolve_search(self._config, None, requested=tool.name, request_options=None, direct_credential=credential)
        return tool

    def _fetch(self, policy: ResolvedWebSearchConfig | None) -> ToolInstance:
        """Resolve the same default and organization credential as a direct fetch."""
        tool = pick_direct_fetch_instance(self._config, None)
        resolve_fetch(
            self._config,
            None,
            requested=tool.name,
            for_search=None,
            direct_credential=policy.fetch_credential if policy is not None else None,
        )
        return tool
