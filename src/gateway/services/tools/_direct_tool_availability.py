"""Whether a direct search or fetch that names no tool would run, asked without running one.

A client that offers web search or page reading on ``POST /api/v1/search`` and
``POST /api/v1/fetch`` needs to know before it offers them, and the answer is not the
one completions get: a direct call needs the key's own scope, and runs on the instance
the direct endpoint picks rather than on the model's tool. These checks are the ones
:class:`DirectToolService` makes before it holds any budget, in its order: the
instance it would pick (or the stand-in an unnamed search falls back to), the key's
scope, then the key's tool list, then whether the organization can pay for a call
on the deployment's account. Pricing and budgets are not checked here; a direct call
still answers 402 or 429 where they refuse it. The workspace's web switch stays
with the caller, which reads it once for every tool it lists.
"""

from collections.abc import Awaitable, Callable

from gateway.core.settings.tools import ToolInstance, ToolKind, ToolSettings
from gateway.exceptions._base import TenancyError
from gateway.exceptions.tools_exceptions import SearchToolSelectionError
from gateway.models.api_keys import APIKey
from gateway.models.tools import ResolvedWebSearchConfig
from gateway.services.tools._web_search._direct import (
    legacy_search_instance,
    organization_key_search_instance,
    pick_direct_search_instance,
)
from gateway.services.tools._web_search._direct_fetch import pick_direct_fetch_instance
from gateway.types.key_scope import KeyScope

ToolRefusal = Callable[..., Awaitable[str | None]]


class DirectToolAvailability:
    """Answers, per caller, whether the unnamed direct search and fetch would be admitted.

    ``tool_refusal`` is the key's tool-list check the direct endpoints make,
    called as ``tool_refusal(api_key, kind=..., provider=..., tool=...)``.
    """

    def __init__(self, config: ToolSettings, tool_refusal: ToolRefusal) -> None:
        self._config = config
        self._tool_refusal = tool_refusal

    async def search(self, api_key: APIKey | None, policy: ResolvedWebSearchConfig | None, *, funded: bool) -> bool:
        """Whether ``POST /api/v1/search`` naming no tool would run for ``api_key`` under ``policy``.

        ``funded`` says whether the organization can pay for a search on the deployment's
        account. A search that falls back to the organization's own key needs no funds.
        """
        instance: ToolInstance | None
        try:
            instance = pick_direct_search_instance(self._config, None)
        except SearchToolSelectionError:
            on_organization_key = organization_key_search_instance(policy)
            if on_organization_key is not None:
                return await self._admits(api_key, on_organization_key, kind="search", scope=KeyScope.WEB_SEARCH)
            instance = legacy_search_instance(self._config)
        except TenancyError:
            return False
        return funded and await self._admits(api_key, instance, kind="search", scope=KeyScope.WEB_SEARCH)

    async def fetch(self, api_key: APIKey | None, *, funded: bool) -> bool:
        """Whether ``POST /api/v1/fetch`` naming no tool would run for ``api_key``; ``funded`` as for :meth:`search`."""
        try:
            instance = pick_direct_fetch_instance(self._config, None)
        except TenancyError:
            return False
        return funded and await self._admits(api_key, instance, kind="fetch", scope=KeyScope.WEB_FETCH)

    async def _admits(
        self, api_key: APIKey | None, instance: ToolInstance | None, *, kind: ToolKind, scope: KeyScope
    ) -> bool:
        if instance is None:
            return False
        if api_key is None:
            return True
        if scope not in api_key.scopes:
            return False
        refusal = await self._tool_refusal(api_key, kind=kind, provider=instance.provider, tool=instance.name)
        return refusal is None
