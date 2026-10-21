"""Where a workspace's web search policy comes from.

The policy says who may search and how far, and nothing here carries a search.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from gateway.exceptions.tools_exceptions import WebSearchPolicyResolutionFailedError
from gateway.models.tools import ResolvedWebSearchConfig, WebTool


@dataclass(frozen=True)
class WebSearchPolicyScope:
    """Whose policy to resolve."""

    workspace_id: uuid.UUID | None


class WebSearchPolicyPort(Protocol):
    """A workspace's web search policy."""

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[WebTool]
    ) -> ResolvedWebSearchConfig | None:
        """The policy that governs ``requested_tools`` for this scope, or ``None`` where the workspace has none.

        Raises:
            WebSearchPolicyResolutionFailedError: the policy could not be resolved, and ``reason`` says why.
        """
        ...


__all__ = [
    "WebSearchPolicyPort",
    "WebSearchPolicyResolutionFailedError",
    "WebSearchPolicyScope",
]
