"""Where a workspace's web search policy comes from.

One deployment holds a workspace's policy itself and another asks a peer that holds it.
Both answer the same question, so a caller reads the policy without knowing which answered.

The policy says who may search and how far, and nothing here carries a search.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from gateway.exceptions.tools_exceptions import WebSearchPolicyResolutionFailedError
from gateway.models.tools import ResolvedWebSearchConfig


@dataclass(frozen=True)
class WebSearchPolicyScope:
    """Whose policy to resolve.

    A deployment that holds the policy reads the workspace ID.
    A deployment that asks a peer reads the caller's token.
    Neither field defaults, so a caller cannot drop the one its deployment reads.
    """

    workspace_id: uuid.UUID | None
    user_token: str | None


class WebSearchPolicyPort(Protocol):
    """A workspace's web search policy."""

    async def resolve(
        self, scope: WebSearchPolicyScope, requested_tools: Sequence[str]
    ) -> ResolvedWebSearchConfig | None:
        """The policy that governs ``requested_tools`` for this scope, or ``None`` where the workspace has none.

        Raises:
            WebSearchPolicyResolutionFailedError: the policy could not be resolved, and ``reason`` says why.
            ControlPlaneError: a peer that holds the policy refused or could not answer.
        """
        ...


__all__ = [
    "WebSearchPolicyPort",
    "WebSearchPolicyResolutionFailedError",
    "WebSearchPolicyScope",
]
