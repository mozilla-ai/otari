"""Per-request caps on the gateway's own calls of its built-in tools.

A caller declaring a tool natively can bound how many calls one request makes
(Anthropic's ``max_uses``). A gateway-run call is billed, so the cap is a spend
control rather than a formatting detail: it applies in every wire format, and only
the shape of the refusal differs between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gateway.services.tool_usage import is_tool_error

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend

# Refusal text the model sees, in the ``[tool error]`` idiom that already reads as
# "ran, not billable".
MAX_USES_EXCEEDED_ERROR = "[tool error] max_uses_exceeded"


class ToolUseBudget:
    """The calls each capped built-in tool has left in one request, by tool name.

    A cap of ``0`` is a cap, not the absence of one.

    NOTE: one budget belongs to one request and must not be shared between two. A request
    spends its caps across every routing attempt it makes, since each attempt re-runs the
    calls and every one of them is billed. Mutation is unsynchronized, which is safe only
    because a request's rounds run one after another.
    """

    def __init__(self, max_uses: Mapping[str, int]) -> None:
        self._remaining = dict(max_uses)

    def caps(self, pool: ToolBackend, name: str) -> bool:
        """Whether ``name`` on ``pool`` is a gateway-run call this budget bounds.

        A call the pool does not own is the caller's own to dispatch and spends nothing
        here. Ownership alone does not tell the gateway's tool from an MCP server's tool
        of the same name, and it does not have to: a request whose MCP servers expose a
        tool named like a built-in tool it declared is refused.
        """
        return name in self._remaining and pool.owns_tool(name)

    def exhausted(self, name: str) -> bool:
        """Whether the next call of ``name`` would exceed its cap."""
        return name in self._remaining and self._remaining[name] <= 0

    def record(self, name: str, result: str) -> None:
        """Charge ``result``'s call against ``name``'s cap unless it failed.

        Failure is read off the ``[tool error]`` sentinel rather than off an exception,
        because the sentinel is what decides billable, and the cap and the bill have to
        agree. The iteration cap, not this one, is what bounds a model that keeps retrying
        a broken backend.
        """
        if name in self._remaining and not is_tool_error(result):
            self._remaining[name] -= 1


def is_capped_call(budget: ToolUseBudget | None, pool: ToolBackend, name: str) -> bool:
    """Whether ``name`` on ``pool`` is a call this request capped."""
    return budget is not None and budget.caps(pool, name)
