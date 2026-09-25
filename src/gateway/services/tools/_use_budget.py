"""Per-request cap on the gateway's own calls of one built-in tool.

A caller declaring a tool natively can bound how many calls one request makes
(Anthropic's ``max_uses``). A gateway-run call is billed, so the cap is a spend
control rather than a formatting detail: it applies in every wire format, and only
the shape of the refusal differs between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gateway.services.tool_usage import is_tool_error

if TYPE_CHECKING:
    from gateway.services._tool_loop import ToolBackend

# Refusal text the model sees, in the ``[tool error]`` idiom that already reads as
# "ran, not billable".
MAX_USES_EXCEEDED_ERROR = "[tool error] max_uses_exceeded"


class ToolUseBudget:
    """The calls of one built-in tool a request has left.

    A cap of ``0`` is a cap, not the absence of one.

    NOTE: one budget belongs to one request and must not be shared between two. A request
    spends its cap across every routing attempt it makes, since each attempt re-runs the
    calls and every one of them is billed. Mutation is unsynchronized, which is safe only
    because a request's rounds run one after another.
    """

    def __init__(self, tool: str, max_uses: int) -> None:
        self._tool = tool
        self._remaining = max_uses

    def caps(self, pool: ToolBackend, name: str) -> bool:
        """Whether ``name`` on ``pool`` is the gateway-run call this budget bounds.

        A call the pool does not own is the caller's own to dispatch and spends nothing
        here. Ownership alone does not tell the gateway's tool from an MCP server's tool
        of the same name, and it does not have to: a request declaring a built-in tool
        alongside an MCP server is refused, so a request holding a budget reaches one
        built-in backend and no MCP pool.
        """
        return name == self._tool and pool.owns_tool(name)

    def exhausted(self) -> bool:
        """Whether the next call would exceed the cap."""
        return self._remaining <= 0

    def record(self, result: str) -> None:
        """Charge ``result``'s call against the cap unless it failed.

        Failure is read off the ``[tool error]`` sentinel rather than off an exception,
        because the sentinel is what decides billable, and the cap and the bill have to
        agree. The iteration cap, not this one, is what bounds a model that keeps retrying
        a broken backend.
        """
        if is_tool_error(result):
            return
        self._remaining -= 1


def is_capped_call(budget: ToolUseBudget | None, pool: ToolBackend, name: str) -> bool:
    """Whether ``name`` on ``pool`` is a call this request capped."""
    return budget is not None and budget.caps(pool, name)
