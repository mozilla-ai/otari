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

# Refusal text the model sees, in the repo-wide ``[tool error]`` idiom that
# ToolUsageTally already reads as "ran, not billable".
MAX_USES_EXCEEDED_ERROR = "[tool error] max_uses_exceeded"


class ToolUseBudget:
    """The calls of one built-in tool a request has left.

    Created whenever the caller declared a cap, ``0`` included, so an uncapped request
    carries no budget and the loops keep their previous behavior. One belongs to one
    request, for the reason :class:`~gateway.services.tool_usage.ToolUsageTally` does: a
    multi-attempt request re-runs its calls and every one of them is billed, so the cap
    has to be spent by the request rather than refilled per attempt. It is built once on
    ``ToolContext`` and handed to whichever loop runs. Requests never share one: the tool
    loops run a request's rounds sequentially, so no locking is needed, but two requests
    must not see each other's count.
    """

    def __init__(self, tool: str, max_uses: int) -> None:
        self._tool = tool
        self._remaining = max_uses

    def caps(self, pool: ToolBackend, name: str) -> bool:
        """Whether ``name`` on ``pool`` is the gateway-run call this budget bounds.

        A call the pool does not own is the caller's own to dispatch and spends nothing
        here. A same-named tool from an MCP server cannot reach this: a request declaring
        a built-in tool and an MCP server is refused, so a request carrying a budget runs
        one built-in backend and no pool.
        """
        return name == self._tool and pool.owns_tool(name)

    def exhausted(self) -> bool:
        """Whether the next call would exceed the cap."""
        return self._remaining <= 0

    def record(self, result: str) -> None:
        """Charge ``result``'s call against the cap unless it failed.

        Failure is read off the ``[tool error]`` sentinel rather than off an exception,
        because that is what decides billable in
        :class:`~gateway.services.tool_usage.ToolUsageTally` and the two numbers have to
        agree: a backend raises when it is unreachable but returns the sentinel for an
        unusable request, and neither is billed. ``max_tool_iterations`` is what bounds a
        model that keeps retrying a broken backend.
        """
        if is_tool_error(result):
            return
        self._remaining -= 1


def is_capped_call(budget: ToolUseBudget | None, pool: ToolBackend, name: str) -> bool:
    """Whether ``name`` on ``pool`` is a call this request capped."""
    return budget is not None and budget.caps(pool, name)
