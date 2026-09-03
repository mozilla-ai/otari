"""Per-request cap on the gateway's own web searches.

A caller declaring web search natively can bound how many searches one request
performs (Anthropic's ``max_uses``). Gateway-run search is billed per successful
call, so the cap is a spend control rather than a formatting detail: it applies in
every wire format, and only the shape of the refusal differs between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gateway.services.tool_usage import is_tool_error
from gateway.services.web_search_backend import WEB_SEARCH_TOOL_NAME

if TYPE_CHECKING:
    from gateway.services._tool_loop import ToolBackend

# Refusal text the model sees, in the repo-wide ``[tool error]`` idiom that
# ToolUsageTally already reads as "ran, not billable".
MAX_USES_EXCEEDED_ERROR = "[tool error] max_uses_exceeded"


class WebSearchBudget:
    """The searches one request has left.

    Created only when the caller declared a positive cap, so an uncapped request
    carries no budget and the loops keep their previous behavior. One belongs to
    one request, for the reason :class:`~gateway.services.tool_usage.ToolUsageTally`
    does: a multi-attempt request re-runs its searches and every one of them is
    billed, so the cap has to be spent by the request rather than refilled per
    attempt. It is built once on ``ToolContext`` and handed to whichever loop runs.
    Requests never share one: the tool loops run a request's rounds sequentially,
    so no locking is needed, but two requests must not see each other's count.
    """

    def __init__(self, max_uses: int) -> None:
        self._remaining = max_uses

    def exhausted(self) -> bool:
        """Whether the next search would exceed the cap."""
        return self._remaining <= 0

    def record(self, result: str) -> None:
        """Charge ``result``'s search against the cap unless it failed.

        Failure is read off the ``[tool error]`` sentinel rather than off an
        exception, because that is what decides billable in
        :class:`~gateway.services.tool_usage.ToolUsageTally` and the two numbers have
        to agree: ``WebSearchBackend`` raises when its backend is unreachable but
        returns the sentinel for an empty query, and neither is billed.
        ``max_tool_iterations`` is what bounds a model that keeps retrying a broken
        backend.
        """
        if is_tool_error(result):
            return
        self._remaining -= 1


def is_capped_search(budget: WebSearchBudget | None, pool: ToolBackend | Any, name: str) -> bool:
    """Whether ``name`` on ``pool`` is a gateway search this request capped.

    The name alone is not enough: an MCP server may expose its own tool called
    ``web_search``, which the caller dispatches nothing for and the cap has no
    business bounding. Ownership is duck-typed on the structured-result buffer only
    the search backend keeps, the same check
    ``mcp_loop_messages._native_blocks_for_call`` uses to decide a call has native
    blocks to report.
    """
    if budget is None or name != WEB_SEARCH_TOOL_NAME:
        return False
    return getattr(pool, "take_last_results", None) is not None
