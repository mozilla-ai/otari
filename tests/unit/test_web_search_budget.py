"""Unit tests for `gateway.services.web_search_budget`.

The loop-level tests in ``test_mcp_loop*.py`` cover each format's refusal shape and
the ``try``/``except``/``else`` placement that keeps a raised search from spending
the cap. What the cap counts, and which calls it applies to at all, is decided here.
"""

from typing import Any

from gateway.services.web_search_budget import (
    MAX_USES_EXCEEDED_ERROR,
    WebSearchBudget,
    is_capped_search,
)


class _SearchBackendLike:
    """Duck-type of the one thing that marks a pool as the gateway's search backend."""

    def take_last_results(self) -> list[dict[str, Any]]:
        return []


class _McpPoolLike:
    """An MCP pool exposing its own ``web_search``, with no structured-result buffer."""


def test_a_fresh_budget_has_room_for_its_first_search() -> None:
    assert not WebSearchBudget(1).exhausted()


def test_a_cap_of_zero_is_exhausted_before_any_search() -> None:
    """A spend control must not read a limit of zero as permission to spend freely."""
    assert WebSearchBudget(0).exhausted()


def test_a_budget_is_spent_by_exactly_its_cap() -> None:
    budget = WebSearchBudget(2)
    budget.record("results")
    assert not budget.exhausted()
    budget.record("results")
    assert budget.exhausted()


def test_a_tool_error_result_does_not_spend_the_cap() -> None:
    """``WebSearchBackend`` returns the sentinel for an empty query rather than raising.

    ``ToolUsageTally`` reads that as not billed, and the cap and the bill have to
    agree, so the same string must leave the budget untouched.
    """
    budget = WebSearchBudget(1)
    budget.record("[tool error] empty query")
    assert not budget.exhausted()


def test_the_refusal_the_gateway_emits_does_not_itself_spend_the_cap() -> None:
    """The refusal is a ``[tool error]``, so replaying one cannot re-charge the caller."""
    budget = WebSearchBudget(1)
    budget.record(MAX_USES_EXCEEDED_ERROR)
    assert not budget.exhausted()


def test_recording_past_the_cap_stays_exhausted() -> None:
    """``exhausted`` is ``<= 0``, so an extra charge cannot wrap back into room."""
    budget = WebSearchBudget(1)
    budget.record("results")
    budget.record("results")
    assert budget.exhausted()


def test_an_uncapped_request_caps_nothing() -> None:
    assert not is_capped_search(None, _SearchBackendLike(), "web_search")


def test_a_tool_that_is_not_the_search_is_not_capped() -> None:
    assert not is_capped_search(WebSearchBudget(1), _SearchBackendLike(), "read_file")


def test_an_mcp_tool_sharing_the_search_name_is_not_capped() -> None:
    """A pool with no structured-result buffer is not the gateway's search backend.

    Its ``web_search`` is the caller's own tool, which the gateway dispatches nothing
    for and the cap has no business bounding.
    """
    assert not is_capped_search(WebSearchBudget(1), _McpPoolLike(), "web_search")


def test_the_gateway_search_backend_is_capped() -> None:
    assert is_capped_search(WebSearchBudget(1), _SearchBackendLike(), "web_search")
