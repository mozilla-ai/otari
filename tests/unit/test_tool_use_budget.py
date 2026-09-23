"""Unit tests for `gateway.services.tools._use_budget`.

The loop-level tests in ``test_mcp_loop*.py`` cover each format's refusal shape and
the ``try``/``except``/``else`` placement that keeps a raised call from spending the
cap. What the cap counts, and which calls it applies to at all, is decided here.
"""

from typing import Any

from gateway.services.tools import MAX_USES_EXCEEDED_ERROR, ToolUseBudget, is_capped_call
from gateway.services.web_retrieval_backend import WEB_SEARCH_TOOL_NAME


def _budget(max_uses: int) -> ToolUseBudget:
    return ToolUseBudget(WEB_SEARCH_TOOL_NAME, max_uses)


class _PoolLike:
    """The tool-backend surface the cap reads, over the names one backend runs."""

    def __init__(self, *names: str) -> None:
        self._names = frozenset(names)

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return []

    def owns_tool(self, name: str) -> bool:
        return name in self._names

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise AssertionError("the cap decides before a call runs")

    def purpose_hints(self) -> list[tuple[str, str]]:
        return []


def _search_backend() -> _PoolLike:
    return _PoolLike(WEB_SEARCH_TOOL_NAME)


def test_a_fresh_budget_has_room_for_its_first_search() -> None:
    assert not _budget(1).exhausted()


def test_a_cap_of_zero_is_exhausted_before_any_search() -> None:
    """A spend control must not read a limit of zero as permission to spend freely."""
    assert _budget(0).exhausted()


def test_a_budget_is_spent_by_exactly_its_cap() -> None:
    budget = _budget(2)
    budget.record("results")
    assert not budget.exhausted()
    budget.record("results")
    assert budget.exhausted()


def test_a_tool_error_result_does_not_spend_the_cap() -> None:
    """The retrieval backend returns the sentinel for an empty query rather than raising.

    ``ToolUsageTally`` reads that as not billed, and the cap and the bill have to
    agree, so the same string must leave the budget untouched.
    """
    budget = _budget(1)
    budget.record("[tool error] empty query")
    assert not budget.exhausted()


def test_the_refusal_the_gateway_emits_does_not_itself_spend_the_cap() -> None:
    """The refusal is a ``[tool error]``, so replaying one cannot re-charge the caller."""
    budget = _budget(1)
    budget.record(MAX_USES_EXCEEDED_ERROR)
    assert not budget.exhausted()


def test_recording_past_the_cap_stays_exhausted() -> None:
    """``exhausted`` is ``<= 0``, so an extra charge cannot wrap back into room."""
    budget = _budget(1)
    budget.record("results")
    budget.record("results")
    assert budget.exhausted()


def test_an_uncapped_request_caps_nothing() -> None:
    assert not is_capped_call(None, _search_backend(), "web_search")


def test_a_tool_that_is_not_the_search_is_not_capped() -> None:
    assert not is_capped_call(_budget(1), _search_backend(), "read_file")


def test_a_call_the_backend_does_not_own_is_not_capped() -> None:
    """The caller dispatches its own tools, so nothing here spends against the cap."""
    assert not is_capped_call(_budget(1), _PoolLike(), "web_search")


def test_the_gateway_search_backend_is_capped() -> None:
    assert is_capped_call(_budget(1), _search_backend(), "web_search")
