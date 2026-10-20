"""The tally keeps each successful call under the label, rate keys and cost it was recorded with (design L)."""

from decimal import Decimal

import pytest

from gateway.models.money import MAX_USD_COST
from gateway.services.tool_usage import MAX_TOOL_NAMES, OVERFLOW_TOOL_NAME, ToolUsageTally
from gateway.types.tool_charges import LabelCalls

_EXA = "exa:exa-search"
_EXA_KEYS = (_EXA, "otari:web_search")
_HITS = "1. Otari\n   https://otari.example"
_NO_HITS = "No results for query: 'otari'"


def _only(tally: ToolUsageTally) -> LabelCalls:
    [entry] = tally.label_calls()
    return entry


def test_a_call_recorded_as_today_keeps_no_label_or_rate_keys() -> None:
    """Callers that pass neither record what they record today: settlement reads ``None`` as ``otari:<tool>``."""
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS)

    assert _only(tally) == LabelCalls(
        tool="web_search",
        label=None,
        rate_keys=None,
        calls=1,
        unreported_calls=1,
        reported_cost=Decimal(0),
        enrichment_cost=Decimal(0),
    )
    assert tally.meters() == {"web_search": {"billed": 1, "errors": 0}}


def test_reported_costs_sum_and_unreported_calls_are_counted_per_label() -> None:
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS, label=_EXA, rate_keys=_EXA_KEYS, cost=Decimal("0.005"))
    tally.record_result("web_search", _HITS, label=_EXA, rate_keys=_EXA_KEYS, cost=Decimal("0.007"))
    tally.record_result("web_search", _HITS, label=_EXA, rate_keys=_EXA_KEYS)

    entry = _only(tally)
    assert (entry.calls, entry.unreported_calls, entry.reported_cost) == (3, 1, Decimal("0.012"))


def test_a_reported_zero_is_a_reported_cost() -> None:
    """A search with no hits that its provider charged nothing for is billed at 0, not at the rate."""
    tally = ToolUsageTally()
    tally.record_result("web_search", _NO_HITS, label=_EXA, rate_keys=_EXA_KEYS, cost=Decimal(0))

    entry = _only(tally)
    assert (entry.calls, entry.unreported_calls, entry.reported_cost) == (1, 0, Decimal(0))
    assert tally.meters() == {"web_search": {"billed": 1, "errors": 0}}


def test_an_unreported_zero_hit_search_is_billed_at_the_rate() -> None:
    """A zero-result success is billed; with no reported cost, a rate prices it."""
    tally = ToolUsageTally()
    tally.record_result("web_search", _NO_HITS)

    entry = _only(tally)
    assert (entry.calls, entry.unreported_calls) == (1, 1)


def test_enrichment_costs_add_up_beside_the_calls_own() -> None:
    tally = ToolUsageTally()
    tally.record_result(
        "web_search", _HITS, label=_EXA, rate_keys=_EXA_KEYS, cost=Decimal("0.005"), enrichment_cost=Decimal("0.002")
    )
    tally.record_result("web_search", _HITS, label=_EXA, rate_keys=_EXA_KEYS, enrichment_cost=Decimal("0.003"))

    entry = _only(tally)
    assert entry.reported_cost == Decimal("0.005")
    assert entry.enrichment_cost == Decimal("0.005")
    assert entry.unreported_calls == 1


@pytest.mark.parametrize(
    "unusable", [Decimal("-0.02"), Decimal("NaN"), Decimal("Infinity"), MAX_USD_COST + Decimal("0.000001")]
)
def test_an_unusable_reported_cost_reads_as_unreported_and_never_credits(unusable: Decimal) -> None:
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS, label=_EXA, rate_keys=_EXA_KEYS, cost=unusable, enrichment_cost=unusable)

    entry = _only(tally)
    assert (entry.calls, entry.unreported_calls) == (1, 1)
    assert entry.reported_cost == Decimal(0)
    assert entry.enrichment_cost == Decimal(0)


def test_a_failed_call_is_counted_and_its_costs_dropped() -> None:
    tally = ToolUsageTally()
    tally.record_result(
        "web_search",
        "[tool error] exa search failed (http_status)",
        label=_EXA,
        rate_keys=_EXA_KEYS,
        cost=Decimal("0.005"),
        enrichment_cost=Decimal("0.001"),
    )
    tally.record_failure("web_search")

    assert tally.label_calls() == []
    assert tally.meters() == {"web_search": {"billed": 0, "errors": 2}}


def test_each_label_of_a_tool_is_kept_apart_under_one_meter() -> None:
    tally = ToolUsageTally()
    tally.record_result("web_fetch", _HITS, label="builtin:builtin_fetch", rate_keys=("builtin:builtin_fetch",))
    tally.record_result("web_fetch", _HITS, label="tabstack:tabstack", rate_keys=("tabstack:tabstack",))
    tally.record_result("web_fetch", _HITS, label="tabstack:tabstack", rate_keys=("tabstack:tabstack",))

    by_label = {entry.label: entry.calls for entry in tally.label_calls()}
    assert by_label == {"builtin:builtin_fetch": 1, "tabstack:tabstack": 2}
    assert tally.meters() == {"web_fetch": {"billed": 3, "errors": 0}}


def test_empty_rate_keys_are_kept_apart_from_none() -> None:
    """A hybrid credential's calls carry no rate keys, which prices nothing, unlike the omitted ``otari:<tool>``."""
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS, label=None, rate_keys=())
    tally.record_result("web_search", _HITS)

    assert {entry.rate_keys for entry in tally.label_calls()} == {(), None}


def test_a_tool_past_the_name_cap_records_its_label_under_the_overflow_meter() -> None:
    tally = ToolUsageTally()
    for index in range(MAX_TOOL_NAMES):
        tally.record_result(f"tool_{index}", "ok")
    tally.record_result("one_more", "ok", label="mcp:one_more", rate_keys=())

    overflow = [entry for entry in tally.label_calls() if entry.tool == OVERFLOW_TOOL_NAME]
    assert [(entry.label, entry.calls) for entry in overflow] == [("mcp:one_more", 1)]
    assert tally.overflowed
