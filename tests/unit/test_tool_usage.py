"""The tally keeps each successful call under the label, rate keys and cost it was recorded with (design L)."""

from decimal import Decimal

import pytest

from gateway.models.money import MAX_USD_COST
from gateway.services.pricing import tool_rate_lookup
from gateway.services.tool_usage import MAX_TOOL_NAME_CHARS, MAX_TOOL_NAMES, OVERFLOW_TOOL_NAME, ToolUsageTally
from gateway.types.tool_charges import LabelCalls, gateway_tool_pricing_key

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


def test_a_success_recorded_without_a_text_is_billed_as_a_call_that_returned_one() -> None:
    """A page read to enrich a search's hit is recorded as the fetch it is, beside the calls the model made."""
    tally = ToolUsageTally()
    tally.record_result(
        "web_fetch", _HITS, label="tabstack:tabstack", rate_keys=("tabstack:tabstack",), cost=Decimal("0.002")
    )
    tally.record_success(
        "web_fetch", label="tabstack:tabstack", rate_keys=("tabstack:tabstack",), cost=Decimal("0.003")
    )
    tally.record_success("web_fetch", label="tabstack:tabstack", rate_keys=("tabstack:tabstack",), deployment_paid=True)

    assert {
        (entry.calls, entry.unreported_calls, entry.reported_cost, entry.deployment_paid)
        for entry in tally.label_calls()
    } == {
        (2, 0, Decimal("0.005"), None),
        (1, 1, Decimal(0), True),
    }
    assert tally.meters() == {"web_fetch": {"billed": 3, "errors": 0}}


@pytest.mark.parametrize(
    "unusable", [Decimal("-0.02"), Decimal("NaN"), Decimal("Infinity"), MAX_USD_COST + Decimal("0.000001")]
)
def test_an_unusable_reported_cost_reads_as_unreported_and_never_credits(unusable: Decimal) -> None:
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS, label=_EXA, rate_keys=_EXA_KEYS, cost=unusable)

    entry = _only(tally)
    assert (entry.calls, entry.unreported_calls) == (1, 1)
    assert entry.reported_cost == Decimal(0)


def test_a_failed_call_is_counted_and_its_cost_dropped() -> None:
    tally = ToolUsageTally()
    tally.record_result(
        "web_search",
        "[tool error] exa search failed (http_status)",
        label=_EXA,
        rate_keys=_EXA_KEYS,
        cost=Decimal("0.005"),
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
    """Calls that carry no rate keys price nothing, unlike the omitted ``otari:<tool>``."""
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


_LONG_TOOL = "mcp__acme_analytics_platform__reporting__generate_quarterly_revenue_pdf"


def test_a_tool_name_past_the_cap_is_still_priced_under_its_full_name() -> None:
    """The meter name is capped, but the operator prices ``otari:<full name>``."""
    assert len(_LONG_TOOL) > MAX_TOOL_NAME_CHARS
    tally = ToolUsageTally()
    tally.record_success(_LONG_TOOL)

    entry = _only(tally)
    assert entry.tool == _LONG_TOOL[:MAX_TOOL_NAME_CHARS]
    lookup = tool_rate_lookup(entry)
    assert lookup is not None
    assert lookup.rate_keys == (gateway_tool_pricing_key(_LONG_TOOL),)


def test_a_tool_name_at_the_cap_and_explicit_rate_keys_are_recorded_as_before() -> None:
    at_cap = "t" * MAX_TOOL_NAME_CHARS
    tally = ToolUsageTally()
    tally.record_success(at_cap)
    tally.record_success(_LONG_TOOL, label="mcp:acme", rate_keys=("mcp:acme",))

    by_tool = {entry.tool: entry for entry in tally.label_calls()}
    assert by_tool[at_cap].rate_keys is None
    assert by_tool[_LONG_TOOL[:MAX_TOOL_NAME_CHARS]].rate_keys == ("mcp:acme",)


def test_two_tools_that_share_a_capped_prefix_keep_their_own_rate_keys() -> None:
    tally = ToolUsageTally()
    tally.record_success(_LONG_TOOL + "_a")
    tally.record_success(_LONG_TOOL + "_b")

    keys = {key for entry in tally.label_calls() if (lookup := tool_rate_lookup(entry)) for key in lookup.rate_keys}
    assert keys == {gateway_tool_pricing_key(_LONG_TOOL + "_a"), gateway_tool_pricing_key(_LONG_TOOL + "_b")}


class _Gate:
    def __init__(self, *, fits: bool) -> None:
        self.fits = fits
        self.asked: list[LabelCalls] = []

    async def admit(self, call: LabelCalls) -> bool:
        self.asked.append(call)
        return self.fits


@pytest.mark.asyncio
async def test_the_spend_gate_prices_a_long_tool_name_under_its_full_name() -> None:
    tally = ToolUsageTally()
    tally.spend_gate = gate = _Gate(fits=True)
    await tally.admit(_LONG_TOOL)

    [asked] = gate.asked
    lookup = tool_rate_lookup(asked)
    assert lookup is not None
    assert lookup.rate_keys == (gateway_tool_pricing_key(_LONG_TOOL),)


@pytest.mark.asyncio
async def test_a_tally_with_no_spend_gate_admits_every_call() -> None:
    assert await ToolUsageTally().admit("web_search")


@pytest.mark.asyncio
async def test_the_spend_gate_is_asked_about_one_unreported_call_under_its_label() -> None:
    """One call with no reported cost, which is what a rate prices before the call has run."""
    tally = ToolUsageTally()
    tally.spend_gate = gate = _Gate(fits=False)

    assert not await tally.admit("web_search", label=_EXA, rate_keys=_EXA_KEYS, deployment_paid=True)
    assert gate.asked == [
        LabelCalls(
            tool="web_search",
            label=_EXA,
            rate_keys=_EXA_KEYS,
            calls=1,
            unreported_calls=1,
            reported_cost=Decimal(0),
            deployment_paid=True,
        )
    ]
    assert tally.is_empty(), "asking records nothing"


@pytest.mark.asyncio
async def test_the_spend_gate_is_asked_about_the_pages_a_search_reads_as_one_hold() -> None:
    """The pages a search reads together are one hold of that many unreported fetches under the fetch instance."""
    tally = ToolUsageTally()
    tally.spend_gate = gate = _Gate(fits=True)

    assert await tally.admit(
        "web_fetch", label="tabstack:tabstack", rate_keys=("tabstack:tabstack",), deployment_paid=True, calls=3
    )
    [asked] = gate.asked
    assert (asked.tool, asked.calls, asked.unreported_calls, asked.reported_cost) == ("web_fetch", 3, 3, Decimal(0))
