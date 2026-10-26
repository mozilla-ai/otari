"""Whose account ran a gateway tool call decides whose prices count and what the organization's funds are charged.

On the deployment's account, only the operator's price applies and the call is
charged to the organization's funds whoever paid for the model call. On a key the
organization brought, its own override may price the call and nothing is charged.
"""

from decimal import Decimal

from gateway.services.pricing import ToolRate, charge_tool_calls, tool_rate_lookup
from gateway.services.tool_usage import ToolUsageTally
from gateway.types.tool_charges import LabelCalls, RateKey

_HITS = "1. Otari\n   https://otari.example"
_CODE_RATE = ToolRate("otari:code_execution", Decimal("0.02"), "deployment")
_SEARCH_RATE = ToolRate("otari:web_search", Decimal("0.01"), "deployment")


def _calls(tally: ToolUsageTally) -> dict[str, LabelCalls]:
    return {entry.tool: entry for entry in tally.label_calls()}


def test_code_execution_on_the_deployments_sandbox_takes_only_the_operators_price() -> None:
    tally = ToolUsageTally()
    tally.record_result("code_execution", "ok", deployment_paid=True)

    lookup = tool_rate_lookup(_calls(tally)["code_execution"])

    assert lookup is not None
    assert lookup.rate_keys == ("otari:code_execution",)
    assert lookup.operator_set_only is True


def test_a_search_on_the_organizations_own_key_takes_its_override() -> None:
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS, deployment_paid=False)

    lookup = tool_rate_lookup(_calls(tally)["web_search"])

    assert lookup is not None
    assert lookup.operator_set_only is False


def test_only_the_calls_the_deployments_account_ran_are_charged_to_the_organization() -> None:
    tally = ToolUsageTally()
    tally.record_result("code_execution", "ok", deployment_paid=True)
    tally.record_result("code_execution", "ok", deployment_paid=True)
    tally.record_result("web_search", _HITS, label="org:tavily", rate_keys=("otari:web_search",))
    calls = tally.label_calls()
    by_key: dict[RateKey, ToolRate] = {"otari:code_execution": _CODE_RATE, "otari:web_search": _SEARCH_RATE}
    rates = {lookup: by_key[lookup.rate_keys[0]] for entry in calls if (lookup := tool_rate_lookup(entry)) is not None}

    charges = charge_tool_calls(calls, rates)

    assert charges.total == Decimal("0.05")
    assert charges.deployment_paid_total == Decimal("0.04")


def test_a_call_whose_payer_was_not_recorded_is_read_off_its_label() -> None:
    """An instance's label is the deployment's; an unlabeled call cannot be told, so it is not charged."""
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS, label="exa:exa-search", rate_keys=("otari:web_search",))
    tally.record_result("web_fetch", "page")
    calls = tally.label_calls()
    rates = {lookup: _SEARCH_RATE for entry in calls if (lookup := tool_rate_lookup(entry)) is not None}

    charges = charge_tool_calls(calls, rates)

    assert charges.total == Decimal("0.02")
    assert charges.deployment_paid_total == Decimal("0.01")


def test_a_reported_cost_on_the_deployments_account_is_charged_as_reported() -> None:
    tally = ToolUsageTally()
    tally.record_result(
        "web_search",
        _HITS,
        label="exa:exa-search",
        rate_keys=("otari:web_search",),
        cost=Decimal("0.007"),
        enrichment_cost=Decimal("0.002"),
        deployment_paid=True,
    )

    charges = charge_tool_calls(tally.label_calls(), {})

    assert charges.deployment_paid_total == Decimal("0.009")


def test_each_charge_line_says_whose_account_ran_its_calls() -> None:
    """One label served on both accounts settles as two lines, so the row shows what the organization was charged."""
    tally = ToolUsageTally()
    tally.record_result("web_search", _HITS, deployment_paid=True)
    tally.record_result("web_search", _HITS, deployment_paid=False)
    calls = tally.label_calls()
    rates = {lookup: _SEARCH_RATE for entry in calls if (lookup := tool_rate_lookup(entry)) is not None}

    charges = charge_tool_calls(calls, rates)

    assert [(line["label"], line["deployment_paid"], line["cost"]) for line in charges.lines] == [
        ("otari:web_search", False, 0.01),
        ("otari:web_search", True, 0.01),
    ]
    assert charges.unit_rates == {"web_search": Decimal("0.01")}
