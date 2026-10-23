"""What a request's gateway-run tool calls cost: one charge line per tool and label.

This is the charge rule, apart from the lookup. A call is priced at the cost its
provider reported, when it reported one, and otherwise at the rate its rate keys
resolve to, which ``services/pricing_service.py`` looks up as it looks up a
model's rate, so tools and models never order their rows differently.

A call the deployment pays for, which an instance or the built-in fetcher
served, takes only an override the operator set, as a deployment-paid model
request does: an override the organization wrote itself would let it name the
deployment's cost, down to zero.
"""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import NamedTuple

from gateway.core.metered_pricing import quantize_cost
from gateway.models.pricing import PriceSource
from gateway.types.tool_charges import (
    GATEWAY_TOOL_PRICING_PROVIDER,
    ORGANIZATION_KEY_LABEL_PREFIX,
    LabelCalls,
    RateKey,
    gateway_tool_pricing_key,
)

# Published per-search prices of searches a provider runs itself, keyed by label.
# Default pricing prices a provider-run search at its list price when the operator
# set no rate, as models.dev prices a model. Anthropic: $10 per 1,000 searches
# (https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-search-tool), read 2026-10-17.
PROVIDER_TOOL_LIST_PRICES: Mapping[str, Decimal] = {
    "anthropic:web_search": Decimal("0.01"),
}


class ToolRate(NamedTuple):
    """The rate a call with no reported cost is priced at, and where it was found."""

    rate_key: RateKey
    #: USD per call.
    unit_rate: Decimal
    source: PriceSource


class ToolRateLookup(NamedTuple):
    """What a label's calls with no reported cost are priced under."""

    rate_keys: tuple[RateKey, ...]
    #: True for a call the deployment pays for, which only the operator's override prices.
    operator_set_only: bool


class UnpricedToolLabel(NamedTuple):
    """A label whose calls reported no cost and whose rate keys found no pricing row."""

    tool: str
    label: str
    #: The row an operator would add: the label's first rate key.
    rate_key: RateKey
    #: The calls that went unpriced: those that reported no cost.
    calls: int


class ToolCharges(NamedTuple):
    """What a request's successful gateway-run tool calls cost.

    ``lines`` holds one charge line per tool and label, ``unit_rates`` each
    tool's cost over its calls, and ``unpriced`` the labels that ran calls no
    row could price. A tool served under several labels has one unit rate, so
    its calls times that rate is still its spend. ``deployment_paid_total`` is
    the part of ``total`` the deployment's account ran, which its organization
    is charged for whoever paid for the model call.
    """

    total: Decimal
    lines: list[dict[str, float | int | str]]
    unit_rates: dict[str, Decimal]
    unpriced: list[UnpricedToolLabel]
    deployment_paid_total: Decimal


def paid_by_the_deployment(label: str | None) -> bool:
    """Whether a call under ``label`` ran on the deployment's account.

    The fallback for a call recorded without saying whose account ran it. An
    instance's label and ``builtin:builtin_fetch`` did. An organization key's
    ``org:<provider>`` did not. Neither can be told for a call with no label or an
    ``otari:<tool>`` one, such as an MCP tool's, so those count as not.
    """
    return label is not None and not label.startswith(
        (ORGANIZATION_KEY_LABEL_PREFIX, f"{GATEWAY_TOOL_PRICING_PROVIDER}:")
    )


def tool_rate_lookup(entry: LabelCalls) -> ToolRateLookup | None:
    """What ``entry``'s calls with no reported cost are priced under, or ``None`` when every call reported one.

    Rate keys of ``None`` mean the tool's own ``otari:<tool>`` row; an empty
    tuple prices nothing.
    """
    if entry.calls <= 0 or entry.unreported_calls <= 0:
        return None
    rate_keys = entry.rate_keys if entry.rate_keys is not None else (gateway_tool_pricing_key(entry.tool),)
    return ToolRateLookup(rate_keys, _deployment_paid(entry))


def _deployment_paid(entry: LabelCalls) -> bool:
    """Whether the deployment's account ran ``entry``'s calls, as recorded or else as its label says."""
    return entry.deployment_paid if entry.deployment_paid is not None else paid_by_the_deployment(entry.label)


def charge_tool_calls(calls: Sequence[LabelCalls], rates: Mapping[ToolRateLookup, ToolRate | None]) -> ToolCharges:
    """Price a request's successful gateway-run tool calls, per tool and label.

    A label costs the costs its calls reported, plus the rate ``rates`` holds for
    its :func:`tool_rate_lookup` for each call that reported none, plus what its
    enrichment fetches reported. A label of ``None`` means the tool's own
    ``otari:<tool>``.

    Charge lines use the shape ``calculate_metered_cost`` produces so both
    kinds share ``UsageLog.pricing_breakdown`` and the dashboard's renderer,
    except that a tool line carries ``unit_rate`` (USD per call, the label's
    cost over its calls) where a token line carries ``rate_per_million``. That
    key is what tells a reader, and the UI, which unit convention applies. A
    tool line also names its ``label``, and ``deployment_paid``: whether the
    deployment's account ran its calls, and so whether the organization's funds
    were charged for them. A label with no reported cost and no
    pricing row still gets a line, at a zero rate, so the work stays on the row
    and in the audit trail even when the operator has not priced it.
    """
    # Keyed by tool, label and whose account ran the calls, so two records of one
    # label with different rate keys still settle as one line, and a line says
    # whether the organization's funds were charged for it.
    by_label: dict[tuple[str, str, bool], tuple[int, Decimal]] = {}
    unpriced: dict[tuple[str, str, RateKey], UnpricedToolLabel] = {}
    for entry in calls:
        if entry.calls <= 0:
            continue
        label = entry.label if entry.label is not None else gateway_tool_pricing_key(entry.tool)
        lookup = tool_rate_lookup(entry)
        rate = rates.get(lookup) if lookup is not None else None
        cost = (
            entry.reported_cost
            + entry.unreported_calls * (rate.unit_rate if rate is not None else Decimal(0))
            + entry.enrichment_cost
        )
        line = (entry.tool, label, _deployment_paid(entry))
        units, label_cost = by_label.get(line, (0, Decimal(0)))
        by_label[line] = (units + entry.calls, label_cost + cost)
        if lookup is not None and lookup.rate_keys and rate is None:
            first = lookup.rate_keys[0]
            previous = unpriced.get((entry.tool, label, first))
            unpriced[(entry.tool, label, first)] = UnpricedToolLabel(
                entry.tool, label, first, entry.unreported_calls + (previous.calls if previous else 0)
            )

    # ``float`` for the JSON charge line; the returned total that settles the
    # row stays Decimal.
    lines: list[dict[str, float | int | str]] = [
        {
            "meter": f"{tool}_calls",
            "label": label,
            "units": units,
            "unit_rate": float(cost / units),
            "cost": float(cost),
            "deployment_paid": paid,
        }
        for (tool, label, paid), (units, cost) in sorted(by_label.items())
    ]
    by_tool: dict[str, tuple[int, Decimal]] = {}
    for (tool, _label, _paid), (units, cost) in by_label.items():
        tool_units, tool_cost = by_tool.get(tool, (0, Decimal(0)))
        by_tool[tool] = (tool_units + units, tool_cost + cost)
    total = quantize_cost(sum((cost for _units, cost in by_tool.values()), Decimal(0)))
    unit_rates = {tool: cost / units for tool, (units, cost) in by_tool.items()}
    deployment_paid = sum((cost for (_tool, _label, paid), (_units, cost) in by_label.items() if paid), Decimal(0))
    return ToolCharges(total, lines, unit_rates, list(unpriced.values()), quantize_cost(deployment_paid))
