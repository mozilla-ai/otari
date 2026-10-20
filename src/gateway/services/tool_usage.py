"""Per-request accounting for tools the gateway runs itself.

A gateway-run tool call (``otari_web_search`` → the tools domain's ``WebSearchBackend``, or the older
:class:`~gateway.services.web_retrieval_backend.WebRetrievalBackend`,
``otari_code_execution`` → :class:`~gateway.services.sandbox_backend.SandboxBackend`,
or any MCP tool) costs the operator real money at a search provider, a sandbox,
or an MCP server, and until now none of it was recorded: the request's usage row
carried tokens only. :class:`ToolUsageTally` is the one place those calls are
counted, so settlement can price them onto the row that caused them.

One tally belongs to one request, never to a backend::

      route ──▶ ToolUsageTally() ──▶ backend(tally=…) ──▶ call_tool() ──▶ record
                      │
                      └──▶ settlement reads meters() ──▶ billing_meters["tools"]

Ownership matters because the streaming path tears the backend down while the
response is still being settled: ``_eager_backend_stream``'s ``finally`` runs as
the stream is exhausted, and ``streaming_generator`` only awaits ``on_complete``
afterwards. A tally stashed on the backend would be read after teardown.

Billable vs failed is decided on the ``[tool error]`` sentinel rather than on an
exception, because the backends disagree about which they use: ``MCPClientPool``,
``SandboxBackend`` and the web search and fetch backends on the libraries return the
sentinel as a normal value, while the older ``WebRetrievalBackend`` returns it for an
empty query but raises ``WebSearchNotReachableError`` when its search backend is
unreachable. A failed call is counted and never billed.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from gateway.core.metered_pricing import to_decimal
from gateway.models.money import MAX_USD_COST
from gateway.types.tool_charges import PROVIDER_WEB_SEARCH_TOOL, LabelCalls, provider_web_search_label

# Every tool backend marks a recoverable failure by prefixing the string it hands
# back to the model. The tool loop's own error path uses the same prefix, so this
# is the single vocabulary for "the call ran but did not work".
TOOL_ERROR_SENTINEL = "[tool error]"

# The meter namespace inside ``UsageLog.billing_meters``. Tool meters are nested
# under one reserved key rather than sitting flat next to the token meters:
# MCP tool names come from a caller-supplied server, and a tool named
# ``completion_tokens`` sitting flat would be picked up by the billed-token SQL
# in ``routes/usage.py`` and corrupt the aggregates for the whole window.
TOOL_METER_NAMESPACE = "tools"

# Bounds on what a caller-influenced name can write into the row. An MCP server
# can advertise any number of tools with any names; the row is JSON, not a
# schema, so the caps are enforced here.
MAX_TOOL_NAMES = 32
MAX_TOOL_NAME_CHARS = 64
# Reserved rather than a plausible tool name: MCP tool names come from a
# caller-supplied server, and a server with a tool literally named "other" would
# otherwise have its counts merged into the overflow bucket.
OVERFLOW_TOOL_NAME = "_other"


def is_tool_error(result: str) -> bool:
    """True when a tool's return value is the recoverable-failure sentinel."""
    return result.startswith(TOOL_ERROR_SENTINEL)


def _usable_cost(value: Decimal | None) -> Decimal | None:
    cost = to_decimal(value)
    return cost if cost is not None and cost <= MAX_USD_COST else None


@dataclass(slots=True)
class _LabelTotals:
    calls: int = 0
    unreported_calls: int = 0
    reported_cost: Decimal = Decimal(0)
    enrichment_cost: Decimal = Decimal(0)


class ToolUsageTally:
    """Counts one request's gateway-run tool calls, split billable vs failed.

    Accumulates across every attempt of a request. A streaming request that
    falls back to a second provider re-runs its tool calls, and both runs cost
    money, so both are counted: the bill follows the work, not the winner.
    """

    __slots__ = ("_counts", "_labels", "_overflowed")

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._labels: dict[tuple[str, str | None, tuple[str, ...] | None, bool | None], _LabelTotals] = {}
        self._overflowed = False

    def record_result(
        self,
        tool: str,
        text: str,
        *,
        label: str | None = None,
        rate_keys: tuple[str, ...] | None = None,
        cost: Decimal | None = None,
        enrichment_cost: Decimal | None = None,
    ) -> None:
        """Record a call that returned, classifying it on the sentinel.

        A successful call is kept under its label with its rate keys (see
        :class:`~gateway.types.tool_charges.LabelCalls`), and with the cost its
        provider reported, where ``None`` means it reported none and a rate
        prices the call. A failed call's costs are dropped, since it is never
        billed. A cost that is negative, not finite, or more than a cost column
        holds is unusable: the call reads as unreported, and an enrichment cost
        as none.
        """
        failed = is_tool_error(text)
        key = self._record(tool, failed=failed)
        if failed:
            return
        totals = self._labels.setdefault((key, label, rate_keys, None), _LabelTotals())
        totals.calls += 1
        reported = _usable_cost(cost)
        if reported is None:
            totals.unreported_calls += 1
        else:
            totals.reported_cost += reported
        totals.enrichment_cost += _usable_cost(enrichment_cost) or Decimal(0)

    def record_provider_searches(self, provider: str, searches: int, *, deployment_paid: bool) -> None:
        """Record searches a model's provider ran itself and bills per search, as Anthropic's search does.

        They are counted under ``web_search`` beside the gateway's own searches and
        priced under the ``<provider>:web_search`` label, at an operator's rate or,
        with default pricing, the provider's list price. ``deployment_paid`` is the
        model call's own, so whose overrides price them follows the credential that
        ran the call, as it does for the call's tokens.
        """
        if searches <= 0:
            return
        key = self._key(PROVIDER_WEB_SEARCH_TOOL)
        self._counts.setdefault(key, {"billed": 0, "errors": 0})["billed"] += searches
        label = provider_web_search_label(provider)
        totals = self._labels.setdefault((key, label, (label,), deployment_paid), _LabelTotals())
        totals.calls += searches
        totals.unreported_calls += searches

    def record_failure(self, tool: str) -> None:
        """Record a call that raised. Counted, never billed."""
        self._record(tool, failed=True)

    def _record(self, tool: str, *, failed: bool) -> str:
        key = self._key(tool)
        entry = self._counts.setdefault(key, {"billed": 0, "errors": 0})
        entry["errors" if failed else "billed"] += 1
        return key

    def _key(self, tool: str) -> str:
        name = (tool or OVERFLOW_TOOL_NAME)[:MAX_TOOL_NAME_CHARS]
        if name in self._counts:
            return name
        if len(self._counts) >= MAX_TOOL_NAMES:
            self._overflowed = True
            return OVERFLOW_TOOL_NAME
        return name

    @property
    def overflowed(self) -> bool:
        """True when distinct tool names hit the cap and folded into ``other``."""
        return self._overflowed

    def is_empty(self) -> bool:
        return not self._counts

    def label_calls(self) -> list[LabelCalls]:
        """Successful calls per tool and label, the units settlement prices."""
        return [
            LabelCalls(
                tool=tool,
                label=label,
                rate_keys=rate_keys,
                calls=totals.calls,
                unreported_calls=totals.unreported_calls,
                reported_cost=totals.reported_cost,
                enrichment_cost=totals.enrichment_cost,
                deployment_paid=deployment_paid,
            )
            for (tool, label, rate_keys, deployment_paid), totals in self._labels.items()
        ]

    def meters(self) -> dict[str, dict[str, float]]:
        """The nested meter payload, sorted so a row's JSON is stable.

        Settlement adds a ``unit_rate`` per tool after pricing, which is why the
        value type is not int-only.
        """
        return {tool: dict(self._counts[tool]) for tool in sorted(self._counts)}
