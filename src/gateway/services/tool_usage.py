"""Per-request accounting for tools the gateway runs itself.

A gateway-run tool call (``otari_web_search`` → :class:`~gateway.services.web_search_backend.WebSearchBackend`,
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
exception, because the backends disagree about which they use: ``MCPClientPool``
and ``SandboxBackend`` return the sentinel as a normal value, ``WebSearchBackend``
returns it for an empty query but raises ``WebSearchNotReachableError`` when the
backend is unreachable. A failed call is counted and never billed.
"""

from __future__ import annotations

from typing import Any

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


class ToolUsageTally:
    """Counts one request's gateway-run tool calls, split billable vs failed.

    Accumulates across every attempt of a request. A streaming request that
    falls back to a second provider re-runs its tool calls, and both runs cost
    money, so both are counted: the bill follows the work, not the winner.
    """

    __slots__ = ("_counts", "_overflowed")

    def __init__(self) -> None:
        self._counts: dict[str, dict[str, int]] = {}
        self._overflowed = False

    def record_result(self, tool: str, result: str) -> None:
        """Record a call that returned, classifying it on the sentinel."""
        self._record(tool, failed=is_tool_error(result))

    def record_failure(self, tool: str) -> None:
        """Record a call that raised. Counted, never billed."""
        self._record(tool, failed=True)

    def _record(self, tool: str, *, failed: bool) -> None:
        key = self._key(tool)
        entry = self._counts.setdefault(key, {"billed": 0, "errors": 0})
        entry["errors" if failed else "billed"] += 1

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

    def billable_calls(self) -> dict[str, int]:
        """Successful calls per tool, the units a charge line is priced on."""
        return {tool: counts["billed"] for tool, counts in self._counts.items() if counts["billed"]}

    def meters(self) -> dict[str, dict[str, float]]:
        """The nested meter payload, sorted so a row's JSON is stable.

        Settlement adds a ``unit_rate`` per tool after pricing, which is why the
        value type is not int-only.
        """
        return {tool: dict(self._counts[tool]) for tool in sorted(self._counts)}


def tool_meters_of(billing_meters: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    """Read the tool meters back off a stored row, tolerating older shapes.

    Rows written before tool metering carry no namespace, and a hand-edited row
    could carry anything, so a non-dict is treated as absent rather than trusted.
    """
    if not isinstance(billing_meters, dict):
        return {}
    nested = billing_meters.get(TOOL_METER_NAMESPACE)
    if not isinstance(nested, dict):
        return {}
    return {tool: counts for tool, counts in nested.items() if isinstance(counts, dict) and isinstance(tool, str)}
