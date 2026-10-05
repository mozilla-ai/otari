"""The ``tools[].type`` values the gateway runs itself, and how one is taken out of a request's tools."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum, auto
from typing import Any


class Tool(StrEnum):
    """Gateway-managed tool types: the only ``type`` values the gateway runs
    itself (everything else is forwarded to the upstream provider).

    Values are derived as ``otari_<member>`` so every gateway tool carries the
    ``otari_`` prefix by construction; registering a new gateway-run tool is a
    one-line addition here.
    """

    @staticmethod
    def _generate_next_value_(name: str, start: int, count: int, last_values: list[Any]) -> str:
        return f"otari_{name.lower()}"

    CODE_EXECUTION = auto()  # -> "otari_code_execution"
    WEB_FETCH = auto()  # -> "otari_web_fetch"
    WEB_SEARCH = auto()  # -> "otari_web_search"


def extract_first_matching_tool(
    tools: list[dict[str, Any]] | None,
    predicate: Callable[[Any], bool],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Pull the first tool entry whose ``type`` matches ``predicate``.

    Returns ``(entry_or_None, remaining_tools_or_None)``. The extracted entry
    is thin (no function schema); the gateway-managed backend's
    ``openai_tools`` provides the full definition during tool-use-loop
    injection. Remaining user-supplied tools pass through unchanged.
    """
    if not tools:
        return None, tools
    entry: dict[str, Any] | None = None
    remaining: list[dict[str, Any]] = []
    for t in tools:
        if entry is None and isinstance(t, dict) and predicate(t.get("type")):
            entry = t
        else:
            remaining.append(t)
    return entry, (remaining or None)
