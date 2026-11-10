"""Web search in the OpenAI Responses server-tool vocabulary."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import ActionSearch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend
    from gateway.services.tools._native import NativeCall

# The gateway's own ``web_search_call`` item ids. OpenAI issues ``ws_`` ids, so a
# reserved prefix is what lets an echoed item be told apart from one describing a
# search OpenAI's own tool ran (see ``routes/responses.py``).
WEB_SEARCH_CALL_ID_PREFIX = "otari_ws_"


def _field(value: Any, name: str) -> Any:
    """``value[name]`` or ``value.name``: an extra arrives as a dict, a declared field as a model."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _reported_searches(usage: Any) -> int | None:
    """``usage.tool_usage.web_search.num_requests``, or ``None`` where it is missing or not a count."""
    extras = getattr(usage, "model_extra", None) or {}
    count = _field(_field(extras.get("tool_usage"), "web_search"), "num_requests")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return count


def provider_web_searches(response: Any) -> int:
    """The searches the provider ran itself during ``response``, which it bills per call.

    OpenAI reports them as ``usage.tool_usage.web_search.num_requests``, an
    extra its SDK does not declare. Where a response reports none, each
    ``web_search_call`` item it returned counts as one, except the gateway's own.
    """
    reported = _reported_searches(getattr(response, "usage", None))
    if reported is not None:
        return reported
    return sum(
        1
        for item in getattr(response, "output", None) or []
        if _field(item, "type") == "web_search_call"
        and not str(_field(item, "id") or "").startswith(WEB_SEARCH_CALL_ID_PREFIX)
    )


def with_provider_web_searches(usage: Any, searches: int) -> Any:
    """A copy of ``usage`` that reports ``searches`` as ``tool_usage.web_search.num_requests``."""
    tool_usage = dict(_field(getattr(usage, "model_extra", None) or {}, "tool_usage") or {})
    tool_usage["web_search"] = {**(tool_usage.get("web_search") or {}), "num_requests": searches}
    return usage.model_copy(update={"tool_usage": tool_usage})


class ResponsesWebSearchRendering:
    """Gateway-run searches as ``web_search_call`` output items.

    This is the one place the gateway's own tool work is expressible in a provider's
    native vocabulary without forging provider-signed content: the item needs only an
    id, an action and a status, all of which the gateway legitimately knows. The
    Anthropic equivalent needs a signed ``encrypted_content`` blob (see docs/tools.md).
    """

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Every caller: the item forges nothing, so any of them can be told the search ran."""
        del tool_entry
        return True

    def started(self, call: NativeCall) -> list[Any]:
        """Nothing: the item is announced once the search has ended, with its status."""
        return []

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The item for one search, ``failed`` where its backend errored and ``completed`` otherwise.

        ``pool`` carries no part of the item: unlike the Messages rendering, this
        vocabulary reports that a search happened rather than what it found, so a
        search that returned no hits is still completed.
        """
        return [
            ResponseFunctionWebSearch(
                id=f"{WEB_SEARCH_CALL_ID_PREFIX}{uuid.uuid4().hex}",
                action=ActionSearch(type="search", query=str(call.arguments.get("query") or "")),
                status="failed" if call.failed else "completed",
                type="web_search_call",
            )
        ]

    def refused(self, call: NativeCall) -> list[Any]:
        """Nothing: no search ran, so there is nothing to announce."""
        return []


RENDERING = ResponsesWebSearchRendering()
