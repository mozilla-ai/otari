"""Web search in the OpenAI Responses server-tool vocabulary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseFunctionWebSearch
from openai.types.responses.response_function_web_search import ActionSearch

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend
    from gateway.services.tools._native import NativeCall


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

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The item for one search, whether or not it returned hits.

        ``pool`` carries no part of the item: unlike the Messages rendering, this
        vocabulary reports that a search happened rather than what it found.
        ``call.failed`` is not read either, so a search whose backend errored is still
        announced as completed. The model is told about the failure in the call's own
        output, and reporting it here as well needs an outcome the non-streaming caller
        does not hold.
        """
        return [
            ResponseFunctionWebSearch(
                id=call.id,
                action=ActionSearch(type="search", query=str(call.arguments.get("query") or "")),
                status="completed",
                type="web_search_call",
            )
        ]

    def refused(self, call: NativeCall) -> list[Any]:
        """Nothing: no search ran, so there is nothing to announce."""
        return []


RENDERING = ResponsesWebSearchRendering()
