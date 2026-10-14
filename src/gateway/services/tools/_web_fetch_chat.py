"""Web fetch in Chat Completions' ``otari_tool_calls`` field (decision 67)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from gateway.schemas.tools import OtariToolCallError
from gateway.services.tools._chat_tool_calls import call_limit_error, chat_entry
from gateway.services.tools._web_display import fetch_display_url

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend
    from gateway.services.tools._native import NativeCall

# The gateway's own fetch entry ids, beside the ``otari_ws_`` ids of its searches.
WEB_FETCH_CALL_ID_PREFIX = "otari_wf_"


class ChatWebFetchRendering:
    """Gateway-run fetches as ``otari_tool_calls`` entries, one when a fetch starts and one when it ends.

    An entry names the URL the model asked for, without its query string and fragment,
    as the Messages fetch blocks show it, and never carries the page.
    """

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Every caller: a client that does not know the field ignores it."""
        del tool_entry
        return True

    def started(self, call: NativeCall) -> list[Any]:
        """The entry for a fetch about to run, with its URL."""
        return [_entry(call, "in_progress")]

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The ended entry, ``failed`` or ``completed``.

        ``pool`` carries no part of it. A fetch that failed on the old backend carries
        no ``error``: that backend returns only a text, and the text is the model's.
        """
        del pool
        return [_entry(call, "failed" if call.failed else "completed")]

    def refused(self, call: NativeCall) -> list[Any]:
        """The ended entry for a fetch the use cap refused before it ran."""
        return [_entry(call, "failed", error=call_limit_error())]


def _entry(
    call: NativeCall,
    status: Literal["in_progress", "completed", "failed"],
    *,
    error: OtariToolCallError | None = None,
) -> dict[str, Any]:
    return chat_entry(
        call,
        id_prefix=WEB_FETCH_CALL_ID_PREFIX,
        kind="web_fetch",
        status=status,
        url=fetch_display_url(call.arguments.get("url")),
        error=error,
    )


RENDERING = ChatWebFetchRendering()
