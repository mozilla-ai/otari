"""Web search in Chat Completions' ``otari_tool_calls`` field (decision 67)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from gateway.schemas.tools import OtariToolCallError, OtariToolCallSource
from gateway.services.tools._chat_tool_calls import call_limit_error, chat_entry
from gateway.services.tools._web_display import hits_to_show, search_query, take_last_hits
from gateway.services.tools._web_names import WEB_RETRIEVAL_RESULT_MAX_BYTES
from gateway.services.tools._web_search_responses import WEB_SEARCH_CALL_ID_PREFIX

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from any_search import SearchHit
    from gateway.services._tool_loop import ToolBackend
    from gateway.services.tools._native import NativeCall


class ChatWebSearchRendering:
    """Gateway-run searches as ``otari_tool_calls`` entries, one when a search starts and one when it ends.

    Both carry the same ``id``, in the Responses rendering's ``otari_ws_`` form, so a
    client replaces the started entry with the ended one.
    """

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Every caller: a client that does not know the field ignores it."""
        del tool_entry
        return True

    def started(self, call: NativeCall) -> list[Any]:
        """The entry for a search about to run, with its query."""
        return [_entry(call, "in_progress")]

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The ended entry: ``failed``, or ``completed`` with the hits the backend kept.

        A failed search's ``error`` is the failure the search backend kept: the
        provider, its status, the tag and whether a retry may help. One that failed
        on the old backend carries none: that backend raises or returns only a text,
        and the text is the model's, not the caller's.
        """
        if call.failed:
            take_last_failure = getattr(pool, "take_last_failure", None)
            failure = take_last_failure() if take_last_failure is not None else None
            error = OtariToolCallError.model_validate(failure.as_dict()) if failure is not None else None
            return [_entry(call, "failed", error=error)]
        return [_entry(call, "completed", sources=_sources(take_last_hits(pool) or []))]

    def refused(self, call: NativeCall) -> list[Any]:
        """The ended entry for a search the use cap refused before it ran."""
        return [_entry(call, "failed", error=call_limit_error())]


def _entry(
    call: NativeCall,
    status: Literal["in_progress", "completed", "failed"],
    *,
    sources: list[OtariToolCallSource] | None = None,
    error: OtariToolCallError | None = None,
) -> dict[str, Any]:
    return chat_entry(
        call,
        id_prefix=WEB_SEARCH_CALL_ID_PREFIX,
        kind="web_search",
        status=status,
        query=search_query(call.arguments),
        sources=sources,
        error=error,
    )


def _sources(results: Iterable[SearchHit | Mapping[str, Any]]) -> list[OtariToolCallSource]:
    """Each hit's URL, title and date, never its text, within the byte limit of the model-facing result.

    The title and date are capped already; the limit is the backstop for the URL,
    which nothing else bounds. A source that would cross it is left out.
    """
    sources: list[OtariToolCallSource] = []
    size = 0
    for url, title, date in hits_to_show(results):
        source = OtariToolCallSource(url=url, title=title, published_date=date)
        source_size = len(source.model_dump_json().encode())
        if size + source_size > WEB_RETRIEVAL_RESULT_MAX_BYTES:
            continue
        size += source_size
        sources.append(source)
    return sources


RENDERING = ChatWebSearchRendering()
