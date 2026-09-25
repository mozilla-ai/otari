"""Web search in the Anthropic Messages server-tool vocabulary."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, cast

from anthropic.types import (
    ServerToolUseBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
    WebSearchToolResultError,
)

from gateway.services.tools._native import SERVER_TOOL_USE_ID_PREFIX, NativeCall
from gateway.services.web_retrieval_backend import (
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    WEB_SEARCH_NATIVE_TYPE_PREFIX,
    WEB_SEARCH_TOOL_NAME,
)
from gateway.services.web_retrieval_network import truncate_utf8

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend

# A search result's recency, collapsed to one line and length-capped before it goes
# on a native block. Same rendering hygiene the text formatter applies: the value is
# whatever a search-API-fronting adapter forwarded, so one overlong or multiline
# entry shouldn't be what makes a citations panel unreadable.
_PAGE_AGE_MAX_CHARS = 128

# Backend-supplied titles are unbounded. Generous for a real page title, and small
# enough that a full result set of them cannot approach the tool-result byte limit
# on its own.
_CITATION_TITLE_MAX_BYTES = 512


def _result_block(tool_use_id: str, citations: list[WebSearchResultBlock]) -> WebSearchToolResultBlock:
    return WebSearchToolResultBlock(
        tool_use_id=tool_use_id,
        type="web_search_tool_result",
        content=citations,
    )


def _result_size(tool_use_id: str, citations: list[WebSearchResultBlock]) -> int:
    block = _result_block(tool_use_id, citations)
    return len(block.model_dump_json(exclude_none=True).encode("utf-8"))


def _append_bounded_citation(
    citations: list[WebSearchResultBlock],
    *,
    tool_use_id: str,
    url: str,
    title: str,
    page_age: str | None,
) -> None:
    """Append one citation, keeping the whole result block inside its byte limit.

    Capping the title is what keeps a normal result set well under the limit, since
    the backend bounds the hit count and ``page_age`` is already capped. The size
    check is the backstop for the remaining unbounded field, the URL.
    """
    candidate = WebSearchResultBlock(
        type="web_search_result",
        url=url,
        title=truncate_utf8(title, _CITATION_TITLE_MAX_BYTES, suffix="…").text,
        page_age=page_age,
        encrypted_content="",
    )
    if _result_size(tool_use_id, [*citations, candidate]) > WEB_RETRIEVAL_RESULT_MAX_BYTES:
        return
    citations.append(candidate)


def _server_tool_use(tool_use_id: str, query: str) -> ServerToolUseBlock:
    return ServerToolUseBlock(
        id=tool_use_id,
        # Anthropic types this field as a Literal of its own server-tool names. The
        # gateway's tool name is one of them, but the shared constant is a plain
        # ``str``, so narrow it here rather than duplicating the literal.
        name=cast('Literal["web_search"]', WEB_SEARCH_TOOL_NAME),
        input={"query": query},
        type="server_tool_use",
    )


def _query_of(call: NativeCall) -> str:
    return str(call.arguments.get("query") or "")


class MessagesWebSearchRendering:
    """Gateway-run searches as ``server_tool_use`` / ``web_search_tool_result`` pairs.

    ``encrypted_content`` is required by the schema but is an Anthropic-signed blob only
    Anthropic can mint, so the gateway sends it empty rather than forging one. A block
    echoed straight back to Anthropic is rejected there, which is the trade-off an
    unsigned block accepts.
    """

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Whether the caller asked in Anthropic's own words, which is what asks for the pair.

        A dated or preview keyword is what the Anthropic SDK, Claude Code and Claude
        Desktop send, and it is what makes them expect these blocks and render citations
        from them. ``otari_web_search`` and the bare ``web_search`` short form imply no
        response shape, so a caller using one is owed only the plain-text result.
        """
        type_value = (tool_entry or {}).get("type")
        return isinstance(type_value, str) and type_value.startswith(WEB_SEARCH_NATIVE_TYPE_PREFIX)

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The pair for one completed search, empty for a search that failed.

        A failed search has nothing to cite. The hits come from the buffer only the
        gateway's own retrieval backend keeps, which is what tells a gateway-run
        search apart from an MCP server's tool of the same name.
        """
        take_last_results = getattr(pool, "take_last_results", None)
        if call.failed or take_last_results is None:
            return []
        tool_use_id = f"{SERVER_TOOL_USE_ID_PREFIX}{uuid.uuid4().hex}"
        citations: list[WebSearchResultBlock] = []
        for result in take_last_results():
            url = str(result.get("url") or "").strip()
            if not url:
                # Nothing to cite. A hit with no URL is unusable to a citations panel.
                continue
            page_age = " ".join(str(result.get("published_date") or "").split())[:_PAGE_AGE_MAX_CHARS]
            _append_bounded_citation(
                citations,
                tool_use_id=tool_use_id,
                url=url,
                title=str(result.get("title") or url).strip(),
                page_age=page_age or None,
            )
        return [_server_tool_use(tool_use_id, _query_of(call)), _result_block(tool_use_id, citations)]

    def refused(self, call: NativeCall) -> list[Any]:
        """The pair for a search the use cap refused, in the vocabulary's own error shape."""
        tool_use_id = f"{SERVER_TOOL_USE_ID_PREFIX}{uuid.uuid4().hex}"
        return [
            _server_tool_use(tool_use_id, _query_of(call)),
            WebSearchToolResultBlock(
                tool_use_id=tool_use_id,
                type="web_search_tool_result",
                content=WebSearchToolResultError(
                    type="web_search_tool_result_error",
                    error_code="max_uses_exceeded",
                ),
            ),
        ]


RENDERING = MessagesWebSearchRendering()
