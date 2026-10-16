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
from gateway.services.tools._web_display import hits_to_show, search_query
from gateway.services.tools._web_names import (
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
    WEB_SEARCH_NATIVE_TYPE_PREFIX,
    WEB_SEARCH_TOOL_NAME,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gateway.services._tool_loop import ToolBackend


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

    The capped title (``hits_to_show``) is what keeps a normal result set well under
    the limit, since the backend bounds the hit count and ``page_age`` is already
    capped. The size check is the backstop for the remaining unbounded field, the URL.
    """
    candidate = WebSearchResultBlock(
        type="web_search_result",
        url=url,
        title=title,
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

    def started(self, call: NativeCall) -> list[Any]:
        """Nothing: the pair describes a search once it has ended."""
        return []

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
        for url, title, page_age in hits_to_show(take_last_results()):
            _append_bounded_citation(citations, tool_use_id=tool_use_id, url=url, title=title, page_age=page_age)
        return [_server_tool_use(tool_use_id, search_query(call.arguments)), _result_block(tool_use_id, citations)]

    def refused(self, call: NativeCall) -> list[Any]:
        """The pair for a search the use cap refused, in the vocabulary's own error shape."""
        tool_use_id = f"{SERVER_TOOL_USE_ID_PREFIX}{uuid.uuid4().hex}"
        return [
            _server_tool_use(tool_use_id, search_query(call.arguments)),
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
