"""Web fetch in the Anthropic Messages server-tool vocabulary."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from anthropic.types import (
    DocumentBlock,
    PlainTextSource,
    ServerToolUseBlock,
    WebFetchBlock,
    WebFetchToolResultBlock,
    WebFetchToolResultErrorBlock,
)

from gateway.services.tools._native import SERVER_TOOL_USE_ID_PREFIX, NativeCall
from gateway.services.tools._web_display import fetch_display_url
from gateway.services.tools._web_names import (
    ANTHROPIC_WEB_FETCH_TYPE,
    WEB_FETCH_TOOL_NAME,
    WEB_RETRIEVAL_RESULT_MAX_BYTES,
)
from gateway.services.web_retrieval_network import truncate_utf8

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anthropic.types import WebFetchToolResultErrorCode

    from gateway.services._tool_loop import ToolBackend


def _pair(call: NativeCall, content: WebFetchBlock | WebFetchToolResultErrorBlock) -> list[Any]:
    tool_use_id = f"{SERVER_TOOL_USE_ID_PREFIX}{uuid.uuid4().hex}"
    return [
        ServerToolUseBlock(
            id=tool_use_id,
            # Narrowed for the same reason as the web-search rendering's name.
            name=cast('Literal["web_fetch"]', WEB_FETCH_TOOL_NAME),
            input={"url": fetch_display_url(call.arguments.get("url"))},
            type="server_tool_use",
        ),
        WebFetchToolResultBlock(tool_use_id=tool_use_id, type="web_fetch_tool_result", content=content),
    ]


def _error(code: WebFetchToolResultErrorCode) -> WebFetchToolResultErrorBlock:
    return WebFetchToolResultErrorBlock(type="web_fetch_tool_result_error", error_code=code)


class MessagesWebFetchRendering:
    """Gateway-run fetches as ``server_tool_use`` / ``web_fetch_tool_result`` pairs."""

    def declared(self, tool_entry: Mapping[str, Any] | None) -> bool:
        """Whether the caller asked in Anthropic's dated keyword, which is what asks for the pair."""
        type_value = (tool_entry or {}).get("type")
        return isinstance(type_value, str) and ANTHROPIC_WEB_FETCH_TYPE.fullmatch(type_value) is not None

    def started(self, call: NativeCall) -> list[Any]:
        """Nothing: the pair describes a fetch once it has ended."""
        return []

    def ran(self, call: NativeCall, pool: ToolBackend) -> list[Any]:
        """The pair for one fetch, in the vocabulary's error shape where it failed.

        The page comes from the buffer only the gateway's own retrieval backend keeps,
        which is what tells a gateway-run fetch apart from an MCP server's tool of the
        same name. The model was told about a failure, so the client sees it too.
        """
        take_last_fetch = getattr(pool, "take_last_fetch", None)
        if take_last_fetch is None:
            return []
        fetched = take_last_fetch()
        if call.failed or fetched is None:
            return _pair(call, _error("unavailable"))
        document = DocumentBlock(
            type="document",
            source=PlainTextSource(
                type="text",
                media_type="text/plain",
                data=truncate_utf8(fetched.text, WEB_RETRIEVAL_RESULT_MAX_BYTES, suffix="…").text,
            ),
        )
        return _pair(
            call,
            WebFetchBlock(
                type="web_fetch_result",
                url=fetched.final_url.display_url,
                content=document,
                retrieved_at=datetime.now(UTC).isoformat(),
            ),
        )

    def refused(self, call: NativeCall) -> list[Any]:
        """The pair for a fetch the use cap refused, in the vocabulary's own error shape."""
        return _pair(call, _error("max_uses_exceeded"))


RENDERING = MessagesWebFetchRendering()
