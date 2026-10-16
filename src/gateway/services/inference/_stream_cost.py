"""Which streamed Chat Completions chunk carries the request's settled cost."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletionChunk


def is_chat_stream_cost_carrier(chunk: ChatCompletionChunk) -> bool:
    """Whether ``chunk`` is a terminal usage chunk, the one ``cost_usd`` is stamped on.

    Providers end a stream's usage in one of two shapes: OpenAI sends a chunk of
    its own with usage and no choices, while others (Cohere, Gemini) report usage
    on the chunk whose choice carries ``finish_reason``. A content chunk that
    reports running usage is neither, so it never becomes the carrier and holds
    the rest of the stream behind it.
    """
    if chunk.usage is None:
        return False
    return not chunk.choices or any(choice.finish_reason is not None for choice in chunk.choices)
