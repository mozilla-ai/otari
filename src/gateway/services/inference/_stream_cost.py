"""Which streamed Chat Completions chunk carries the request's settled cost."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletionChunk


def _is_empty_choice(choice: Any) -> bool:
    """A choice that carries nothing: no finish reason and no content, tool call or refusal in its delta.

    Bedrock's adapter puts the usage on such a choice: a chunk after the finish chunk whose only
    choice has ``delta: {"role": "assistant"}`` and ``finish_reason: null``.
    """
    if choice.finish_reason is not None:
        return False
    delta = getattr(choice, "delta", None)
    if delta is None:
        return True
    return not any(
        getattr(delta, field, None)
        for field in ("content", "tool_calls", "function_call", "refusal", "reasoning", "reasoning_content")
    )


def is_usage_only_chunk(chunk: ChatCompletionChunk) -> bool:
    """Whether ``chunk`` reports usage and nothing else.

    OpenAI's ``stream_options.include_usage`` chunk has no choices; any-llm's Bedrock adapter
    sends the same report on a chunk whose choices are all empty (see ``_is_empty_choice``).
    Both are the stream's usage report, not content.
    """
    if chunk.usage is None:
        return False
    return not chunk.choices or all(_is_empty_choice(choice) for choice in chunk.choices)


def is_chat_stream_cost_carrier(chunk: ChatCompletionChunk) -> bool:
    """Whether ``chunk`` is a terminal usage chunk, the one ``cost_usd`` is stamped on.

    Providers end a stream's usage in one of three shapes: OpenAI sends a chunk of
    its own with usage and no choices, Bedrock (through any-llm) sends usage on a
    chunk whose only choice is empty, while others (Cohere, Gemini) report usage
    on the chunk whose choice carries ``finish_reason``. A content chunk that
    reports running usage is none of these, so it never becomes the carrier and holds
    the rest of the stream behind it.
    """
    if chunk.usage is None:
        return False
    return is_usage_only_chunk(chunk) or any(choice.finish_reason is not None for choice in chunk.choices)
