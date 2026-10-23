"""Answer a load test's ``mock_response`` without calling the provider."""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
)

from gateway.models.api_keys import APIKey


def mock_response_allowed(api_key: APIKey | None) -> bool:
    """Whether a request may carry ``mock_response``: only on an API key an operator turned it on for."""
    return api_key is not None and api_key.allow_mock_response


def mock_completion(kwargs: dict[str, Any], text: str) -> ChatCompletion:
    """The completion a request's ``mock_response`` stands for, with no provider called.

    Usage is estimated at four characters a token, as the budget hold estimates a
    prompt, so the request is priced and settled like any other.
    """
    prompt_chars = sum(len(str(message.get("content") or "")) for message in kwargs.get("messages") or [])
    prompt_tokens = (prompt_chars + 3) // 4
    completion_tokens = (len(text) + 3) // 4
    return ChatCompletion(
        id=f"chatcmpl-mock-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=str(kwargs.get("model") or ""),
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content=text), finish_reason="stop")],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def mock_stream(completion: ChatCompletion) -> AsyncIterator[ChatCompletionChunk]:
    """``completion`` as a stream: its text in one chunk, then the usage chunk ``include_usage`` asks for."""
    content = completion.choices[0].message.content
    yield ChatCompletionChunk(
        id=completion.id,
        object="chat.completion.chunk",
        created=completion.created,
        model=completion.model,
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content=content), finish_reason="stop")],
    )
    yield ChatCompletionChunk(
        id=completion.id,
        object="chat.completion.chunk",
        created=completion.created,
        model=completion.model,
        choices=[],
        usage=completion.usage,
    )
