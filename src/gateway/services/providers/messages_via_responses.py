"""Serve a Messages request through the Responses API when Chat Completions refuses it.

any-llm bridges a Messages request to every provider without a Messages API of
its own through Chat Completions. OpenAI's Chat Completions refuses function
tools combined with a reasoning effort on some reasoning models, and its 400
names ``/v1/responses`` as the endpoint that serves the combination.
:func:`amessages_with_responses_fallback` reacts to exactly that refusal by
sending the same request to the Responses API and translating the answer back,
so the caller keeps both the tools and the reasoning effort they asked for.

It reacts to the refusal rather than predicting it: OpenAI exposes no
capability flag for the combination, and a list of model names would go stale
with every release (mozilla-ai/any-llm#1189). The cost is one rejected round
trip, which bills no tokens, on each request that needs it.

The request side reuses any-llm's own Messages-to-Chat conversion and then
maps Chat messages onto Responses input items, so a request reaches Responses
with the same messages, tools and effort Chat Completions would have seen. A
request carrying anything that mapping does not cover (``stop_sequences``,
structured output, a content part other than text or an image, a tool
result or assistant turn holding anything but text) keeps its original error
rather than being sent with a field quietly missing. Replayed ``thinking`` text
is not sent: Responses takes prior reasoning only as the opaque items it
issued, which a Messages transcript does not hold. No reasoning summary is
requested, so the answer carries no ``thinking`` block, which is what the Chat
Completions path returns for these models too.

Stopgap: which provider API serves a bridged Messages request is any-llm's to
decide. Remove this once any-llm serves these OpenAI requests through
Responses and the SDK pin moves (otari#1630).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from any_llm import AnyLLM, aresponses
from any_llm.exceptions import AnyLLMError
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    InputJSONDelta,
    MessageContentBlock,
    MessageDelta,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessagesParams,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
    StopReason,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from any_llm.utils.aio import aclose_quietly
from any_llm.utils.messages_compat import messages_params_to_completion_params, split_cached_input_tokens
from openai import APIStatusError

from gateway.log_config import logger

# The Chat Completions fields a translated request may carry. Anything else the
# any-llm bridge emits has no mapping below.
_TRANSLATABLE_FIELDS = frozenset(
    {
        "model_id",
        "messages",
        "max_tokens",
        "reasoning_effort",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "prompt_cache_key",
        "service_tier",
        "stream",
        "stream_options",
    }
)
# Forwarded to Responses under the same name.
_SAME_NAME_FIELDS = ("parallel_tool_calls", "temperature", "top_p", "prompt_cache_key", "service_tier", "stream")


class ResponsesStreamFailedError(Exception):
    """The Responses stream reported a failure after it opened."""

    def __init__(self) -> None:
        super().__init__("The Responses API stream failed")


async def amessages_with_responses_fallback(
    call: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
) -> Any:
    """Return ``call(**kwargs)``, or the Responses API's answer when Chat Completions refused the request.

    ``call`` is the caller's ``amessages``, passed in so a test that patches the
    caller's module global still intercepts the first attempt.
    """
    try:
        return await call(**kwargs)
    except (AnyLLMError, APIStatusError) as exc:
        request = _responses_request(exc, kwargs)
        if request is None:
            raise
        logger.info(
            "Chat Completions refused function tools with a reasoning effort for %s; retrying through Responses",
            kwargs.get("model"),
        )
        result = await aresponses(**request)
        if request.get("stream"):
            return _responses_stream_to_messages(result)
        return _response_to_message(result)


def _is_responses_only_rejection(exc: BaseException) -> bool:
    """Whether ``exc`` is OpenAI refusing tools with ``reasoning_effort`` in favor of Responses.

    Walks ``original_exception`` so the check holds whether any-llm hands back the
    raw SDK error or its unified wrapper, which carries the same fields.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = getattr(current, "message", None)
        if (
            getattr(current, "status_code", None) == 400
            and getattr(current, "param", None) == "reasoning_effort"
            and isinstance(message, str)
            and "/v1/responses" in message
        ):
            return True
        current = getattr(current, "original_exception", None)
    return False


def _serves_responses(model: Any) -> bool:
    if not isinstance(model, str):
        return False
    try:
        provider, _ = AnyLLM.split_model_provider(model)
        return bool(AnyLLM.get_provider_class(provider).SUPPORTS_RESPONSES)
    except (ValueError, ImportError, AnyLLMError):
        return False


def _responses_request(exc: BaseException, kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """The ``aresponses`` kwargs for the refused request, or ``None`` when it should keep its error."""
    if not _is_responses_only_rejection(exc) or not _serves_responses(kwargs.get("model")):
        return None
    message_fields = {key: value for key, value in kwargs.items() if key in MessagesParams.model_fields}
    chat = messages_params_to_completion_params(MessagesParams(**message_fields))
    if not chat.keys() <= _TRANSLATABLE_FIELDS or "reasoning_effort" not in chat:
        return None
    input_items: list[dict[str, Any]] = []
    for message in chat["messages"]:
        items = _input_items(message)
        if items is None:
            return None
        input_items.extend(items)

    # Credentials and client options pass through exactly as ``amessages`` got them.
    request: dict[str, Any] = {key: value for key, value in kwargs.items() if key not in MessagesParams.model_fields}
    request.update(
        model=kwargs["model"],
        input_data=input_items,
        max_output_tokens=chat["max_tokens"],
        reasoning={"effort": chat["reasoning_effort"]},
        # Chat Completions stores nothing by default; neither does this.
        store=False,
    )
    request.update({field: chat[field] for field in _SAME_NAME_FIELDS if field in chat})
    if chat.get("tools"):
        request["tools"] = [_function_tool(tool) for tool in chat["tools"]]
    if "tool_choice" in chat:
        request["tool_choice"] = _tool_choice(chat["tool_choice"])
    return request


def _function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Chat function tool, keeping Chat's non-strict schema handling.

    Responses validates function schemas strictly unless told otherwise, which
    rejects the ordinary schemas an Anthropic client sends.
    """
    function = tool.get("function") or {}
    return {
        "type": "function",
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        "strict": False,
    }


def _tool_choice(choice: Any) -> Any:
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "function", "name": (choice.get("function") or {}).get("name", "")}
    return choice


def _input_items(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Map one Chat message onto Responses input items, or ``None`` when it holds an unmapped part."""
    role = message.get("role")
    content = message.get("content")
    if role == "tool":
        output = _joined_text(content)
        if output is None:
            return None
        return [{"type": "function_call_output", "call_id": message.get("tool_call_id", ""), "output": output}]
    if role == "assistant":
        text = "" if content is None else _joined_text(content)
        if text is None:
            return None
        items: list[dict[str, Any]] = []
        if text:
            items.append({"role": "assistant", "content": text})
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments") or "{}",
                }
            )
        return items
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return None
    parts: list[dict[str, Any]] = []
    for part in content:
        converted = _input_part(part)
        if converted is None:
            return None
        parts.append(converted)
    return [{"role": role, "content": parts}]


def _joined_text(content: Any) -> str | None:
    """``content`` as one string when it is text alone, else ``None``.

    A list of text parts joins without loss, which is how any-llm flattens them
    itself; any other part has no place in a string.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    texts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            return None
        texts.append(part["text"])
    return "".join(texts)


def _input_part(part: Any) -> dict[str, Any] | None:
    if not isinstance(part, dict):
        return None
    if part.get("type") == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if part.get("type") == "image_url":
        image = part.get("image_url") or {}
        return {"type": "input_image", "image_url": image.get("url", ""), "detail": image.get("detail") or "auto"}
    return None


def _usage(usage: Any) -> tuple[int, int | None, int]:
    """(input, cache read, output) tokens, with the cached share split out of the input as Messages reports it."""
    if usage is None:
        return 0, None, 0
    details = getattr(usage, "input_tokens_details", None)
    input_tokens, cache_read = split_cached_input_tokens(
        getattr(usage, "input_tokens", 0) or 0,
        getattr(details, "cached_tokens", None) if details is not None else None,
    )
    return input_tokens, cache_read, getattr(usage, "output_tokens", 0) or 0


def _stop_reason(response: Any, *, tool_use: bool, refusal: bool) -> StopReason:
    reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
    if refusal or reason == "content_filter":
        return "refusal"
    if reason == "max_output_tokens":
        return "max_tokens"
    return "tool_use" if tool_use else "end_turn"


def _tool_input(arguments: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments) if arguments else {}
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _response_to_message(response: Any) -> MessageResponse:
    """Translate a Responses answer into a Messages response."""
    blocks: list[MessageContentBlock] = []
    tool_use = refusal = False
    for item in getattr(response, "output", None) or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            for part in getattr(item, "content", None) or []:
                part_type = getattr(part, "type", None)
                if part_type == "output_text":
                    blocks.append(TextBlock(type="text", text=part.text))
                elif part_type == "refusal":
                    refusal = True
                    blocks.append(TextBlock(type="text", text=part.refusal))
        elif item_type == "function_call":
            tool_use = True
            blocks.append(
                ToolUseBlock(type="tool_use", id=item.call_id, name=item.name, input=_tool_input(item.arguments))
            )
    if not blocks:
        blocks.append(TextBlock(type="text", text=""))
    input_tokens, cache_read, output_tokens = _usage(getattr(response, "usage", None))
    return MessageResponse(
        id=response.id,
        type="message",
        role="assistant",
        content=blocks,
        model=response.model,
        stop_reason=_stop_reason(response, tool_use=tool_use, refusal=refusal),
        usage=MessageUsage(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=cache_read),
    )


def _closing_delta(response: Any, stop_reason: StopReason | None) -> MessageDeltaEvent:
    input_tokens, cache_read, output_tokens = _usage(getattr(response, "usage", None))
    return MessageDeltaEvent(
        type="message_delta",
        delta=MessageDelta(stop_reason=stop_reason),
        usage=MessageDeltaUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
        ),
    )


async def _responses_stream_to_messages(stream: Any) -> AsyncIterator[MessageStreamEvent]:
    """Translate a Responses event stream into Messages stream events.

    Each output item that carries content becomes one content block, numbered in
    the order the items open. The closing ``message_delta`` carries the usage
    Responses reports on its terminal event.
    """
    block_indexes: dict[int, int] = {}
    next_index = 0
    tool_use = refusal = False
    try:
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.created":
                response = event.response
                yield MessageStartEvent(
                    type="message_start",
                    message=MessageResponse(
                        id=response.id,
                        type="message",
                        role="assistant",
                        content=[],
                        model=response.model,
                        stop_reason=None,
                        usage=MessageUsage(input_tokens=0, output_tokens=0),
                    ),
                )
            elif event_type == "response.output_item.added":
                item = event.item
                block: MessageContentBlock
                if item.type == "message":
                    block = TextBlock(type="text", text="")
                elif item.type == "function_call":
                    tool_use = True
                    block = ToolUseBlock(type="tool_use", id=item.call_id, name=item.name, input={})
                else:
                    continue
                block_indexes[event.output_index] = next_index
                yield ContentBlockStartEvent(type="content_block_start", index=next_index, content_block=block)
                next_index += 1
            elif event_type in ("response.output_text.delta", "response.refusal.delta"):
                refusal = refusal or event_type == "response.refusal.delta"
                if (index := block_indexes.get(event.output_index)) is not None:
                    yield ContentBlockDeltaEvent(
                        type="content_block_delta", index=index, delta=TextDelta(type="text_delta", text=event.delta)
                    )
            elif event_type == "response.function_call_arguments.delta":
                if (index := block_indexes.get(event.output_index)) is not None:
                    yield ContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=index,
                        delta=InputJSONDelta(type="input_json_delta", partial_json=event.delta),
                    )
            elif event_type == "response.output_item.done":
                if (index := block_indexes.pop(event.output_index, None)) is not None:
                    yield ContentBlockStopEvent(type="content_block_stop", index=index)
            elif event_type in ("response.completed", "response.incomplete"):
                for index in sorted(block_indexes.values()):
                    yield ContentBlockStopEvent(type="content_block_stop", index=index)
                block_indexes.clear()
                response = event.response
                yield _closing_delta(response, _stop_reason(response, tool_use=tool_use, refusal=refusal))
                yield MessageStopEvent(type="message_stop")
                return
            elif event_type in ("response.failed", "error"):
                # Report what the failed attempt consumed before failing, as the
                # Chat Completions bridge does, so it is still billed.
                response = getattr(event, "response", None)
                if getattr(response, "usage", None) is not None:
                    yield _closing_delta(response, None)
                raise ResponsesStreamFailedError
    finally:
        await aclose_quietly(stream)
