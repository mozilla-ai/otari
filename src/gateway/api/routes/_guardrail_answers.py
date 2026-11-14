"""The model's answer, as a request's guardrails read it, in each wire format.

The guardrails that read the answer (``output``, and ``tool_calls`` for the
calls it hands the client) run over its parts once it is complete
(:func:`answer_parts`). They need all of a streamed answer, so the stream is
held to its end, checked, and then sent on unchanged (:func:`guarded_stream`).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from any_llm.types.completion import CompletionUsage

from gateway.api.routes._guardrail_targets import AnswerPart
from gateway.api.routes._guardrails import RequestGuardrails
from gateway.exceptions.guardrails_exceptions import AnswerRefused
from gateway.streaming import close_stream_quietly, merge_stream_usage

# The Responses items that are a call the client runs, each with the field its arguments are in.
_RESPONSES_CALLS = {"function_call": "arguments", "custom_tool_call": "input"}


def _call_arguments(item: Any) -> str | None:
    """A Responses output item's arguments, where it is a call the client runs."""
    field = _RESPONSES_CALLS.get(getattr(item, "type", None) or "")
    arguments = getattr(item, field, None) if field is not None else None
    return arguments if isinstance(arguments, str) else None


def answer_parts(dialect: str, result: Any) -> list[AnswerPart]:
    """The text and the tool calls of a complete answer, in ``dialect`` (``chat``, ``messages`` or ``responses``)."""
    parts: list[AnswerPart] = []
    if dialect == "chat":
        for choice in getattr(result, "choices", None) or []:
            message = choice.message
            if isinstance(message.content, str) and message.content:
                parts.append(AnswerPart("text", message.content))
            for call in message.tool_calls or []:
                function = getattr(call, "function", None)
                if function is not None and isinstance(function.arguments, str):
                    parts.append(AnswerPart("tool_call", function.arguments, tool=function.name))
    elif dialect == "messages":
        for block in getattr(result, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "text" and block.text:
                parts.append(AnswerPart("text", block.text))
            elif kind == "tool_use" and isinstance(block.input, dict):
                parts.append(AnswerPart("tool_call", json.dumps(block.input), tool=block.name))
    else:
        for item in getattr(result, "output", None) or []:
            kind = getattr(item, "type", None)
            if kind == "message":
                parts.extend(
                    AnswerPart("text", content.text)
                    for content in item.content or []
                    if getattr(content, "type", None) == "output_text" and content.text
                )
            elif (arguments := _call_arguments(item)) is not None:
                parts.append(AnswerPart("tool_call", arguments, tool=item.name))
    return parts


async def finish_result(guardrails: RequestGuardrails, dialect: str, result: Any) -> None:
    """Run the guardrails that read a complete answer.

    Raises:
        AnswerRefused: a guardrail on the answer refused it.
    """
    if guardrails.reads_answer:
        await guardrails.finish_answer(answer_parts(dialect, result))


# -- streams --------------------------------------------------------------------


@dataclass
class _Delta:
    """One piece of text a stream event adds to the answer, keyed by the part it belongs to."""

    channel: tuple[Any, ...]
    text: str


def _deltas(dialect: str, event: Any) -> list[_Delta]:
    """The text pieces ``event`` adds to the answer. A key starting with ``call`` is a tool call's arguments."""
    deltas: list[_Delta] = []
    if dialect == "chat":
        for choice in getattr(event, "choices", None) or []:
            delta = choice.delta
            if delta is None:
                continue
            if isinstance(delta.content, str) and delta.content:
                deltas.append(_Delta(("text", choice.index), delta.content))
            for call in delta.tool_calls or []:
                function = call.function
                if function is not None and isinstance(function.arguments, str) and function.arguments:
                    deltas.append(_Delta(("call", choice.index, call.index), function.arguments))
    elif dialect == "messages":
        if getattr(event, "type", None) == "content_block_delta":
            delta = event.delta
            if getattr(delta, "type", None) == "text_delta":
                deltas.append(_Delta(("text", event.index), delta.text))
            elif getattr(delta, "type", None) == "input_json_delta":
                deltas.append(_Delta(("call", event.index), delta.partial_json))
    else:
        kind = getattr(event, "type", None)
        if kind == "response.output_text.delta":
            deltas.append(_Delta(("text", event.item_id, event.content_index), event.delta))
        elif kind in ("response.function_call_arguments.delta", "response.custom_tool_call_input.delta"):
            deltas.append(_Delta(("call", event.item_id), event.delta))
    return deltas


def _tool_name(dialect: str, event: Any) -> tuple[tuple[Any, ...], str] | None:
    """The tool a starting tool call names, keyed as its arguments' deltas are."""
    if dialect == "chat":
        for choice in getattr(event, "choices", None) or []:
            for call in (choice.delta.tool_calls if choice.delta is not None else None) or []:
                if call.function is not None and call.function.name:
                    return ("call", choice.index, call.index), call.function.name
    elif dialect == "messages":
        block = getattr(event, "content_block", None)
        if block is not None and getattr(event, "type", None) == "content_block_start" and block.type == "tool_use":
            return ("call", event.index), block.name
    else:
        item = getattr(event, "item", None)
        if item is not None and item.type in _RESPONSES_CALLS and event.type == "response.output_item.added":
            return ("call", item.id), item.name
    return None


def _complete_call(dialect: str, event: Any) -> tuple[tuple[Any, ...], str, str] | None:
    """A Responses call the stream hands over whole, keyed as its deltas are, with its name and arguments.

    A provider may send a call's arguments only here, without deltas, so they
    replace whatever the deltas assembled rather than adding to it.
    """
    item = getattr(event, "item", None)
    if dialect != "responses" or item is None or getattr(event, "type", None) != "response.output_item.done":
        return None
    arguments = _call_arguments(item)
    return None if arguments is None else (("call", item.id), item.name, arguments)


def _provider_call(dialect: str, event: Any) -> tuple[Any, ...] | None:
    """The key of a Messages block that starts a call the provider runs itself, such as its own web search.

    Its arguments stream as a client call's do, but it never reaches the client,
    so it is not read, as :func:`answer_parts` does not read it either.
    """
    block = getattr(event, "content_block", None)
    if dialect != "messages" or block is None or getattr(event, "type", None) != "content_block_start":
        return None
    return None if block.type in ("tool_use", "text") else ("call", event.index)


async def guarded_stream(
    stream: AsyncIterator[Any],
    guardrails: RequestGuardrails,
    dialect: str,
    usage_of: Callable[[Any], CompletionUsage | None],
) -> AsyncIterator[Any]:
    """``stream``, held to its end and sent on once the guardrails that read the answer let it through.

    Raises:
        AnswerRefused: a guardrail on the answer refused it, carrying the usage
            the stream reported so the request is still billed for it.
    """
    try:
        events: list[Any] = []
        texts: dict[tuple[Any, ...], str] = {}
        names: dict[tuple[Any, ...], str] = {}
        provider_calls: set[tuple[Any, ...]] = set()
        usage: CompletionUsage | None = None
        async for event in stream:
            events.append(event)
            if (reported := usage_of(event)) is not None:
                usage = reported if usage is None else merge_stream_usage(usage, reported)
            if (named := _tool_name(dialect, event)) is not None:
                names[named[0]] = named[1]
            if (provider_call := _provider_call(dialect, event)) is not None:
                provider_calls.add(provider_call)
            for delta in _deltas(dialect, event):
                texts[delta.channel] = texts.get(delta.channel, "") + delta.text
            if (complete := _complete_call(dialect, event)) is not None:
                key, name, arguments = complete
                names[key], texts[key] = name, arguments
        parts = [
            AnswerPart("tool_call" if key[0] == "call" else "text", text, tool=names.get(key))
            for key, text in texts.items()
            if text and key not in provider_calls
        ]
        try:
            await guardrails.finish_answer(parts)
        except AnswerRefused as refused:
            refused.usage = usage
            raise
        for event in events:
            yield event
    finally:
        # Closing this generator does not close the stream it reads.
        await close_stream_quietly(stream)
