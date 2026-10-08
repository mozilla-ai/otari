"""Serve a Responses request from a provider that only speaks Chat Completions.

A provider without a Responses API (Mistral, Anthropic, Bedrock, ...) still
answers ``/v1/responses``: the request is translated into a chat completion, the
completion is called through any-llm, and the result (or its chunk stream) is
translated back into a Responses object (or event stream).

The translation is direct, with no intermediate format. A request that relies
on server-side state (``previous_response_id``, ``conversation``,
``background``), on a hosted tool, or on an input item with no chat equivalent
is refused with ``UnsupportedParameterError``, which the completion routes
already render as a 400 naming the field. Fields that only steer what OpenAI
stores or returns (``store``, ``metadata``, ``include``, ...) are dropped.
Reasoning items on an inbound ``input`` are dropped too: chat completions have
no way to send them back.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.types.responses import Response, ResponseStreamEvent
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from pydantic import BaseModel

from gateway.log_config import logger

# Same name and meaning on both APIs.
_SHARED_FIELDS = frozenset(
    {
        "api_key",
        "api_base",
        "client_args",
        "timeout",
        "temperature",
        "top_p",
        "parallel_tool_calls",
        "presence_penalty",
        "frequency_penalty",
        "user",
        "service_tier",
        "prompt_cache_key",
        "extra_body",
        "extra_headers",
        "response_format",
    }
)

# Handled by the translation itself.
_TRANSLATED_FIELDS = frozenset(
    {
        "model",
        "provider",
        "input_data",
        "instructions",
        "tools",
        "tool_choice",
        "max_output_tokens",
        "reasoning",
        "text",
        "stream",
    }
)

# Meaningful only to a server that keeps state or runs work on its own.
_REFUSED_FIELDS = ("previous_response_id", "conversation", "background", "context_management", "prompt")

_BRIDGE_NOTE = "This provider has no Responses API, so the gateway serves the request as a chat completion."

_TEXT_PART_TYPES = frozenset({"input_text", "output_text", "text"})

_STREAM_EVENTS: dict[str, type[BaseModel]] = {
    "response.created": ResponseCreatedEvent,
    "response.in_progress": ResponseInProgressEvent,
    "response.completed": ResponseCompletedEvent,
    "response.incomplete": ResponseIncompleteEvent,
    "response.output_item.added": ResponseOutputItemAddedEvent,
    "response.output_item.done": ResponseOutputItemDoneEvent,
    "response.content_part.added": ResponseContentPartAddedEvent,
    "response.content_part.done": ResponseContentPartDoneEvent,
    "response.output_text.delta": ResponseTextDeltaEvent,
    "response.output_text.done": ResponseTextDoneEvent,
    "response.reasoning_text.delta": ResponseReasoningTextDeltaEvent,
    "response.reasoning_text.done": ResponseReasoningTextDoneEvent,
    "response.function_call_arguments.delta": ResponseFunctionCallArgumentsDeltaEvent,
    "response.function_call_arguments.done": ResponseFunctionCallArgumentsDoneEvent,
}


def uses_chat_completions_bridge(provider: str | LLMProvider | None) -> bool:
    """Whether a Responses request to ``provider`` is served as a chat completion.

    A provider any-llm cannot load answers False, leaving the native call to
    report the failure as it always has.
    """
    if provider is None:
        return False
    try:
        provider_class = AnyLLM.get_provider_class(LLMProvider(provider))
    except (ValueError, ImportError):
        return False
    return not getattr(provider_class, "SUPPORTS_RESPONSES", False) and bool(
        getattr(provider_class, "SUPPORTS_COMPLETION", False)
    )


def serves_responses(provider: str | LLMProvider) -> bool:
    """Whether ``provider`` can answer a Responses request, natively or as a chat completion."""
    provider_class = AnyLLM.get_provider_class(LLMProvider(provider))
    return bool(
        getattr(provider_class, "SUPPORTS_RESPONSES", False) or getattr(provider_class, "SUPPORTS_COMPLETION", False)
    )


async def call_responses(native: Callable[..., Awaitable[Any]], kwargs: dict[str, Any]) -> Any:
    """Run ``aresponses`` keyword arguments natively, or through the bridge for a provider without the API.

    ``native`` is the caller's ``aresponses``, taken as an argument so the
    caller's module global stays the one place a test replaces it.
    """
    if uses_chat_completions_bridge(kwargs.get("provider")):
        return await aresponses_via_chat_completions(**kwargs)
    return await native(**kwargs)


async def aresponses_via_chat_completions(**kwargs: Any) -> Response | AsyncIterator[ResponseStreamEvent]:
    """Run ``aresponses`` keyword arguments as a chat completion, answering in Responses shape."""
    provider = str(LLMProvider(kwargs["provider"]).value)
    completion_kwargs = _completion_kwargs(kwargs, provider)
    echo = _ResponseEcho.from_request(kwargs)
    if completion_kwargs.get("stream"):
        chunks = await acompletion(**completion_kwargs)
        assert not isinstance(chunks, ChatCompletion)
        return _stream_events(chunks, echo)
    completion = await acompletion(**completion_kwargs)
    assert isinstance(completion, ChatCompletion)
    return _completion_to_response(completion, echo)


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def _completion_kwargs(kwargs: dict[str, Any], provider: str) -> dict[str, Any]:
    for field in _REFUSED_FIELDS:
        if kwargs.get(field):
            raise UnsupportedParameterError(field, provider, _BRIDGE_NOTE)

    out: dict[str, Any] = {key: value for key, value in kwargs.items() if key in _SHARED_FIELDS}
    out["provider"] = kwargs["provider"]
    out["model"] = kwargs["model"]
    out["messages"] = _messages(kwargs.get("input_data"), kwargs.get("instructions"), provider)

    if tools := kwargs.get("tools"):
        out["tools"] = [_chat_tool(tool, provider) for tool in tools]
    if (tool_choice := kwargs.get("tool_choice")) is not None:
        out["tool_choice"] = _chat_tool_choice(tool_choice, provider)
    if (max_output_tokens := kwargs.get("max_output_tokens")) is not None:
        out["max_tokens"] = max_output_tokens
    if isinstance(reasoning := _as_dict(kwargs.get("reasoning")), dict) and reasoning.get("effort"):
        out["reasoning_effort"] = reasoning["effort"]
    if "response_format" not in out and (response_format := _chat_response_format(kwargs.get("text"))):
        out["response_format"] = response_format
    if kwargs.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    # A cache hint changes nothing about the answer, so a provider that has no
    # prompt cache key loses nothing when it is left out.
    provider_class = AnyLLM.get_provider_class(LLMProvider(kwargs["provider"]))
    if getattr(provider_class, "PROMPT_CACHE_KEY_SUPPORT", None) == "unsupported":
        out.pop("prompt_cache_key", None)

    dropped = sorted(set(kwargs) - _SHARED_FIELDS - _TRANSLATED_FIELDS - set(_REFUSED_FIELDS))
    if dropped:
        logger.debug("Responses bridge to %s dropped fields with no chat equivalent: %s", provider, dropped)
    return out


def _as_dict(value: Any) -> Any:
    return value.model_dump(exclude_none=True) if isinstance(value, BaseModel) else value


def _messages(input_data: Any, instructions: Any, provider: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages
    for raw in input_data or []:
        item = _as_dict(raw)
        if not isinstance(item, dict):
            raise UnsupportedParameterError("input", provider, _BRIDGE_NOTE)
        item_type = item.get("type") or ("message" if "role" in item else None)
        if item_type == "message":
            messages.append(_chat_message(item, provider))
        elif item_type == "function_call":
            _append_tool_call(messages, item)
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": _text_of(item.get("output"), provider),
                }
            )
        elif item_type == "reasoning":
            continue
        else:
            raise UnsupportedParameterError(f"input item type '{item_type}'", provider, _BRIDGE_NOTE)
    return messages


def _chat_message(item: dict[str, Any], provider: str) -> dict[str, Any]:
    role = item.get("role")
    content = item.get("content")
    if role == "developer":
        role = "system"
    if role in ("assistant", "system") or isinstance(content, str):
        return {"role": role, "content": _text_of(content, provider)}
    return {"role": role, "content": [_chat_part(part, provider) for part in content or []]}


def _chat_part(raw: Any, provider: str) -> dict[str, Any]:
    part = _as_dict(raw)
    part_type = part.get("type") if isinstance(part, dict) else None
    if part_type in _TEXT_PART_TYPES:
        return {"type": "text", "text": part.get("text") or ""}
    if part_type == "input_image" and part.get("image_url"):
        image: dict[str, Any] = {"url": part["image_url"]}
        if part.get("detail"):
            image["detail"] = part["detail"]
        return {"type": "image_url", "image_url": image}
    if part_type == "input_file" and (part.get("file_data") or part.get("file_id")):
        file = {key: part[key] for key in ("file_data", "file_id", "filename") if part.get(key)}
        return {"type": "file", "file": file}
    raise UnsupportedParameterError(f"input content type '{part_type}'", provider, _BRIDGE_NOTE)


def _text_of(content: Any, provider: str) -> str:
    """Flatten a content value to the plain string chat completions take.

    Only text and refusal parts flatten; any other part is refused rather than
    dropped, because the chat message it would land in takes a string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for raw in content:
        part = _as_dict(raw)
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type in _TEXT_PART_TYPES:
            texts.append(str(part.get("text") or ""))
        elif part_type == "refusal":
            texts.append(str(part.get("refusal") or ""))
        else:
            raise UnsupportedParameterError(f"input content type '{part_type}'", provider, _BRIDGE_NOTE)
    return "".join(texts)


def _append_tool_call(messages: list[dict[str, Any]], item: dict[str, Any]) -> None:
    """Attach a ``function_call`` item to the assistant turn it belongs to."""
    call = {
        "id": item.get("call_id"),
        "type": "function",
        "function": {"name": item.get("name"), "arguments": item.get("arguments") or "{}"},
    }
    if messages and messages[-1].get("role") == "assistant":
        messages[-1].setdefault("tool_calls", []).append(call)
        return
    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})


def _chat_tool(raw: Any, provider: str) -> dict[str, Any]:
    tool = _as_dict(raw)
    if not isinstance(tool, dict) or tool.get("type") != "function":
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        raise UnsupportedParameterError(f"tool type '{tool_type}'", provider, _BRIDGE_NOTE)
    if "function" in tool:
        return tool
    function = {key: tool[key] for key in ("name", "description", "parameters", "strict") if tool.get(key) is not None}
    return {"type": "function", "function": function}


def _chat_tool_choice(raw: Any, provider: str) -> Any:
    choice = _as_dict(raw)
    if isinstance(choice, str):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        name = choice.get("name") or (choice.get("function") or {}).get("name")
        return {"type": "function", "function": {"name": name}}
    raise UnsupportedParameterError("tool_choice", provider, _BRIDGE_NOTE)


def _chat_response_format(raw: Any) -> dict[str, Any] | None:
    text = _as_dict(raw)
    fmt = _as_dict(text.get("format")) if isinstance(text, dict) else None
    if not isinstance(fmt, dict):
        return None
    if fmt.get("type") == "json_schema":
        schema = {key: fmt[key] for key in ("name", "description", "schema", "strict") if fmt.get(key) is not None}
        return {"type": "json_schema", "json_schema": schema}
    if fmt.get("type") == "json_object":
        return {"type": "json_object"}
    return None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class _ResponseEcho:
    """The request fields a Responses object repeats back, in Responses shape."""

    def __init__(self, fields: dict[str, Any]) -> None:
        self.fields = fields

    @classmethod
    def from_request(cls, kwargs: dict[str, Any]) -> _ResponseEcho:
        tool_choice = _as_dict(kwargs.get("tool_choice"))
        tools = [
            tool
            for tool in (_as_dict(raw) for raw in kwargs.get("tools") or [])
            if isinstance(tool, dict) and tool.get("type") == "function" and "function" not in tool
        ]
        return cls(
            {
                "model": kwargs["model"],
                "instructions": kwargs.get("instructions") if isinstance(kwargs.get("instructions"), str) else None,
                "tools": tools,
                "tool_choice": tool_choice if tool_choice is not None else "auto",
                "parallel_tool_calls": kwargs.get("parallel_tool_calls", True),
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
                "max_output_tokens": kwargs.get("max_output_tokens"),
            }
        )

    def response(
        self,
        *,
        response_id: str,
        created_at: float,
        model: str | None,
        status: str,
        output: list[dict[str, Any]],
        finish_reason: str | None = None,
        usage: Any = None,
    ) -> Response:
        incomplete = _INCOMPLETE_REASONS.get(finish_reason or "") if status != "in_progress" else None
        body: dict[str, Any] = {
            **self.fields,
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model or self.fields["model"],
            "status": "incomplete" if incomplete else status,
            "incomplete_details": {"reason": incomplete} if incomplete else None,
            "output": output,
            "usage": _responses_usage(usage),
        }
        return Response.model_validate(body)


_INCOMPLETE_REASONS = {"length": "max_output_tokens", "content_filter": "content_filter"}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _responses_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": usage.prompt_tokens or 0,
        "input_tokens_details": {"cached_tokens": getattr(prompt_details, "cached_tokens", None) or 0},
        "output_tokens": usage.completion_tokens or 0,
        "output_tokens_details": {"reasoning_tokens": getattr(completion_details, "reasoning_tokens", None) or 0},
        "total_tokens": usage.total_tokens or 0,
    }


def _reasoning_item(item_id: str, text: str, status: str) -> dict[str, Any]:
    content = [{"type": "reasoning_text", "text": text}] if text or status == "completed" else []
    return {"type": "reasoning", "id": item_id, "summary": [], "content": content, "status": status}


def _message_item(item_id: str, text: str | None, status: str, refusal: str | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text is not None:
        content.append({"type": "output_text", "text": text, "annotations": [], "logprobs": []})
    if refusal:
        content.append({"type": "refusal", "refusal": refusal})
    return {"type": "message", "id": item_id, "role": "assistant", "status": status, "content": content}


def _function_call_item(item_id: str, call_id: str, name: str, arguments: str, status: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def _completion_to_response(completion: ChatCompletion, echo: _ResponseEcho) -> Response:
    output: list[dict[str, Any]] = []
    finish_reason: str | None = None
    if completion.choices:
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        message = choice.message
        reasoning = getattr(message, "reasoning", None)
        if reasoning is not None and reasoning.content:
            output.append(_reasoning_item(_new_id("rs"), reasoning.content, "completed"))
        if message.content or message.refusal:
            output.append(_message_item(_new_id("msg"), message.content or None, "completed", message.refusal))
        for call in message.tool_calls or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            output.append(
                _function_call_item(_new_id("fc"), call.id, function.name, function.arguments or "", "completed")
            )
    return echo.response(
        response_id=_new_id("resp"),
        created_at=float(completion.created or time.time()),
        model=completion.model,
        status="completed",
        output=output,
        finish_reason=finish_reason,
        usage=completion.usage,
    )


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------


class _ToolCallState:
    def __init__(self) -> None:
        self.item_id = _new_id("fc")
        self.call_id = ""
        self.name = ""
        self.arguments = ""
        self.output_index: int | None = None


class _StreamTranslator:
    """Turn chat completion chunks into the Responses event sequence.

    Items are opened in the order the model starts them (reasoning, then text,
    then each tool call) and a text or reasoning item is closed as soon as a
    later item opens. A tool call opens once its name is known, and every tool
    call closes when the stream ends, since chunks may interleave their
    argument fragments.
    """

    def __init__(self, echo: _ResponseEcho) -> None:
        self.echo = echo
        self.response_id = _new_id("resp")
        self.created_at = time.time()
        self.model: str | None = None
        self.sequence = 0
        self.output: list[dict[str, Any]] = []
        self.reasoning: tuple[int, str] | None = None
        self.reasoning_text = ""
        self.text: tuple[int, str] | None = None
        self.text_value = ""
        self.tool_calls: dict[int, _ToolCallState] = {}
        self.finish_reason: str | None = None
        self.usage: Any = None

    def _event(self, event_type: str, **fields: Any) -> ResponseStreamEvent:
        payload = {"type": event_type, "sequence_number": self.sequence, **fields}
        self.sequence += 1
        return _STREAM_EVENTS[event_type].model_validate(payload)  # type: ignore[return-value]

    def _snapshot(self, status: str) -> Response:
        return self.echo.response(
            response_id=self.response_id,
            created_at=self.created_at,
            model=self.model,
            status=status,
            output=list(self.output),
            finish_reason=self.finish_reason,
            usage=self.usage if status != "in_progress" else None,
        )

    def start(self) -> list[ResponseStreamEvent]:
        snapshot = self._snapshot("in_progress")
        return [
            self._event("response.created", response=snapshot),
            self._event("response.in_progress", response=snapshot),
        ]

    def feed(self, chunk: ChatCompletionChunk) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        self.model = self.model or chunk.model
        if chunk.usage is not None:
            self.usage = chunk.usage
        for choice in chunk.choices[:1]:
            delta = choice.delta
            reasoning = getattr(delta, "reasoning", None)
            if reasoning is not None and reasoning.content:
                events.extend(self._reasoning_delta(reasoning.content))
            text = (delta.content or "") + (delta.refusal or "")
            if text:
                events.extend(self._text_delta(text))
            for call in delta.tool_calls or []:
                events.extend(self._tool_call_delta(call))
            if choice.finish_reason:
                self.finish_reason = choice.finish_reason
        return events

    def finish(self) -> list[ResponseStreamEvent]:
        events = self._close_reasoning() + self._close_text()
        for state in self.tool_calls.values():
            if state.output_index is None:
                events.extend(self._open_tool_call(state))
            item = _function_call_item(state.item_id, state.call_id, state.name, state.arguments, "completed")
            self.output[state.output_index] = item  # type: ignore[index]
            events.append(
                self._event(
                    "response.function_call_arguments.done",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    name=state.name,
                    arguments=state.arguments,
                )
            )
            events.append(self._event("response.output_item.done", output_index=state.output_index, item=item))
        snapshot = self._snapshot("completed")
        terminal = "response.incomplete" if snapshot.status == "incomplete" else "response.completed"
        events.append(self._event(terminal, response=snapshot))
        return events

    def _open_item(self, item: dict[str, Any]) -> tuple[int, ResponseStreamEvent]:
        index = len(self.output)
        self.output.append(item)
        return index, self._event("response.output_item.added", output_index=index, item=item)

    def _reasoning_delta(self, delta: str) -> list[ResponseStreamEvent]:
        events: list[ResponseStreamEvent] = []
        if self.reasoning is None:
            item_id = _new_id("rs")
            index, added = self._open_item(_reasoning_item(item_id, "", "in_progress"))
            self.reasoning = (index, item_id)
            events += [
                added,
                self._event(
                    "response.content_part.added",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    part={"type": "reasoning_text", "text": ""},
                ),
            ]
        index, item_id = self.reasoning
        self.reasoning_text += delta
        events.append(
            self._event(
                "response.reasoning_text.delta", item_id=item_id, output_index=index, content_index=0, delta=delta
            )
        )
        return events

    def _close_reasoning(self) -> list[ResponseStreamEvent]:
        if self.reasoning is None:
            return []
        index, item_id = self.reasoning
        self.reasoning = None
        item = _reasoning_item(item_id, self.reasoning_text, "completed")
        self.output[index] = item
        part = {"type": "reasoning_text", "text": self.reasoning_text}
        return [
            self._event(
                "response.reasoning_text.done",
                item_id=item_id,
                output_index=index,
                content_index=0,
                text=self.reasoning_text,
            ),
            self._event("response.content_part.done", item_id=item_id, output_index=index, content_index=0, part=part),
            self._event("response.output_item.done", output_index=index, item=item),
        ]

    def _text_delta(self, delta: str) -> list[ResponseStreamEvent]:
        events = self._close_reasoning()
        if self.text is None:
            item_id = _new_id("msg")
            index, added = self._open_item(_message_item(item_id, None, "in_progress"))
            self.text = (index, item_id)
            events += [
                added,
                self._event(
                    "response.content_part.added",
                    item_id=item_id,
                    output_index=index,
                    content_index=0,
                    part={"type": "output_text", "text": "", "annotations": [], "logprobs": []},
                ),
            ]
        index, item_id = self.text
        self.text_value += delta
        events.append(
            self._event(
                "response.output_text.delta",
                item_id=item_id,
                output_index=index,
                content_index=0,
                delta=delta,
                logprobs=[],
            )
        )
        return events

    def _close_text(self) -> list[ResponseStreamEvent]:
        if self.text is None:
            return []
        index, item_id = self.text
        self.text = None
        item = _message_item(item_id, self.text_value, "completed")
        self.output[index] = item
        return [
            self._event(
                "response.output_text.done",
                item_id=item_id,
                output_index=index,
                content_index=0,
                text=self.text_value,
                logprobs=[],
            ),
            self._event(
                "response.content_part.done",
                item_id=item_id,
                output_index=index,
                content_index=0,
                part=item["content"][0],
            ),
            self._event("response.output_item.done", output_index=index, item=item),
        ]

    def _tool_call_delta(self, call: Any) -> list[ResponseStreamEvent]:
        state = self.tool_calls.setdefault(call.index, _ToolCallState())
        function = call.function
        state.call_id = state.call_id or call.id or ""
        if function is not None and function.name:
            state.name = state.name or function.name
        fragment = (function.arguments if function is not None else None) or ""
        state.arguments += fragment
        if state.output_index is None:
            return self._open_tool_call(state) if state.name else []
        if not fragment:
            return []
        return [
            self._event(
                "response.function_call_arguments.delta",
                item_id=state.item_id,
                output_index=state.output_index,
                delta=fragment,
            )
        ]

    def _open_tool_call(self, state: _ToolCallState) -> list[ResponseStreamEvent]:
        events = self._close_reasoning() + self._close_text()
        state.call_id = state.call_id or _new_id("call")
        index, added = self._open_item(_function_call_item(state.item_id, state.call_id, state.name, "", "in_progress"))
        state.output_index = index
        events.append(added)
        if state.arguments:
            events.append(
                self._event(
                    "response.function_call_arguments.delta",
                    item_id=state.item_id,
                    output_index=index,
                    delta=state.arguments,
                )
            )
        return events


async def _stream_events(
    chunks: AsyncIterator[ChatCompletionChunk], echo: _ResponseEcho
) -> AsyncIterator[ResponseStreamEvent]:
    translator = _StreamTranslator(echo)
    try:
        # Wait for the first chunk before announcing the response, so a provider
        # that fails on its first read fails the attempt before anything reached
        # the client, as a native Responses stream does.
        first = await anext(chunks, None)
        for event in translator.start():
            yield event
        if first is not None:
            for event in translator.feed(first):
                yield event
        async for chunk in chunks:
            for event in translator.feed(chunk):
                yield event
        for event in translator.finish():
            yield event
    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()
