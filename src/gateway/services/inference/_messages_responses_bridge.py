"""Serve a Messages request from a provider's Responses API.

An operator opts a provider instance in with ``messages_api: responses``. A
Messages request routed to that instance is then translated into a Responses
request, called through any-llm, and the result (or its event stream) is
translated back into a Messages response (or Messages stream events). Without
the setting, any-llm serves the request as a chat completion, as it always has.

The point of the detour is reasoning. A Responses ``reasoning`` item comes back
as a ``thinking`` block whose ``signature`` carries the item's encrypted content
behind :data:`REASONING_SIGNATURE_PREFIX` and a tag naming the instance that
served it. A Messages client resends thinking blocks on the next turn, because
Anthropic's API requires it, and the signature tells this module which ones to
hand back to the provider as reasoning items. Only the instance that minted a
block can use it: the content is encrypted for that provider account, and
another account refuses the whole call over it (``invalid_encrypted_content``),
so a block minted by another instance is dropped like any other provider's
thinking. Any model on the same account accepts it. A block minted here means
nothing to any other provider, so :func:`call_messages` drops those before
every other call.

Every request is sent with ``store: false``: nothing is kept upstream, and a
Messages client resends the whole conversation each turn anyway.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageDeltaUsage,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    MessageStreamEvent,
    MessageUsage,
)
from pydantic import BaseModel

from gateway.log_config import logger
from gateway.services.providers import aresponses
from gateway.services.traces import resolve_session

RESPONSES_INSTANCE_FIELD = "_otari_responses_instance"
"""Names the instance whose Responses API serves this call, set only for one that opted in.

Never a caller field: :func:`call_messages` consumes it before any provider sees the kwargs.
"""

CONVERSATION_CACHE_KEY_FIELD = "_otari_conversation_cache_key"
"""Carries a prompt cache key derived from the request's session; the route scopes it to the caller.

Used only as a Responses call's ``prompt_cache_key`` when the caller sent none, so
the turns of one conversation land on the same cache. Never reaches another call.
"""

REASONING_SIGNATURE_PREFIX = "otari-rsn1."
"""Marks a thinking block's signature as ``otari-rsn1.<instance tag>.<encrypted reasoning>``."""

_PRIVATE_FIELDS = frozenset({RESPONSES_INSTANCE_FIELD, CONVERSATION_CACHE_KEY_FIELD})

_BRIDGE_NOTE = "This instance serves Messages requests through the provider's Responses API."

# Fields with the same name and meaning on both APIs.
_SHARED_FIELDS = (
    "api_key",
    "api_base",
    "client_args",
    "timeout",
    "temperature",
    "top_p",
    "service_tier",
    "prompt_cache_key",
    "extra_headers",
    "extra_body",
)

# Handled by the translation itself.
_TRANSLATED_FIELDS = frozenset(
    {
        "model",
        "provider",
        "messages",
        "system",
        "max_tokens",
        "tools",
        "tool_choice",
        "thinking",
        "output_format",
        "stream",
        "user",
        "safety_identifier",
    }
)

# Anthropic-only steering a Responses call has no use for. ``context_management`` and
# ``betas`` are normally gone before this point (see the route's ``_split_client_betas``).
_DROPPED_FIELDS = frozenset({"metadata", "cache_control", "container", "context_management", "betas", "top_k"})

# The Anthropic budget a Responses effort stands for. ``minimal`` and ``none`` are
# left out: neither is accepted by every reasoning model, while these three are.
_EFFORT_BY_BUDGET = ((2048, "low"), (16384, "medium"))
_EFFORT_ABOVE_BUDGETS = "high"
_EFFORTS = frozenset({"low", "medium", "high"})

# Reasoning models. Sending ``reasoning`` or asking for encrypted reasoning makes
# any other model fail the call (``gpt-4.1``, ``gpt-5-chat-latest``), even when the
# caller asked for thinking, so a model must match this to get either.
_REASONING_MODEL = re.compile(r"^(o\d|gpt-5)(?!.*-chat)|codex", re.IGNORECASE)

_STOP_FOR_INCOMPLETE = {"max_output_tokens": "max_tokens", "content_filter": "refusal"}

_warned_unsupported: set[str] = set()


def serves_responses_natively(provider: str | LLMProvider) -> bool:
    """Whether any-llm calls ``provider``'s own Responses API (rather than none at all)."""
    try:
        provider_class = AnyLLM.get_provider_class(LLMProvider(provider))
    except (ValueError, ImportError):
        return False
    return bool(getattr(provider_class, "SUPPORTS_RESPONSES", False))


def with_responses_instance(kwargs: dict[str, Any], instance: str | None) -> dict[str, Any]:
    """``kwargs`` marked to go through ``instance``'s Responses API, or unchanged for ``None``."""
    if instance is None:
        return kwargs
    return {**kwargs, RESPONSES_INSTANCE_FIELD: instance}


def conversation_cache_key(headers: Mapping[str, str], *, session_label: str | None) -> dict[str, str]:
    """The private field keying a Responses call's prompt cache by the request's session, if it has one.

    The turns of one conversation then share a cache. The caller's own
    ``prompt_cache_key`` still wins (see :func:`call_messages`), and the route scopes
    this key to the caller as it scopes that one.
    """
    session = resolve_session(headers, session_label=session_label, tags=None)
    if session is None:
        return {}
    return {CONVERSATION_CACHE_KEY_FIELD: f"{session.source}:{session.key}"}


async def call_messages(native: Callable[..., Awaitable[Any]], kwargs: dict[str, Any]) -> Any:
    """Run ``amessages`` keyword arguments natively, or through the provider's Responses API when marked.

    ``native`` is the caller's ``amessages``, taken as an argument so the caller's
    module global stays the one place a test replaces it.
    """
    instance = kwargs.get(RESPONSES_INSTANCE_FIELD)
    conversation_key = kwargs.get(CONVERSATION_CACHE_KEY_FIELD)
    kwargs = {key: value for key, value in kwargs.items() if key not in _PRIVATE_FIELDS}
    if isinstance(instance, str):
        provider = _provider_of(kwargs)
        if provider is not None and serves_responses_natively(provider):
            if isinstance(conversation_key, str) and not kwargs.get("prompt_cache_key"):
                kwargs["prompt_cache_key"] = conversation_key
            return await amessages_via_responses(kwargs, instance=instance)
        if str(provider) not in _warned_unsupported:
            _warned_unsupported.add(str(provider))
            logger.warning(
                "messages_api: responses is set for a %s instance, which has no Responses API; "
                "serving Messages requests as before",
                provider,
            )
    if isinstance(messages := kwargs.get("messages"), list):
        if kwargs.get("thinking") and _continues_tool_use_after_minted_thinking(messages):
            # Anthropic requires a tool-use turn that continues under thinking to open
            # with its own thinking block, and the one there is about to be dropped.
            kwargs.pop("thinking")
        kwargs["messages"] = _without_minted_thinking(messages)
    return await native(**kwargs)


async def amessages_via_responses(
    kwargs: dict[str, Any], *, instance: str
) -> MessageResponse | AsyncIterator[MessageStreamEvent]:
    """Run ``amessages`` keyword arguments as a call to ``instance``'s Responses API, answering in Messages shape."""
    responses_kwargs = responses_kwargs_for(kwargs, instance=instance)
    if responses_kwargs.get("stream"):
        events = await aresponses(**responses_kwargs)
        return _stream_events(events, instance=instance)  # type: ignore[arg-type]
    response = await aresponses(**responses_kwargs)
    return message_from_response(_as_dict(response), instance=instance)


def _provider_of(kwargs: dict[str, Any]) -> str | None:
    if kwargs.get("provider"):
        return str(AnyLLM.resolve_provider_key(kwargs["provider"]))
    try:
        provider, _ = AnyLLM.split_model_provider(str(kwargs.get("model")))
    except Exception:  # noqa: BLE001 - any-llm raises its own types for an unknown provider
        return None
    return provider.value if isinstance(provider, LLMProvider) else str(provider)


def _as_dict(value: Any) -> Any:
    return value.model_dump(exclude_none=True, warnings=False) if isinstance(value, BaseModel) else value


def _instance_tag(instance: str) -> str:
    """A short stand-in for ``instance`` in a signature, which the client sees, so the config name stays private."""
    return hashlib.sha256(instance.encode()).hexdigest()[:16]


def reasoning_signature(instance: str, encrypted_content: str) -> str:
    """The thinking-block signature carrying ``instance``'s encrypted reasoning."""
    return f"{REASONING_SIGNATURE_PREFIX}{_instance_tag(instance)}.{encrypted_content}"


def _minted_reasoning(block: dict[str, Any], instance: str) -> str | None:
    """The encrypted reasoning a block minted by ``instance`` carries, or ``None`` for another's."""
    tag, _, encrypted = str(block["signature"])[len(REASONING_SIGNATURE_PREFIX) :].partition(".")
    return encrypted if tag == _instance_tag(instance) and encrypted else None


def _is_minted_thinking(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "thinking"
        and str(block.get("signature") or "").startswith(REASONING_SIGNATURE_PREFIX)
    )


def _continues_tool_use_after_minted_thinking(messages: list[Any]) -> bool:
    """Whether the request answers tool calls from an assistant turn that opened with a block minted here."""
    last = messages[-1] if messages else None
    if not isinstance(last, dict) or last.get("role") != "user" or not isinstance(last.get("content"), list):
        return False
    if not any(isinstance(block, dict) and block.get("type") == "tool_result" for block in last["content"]):
        return False
    previous = messages[-2] if len(messages) > 1 else None
    content = previous.get("content") if isinstance(previous, dict) else None
    return isinstance(content, list) and any(_is_minted_thinking(block) for block in content)


def _without_minted_thinking(messages: list[Any]) -> list[Any]:
    """``messages`` without the thinking blocks this module signed, which no other provider can verify."""
    kept: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list) or not any(_is_minted_thinking(block) for block in content):
            kept.append(message)
            continue
        blocks = [block for block in content if not _is_minted_thinking(block)]
        if blocks:
            kept.append({**message, "content": blocks})
    return kept


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def responses_kwargs_for(kwargs: dict[str, Any], *, instance: str) -> dict[str, Any]:
    """Translate ``amessages`` keyword arguments into ``aresponses`` ones for a call to ``instance``.

    Raises ``UnsupportedParameterError`` for a field the Responses API cannot carry.
    """
    provider = _provider_of(kwargs) or "unknown"
    if kwargs.get("stop_sequences"):
        raise UnsupportedParameterError("stop_sequences", provider, _BRIDGE_NOTE)

    out: dict[str, Any] = {key: kwargs[key] for key in _SHARED_FIELDS if kwargs.get(key) is not None}
    out["model"] = kwargs["model"]
    if kwargs.get("provider"):
        out["provider"] = kwargs["provider"]
    out["input_data"] = _input_items(kwargs.get("messages") or [], provider, instance)
    out["store"] = False
    if instructions := _system_text(kwargs.get("system")):
        out["instructions"] = instructions
    if (max_tokens := kwargs.get("max_tokens")) is not None:
        out["max_output_tokens"] = max_tokens
    if identifier := kwargs.get("safety_identifier") or kwargs.get("user"):
        out["safety_identifier"] = identifier

    if tools := [tool for tool in kwargs.get("tools") or [] if _is_function_tool(tool)]:
        out["tools"] = [_function_tool(tool) for tool in tools]
    if isinstance(tool_choice := kwargs.get("tool_choice"), dict):
        if (choice := _tool_choice(tool_choice)) is not None:
            out["tool_choice"] = choice
        if tool_choice.get("disable_parallel_tool_use") is True:
            out["parallel_tool_calls"] = False

    output_config = kwargs.get("output_format")
    if output_config is not None and not isinstance(output_config, dict):
        raise UnsupportedParameterError("output_format", provider, _BRIDGE_NOTE)
    if text := _text_format(output_config):
        out["text"] = text

    if (reasoning := _reasoning(kwargs.get("thinking"), output_config, str(kwargs["model"]))) is not None:
        if reasoning:
            out["reasoning"] = reasoning
        out["include"] = ["reasoning.encrypted_content"]

    if kwargs.get("stream"):
        out["stream"] = True

    dropped = sorted(set(kwargs) - set(_SHARED_FIELDS) - _TRANSLATED_FIELDS - _DROPPED_FIELDS - {"stop_sequences"})
    if dropped:
        logger.debug("Messages-to-Responses bridge dropped fields with no Responses equivalent: %s", dropped)
    return out


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            str(block.get("text") or "") for block in system if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _is_function_tool(tool: Any) -> bool:
    """Whether ``tool`` is one the caller runs. Anthropic's server tools are not forwarded."""
    return isinstance(tool, dict) and tool.get("type") in (None, "custom") and bool(tool.get("name"))


def _function_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function: dict[str, Any] = {
        "type": "function",
        "name": tool["name"],
        "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        "strict": False,
    }
    if tool.get("description"):
        function["description"] = tool["description"]
    return function


def _tool_choice(tool_choice: dict[str, Any]) -> str | dict[str, Any] | None:
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool":
        return {"type": "function", "name": tool_choice.get("name", "")}
    return None


def _text_format(output_config: dict[str, Any] | None) -> dict[str, Any] | None:
    """The Responses ``text`` for an Anthropic ``output_config``, either of its two shapes."""
    if not output_config:
        return None
    fmt = output_config.get("format") if "format" in output_config else output_config
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
        return None
    return {"format": {"type": "json_schema", "name": "output", "schema": fmt["schema"], "strict": False}}


def _reasoning(thinking: Any, output_config: Any, model: str) -> dict[str, Any] | None:
    """The Responses ``reasoning`` for a Messages request, or ``None`` for a model that does not reason.

    An explicit ``output_config.effort`` wins over a thinking budget. A summary is asked
    for unless the caller turned thinking off, so a thinking block has text to show.
    """
    if not _REASONING_MODEL.search(model.partition(":")[2].rpartition("/")[2] or model):
        return None
    thinking = thinking if isinstance(thinking, dict) else {}
    thinking_type = thinking.get("type")
    effort = output_config.get("effort") if isinstance(output_config, dict) else None
    if effort == "max":
        effort = "high"
    if effort not in _EFFORTS:
        effort = None
    if effort is None and thinking_type == "enabled":
        effort = _effort_for_budget(thinking.get("budget_tokens"))

    reasoning: dict[str, Any] = {}
    if effort is not None:
        reasoning["effort"] = effort
    if thinking_type != "disabled":
        reasoning["summary"] = "auto"
    return reasoning


def _effort_for_budget(budget: Any) -> str:
    if not isinstance(budget, int) or isinstance(budget, bool):
        return "medium"
    for ceiling, effort in _EFFORT_BY_BUDGET:
        if budget <= ceiling:
            return effort
    return _EFFORT_ABOVE_BUDGETS


def _input_items(messages: list[Any], provider: str, instance: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise UnsupportedParameterError("messages", provider, _BRIDGE_NOTE)
        role = message.get("role")
        content = message.get("content")
        if isinstance(content, str):
            items.append({"role": role, "content": content})
        elif role == "assistant":
            items.extend(_assistant_items(content or [], instance))
        else:
            items.extend(_user_items(content or [], provider))
    return items


def _assistant_items(blocks: list[Any], instance: str) -> list[dict[str, Any]]:
    """Assistant blocks in order, consecutive text joined into one output message."""
    items: list[dict[str, Any]] = []
    text: list[str] = []

    def flush() -> None:
        if text:
            items.append({"role": "assistant", "content": "".join(text)})
            text.clear()

    for block in blocks:
        block_type = block.get("type") if isinstance(block, dict) else None
        if block_type == "text":
            text.append(str(block.get("text") or ""))
        elif block_type == "tool_use":
            flush()
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {}),
                }
            )
        elif _is_minted_thinking(block) and (encrypted := _minted_reasoning(block, instance)) is not None:
            flush()
            summary = str(block.get("thinking") or "")
            items.append(
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": summary}] if summary else [],
                    "encrypted_content": encrypted,
                }
            )
        elif block_type in ("thinking", "redacted_thinking"):
            # Another provider's signature, or another instance's, which this one cannot read.
            continue
        else:
            # Server-tool activity a provider ran on an earlier Anthropic turn.
            logger.debug("Messages-to-Responses bridge dropped an assistant %r block", block_type)
    flush()
    return items


def _user_items(blocks: list[Any], provider: str) -> list[dict[str, Any]]:
    """User blocks in order; each tool result becomes its own item between message parts."""
    items: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []

    def flush() -> None:
        if parts:
            items.append({"role": "user", "content": list(parts)})
            parts.clear()

    for block in blocks:
        block_type = block.get("type") if isinstance(block, dict) else None
        if block_type == "tool_result":
            flush()
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id"),
                    "output": _tool_output(block, provider),
                }
            )
        else:
            parts.append(_input_part(block, provider))
    flush()
    return items


def _tool_output(block: dict[str, Any], provider: str) -> str | list[dict[str, Any]]:
    """A ``tool_result``'s content as a Responses output: a string, or parts when it holds more than text.

    Responses has no error flag on a function output, so ``is_error`` is said in the
    text, as ``Error: ...``, the way any-llm's chat-completions conversion says it.
    """
    content = block.get("content")
    parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else []
    if isinstance(content, list):
        parts = [_input_part(part, provider) for part in content]
    if block.get("is_error") is True:
        if parts and parts[0]["type"] == "input_text" and parts[0]["text"]:
            if not parts[0]["text"].startswith("Error:"):
                parts[0] = {"type": "input_text", "text": f"Error: {parts[0]['text']}"}
        else:
            parts.insert(0, {"type": "input_text", "text": "Error"})
    if all(part["type"] == "input_text" for part in parts):
        return "".join(part["text"] for part in parts)
    return parts


def _input_part(block: Any, provider: str) -> dict[str, Any]:
    block_type = block.get("type") if isinstance(block, dict) else None
    if block_type == "text":
        return {"type": "input_text", "text": str(block.get("text") or "")}
    if block_type == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64" and source.get("data"):
            url = f"data:{source.get('media_type') or 'image/png'};base64,{source['data']}"
            return {"type": "input_image", "image_url": url, "detail": "auto"}
        if source.get("type") == "url" and source.get("url"):
            return {"type": "input_image", "image_url": source["url"], "detail": "auto"}
    if block_type == "document":
        source = block.get("source") or {}
        if source.get("type") == "text":
            return {"type": "input_text", "text": str(source.get("data") or "")}
        if source.get("type") == "base64" and source.get("data"):
            media_type = source.get("media_type") or "application/pdf"
            return {
                "type": "input_file",
                "filename": str(block.get("title") or "document.pdf"),
                "file_data": f"data:{media_type};base64,{source['data']}",
            }
        if source.get("type") == "url" and source.get("url"):
            return {"type": "input_file", "file_url": source["url"]}
    raise UnsupportedParameterError(f"content block type '{block_type}'", provider, _BRIDGE_NOTE)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def _thinking_block(item: dict[str, Any], instance: str) -> dict[str, Any]:
    summary = "\n\n".join(str(part.get("text") or "") for part in item.get("summary") or [] if isinstance(part, dict))
    encrypted = item.get("encrypted_content")
    return {
        "type": "thinking",
        "thinking": summary,
        "signature": reasoning_signature(instance, str(encrypted)) if encrypted else "",
    }


def _tool_input(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        logger.warning("Messages-to-Responses bridge received function call arguments that are not JSON")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stop_reason(response: dict[str, Any], *, tool_called: bool, refused: bool) -> str:
    if response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        return _STOP_FOR_INCOMPLETE.get(str(reason), "end_turn")
    if refused:
        return "refusal"
    return "tool_use" if tool_called else "end_turn"


class UnreportedUsage(MessageUsage):
    """Zero usage on the wire for a response whose provider reported none; see :func:`usage_unreported`.

    A subclass rather than ``None`` because any-llm's Messages types require ``usage``.
    """


class UnreportedDeltaUsage(MessageDeltaUsage):
    """The streamed counterpart of :class:`UnreportedUsage`."""


def usage_unreported(usage: Any) -> bool:
    """Whether ``usage`` stands for counts the provider never sent, so metering must treat it as missing.

    A tool loop folds earlier turns' tokens into the last turn's usage object; once it
    holds any, those are real counts and are billed.
    """
    return isinstance(usage, UnreportedUsage | UnreportedDeltaUsage) and not (usage.output_tokens or usage.input_tokens)


def _usage(usage: Any) -> dict[str, Any] | None:
    """Responses usage in Messages terms: cached input is reported apart, as Anthropic does.

    ``None`` when the provider reported none, which is not the same as zero tokens.
    """
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
    out: dict[str, Any] = {
        "input_tokens": max(input_tokens - cached, 0),
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": int(usage.get("output_tokens") or 0),
    }
    if reasoning:
        out["output_tokens_details"] = {"thinking_tokens": reasoning}
    return out


def _message_id(response: dict[str, Any]) -> str:
    return f"msg_{response.get('id') or uuid.uuid4().hex}"


def message_from_response(response: dict[str, Any], *, instance: str) -> MessageResponse:
    """A Responses ``Response`` (as a dict) from ``instance`` as a Messages response."""
    content: list[dict[str, Any]] = []
    tool_called = refused = False
    for item in response.get("output") or []:
        item_type = item.get("type")
        if item_type == "reasoning":
            block = _thinking_block(item, instance)
            if block["thinking"] or block["signature"]:
                content.append(block)
        elif item_type == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    content.append({"type": "text", "text": part.get("text") or ""})
                elif part.get("type") == "refusal":
                    refused = True
                    content.append({"type": "text", "text": part.get("refusal") or ""})
        elif item_type == "function_call":
            tool_called = True
            content.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "input": _tool_input(item.get("arguments")),
                }
            )
    return MessageResponse.model_validate(
        {
            "id": _message_id(response),
            "type": "message",
            "role": "assistant",
            "model": response.get("model") or "",
            "content": content,
            "stop_reason": _stop_reason(response, tool_called=tool_called, refused=refused),
            "stop_sequence": None,
            "usage": _usage(response.get("usage"))
            or UnreportedUsage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0),
        }
    )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class ResponsesStreamFailedError(Exception):
    """The provider ended a Responses stream with ``response.failed`` or an ``error`` event.

    ``code`` is the provider's error code (``context_length_exceeded``, ...), when it sent one.
    """

    def __init__(self, event_type: str, code: str | None) -> None:
        super().__init__(f"{event_type} ({code})" if code else event_type)
        self.code = code


def _stream_error_code(event: dict[str, Any]) -> str | None:
    error = (event.get("response") or {}).get("error") if event.get("type") == "response.failed" else event
    code = error.get("code") if isinstance(error, dict) else None
    return str(code) if code else None


class _StreamTranslator:
    """Turns one Responses event stream from ``instance`` into Messages events.

    Messages numbers content blocks from zero in the order they open, so each
    Responses output item (or, for a message, each content part) is given the
    next index when its first block opens. A reasoning item opens its thinking
    block only once it has something to show, so an empty item leaves no block.
    """

    def __init__(self, instance: str) -> None:
        self._instance = instance
        self._pending_reasoning: set[int] = set()
        self._next_index = 0
        self._open: dict[tuple[int, int], int] = {}
        self._kinds: dict[tuple[int, int], str] = {}
        self._arguments_streamed: set[int] = set()
        self._tool_called = False
        self._refused = False
        self._started = False

    def _start(self, key: tuple[int, int], block: dict[str, Any]) -> MessageStreamEvent:
        index = self._next_index
        self._next_index += 1
        self._open[key] = index
        self._kinds[key] = block["type"]
        return ContentBlockStartEvent.model_validate(
            {"type": "content_block_start", "index": index, "content_block": block}
        )

    def _delta(self, key: tuple[int, int], delta: dict[str, Any]) -> MessageStreamEvent:
        return ContentBlockDeltaEvent.model_validate(
            {"type": "content_block_delta", "index": self._open[key], "delta": delta}
        )

    def _stop(self, key: tuple[int, int]) -> MessageStreamEvent:
        self._kinds.pop(key, None)
        return ContentBlockStopEvent.model_validate({"type": "content_block_stop", "index": self._open.pop(key)})

    def _text_delta(self, event: dict[str, Any], text: str) -> list[MessageStreamEvent]:
        key = (int(event.get("output_index") or 0), int(event.get("content_index") or 0))
        out: list[MessageStreamEvent] = []
        if key not in self._open:
            out.append(self._start(key, {"type": "text", "text": ""}))
        if text:
            out.append(self._delta(key, {"type": "text_delta", "text": text}))
        return out

    def _message_start(self, response: dict[str, Any]) -> MessageStreamEvent:
        self._started = True
        return MessageStartEvent.model_validate(
            {
                "type": "message_start",
                "message": {
                    "id": _message_id(response),
                    "type": "message",
                    "role": "assistant",
                    "model": response.get("model") or "",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        )

    def translate(self, event: dict[str, Any]) -> list[MessageStreamEvent]:
        event_type = event.get("type")
        output_index = int(event.get("output_index") or 0)
        whole = (output_index, 0)
        out: list[MessageStreamEvent] = []
        if event_type in ("response.created", "response.in_progress"):
            if not self._started:
                out.append(self._message_start(event.get("response") or {}))
        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                self._tool_called = True
                block: dict[str, Any] = {
                    "type": "tool_use",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "input": {},
                }
                out.append(self._start(whole, block))
            elif item.get("type") == "reasoning":
                self._pending_reasoning.add(output_index)
        elif event_type == "response.output_text.delta":
            out.extend(self._text_delta(event, str(event.get("delta") or "")))
        elif event_type == "response.refusal.delta":
            self._refused = True
            out.extend(self._text_delta(event, str(event.get("delta") or "")))
        elif event_type == "response.function_call_arguments.delta" and whole in self._open:
            self._arguments_streamed.add(output_index)
            out.append(self._delta(whole, {"type": "input_json_delta", "partial_json": str(event.get("delta") or "")}))
        elif event_type == "response.reasoning_summary_part.added" and int(event.get("summary_index") or 0) > 0:
            out.extend(self._thinking_delta(output_index, "\n\n"))
        elif event_type == "response.reasoning_summary_text.delta":
            out.extend(self._thinking_delta(output_index, str(event.get("delta") or "")))
        elif event_type == "response.output_item.done":
            out.extend(self._item_done(output_index, event.get("item") or {}))
        elif event_type in ("response.completed", "response.incomplete"):
            out.extend(self._finish(event.get("response") or {}))
        elif event_type in ("response.failed", "error"):
            code = _stream_error_code(event)
            logger.warning("Responses stream for a Messages request failed: %s code=%s", event_type, code)
            raise ResponsesStreamFailedError(str(event_type), code)
        return out

    def _thinking_delta(self, output_index: int, text: str) -> list[MessageStreamEvent]:
        """A thinking delta for a reasoning item, opening its block first if this is its first text."""
        whole = (output_index, 0)
        out: list[MessageStreamEvent] = []
        if whole not in self._open:
            if output_index not in self._pending_reasoning or not text:
                return out
            self._pending_reasoning.discard(output_index)
            out.append(self._start(whole, {"type": "thinking", "thinking": "", "signature": ""}))
        if text:
            out.append(self._delta(whole, {"type": "thinking_delta", "thinking": text}))
        return out

    def _item_done(self, output_index: int, item: dict[str, Any]) -> list[MessageStreamEvent]:
        whole = (output_index, 0)
        out: list[MessageStreamEvent] = []
        item_type = item.get("type")
        if item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            if encrypted and output_index in self._pending_reasoning:
                self._pending_reasoning.discard(output_index)
                out.append(self._start(whole, {"type": "thinking", "thinking": "", "signature": ""}))
            self._pending_reasoning.discard(output_index)
            if encrypted and whole in self._open:
                signature = reasoning_signature(self._instance, str(encrypted))
                out.append(self._delta(whole, {"type": "signature_delta", "signature": signature}))
        elif item_type == "function_call" and whole in self._open and output_index not in self._arguments_streamed:
            if arguments := item.get("arguments"):
                out.append(self._delta(whole, {"type": "input_json_delta", "partial_json": str(arguments)}))
        elif item_type == "message":
            # A message whose text never streamed still says what it said.
            for content_index, part in enumerate(item.get("content") or []):
                key = (output_index, content_index)
                if key in self._open:
                    continue
                if part.get("type") == "output_text" and part.get("text"):
                    out.extend(
                        self._text_delta({"output_index": output_index, "content_index": content_index}, part["text"])
                    )
                elif part.get("type") == "refusal" and part.get("refusal"):
                    self._refused = True
                    out.extend(
                        self._text_delta(
                            {"output_index": output_index, "content_index": content_index}, part["refusal"]
                        )
                    )
        out.extend(self._stop(key) for key in sorted(self._open, key=self._open.__getitem__) if key[0] == output_index)
        return out

    def _finish(self, response: dict[str, Any]) -> list[MessageStreamEvent]:
        out: list[MessageStreamEvent] = []
        if not self._started:
            out.append(self._message_start(response))
        out.extend(self._stop(key) for key in sorted(self._open, key=self._open.__getitem__))
        stop_reason = _stop_reason(response, tool_called=self._tool_called, refused=self._refused)
        out.append(
            MessageDeltaEvent.model_validate(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": _usage(response.get("usage"))
                    or UnreportedDeltaUsage(output_tokens=0, input_tokens=0, cache_read_input_tokens=0),
                }
            )
        )
        out.append(MessageStopEvent.model_validate({"type": "message_stop"}))
        return out


async def _stream_events(events: AsyncIterator[Any], *, instance: str) -> AsyncIterator[MessageStreamEvent]:
    translator = _StreamTranslator(instance)
    async for raw in events:
        event = _as_dict(raw)
        if not isinstance(event, dict):
            continue
        for translated in translator.translate(event):
            yield translated
