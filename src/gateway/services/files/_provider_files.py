"""The files a provider's own code execution produced, as its response announces them.

A provider-native code execution keeps what it wrote in the provider's container
and answers with the provider's file ID.
The provider does not keep it for long: OpenAI discards a container 20 minutes after its last use.
Gemini is the exception, answering with the bytes themselves, which are stored
and named by an ID of Otari's own (``store_inline_outputs``).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Awaitable, Callable, Iterable
from datetime import timedelta
from typing import Any

from anthropic.types import CodeExecutionOutputBlock
from any_llm import LLMProvider

from gateway.types.provider_file import ProviderFile

# The shortest life a provider will give a file it stores. Anthropic's Files
# API takes an expiry between one hour and 90 days, so a copy cannot be asked
# for less than an hour. A provider absent here sets no floor of its own.
_MINIMUM_COPY_LIFETIMES = {LLMProvider.ANTHROPIC: timedelta(hours=1)}


def _anthropic_files_in(blocks: list[Any]) -> list[ProviderFile]:
    files: dict[str, ProviderFile] = {}
    for block in blocks:
        content = getattr(block, "content", None)
        for output in getattr(content, "content", None) or []:
            file_id = getattr(output, "file_id", None)
            if isinstance(file_id, str) and file_id and file_id not in files:
                files[file_id] = ProviderFile(file_id=file_id)
    return list(files.values())


def anthropic_produced_files(result: Any) -> list[ProviderFile]:
    """Provider file ids in an Anthropic Messages response's tool results.

    Both the python and the bash variant of the tool name their outputs in a
    block of their own, and neither gives a filename, so the shape is what this
    looks for rather than a block name.
    """
    return _anthropic_files_in(list(getattr(result, "content", None) or []))


def _field(obj: Any, name: str) -> Any:
    """``obj``'s ``name``, whether a stream event carried it as an object or as a plain dict."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _cited_files(annotations: Iterable[Any]) -> list[ProviderFile]:
    """The files the ``container_file_citation`` annotations among ``annotations`` name, each once."""
    files: dict[str, ProviderFile] = {}
    for note in annotations:
        if _field(note, "type") != "container_file_citation":
            continue
        file_id = _field(note, "file_id")
        if not isinstance(file_id, str) or not file_id or file_id in files:
            continue
        filename, container_id = _field(note, "filename"), _field(note, "container_id")
        files[file_id] = ProviderFile(
            file_id=file_id,
            filename=filename if isinstance(filename, str) and filename else None,
            container_id=container_id if isinstance(container_id, str) and container_id else None,
        )
    return list(files.values())


def _annotations_in(parts: Any) -> list[Any]:
    return [note for part in parts or [] for note in _field(part, "annotations") or []]


def responses_produced_files(result: Any) -> list[ProviderFile]:
    """Provider file ids an OpenAI Responses reply cites from its container.

    OpenAI announces a produced file as a ``container_file_citation``
    annotation on the message it wrote, which carries the container and the
    name as well as the id.
    """
    items = _field(result, "output") or []
    return _cited_files(note for item in items for note in _annotations_in(_field(item, "content")))


def produced_files_for(dialect: str, obj: Any) -> list[ProviderFile]:
    """Provider file ids in a completed reply, or in one streamed event, of ``dialect``.

    A Messages stream delivers a server tool result whole in its
    ``content_block_start`` event. A Responses stream names a cited file in the
    annotation, content part and output item events before it repeats the whole
    response on ``response.completed``. Anything else (a delta, a chat
    completion, which has no native code tool) names no file.
    """
    kind = getattr(obj, "type", None)
    if dialect == "messages":
        if kind == "content_block_start":
            return _anthropic_files_in([getattr(obj, "content_block", None)])
        return anthropic_produced_files(obj)
    if dialect == "responses":
        if kind == "response.output_text.annotation.added":
            return _cited_files([_field(obj, "annotation")])
        if kind == "response.content_part.done":
            return _cited_files(_annotations_in([_field(obj, "part")]))
        if kind == "response.output_item.done":
            return _cited_files(_annotations_in(_field(_field(obj, "item"), "content")))
        if kind == "response.completed":
            return responses_produced_files(_field(obj, "response"))
        return responses_produced_files(obj)
    return []


# Stores one inline output's bytes and returns the file ID it is served under, or ``None``.
InlineOutputStore = Callable[[bytes, str], Awaitable[str | None]]


def _decoded(data: Any) -> bytes | None:
    if not isinstance(data, str) or not data:
        return None
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None


async def _store_messages_outputs(block: Any, store: InlineOutputStore) -> None:
    """Swap a ``code_execution_tool_result``'s ``inline_outputs`` for ``code_execution_output`` file blocks.

    The Messages bridge carries what Gemini's code produced as bytes on an extra
    field, because Anthropic's shape names a produced file by an ID only a file
    store can mint. This is that store, so the caller gets the shape Anthropic's
    own replies have. An output that could not be stored stays on
    ``inline_outputs``, as the bytes on Chat Completions do.
    """
    if _field(block, "type") != "code_execution_tool_result":
        return
    result = _field(block, "content")
    extra = result if isinstance(result, dict) else getattr(result, "__pydantic_extra__", None)
    if not isinstance(extra, dict) or not isinstance(outputs := extra.get("inline_outputs"), list):
        return
    produced: list[CodeExecutionOutputBlock] = []
    unstored: list[Any] = []
    for output in outputs:
        data = _decoded(_field(output, "data"))
        file_id = await store(data, str(_field(output, "mime_type") or "")) if data is not None else None
        if file_id is None:
            unstored.append(output)
        else:
            produced.append(CodeExecutionOutputBlock(type="code_execution_output", file_id=file_id))
    if unstored:
        extra["inline_outputs"] = unstored
    else:
        del extra["inline_outputs"]
    if not produced:
        return
    if isinstance(result, dict):
        result["content"] = [*(result.get("content") or []), *(output.model_dump() for output in produced)]
    else:
        result.content = [*(result.content or []), *produced]


async def _store_chat_outputs(extra_content: Any, store: InlineOutputStore) -> None:
    """Swap the bytes on Gemini's ``code_execution_output`` items for the file ID they are stored under."""
    google = extra_content.get("google") if isinstance(extra_content, dict) else None
    items = google.get("code_execution") if isinstance(google, dict) else None
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or item.get("type") != "code_execution_output":
            continue
        data = _decoded(item.get("data"))
        file_id = await store(data, str(item.get("mime_type") or "")) if data is not None else None
        if file_id is not None:
            item.pop("data", None)
            item["file_id"] = file_id


async def store_inline_outputs(dialect: str, obj: Any, store: InlineOutputStore) -> None:
    """Store the files a completed reply, or one streamed event, of ``dialect`` carries inline, in place.

    Only Gemini returns produced files as bytes, on Chat Completions (on the
    ``code_execution`` items in ``extra_content``) and on Messages (on the
    ``code_execution_tool_result`` blocks the bridge builds, whole in a stream's
    ``content_block_start``).
    """
    if dialect == "messages":
        blocks = (
            [_field(obj, "content_block")] if _field(obj, "type") == "content_block_start" else _field(obj, "content")
        )
        for block in blocks or []:
            await _store_messages_outputs(block, store)
    elif dialect == "chat":
        for choice in _field(obj, "choices") or []:
            holder = _field(choice, "message") or _field(choice, "delta")
            await _store_chat_outputs(_field(holder, "extra_content"), store)


def minimum_copy_lifetime(provider: LLMProvider) -> timedelta:
    """The shortest life ``provider`` will give a file it stores, or zero where it sets no floor."""
    return _MINIMUM_COPY_LIFETIMES.get(provider, timedelta(0))
