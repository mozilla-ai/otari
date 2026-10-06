"""The files a provider's own code execution produced, as its response announces them.

A provider-native code execution keeps what it wrote in the provider's container
and answers with the provider's file ID.
The provider does not keep it for long: OpenAI discards a container 20 minutes after its last use.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from typing import Any

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


def minimum_copy_lifetime(provider: LLMProvider) -> timedelta:
    """The shortest life ``provider`` will give a file it stores, or zero where it sets no floor."""
    return _MINIMUM_COPY_LIFETIMES.get(provider, timedelta(0))
