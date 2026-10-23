"""Unit tests for when a stream forwards an event that names a provider's file.

The pipeline copies the files an event cites before it forwards the event, so
a caller never sees a file ID before Otari holds the file's bytes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.api.routes._pipeline import _copying_produced_files
from gateway.services.files import ProviderFile

_CITATION = {"type": "container_file_citation", "file_id": "cfile_1", "filename": "plot.png", "container_id": "c_1"}


async def _order_of(dialect: str, *events: Any) -> list[str]:
    order: list[str] = []

    async def upstream() -> AsyncIterator[Any]:
        for event in events:
            yield event

    async def copy(files: list[ProviderFile]) -> None:
        order.append("copied " + ",".join(file.file_id for file in files))

    async def store(data: bytes, mime_type: str) -> str:
        order.append(f"stored {len(data)} bytes")
        return "file-stored"

    bridge = SimpleNamespace(store_provider_output=store)
    async for event in _copying_produced_files(upstream(), dialect, copy, cast(Any, bridge)):
        order.append(f"sent {event.type}")  # noqa: PERF401 - the test checks when each append happens
    return order


@pytest.mark.asyncio
async def test_a_file_gemini_sent_inline_is_stored_before_its_event_goes_on() -> None:
    result = {
        "type": "code_execution_result",
        "content": [],
        "inline_outputs": [{"mime_type": "image/png", "data": "iVBORw=="}],
    }
    event = SimpleNamespace(
        type="content_block_start", index=0, content_block={"type": "code_execution_tool_result", "content": result}
    )

    order = await _order_of("messages", event)

    assert order == ["stored 4 bytes", "sent content_block_start"]
    assert result["content"] == [{"type": "code_execution_output", "file_id": "file-stored"}]
    assert "inline_outputs" not in result


@pytest.mark.asyncio
async def test_a_messages_event_goes_on_only_after_its_files_are_copied() -> None:
    block = SimpleNamespace(content=SimpleNamespace(content=[SimpleNamespace(file_id="file_01abc")]))
    cites = SimpleNamespace(type="content_block_start", index=0, content_block=block)

    order = await _order_of("messages", cites, SimpleNamespace(type="message_stop"))

    assert order == ["copied file_01abc", "sent content_block_start", "sent message_stop"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event",
    [
        SimpleNamespace(type="response.output_text.annotation.added", annotation=_CITATION),
        SimpleNamespace(type="response.content_part.done", part=SimpleNamespace(annotations=[_CITATION])),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(content=[SimpleNamespace(annotations=[_CITATION])]),
        ),
    ],
)
async def test_a_responses_event_goes_on_only_after_its_files_are_copied(event: Any) -> None:
    order = await _order_of("responses", SimpleNamespace(type="response.output_text.delta"), event)

    assert order == ["sent response.output_text.delta", "copied cfile_1", f"sent {event.type}"]
