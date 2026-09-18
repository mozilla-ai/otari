"""Anthropic message blocks and SSE buffering around shared output registration."""

import json
from collections.abc import AsyncIterator
from typing import Any

from gateway.services.provider_files.contracts import FilesError
from gateway.services.provider_files.inference import FileOutputBinder
from gateway.services.provider_files.references import collect_anthropic_file_references


class AnthropicFileOutputBinder(FileOutputBinder):
    async def register(self, value: Any) -> None:
        await self.register_ids(collect_anthropic_file_references(value))

    async def stream(self, source: AsyncIterator[Any]) -> AsyncIterator[Any]:
        held: list[Any] = []
        block: dict[str, Any] | None = None
        size = 0
        partial = ""
        try:
            async for event in source:
                payload = event.model_dump(exclude_unset=True)
                kind = payload.get("type")
                if kind == "content_block_start":
                    candidate = payload.get("content_block", {})
                    # Hold structured provider results, keeping all later events behind them.
                    if isinstance(candidate, dict) and (
                        "tool_result" in candidate.get("type", "") or collect_anthropic_file_references(candidate)
                    ):
                        block = candidate
                if block is not None:
                    size += len(event.model_dump_json())
                    if size > 1024 * 1024 or len(held) >= 4096:
                        raise FilesError(502, "Provider file output exceeds registration limits")
                    held.append(event)
                    delta = payload.get("delta", {})
                    if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
                        partial += delta.get("partial_json", "")
                    if kind == "content_block_stop":
                        if partial:
                            try:
                                decoded = json.loads(partial)
                            except ValueError:
                                raise FilesError(502, "Invalid provider file output") from None
                            await self.register(decoded)
                        await self.register(block)
                        for buffered in held:
                            yield buffered
                        held, block, size, partial = [], None, 0, ""
                else:
                    await self.register(payload)
                    yield event
            if held:
                raise FilesError(502, "Incomplete provider file output")
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()
            await self.complete()
