"""Withhold provider output references until their durable binding is active."""

import json
from collections.abc import AsyncIterator
from typing import Any

from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import FileMetadata, FilesError, Operation, OutputCleanup, WireModel
from gateway.services.provider_files.references import collect_file_references
from gateway.services.provider_files.transport import provider_client, provider_error


class FileOutputBinder:
    def __init__(self, client: PlatformFilesClient, operation: Operation, existing_ids: list[str]) -> None:
        self.client = client
        self.operation = operation
        self.bound = set(existing_ids)

    async def register(self, value: Any) -> None:
        for file_id in collect_file_references(value):
            if file_id in self.bound:
                continue
            try:
                async with provider_client(self.operation.account) as provider:
                    result = await provider.aretrieve_file(file_id, max_retries=0)
                    metadata = FileMetadata.model_validate(result.model_dump(exclude_unset=True))
            except Exception:
                await self.compensate(None, file_id)
                raise FilesError(502, "Provider file metadata could not be registered") from None
            try:
                await self.client.retry(
                    "outputs/register",
                    {
                        "operation_id": str(self.operation.id),
                        "metadata": metadata.model_dump(mode="json", exclude_unset=True),
                    },
                    FileMetadata,
                )
            except FilesError:
                await self.compensate(metadata)
                raise
            self.bound.add(file_id)

    async def complete(self) -> None:
        try:
            await self.client.retry(
                f"outputs/{self.operation.id}/complete",
                {"cleanup_token": self.operation.cleanup_token.get_secret_value(), "deleted": False},
                WireModel,
            )
        except FilesError:
            # The durable deadline releases reserved capacity if the peer is unavailable.
            pass

    async def compensate(self, metadata: FileMetadata | None, file_id: str | None = None) -> None:
        try:
            cleanup = await self.client.retry(
                f"outputs/{self.operation.id}/abandon",
                {
                    "cleanup_token": self.operation.cleanup_token.get_secret_value(),
                    "metadata": metadata.model_dump(mode="json", exclude_unset=True) if metadata else None,
                    "file_id": file_id,
                },
                OutputCleanup,
            )
            # The authority proves this ID belongs to this operation before deletion.
            if cleanup.operation_id is None or cleanup.cleanup_token is None:
                return
            deleted = False
            try:
                async with provider_client(self.operation.account) as provider:
                    await provider.adelete_file(metadata.id if metadata else file_id, max_retries=0)
                deleted = True
            except Exception as exc:
                deleted = provider_error(exc).status_code == 404
            await self.client.retry(
                f"{cleanup.operation_id}/cleanup-result",
                {"cleanup_token": cleanup.cleanup_token.get_secret_value(), "deleted": deleted},
                WireModel,
            )
        except FilesError:
            pass

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
                        "tool_result" in candidate.get("type", "") or collect_file_references(candidate)
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
