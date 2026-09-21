"""Withhold provider output references until their durable binding is active."""

import asyncio
from collections.abc import Iterable
from contextlib import AsyncExitStack
from itertools import batched
from typing import Any

from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import FileMetadata, FilesError, Operation, OutputCleanup, WireModel
from gateway.services.provider_files.transport import provider_client, provider_error


class FileOutputBinder:
    def __init__(self, client: PlatformFilesClient, operation: Operation, existing_ids: list[str]) -> None:
        self.client = client
        self.operation = operation
        self.bound = set(existing_ids)

    async def register_ids(self, ids: Iterable[str]) -> None:
        pending = [file_id for file_id in dict.fromkeys(ids) if file_id not in self.bound]
        if not pending:
            return
        async with AsyncExitStack() as stack:
            try:
                provider = await stack.enter_async_context(provider_client(self.operation.account))
            except Exception:
                for file_id in pending:
                    await self.compensate(None, file_id)
                raise FilesError(502, "Provider file metadata could not be registered") from None
            for batch in batched(pending, 4):
                # Finish every sibling's registration or compensation before releasing the client.
                results = await asyncio.gather(
                    *(self._register_id(provider, file_id) for file_id in batch), return_exceptions=True
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result

    async def _register_id(self, provider: Any, file_id: str) -> None:
        try:
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
