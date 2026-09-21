"""Provider-native Files with thin native API envelopes on a stateless gateway."""

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import AsyncExitStack
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from starlette.types import Receive, Scope, Send

from gateway.api.deps import extract_credential_token, get_config
from gateway.api.routes._file_formats import (
    AnthropicFileDeleted,
    AnthropicFileMetadata,
    AnthropicFilePage,
    OpenAIFileDeleted,
    OpenAIFileMetadata,
    OpenAIFilePage,
    files_format,
)
from gateway.core.config import GatewayConfig
from gateway.inflight import track_request
from gateway.services.provider_files.capabilities import check_file_account, require_download, require_file_operation
from gateway.services.provider_files.client import PlatformFilesClient
from gateway.services.provider_files.contracts import (
    FILES_PROTOCOL_VERSION,
    FileMetadata,
    FilePage,
    FilesError,
    Operation,
    ResolvedFile,
    WireModel,
)
from gateway.services.provider_files.transfers import UploadAdmission, receive_upload
from gateway.services.provider_files.transport import provider_client, provider_error


class FileDownloadResponse(StreamingResponse):
    def __init__(self, *args: Any, stack: AsyncExitStack, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stack = stack

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.stack.aclose()


class FilesRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def handle(request: Request) -> Response:
            try:
                result = await handler(request)
            except RequestValidationError:
                raise HTTPException(
                    400,
                    "Invalid file operation",
                    headers={
                        "Cache-Control": "private, no-store",
                        "X-Otari-Files-Protocol": FILES_PROTOCOL_VERSION,
                    },
                ) from None
            except FilesError as exc:
                raise HTTPException(
                    exc.status_code,
                    exc.detail,
                    headers={
                        "Cache-Control": "private, no-store",
                        "X-Otari-Files-Protocol": FILES_PROTOCOL_VERSION,
                        **exc.headers,
                    },
                ) from None
            except TimeoutError:
                raise HTTPException(
                    504,
                    "File transfer timed out",
                    headers={
                        "Cache-Control": "private, no-store",
                        "X-Otari-Files-Protocol": FILES_PROTOCOL_VERSION,
                    },
                ) from None
            result.headers["Cache-Control"] = "private, no-store"
            result.headers["X-Otari-Files-Protocol"] = FILES_PROTOCOL_VERSION
            return result

        return handle


router = APIRouter(tags=["files"], route_class=FilesRoute)
Config = Annotated[GatewayConfig, Depends(get_config)]


def files_client(request: Request, config: GatewayConfig) -> PlatformFilesClient:
    if not config.files_enabled or not config.files_provider_native_enabled:
        raise FilesError(404, "Provider-native Files are not enabled")
    files_format(request)
    token = extract_credential_token(request)
    base = config.platform.get("base_url")
    if not base or not config.platform_token:
        raise FilesError(502, "Authorization service unavailable")
    return PlatformFilesClient(
        base, config.platform_token, token, timeout=int(config.platform.get("resolve_timeout_ms", 5000)) / 1000
    )


def _admission(request: Request, config: GatewayConfig) -> UploadAdmission:
    admission = getattr(request.app.state, "provider_file_upload_admission", None)
    if not isinstance(admission, UploadAdmission):
        admission = UploadAdmission(config.files_temporary_capacity_bytes)
        request.app.state.provider_file_upload_admission = admission
    return admission


@router.post("/files", response_model=AnthropicFileMetadata | OpenAIFileMetadata, response_model_exclude_unset=True)
async def upload_file(request: Request, config: Config) -> AnthropicFileMetadata | OpenAIFileMetadata:
    client = files_client(request, config)
    envelope = files_format(request)
    require_file_operation(envelope.provider, "upload")
    require_file_operation(envelope.provider, "delete")
    headers = envelope.headers(request)
    operation: Operation | None = None
    metadata: FileMetadata | None = None
    started = committed = False
    try:
        async with asyncio.timeout(config.files_transfer_timeout_seconds):
            operation = await client.retry(
                "uploads/prepare",
                {
                    "operation_id": str(uuid.uuid4()),
                    "size_bytes": config.files_max_bytes,
                    "provider": envelope.provider,
                },
                Operation,
            )
            check_file_account(operation.account, envelope.provider)
            maximum = min(config.files_max_bytes, operation.max_bytes)
            track_request(request, endpoint="/files", model="files", provider=operation.account.provider)
            async with _admission(request, config).reserve(maximum + 65536):
                async with receive_upload(
                    request.headers,
                    request.stream(),
                    max_bytes=maximum,
                    idle_seconds=config.files_idle_timeout_seconds,
                    allowed_fields=envelope.upload_fields,
                ) as (upload, fields):
                    duration, purpose = envelope.upload_options(fields)
                    retention = min(
                        duration or operation.expires_in_seconds,
                        operation.expires_in_seconds,
                        envelope.max_retention_seconds,
                    )
                    async with provider_client(
                        operation.account, idle_timeout=config.files_idle_timeout_seconds
                    ) as provider:
                        started = True
                        result = await provider.aupload_file(
                            upload.file,
                            filename=upload.filename,
                            mime_type=upload.content_type,
                            expires_in=retention,
                            purpose=purpose,
                            max_retries=0,
                            extra_headers=headers,
                        )
                        metadata = FileMetadata.model_validate(result.model_dump(exclude_unset=True))
                    finalized = await client.retry(
                        f"uploads/{operation.id}/finalize",
                        {
                            "metadata": metadata.model_dump(mode="json", exclude_unset=True),
                            "expires_in_seconds": retention,
                        },
                        FileMetadata,
                    )
                    committed = True
                    return envelope.metadata(finalized)
    except BaseException as exc:
        # Past finalize the binding is committed, so a failure here is only the
        # response's; compensating would delete an upload that succeeded.
        if operation is not None and not committed:
            await _compensate_upload(client, operation, metadata, headers, started, exc)
        if isinstance(exc, (FilesError, asyncio.CancelledError, TimeoutError)):
            raise
        if isinstance(exc, Exception):
            raise provider_error(exc) from None
        raise


async def _compensate_upload(
    client: PlatformFilesClient,
    operation: Operation,
    metadata: FileMetadata | None,
    headers: dict[str, str],
    started: bool,
    failure: BaseException,
) -> None:
    async def compensate() -> None:
        deleted = False
        if metadata is not None and not (isinstance(failure, FilesError) and failure.status_code == 409):
            try:
                async with provider_client(operation.account) as provider:
                    await provider.adelete_file(metadata.id, max_retries=0, extra_headers=headers)
                deleted = True
            except Exception as exc:
                deleted = provider_error(exc).status_code == 404
        try:
            await client.retry(
                f"uploads/{operation.id}/abandon",
                {
                    "cleanup_token": operation.cleanup_token.get_secret_value(),
                    "metadata": metadata.model_dump(mode="json", exclude_unset=True) if metadata else None,
                    "deleted": deleted,
                    "outcome_unknown": started
                    and metadata is None
                    and (
                        not isinstance(failure, Exception)
                        or provider_error(failure).status_code not in {400, 404, 413, 429}
                    ),
                },
                WireModel,
            )
        except FilesError:
            pass

    async def bounded_compensate() -> None:
        try:
            await asyncio.wait_for(compensate(), timeout=20)
        except TimeoutError:
            pass

    await asyncio.shield(asyncio.create_task(bounded_compensate()))


@router.get("/files", response_model=AnthropicFilePage | OpenAIFilePage, response_model_exclude_unset=True)
async def list_files(request: Request, config: Config) -> AnthropicFilePage | OpenAIFilePage:
    client = files_client(request, config)
    envelope = files_format(request)
    parsed = envelope.list_request(request)
    result = await client.post("list", parsed.model_dump(exclude_none=True), FilePage)
    return envelope.page(result)


@router.get(
    "/files/{file_id}", response_model=AnthropicFileMetadata | OpenAIFileMetadata, response_model_exclude_unset=True
)
async def retrieve_file(file_id: str, request: Request, config: Config) -> AnthropicFileMetadata | OpenAIFileMetadata:
    client = files_client(request, config)
    envelope = files_format(request)
    resolved = await client.post(
        f"{quote(file_id, safe='')}/resolve", {"operation": "metadata", "provider": envelope.provider}, ResolvedFile
    )
    return envelope.metadata(resolved.metadata)


def _download_headers(upstream: Mapping[str, str], max_bytes: int) -> dict[str, str]:
    """Reject an oversized download before the 200 is committed; keep the descriptive headers."""
    lowered = {name.lower(): value for name, value in upstream.items()}
    headers = {name: lowered[name] for name in ("content-type", "content-disposition") if name in lowered}
    declared = lowered.get("content-length")
    if declared is None:
        return headers
    try:
        length = int(declared)
    except ValueError:
        raise FilesError(502, "Provider returned an invalid download") from None
    if length > max_bytes:
        raise FilesError(413, "File size limit exceeded")
    # The SDK decodes the body, so the length only describes it when nothing was encoded.
    if lowered.get("content-encoding", "identity") == "identity":
        headers["content-length"] = declared
    return headers


@router.get("/files/{file_id}/content")
async def download_file(file_id: str, request: Request, config: Config) -> Response:
    client = files_client(request, config)
    envelope = files_format(request)
    resolved = await client.post(
        f"{quote(file_id, safe='')}/resolve", {"operation": "download", "provider": envelope.provider}, ResolvedFile
    )
    if resolved.account is None:
        raise FilesError(502, "Authorization service returned an invalid file account")
    check_file_account(resolved.account, envelope.provider)
    require_download(envelope.provider, resolved.metadata)
    if resolved.metadata.size_bytes is not None and resolved.metadata.size_bytes > config.files_max_bytes:
        raise FilesError(413, "File size limit exceeded")
    track_request(request, endpoint="/files", model="files", provider=resolved.account.provider)
    stack = AsyncExitStack()
    deadline = asyncio.get_running_loop().time() + config.files_transfer_timeout_seconds
    try:
        provider = await stack.enter_async_context(
            provider_client(resolved.account, idle_timeout=config.files_idle_timeout_seconds)
        )
        async with asyncio.timeout(config.files_idle_timeout_seconds):
            download = await stack.enter_async_context(
                provider.adownload_file(file_id, max_retries=0, extra_headers=envelope.headers(request))
            )
    except BaseException as exc:
        await stack.aclose()
        if not isinstance(exc, Exception):
            raise
        raise provider_error(exc) from None
    try:
        headers = _download_headers(download.headers, config.files_max_bytes)
    except BaseException:
        await stack.aclose()
        raise

    async def chunks() -> AsyncIterator[bytes]:
        total = 0
        try:
            iterator = aiter(download)
            while True:
                try:
                    async with asyncio.timeout_at(
                        min(deadline, asyncio.get_running_loop().time() + config.files_idle_timeout_seconds)
                    ):
                        chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                total += len(chunk)
                if total > config.files_max_bytes:
                    # Past the headers by now; the abort leaves the body
                    # unterminated so the client sees a failed transfer.
                    raise FilesError(413, "File size limit exceeded")
                yield chunk
        finally:
            await stack.aclose()

    return FileDownloadResponse(chunks(), stack=stack, headers=headers, media_type="application/octet-stream")


@router.delete("/files/{file_id}", response_model=AnthropicFileDeleted | OpenAIFileDeleted)
async def delete_file(file_id: str, request: Request, config: Config) -> AnthropicFileDeleted | OpenAIFileDeleted:
    client = files_client(request, config)
    envelope = files_format(request)
    require_file_operation(envelope.provider, "delete")
    resolved = await client.post(
        f"{quote(file_id, safe='')}/resolve", {"operation": "delete", "provider": envelope.provider}, ResolvedFile
    )
    if resolved.account is None or resolved.cleanup_token is None or resolved.operation_id is None:
        raise FilesError(502, "Authorization service returned an invalid cleanup response")
    check_file_account(resolved.account, envelope.provider)
    track_request(request, endpoint="/files", model="files", provider=resolved.account.provider)
    failure = None
    try:
        async with (
            asyncio.timeout(config.files_transfer_timeout_seconds),
            provider_client(resolved.account) as provider,
        ):
            await provider.adelete_file(file_id, max_retries=0, extra_headers=envelope.headers(request))
    except Exception as exc:
        failure = provider_error(exc)
        if failure.status_code == 404:
            failure = None
    await client.retry(
        f"{resolved.operation_id}/cleanup-result",
        {
            "cleanup_token": resolved.cleanup_token.get_secret_value(),
            "deleted": failure is None,
        },
        WireModel,
    )
    if failure is not None:
        raise failure
    return envelope.deleted(file_id)
