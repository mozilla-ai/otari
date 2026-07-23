"""OpenAI-compatible file upload/storage endpoints.

Stores uploaded files so they can later be referenced from chat messages by
``file_id``. The content normalizer (gateway.services.content_normalizer)
resolves those references and either forwards them to natively-capable
providers or extracts them to text for text-only local models.
"""

import mimetypes
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_file_store, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.entities import APIKey, FileObject
from gateway.services.file_service import fetch_file
from gateway.services.file_store import FileStore

router = APIRouter(prefix="/v1", tags=["files"])

# OpenAI's documented file purposes plus a generic default. We don't enforce the
# enum (forward-compat), but normalise the empty case to "user_data".
_DEFAULT_PURPOSE = "user_data"


def _resolve_user(
    auth_result: tuple[APIKey | None, bool],
    user: str | None,
    config: GatewayConfig,
) -> str:
    api_key, is_master_key = auth_result
    return resolve_user_id(
        user_id_from_request=user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When using master key, 'user' field is required",
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation failed",
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key has no associated user",
        ),
        forbidden_user_error=HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="'user' field does not match the authenticated API key's user",
        ),
        reject_mismatch=config.reject_user_mismatch,
    )


_READ_CHUNK_BYTES = 1024 * 1024


async def _capped_chunks(file: UploadFile, max_bytes: int) -> AsyncIterator[bytes]:
    """Yield an upload's bytes in chunks, aborting with 413 past ``max_bytes``.

    This is what keeps the size cap an HTTP concern living in the route: it
    yields chunks straight through to the storage backend (via
    ``FileStore.put_stream``) instead of accumulating them, so the cap is
    enforced as bytes flow rather than after a full buffer is built.
    """
    total = 0
    while chunk := await file.read(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds maximum upload size of {max_bytes // (1024 * 1024)} MB",
            )
        yield chunk


async def _prime(chunks: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
    """Eagerly read ``chunks``' first item so a read failure raises here, not later.

    ``StreamingResponse`` only touches its body iterator after the 200 status
    and headers are already flushed. Without this, a missing or unreadable blob
    (DB record present, disk blob gone or corrupted) would truncate an
    already-started 200 response instead of failing with a clean 500. Awaiting
    the first chunk before constructing the response surfaces that failure as
    a normal exception in the route.
    """
    try:
        first: bytes | None = await chunks.__anext__()
    except StopAsyncIteration:
        first = None

    async def _rest() -> AsyncGenerator[bytes, None]:
        # StreamingResponse closes this outer generator on early client
        # disconnect, but that doesn't automatically propagate to closing
        # `chunks` (no such thing for a bare `async for`) — without this
        # try/finally, `get_stream`'s file handle stays open until GC
        # eventually gets to the abandoned generator, which under real
        # traffic (cancelled downloads, closed tabs) means fds pile up.
        try:
            if first is not None:
                yield first
                async for chunk in chunks:
                    yield chunk
        finally:
            await chunks.aclose()

    return _rest()


def _content_disposition(filename: str) -> str:
    """Build a Content-Disposition header value that is safe from injection.

    ``filename`` is user-controlled (set at upload), so interpolating it raw
    would allow CRLF/quote header injection. We emit an ASCII-sanitized
    ``filename`` for legacy clients plus an RFC 5987 percent-encoded
    ``filename*`` for the real (possibly non-ASCII) name.
    """
    ascii_name = "".join(c for c in filename if c.isprintable() and c not in '"\\').encode(
        "ascii", "ignore"
    ).decode("ascii")
    ascii_name = ascii_name.strip() or "download"
    encoded = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"


def _guess_mime(filename: str | None, declared: str | None) -> str:
    if declared and declared != "application/octet-stream":
        return declared
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return declared or "application/octet-stream"


@router.post("/files")
async def create_file(
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    file_store: Annotated[FileStore, Depends(get_file_store)],
    file: UploadFile = File(...),
    purpose: str = Form(_DEFAULT_PURPOSE),
    user: str | None = Form(None),
) -> dict[str, Any]:
    """OpenAI-compatible file upload endpoint."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)

    file_id = f"file-{uuid.uuid4().hex}"
    storage_ref, size = await file_store.put_stream(file_id, _capped_chunks(file, config.files_max_bytes))
    if size == 0:
        # The empty check now happens after the stream drains (we don't know
        # the size until then), so a zero-byte blob must be cleaned up before
        # rejecting the upload.
        await file_store.delete(storage_ref)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    expires_at: datetime | None = None
    if config.files_retention_hours is not None:
        expires_at = datetime.now(UTC) + timedelta(hours=config.files_retention_hours)

    record = FileObject(
        id=file_id,
        user_id=user_id,
        filename=file.filename or file_id,
        mime_type=_guess_mime(file.filename, file.content_type),
        bytes=size,
        purpose=purpose or _DEFAULT_PURPOSE,
        storage_ref=storage_ref,
        created_at=datetime.now(UTC),
        expires_at=expires_at,
    )
    db.add(record)
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        # The bytes were written before the metadata commit; drop them so a
        # failed insert doesn't leak an unreferenced blob.
        await file_store.delete(storage_ref)
        logger.error("Failed to persist file metadata for %s: %s", file_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store file",
        ) from exc

    logger.info("Stored file %s (%d bytes) for user %s", file_id, size, user_id)
    return record.to_dict()


@router.get("/files")
async def list_files(
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user: str | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    """List the authenticated user's uploaded files."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    stmt = select(FileObject).where(
        FileObject.user_id == user_id,
        FileObject.deleted_at.is_(None),
    )
    if purpose is not None:
        stmt = stmt.where(FileObject.purpose == purpose)
    stmt = stmt.order_by(FileObject.created_at.desc())

    result = await db.execute(stmt)
    records = result.scalars().all()
    return {"object": "list", "data": [r.to_dict() for r in records]}


@router.get("/files/{file_id}")
async def get_file(
    file_id: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user: str | None = None,
) -> dict[str, Any]:
    """Retrieve metadata for a single file."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    record = await fetch_file(db, file_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return record.to_dict()


@router.get("/files/{file_id}/content")
async def get_file_content(
    file_id: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    file_store: Annotated[FileStore, Depends(get_file_store)],
    user: str | None = None,
) -> Response:
    """Download the raw bytes of a file, streamed rather than buffered whole."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    record = await fetch_file(db, file_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    # No Content-Length: it would come from record.bytes (DB) while the body
    # comes from the storage backend (disk). If those ever diverge (partial
    # write, corruption), a length header derived from the DB value would be
    # wrong, and clients trust that header over what actually arrives. Chunked
    # transfer encoding doesn't need to declare a length up front.
    try:
        body = await _prime(file_store.get_stream(record.storage_ref))
    except OSError as exc:
        logger.error("Failed to read blob for file %s (ref=%s): %s", file_id, record.storage_ref, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read file",
        ) from exc

    return StreamingResponse(
        body,
        media_type=record.mime_type,
        headers={"Content-Disposition": _content_disposition(record.filename)},
    )


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    file_store: Annotated[FileStore, Depends(get_file_store)],
    user: str | None = None,
) -> dict[str, Any]:
    """Soft-delete a file's metadata and remove its bytes from the backend."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    record = await fetch_file(db, file_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    storage_ref = record.storage_ref
    record.deleted_at = datetime.now(UTC)
    try:
        await db.commit()
    except SQLAlchemyError as exc:
        await db.rollback()
        logger.error("Failed to delete file %s: %s", file_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete file",
        ) from exc

    # The soft-delete already committed, so the file is gone from the user's
    # view. Removing the blob is best-effort cleanup: a backend failure must not
    # turn a successful delete into a 500 (it would only leave an orphaned blob).
    try:
        await file_store.delete(storage_ref)
    except OSError as exc:
        logger.warning("Soft-deleted file %s but failed to remove its blob %s: %s", file_id, storage_ref, exc)

    return {"id": file_id, "object": "file", "deleted": True}
