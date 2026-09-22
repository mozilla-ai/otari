"""OpenAI-compatible file upload/storage endpoints.

Stores uploaded files so they can later be referenced from chat messages by
``file_id``. The content normalizer (gateway.services.content_normalizer)
resolves those references and either forwards them to natively-capable
providers or extracts them to text for text-only local models.

Files carry two scopes. ``user_id`` is the owner, resolved from the authenticated
principal. ``workspace_id`` is the workspace the upload was made in, taken off the
API key that authenticated it and never from a header, exactly as every other
request-plane row does (``services/workspace_scope``). A keyed request is confined
to its own key's workspace on every verb; a master-key request is the operator
acting deployment-wide and sees every workspace, narrowable on the listing with
``workspace_id``, matching ``GET /api/v1/keys``.

The same five routes serve two SDKs. OpenAI's and Anthropic's Files APIs share
their paths and verbs and differ only in the JSON they return, so the response
shape follows the caller: a request carrying Anthropic's ``anthropic-version``
header (which its SDK sends on every call) gets ``FileMetadata``, everything
else gets the OpenAI file object.
The Anthropic flavor is its GA shape only, so a request for the Files API beta is a 400.
"""

import base64
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, get_file_store, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.core.config import GatewayConfig
from gateway.log_config import logger
from gateway.models.api_keys import APIKey
from gateway.models.tools import FileObject
from gateway.ports.file_storage_port import FileStoragePort
from gateway.services.file_service import expiry_for, fetch_file, guess_mime_type
from gateway.services.workspace_scope import default_workspace_id

_FILES_BETA = "files-api-2025-04-14"


async def _refuse_files_beta(raw_request: Request) -> None:
    """Refuse Anthropic's Files API beta, whose shapes differ from the GA shapes served here."""
    for header_value in raw_request.headers.getlist("anthropic-beta"):
        if _FILES_BETA in (beta.strip() for beta in header_value.split(",")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The {_FILES_BETA} beta is not supported: send Files API requests without it in anthropic-beta",
            )


router = APIRouter(tags=["files"], dependencies=[Depends(_refuse_files_beta)])

# OpenAI's documented file purposes plus a generic default. We don't enforce the
# enum (forward-compat), but normalise the empty case to "user_data".
_DEFAULT_PURPOSE = "user_data"

# Listing page bounds. The default is OpenAI's; the ceiling is well under
# OpenAI's 10000 because a page is one query and one JSON body.
_DEFAULT_LIST_LIMIT = 100
_MAX_LIST_LIMIT = 1000
_MAX_LIST_IDS = 100

_PAGE_TOKEN_PREFIX = "page_"


def _anthropic_shape(raw_request: Request) -> bool:
    """Whether the caller speaks Anthropic's Files API rather than OpenAI's."""
    return "anthropic-version" in raw_request.headers


def _page_token(file_id: str) -> str:
    """The opaque Anthropic ``next_page`` token that resumes a listing after ``file_id``."""
    return _PAGE_TOKEN_PREFIX + base64.urlsafe_b64encode(file_id.encode()).decode().rstrip("=")


def _could_name_a_file(value: str) -> bool:
    # Every file ID is printable ASCII, and PostgreSQL rejects a NUL in a text parameter.
    return value.isascii() and value.isprintable()


def _invalid_page_token() -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid page token")


def _page_token_file_id(token: str) -> str:
    """The file ID a ``page`` token resumes after, raising a 400 for a token this gateway did not issue."""
    if not token.startswith(_PAGE_TOKEN_PREFIX):
        raise _invalid_page_token()
    encoded = token.removeprefix(_PAGE_TOKEN_PREFIX)
    try:
        file_id = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except ValueError as exc:
        raise _invalid_page_token() from exc
    if not _could_name_a_file(file_id):
        raise _invalid_page_token()
    return file_id


def _check_anthropic_list_params(raw_request: Request, page: str | None, ids: list[str] | None) -> None:
    """Refuse the parameter combinations Anthropic's GA listing refuses, given de-duplicated ``ids``."""
    params = raw_request.query_params
    if "after_id" in params or "before_id" in params:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="after_id and before_id are not supported: pass next_page back as page instead",
        )
    if ids is None:
        return
    if page is not None or "limit" in params:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="ids[] cannot be combined with page or limit"
        )
    if len(ids) > _MAX_LIST_IDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"ids[] takes at most {_MAX_LIST_IDS} file IDs"
        )


def _serialize(record: FileObject, raw_request: Request) -> dict[str, Any]:
    return record.to_anthropic_dict() if _anthropic_shape(raw_request) else record.to_dict()


def _request_workspace_id(auth_result: tuple[APIKey | None, bool]) -> uuid.UUID | None:
    """The workspace a keyed request is confined to, or ``None`` for the master key.

    Read off the key rather than from a header: a caller controls its headers and
    not which key it holds, so a header here would let anyone reach another
    workspace's files. ``None`` for the master key is deliberate, and is what keeps
    an existing deployment's operator tooling working: the master key is the
    operator acting deployment-wide, so it is not narrowed to any one workspace.
    """
    api_key, _is_master_key = auth_result
    return api_key.workspace_id if api_key is not None else None


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
    ``FileStoragePort.put_stream``) instead of accumulating them, so the cap is
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


@router.post("/files")
async def create_file(
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    file_store: Annotated[FileStoragePort, Depends(get_file_store)],
    file: UploadFile = File(...),
    purpose: str = Form(_DEFAULT_PURPOSE),
    user: str | None = Form(None),
) -> dict[str, Any]:
    """Upload a file. Answers in the OpenAI or Anthropic file shape, following the caller's headers."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    # A master-key upload has no key to read a workspace off, so it lands in the
    # default workspace: the operator acting deployment-wide, which is the same
    # answer `resolve_workspace_id` gives every other master-key write.
    workspace_id = _request_workspace_id(auth_result) or await default_workspace_id(db)

    file_id = f"file-{uuid.uuid4().hex}"
    storage_ref, size = await file_store.put_stream(file_id, _capped_chunks(file, config.files_max_bytes))
    if size == 0:
        # The empty check now happens after the stream drains (we don't know
        # the size until then), so a zero-byte blob must be cleaned up before
        # rejecting the upload.
        await file_store.delete(storage_ref)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    now = datetime.now(UTC)
    record = FileObject(
        id=file_id,
        user_id=user_id,
        workspace_id=workspace_id,
        filename=file.filename or file_id,
        mime_type=guess_mime_type(file.filename, file.content_type),
        bytes=size,
        purpose=purpose or _DEFAULT_PURPOSE,
        storage_ref=storage_ref,
        created_at=now,
        expires_at=expiry_for(config, now),
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

    logger.info(
        "Stored file %s (%d bytes) for user %s in workspace %s", file_id, size, user_id, workspace_id
    )
    return _serialize(record, raw_request)


@router.get("/files")
async def list_files(
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user: str | None = None,
    purpose: str | None = None,
    workspace_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=_MAX_LIST_LIMIT)] = _DEFAULT_LIST_LIMIT,
    after: str | None = None,
    order: Literal["asc", "desc"] = "desc",
    page: str | None = None,
    ids: Annotated[list[str] | None, Query(alias="ids[]")] = None,
) -> dict[str, Any]:
    """List the authenticated user's uploaded files in the request's workspace.

    ``workspace_id`` narrows a master-key listing to one workspace; a keyed
    request is already confined to its key's own and cannot widen or move it.

    Each flavor pages with its own cursor.
    OpenAI's ``after`` names the last file of the previous page, and ``has_more`` says whether to ask again.
    Anthropic's ``next_page`` is passed back as ``page``, and ``ids[]`` reads up to 100 named files in one page.
    A cursor whose file has since been deleted or has expired is still a position.
    An ``after`` the caller never owned is a 404, and a ``page`` token this gateway did not issue is a 400.
    """
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    anthropic = _anthropic_shape(raw_request)
    if anthropic:
        named_ids = None if ids is None else list(dict.fromkeys(ids))
        _check_anthropic_list_params(raw_request, page, named_ids)
        cursor_id = _page_token_file_id(page) if page is not None else None
    else:
        named_ids, cursor_id = None, after
    user_id = _resolve_user(auth_result, user, config)
    # The key's own workspace wins over anything the caller sent, rather than
    # 400ing on a mismatch: the parameter is a master-key narrowing, and a keyed
    # request is confined either way, so refusing it would only add a way to get
    # an error instead of the same answer.
    scope = _request_workspace_id(auth_result) or workspace_id
    # Expired rows are excluded for the same reason ``fetch_file`` excludes
    # them: the sweep reclaims them on a timer, so between expiry and the next
    # tick a listing would otherwise offer files that every other verb 404s.
    stmt = select(FileObject).where(
        FileObject.user_id == user_id,
        FileObject.deleted_at.is_(None),
        or_(FileObject.expires_at.is_(None), FileObject.expires_at > datetime.now(UTC)),
    )
    if scope is not None:
        stmt = stmt.where(FileObject.workspace_id == scope)
    if purpose is not None:
        stmt = stmt.where(FileObject.purpose == purpose)
    if named_ids is not None:
        stmt = stmt.where(FileObject.id.in_([file_id for file_id in named_ids if _could_name_a_file(file_id)]))

    if cursor_id is not None:
        # A position, not a file: the row is read with the tenant predicates
        # only, so a cursor that was deleted or expired between two pages (the
        # usual "list, delete each, list again" loop) still says where the next
        # page starts. Another user's ID is refused.
        cursor_conditions = [FileObject.id == cursor_id, FileObject.user_id == user_id]
        if scope is not None:
            cursor_conditions.append(FileObject.workspace_id == scope)
        cursor = (await db.execute(select(FileObject).where(*cursor_conditions))).scalar_one_or_none()
        if cursor is None and anthropic:
            raise _invalid_page_token()
        if cursor is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
        # (created_at, id) is the sort key, so the page after the cursor is
        # everything strictly past it in that order. Spelled as two clauses
        # rather than a row-value comparison, which SQLite only partly supports.
        if order == "desc":
            past = or_(
                FileObject.created_at < cursor.created_at,
                and_(FileObject.created_at == cursor.created_at, FileObject.id < cursor.id),
            )
        else:
            past = or_(
                FileObject.created_at > cursor.created_at,
                and_(FileObject.created_at == cursor.created_at, FileObject.id > cursor.id),
            )
        stmt = stmt.where(past)

    if order == "desc":
        stmt = stmt.order_by(FileObject.created_at.desc(), FileObject.id.desc())
    else:
        stmt = stmt.order_by(FileObject.created_at.asc(), FileObject.id.asc())
    # One past the page tells us whether there is a next one without a count.
    records = list((await db.execute(stmt.limit(limit + 1))).scalars().all())
    has_more = len(records) > limit
    records = records[:limit]
    data = [_serialize(r, raw_request) for r in records]

    if anthropic:
        return {"data": data, "next_page": _page_token(records[-1].id) if has_more else None}
    return {
        "object": "list",
        "data": data,
        "has_more": has_more,
        "first_id": records[0].id if records else None,
        "last_id": records[-1].id if records else None,
    }


@router.get("/files/{file_id}")
async def get_file(
    file_id: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    user: str | None = None,
) -> dict[str, Any]:
    """Retrieve metadata for a single file."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    record = await fetch_file(db, file_id, user_id, workspace_id=_request_workspace_id(auth_result))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return _serialize(record, raw_request)


@router.get(
    "/files/{file_id}/content",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": (
                "File content. Content-Type reflects the stored media type; application/octet-stream is the fallback."
            ),
            "content": {
                "application/octet-stream": {"schema": {"type": "string", "format": "binary"}},
                "*/*": {"schema": {"type": "string", "format": "binary"}},
            },
            "headers": {
                "Content-Disposition": {
                    "description": "Attachment filename, with a UTF-8 filename* parameter for non-ASCII names.",
                    "schema": {"type": "string"},
                }
            },
        }
    },
)
async def get_file_content(
    file_id: str,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    file_store: Annotated[FileStoragePort, Depends(get_file_store)],
    user: str | None = None,
) -> Response:
    """Download the raw bytes of a file, streamed rather than buffered whole."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    record = await fetch_file(db, file_id, user_id, workspace_id=_request_workspace_id(auth_result))
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    # No Content-Length: it would come from record.bytes (DB) while the body
    # comes from the storage backend (disk). If those ever diverge (partial
    # write, corruption), a length header derived from the DB value would be
    # wrong, and clients trust that header over what actually arrives. Chunked
    # transfer encoding doesn't need to declare a length up front.
    if record.storage_ref is None:
        logger.error("File %s has no stored bytes", file_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

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
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    file_store: Annotated[FileStoragePort, Depends(get_file_store)],
    user: str | None = None,
) -> dict[str, Any]:
    """Soft-delete a file's metadata and remove its bytes from the backend."""
    if not config.files_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File uploads are disabled")

    user_id = _resolve_user(auth_result, user, config)
    record = await fetch_file(db, file_id, user_id, workspace_id=_request_workspace_id(auth_result))
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
        if storage_ref is not None:
            await file_store.delete(storage_ref)
    except OSError as exc:
        logger.warning("Soft-deleted file %s but failed to remove its blob %s: %s", file_id, storage_ref, exc)

    if _anthropic_shape(raw_request):
        return {"id": file_id, "type": "file_deleted"}
    return {"id": file_id, "object": "file", "deleted": True}
