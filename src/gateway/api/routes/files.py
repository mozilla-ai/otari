"""OpenAI-compatible file upload/storage endpoints.

Stores uploaded files so they can later be referenced from chat messages by
``file_id``. The content normalizer resolves those references and either
forwards them to natively-capable providers or extracts them to text for
text-only local models.

Files carry two scopes. ``user_id`` is the owner, resolved from the authenticated
principal. ``workspace_id`` is the workspace the upload was made in, taken off the
API key that authenticated it and never from a header, exactly as every other
request-plane row does. A keyed request is confined to its own key's workspace on
every verb; a master-key request is the operator acting deployment-wide and sees
every workspace, narrowable on the listing with ``workspace_id``, matching
``GET /api/v1/keys``.

The same five routes serve two SDKs. OpenAI's and Anthropic's Files APIs share
their paths and verbs and differ only in the JSON they return, so the response
shape follows the caller: a request carrying Anthropic's ``anthropic-version``
header (which its SDK sends on every call) gets ``FileMetadata``, everything
else gets the OpenAI file object.
The Anthropic flavor is its GA shape only, so a request for the Files API beta is a 400.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse

from gateway.api.deps import FileServiceDep, get_config, verify_api_key_or_master_key
from gateway.api.routes._helpers import resolve_user_id
from gateway.core.config import GatewayConfig
from gateway.exceptions.files_exceptions import FilesDisabledError
from gateway.models.api_keys import APIKey
from gateway.schemas.files import (
    AnthropicFileDeleted,
    AnthropicFileList,
    AnthropicFileMetadata,
    OpenAIFileDeleted,
    OpenAIFileList,
    OpenAIFileObject,
)
from gateway.services.files import (
    DEFAULT_LIST_LIMIT,
    MAX_LIST_LIMIT,
    FileDialect,
    FileListing,
    FileScope,
    NewFile,
)

_FILES_BETA = "files-api-2025-04-14"


async def _refuse_files_beta(raw_request: Request) -> None:
    """Refuse Anthropic's Files API beta, whose shapes differ from the GA shapes served here."""
    for header_value in raw_request.headers.getlist("anthropic-beta"):
        if _FILES_BETA in (beta.strip() for beta in header_value.split(",")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The {_FILES_BETA} beta is not supported: send Files API requests without it in anthropic-beta",
            )


async def _require_files_enabled(config: Annotated[GatewayConfig, Depends(get_config)]) -> None:
    """Refuse every verb when the deployment does not serve files.

    A router dependency, so the refusal comes before the request is parsed and
    before the caller is authenticated. That is what makes a switched-off
    feature answer like an unmounted one instead of leaking, through the
    refusal it picks, that the paths are there at all.
    """
    if not config.files_enabled:
        raise FilesDisabledError


router = APIRouter(
    tags=["files"], dependencies=[Depends(_require_files_enabled), Depends(_refuse_files_beta)]
)

# OpenAI's documented file purposes plus a generic default. We don't enforce the
# enum (forward-compat), but normalise the empty case to "user_data".
_DEFAULT_PURPOSE = "user_data"

# The most files ``ids[]`` may name in one page.
_MAX_LIST_IDS = 100

_READ_CHUNK_BYTES = 1024 * 1024


def _dialect(raw_request: Request) -> FileDialect:
    """Which SDK's Files API the caller speaks."""
    if "anthropic-version" in raw_request.headers:
        return FileDialect.ANTHROPIC
    return FileDialect.OPENAI


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


def _scope(auth_result: tuple[APIKey | None, bool], user: str | None, config: GatewayConfig) -> FileScope:
    """The rows this request may reach: its billed user, and its key's workspace."""
    return FileScope(
        user_id=_resolve_user(auth_result, user, config),
        workspace_id=_request_workspace_id(auth_result),
    )


async def _upload_chunks(file: UploadFile) -> AsyncIterator[bytes]:
    """Read the upload off the request a chunk at a time, so it is never held whole."""
    while chunk := await file.read(_READ_CHUNK_BYTES):
        yield chunk


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
    config: Annotated[GatewayConfig, Depends(get_config)],
    files: FileServiceDep,
    file: UploadFile = File(...),
    purpose: str = Form(_DEFAULT_PURPOSE),
    user: str | None = Form(None),
) -> OpenAIFileObject | AnthropicFileMetadata:
    """Upload a file. Answers in the OpenAI or Anthropic file shape, following the caller's headers."""
    record = await files.store(
        NewFile(
            user_id=_resolve_user(auth_result, user, config),
            workspace_id=_request_workspace_id(auth_result),
            filename=file.filename,
            content_type=file.content_type,
            purpose=purpose or _DEFAULT_PURPOSE,
            chunks=_upload_chunks(file),
        )
    )
    if _dialect(raw_request) is FileDialect.ANTHROPIC:
        return AnthropicFileMetadata.of(record)
    return OpenAIFileObject.of(record)


@router.get("/files")
async def list_files(
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    files: FileServiceDep,
    user: str | None = None,
    purpose: str | None = None,
    workspace_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIST_LIMIT)] = DEFAULT_LIST_LIMIT,
    after: str | None = None,
    order: Literal["asc", "desc"] = "desc",
    page: str | None = None,
    ids: Annotated[list[str] | None, Query(alias="ids[]")] = None,
) -> OpenAIFileList | AnthropicFileList:
    """List the authenticated user's uploaded files in the request's workspace.

    ``workspace_id`` narrows a master-key listing to one workspace; a keyed
    request is already confined to its key's own and cannot widen or move it.

    Each flavor pages with its own cursor.
    OpenAI's ``after`` names the last file of the previous page, and ``has_more`` says whether to ask again.
    Anthropic's ``next_page`` is passed back as ``page``, and ``ids[]`` reads up to 100 named files in one page.
    A cursor whose file has since been deleted or has expired is still a position.
    An ``after`` the caller never owned is a 404, and a ``page`` token this gateway did not issue is a 400.
    """
    dialect = _dialect(raw_request)
    if dialect is FileDialect.ANTHROPIC:
        named_ids = None if ids is None else list(dict.fromkeys(ids))
        _check_anthropic_list_params(raw_request, page, named_ids)
        cursor = page
    else:
        named_ids, cursor = None, after
    # The key's own workspace wins over anything the caller sent, rather than
    # 400ing on a mismatch: the parameter is a master-key narrowing, and a keyed
    # request is confined either way, so refusing it would only add a way to get
    # an error instead of the same answer.
    scope = FileScope(
        user_id=_resolve_user(auth_result, user, config),
        workspace_id=_request_workspace_id(auth_result) or workspace_id,
    )

    result = await files.page(
        FileListing(
            scope=scope,
            dialect=dialect,
            limit=limit,
            ascending=order == "asc",
            purpose=purpose,
            file_ids=named_ids,
            cursor=cursor,
        )
    )
    if dialect is FileDialect.ANTHROPIC:
        return AnthropicFileList(
            data=[AnthropicFileMetadata.of(record) for record in result.files],
            next_page=result.next_cursor,
        )
    return OpenAIFileList(
        data=[OpenAIFileObject.of(record) for record in result.files],
        has_more=result.next_cursor is not None,
        first_id=result.files[0].id if result.files else None,
        last_id=result.files[-1].id if result.files else None,
    )


@router.get("/files/{file_id}")
async def get_file(
    file_id: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    files: FileServiceDep,
    user: str | None = None,
) -> OpenAIFileObject | AnthropicFileMetadata:
    """Retrieve metadata for a single file."""
    record = await files.stored_file(file_id, _scope(auth_result, user, config))
    if _dialect(raw_request) is FileDialect.ANTHROPIC:
        return AnthropicFileMetadata.of(record)
    return OpenAIFileObject.of(record)


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
    config: Annotated[GatewayConfig, Depends(get_config)],
    files: FileServiceDep,
    user: str | None = None,
) -> Response:
    """Download the raw bytes of a file, streamed rather than buffered whole."""
    content = await files.content(file_id, _scope(auth_result, user, config))
    # No Content-Length: it would come from the row's byte count while the body
    # comes from the storage backend. If those ever diverge (partial write,
    # corruption), a length header derived from the row would be wrong, and
    # clients trust that header over what actually arrives. Chunked transfer
    # encoding doesn't need to declare a length up front.
    return StreamingResponse(
        content.chunks,
        media_type=content.mime_type,
        headers={"Content-Disposition": _content_disposition(content.filename)},
    )


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    raw_request: Request,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    files: FileServiceDep,
    user: str | None = None,
) -> OpenAIFileDeleted | AnthropicFileDeleted:
    """Soft-delete a file's metadata and remove its bytes from the backend."""
    await files.discard(file_id, _scope(auth_result, user, config))
    if _dialect(raw_request) is FileDialect.ANTHROPIC:
        return AnthropicFileDeleted(id=file_id)
    return OpenAIFileDeleted(id=file_id)
