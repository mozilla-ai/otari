"""Public Files envelopes; provider SDK translation belongs to any-llm."""

from typing import Any, Literal

from fastapi import Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from gateway.services.provider_files.capabilities import require_file_operation
from gateway.services.provider_files.contracts import FileListRequest, FileMetadata, FilePage, FilesError


class AnthropicFileMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["file"] = "file"
    filename: str
    mime_type: str
    size_bytes: int
    created_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    downloadable: bool


class OpenAIFileMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["file"] = "file"
    filename: str | None = None
    bytes: int | None = None
    created_at: int | None = None
    expires_at: int | None = None
    purpose: str | None = None
    status: str | None = None


class AnthropicFilePage(BaseModel):
    data: list[AnthropicFileMetadata]
    next_page: str | None = None


class OpenAIFilePage(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIFileMetadata]
    first_id: str | None
    last_id: str | None
    has_more: bool


class AnthropicFileDeleted(BaseModel):
    id: str
    type: Literal["file_deleted"] = "file_deleted"


class OpenAIFileDeleted(BaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: Literal[True] = True


class AnthropicFilesFormat:
    provider = "anthropic"
    upload_fields = frozenset({"expires_in_seconds"})
    max_retention_seconds = 7776000

    def headers(self, request: Request) -> dict[str, str]:
        version = request.headers.get("anthropic-version")
        if not version:
            raise FilesError(400, "Hybrid provider-native Files require the Anthropic API contract (anthropic-version)")
        beta = request.headers.get("anthropic-beta", "")
        if "files-api-2025-04-14" in {value.strip() for value in beta.split(",")}:
            raise FilesError(400, "Hybrid Files require the GA API; the legacy Files beta is unsupported")
        return {"anthropic-version": version, **({"anthropic-beta": beta} if beta else {})}

    def upload_options(self, fields: dict[str, str]) -> tuple[int | None, str | None]:
        duration = _duration(fields.get("expires_in_seconds"))
        if duration is not None and not 3600 <= duration <= 7776000:
            raise FilesError(400, "File retention must be between one hour and 90 days")
        return duration, None

    def list_request(self, request: Request) -> FileListRequest:
        query = request.query_params
        if set(query) - {"page", "limit", "ids[]"}:
            raise FilesError(400, "Hybrid Files require GA pagination (page, limit, ids[])")
        _unique_query(request, {"page", "limit"})
        try:
            return FileListRequest(
                provider=self.provider,
                page=query.get("page"),
                limit=int(query["limit"]) if "limit" in query else None,
                ids=query.getlist("ids[]") if "ids[]" in query else None,
            )
        except (ValueError, ValidationError):
            raise FilesError(400, "Invalid Files pagination") from None

    def metadata(self, value: FileMetadata) -> AnthropicFileMetadata:
        data = value.model_dump(exclude_unset=True)
        for key in ("purpose", "status", "object"):
            data.pop(key, None)
        try:
            return AnthropicFileMetadata.model_validate({**data, "type": "file"})
        except ValidationError:
            raise FilesError(502, "Provider returned invalid file metadata") from None

    def page(self, value: FilePage) -> AnthropicFilePage:
        return AnthropicFilePage(data=[self.metadata(item) for item in value.data], next_page=value.next_page)

    def deleted(self, file_id: str) -> AnthropicFileDeleted:
        return AnthropicFileDeleted(id=file_id)


class OpenAIFilesFormat:
    provider = "openai"
    upload_fields = frozenset({"purpose", "expires_after[anchor]", "expires_after[seconds]"})
    max_retention_seconds = 2592000

    def headers(self, request: Request) -> dict[str, str]:
        if any(name in request.headers for name in ("anthropic-version", "anthropic-beta")):
            raise FilesError(400, "Conflicting Files API headers")
        return {}

    def upload_options(self, fields: dict[str, str]) -> tuple[int | None, str | None]:
        purpose = fields.get("purpose")
        if not purpose or not purpose.strip() or len(purpose) > 255:
            raise FilesError(400, "File purpose is required")
        anchor, seconds = fields.get("expires_after[anchor]"), fields.get("expires_after[seconds]")
        if (anchor is not None or seconds is not None) and (anchor != "created_at" or seconds is None):
            raise FilesError(400, "Invalid file retention")
        duration = _duration(seconds)
        if duration is not None and not 3600 <= duration <= self.max_retention_seconds:
            raise FilesError(400, "File retention must be between one hour and 30 days")
        return duration, purpose

    def list_request(self, request: Request) -> FileListRequest:
        query = request.query_params
        allowed = {"after", "before", "limit", "order", "purpose"}
        if set(query) - allowed:
            raise FilesError(400, "Unsupported Files pagination parameter")
        _unique_query(request, allowed)
        try:
            return FileListRequest.model_validate(
                {
                    "provider": self.provider,
                    "after_id": query.get("after"),
                    "before_id": query.get("before"),
                    "limit": int(query["limit"]) if "limit" in query else None,
                    "order": query.get("order", "desc"),
                    "sort_by": "provider_created_at",
                    "purpose": query.get("purpose"),
                }
            )
        except (ValueError, ValidationError):
            raise FilesError(400, "Invalid Files pagination") from None

    def metadata(self, value: FileMetadata) -> OpenAIFileMetadata:
        data: dict[str, Any] = value.model_dump(exclude_unset=True)
        for key in ("type", "mime_type", "downloadable"):
            data.pop(key, None)
        if "size_bytes" in data:
            data["bytes"] = data.pop("size_bytes")
        for key in ("created_at", "expires_at"):
            if data.get(key) is not None:
                data[key] = int(data[key].timestamp())
        try:
            return OpenAIFileMetadata.model_validate({**data, "object": "file"})
        except ValidationError:
            raise FilesError(502, "Provider returned invalid file metadata") from None

    def page(self, value: FilePage) -> OpenAIFilePage:
        return OpenAIFilePage(
            object="list",
            data=[self.metadata(item) for item in value.data],
            first_id=value.data[0].id if value.data else None,
            last_id=value.data[-1].id if value.data else None,
            has_more=value.next_page is not None,
        )

    def deleted(self, file_id: str) -> OpenAIFileDeleted:
        return OpenAIFileDeleted(id=file_id)


def _duration(raw: str | None) -> int | None:
    if raw is None:
        return None
    if not raw.isascii() or not raw.isdigit() or len(raw) > 8 or int(raw) <= 0:
        raise FilesError(400, "Invalid file retention")
    return int(raw)


def _unique_query(request: Request, names: set[str]) -> None:
    if any(len(request.query_params.getlist(name)) > 1 for name in names):
        raise FilesError(400, "Duplicate pagination parameter")


FilesFormat = AnthropicFilesFormat | OpenAIFilesFormat
_FORMATS: dict[str, FilesFormat] = {"anthropic": AnthropicFilesFormat(), "openai": OpenAIFilesFormat()}


def files_format(request: Request) -> FilesFormat:
    values = request.headers.getlist("x-otari-files-provider")
    if len(values) > 1:
        raise FilesError(400, "Duplicate Files provider header")
    provider = values[0] if values else "anthropic"
    require_file_operation(provider, "upload")
    result = _FORMATS.get(provider)
    if result is None:
        raise FilesError(400, "No public Files API format is configured for this provider")
    result.headers(request)
    return result
