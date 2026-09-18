"""Additive gateway/control-plane contracts for provider-native files."""

import uuid
from datetime import datetime
from typing import Annotated, Literal, Self

from any_llm.types.files import FileMetadata as SDKFileMetadata
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr, model_validator


class FilesError(Exception):
    """A fixed, caller-safe file operation failure."""

    def __init__(self, status_code: int, detail: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or ({"Retry-After": "60"} if status_code == 429 else {})


class WireModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


ProviderFileId = Annotated[str, Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_-]+$")]


ProviderName = Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_]*$")]
FILES_PROTOCOL_VERSION = "2"


class FileMetadata(SDKFileMetadata, WireModel):
    """Bounded any-llm metadata; absent provider fields remain unknown."""

    model_config = ConfigDict(extra="allow")

    id: ProviderFileId
    filename: str | None = Field(default=None, max_length=1024)
    mime_type: str | None = Field(default=None, max_length=255)
    size_bytes: int | None = Field(default=None, ge=0)
    created_at: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None
    purpose: str | None = Field(default=None, max_length=255)
    status: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def bounded_metadata(self) -> Self:
        if len(self.model_dump_json().encode()) > 16384:
            raise ValueError("Provider file metadata exceeds the size limit")
        return self


class FilePage(WireModel):
    data: list[FileMetadata]
    next_page: str | None = None


class FileListRequest(WireModel):
    provider: ProviderName = "anthropic"
    purpose: str | None = Field(default=None, max_length=255)
    after_id: ProviderFileId | None = None
    before_id: ProviderFileId | None = None
    order: Literal["asc", "desc"] = "desc"
    sort_by: Literal["binding_created_at", "provider_created_at"] = "binding_created_at"
    page: str | None = Field(default=None, max_length=4096)
    limit: int | None = Field(default=None, ge=1, le=1000)
    ids: list[ProviderFileId] | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def compatible_filters(self) -> Self:
        if sum(value is not None for value in (self.page, self.after_id, self.before_id)) > 1:
            raise ValueError("Only one pagination cursor is allowed")
        if self.ids is not None and any(
            value is not None for value in (self.page, self.limit, self.after_id, self.before_id)
        ):
            raise ValueError("ids[] cannot be combined with page or limit")
        return self


class FileScope(WireModel):
    """Derived from authenticated gateway and workspace API key, never a public body."""

    organization_id: uuid.UUID
    workspace_id: uuid.UUID
    user_id: str = Field(max_length=255)
    gateway_id: str = Field(max_length=255)
    default_gateway: bool = False


class FileAccount(WireModel):
    generation_id: uuid.UUID
    provider: ProviderName = "anthropic"
    api_key: SecretStr
    api_base: str | None = None
    workspace: str | None = None
    managed: bool = False


class Operation(WireModel):
    id: uuid.UUID
    cleanup_token: SecretStr
    deadline: datetime
    account: FileAccount
    max_bytes: int = Field(gt=0)
    expires_in_seconds: int = Field(ge=3600, le=7776000)


class PrepareUpload(WireModel):
    provider: ProviderName = "anthropic"
    operation_id: uuid.UUID
    size_bytes: int = Field(ge=0)
    expires_in_seconds: int | None = Field(default=None, ge=3600, le=7776000)


class FinalizeUpload(WireModel):
    expires_in_seconds: int | None = Field(default=None, ge=3600, le=7776000)
    metadata: FileMetadata


class AbandonUpload(WireModel):
    file_id: str | None = Field(default=None, max_length=255, pattern=r"^[A-Za-z0-9_-]+$")
    cleanup_token: SecretStr
    metadata: FileMetadata | None = None
    deleted: bool = False
    outcome_unknown: bool = False


class ResolveFile(WireModel):
    provider: ProviderName = "anthropic"
    operation: Literal["metadata", "download", "delete"]


class ResolvedFile(WireModel):
    metadata: FileMetadata
    account: FileAccount | None = None
    operation_id: uuid.UUID | None = None
    cleanup_token: SecretStr | None = None


class References(WireModel):
    provider: ProviderName = "anthropic"
    ids: list[ProviderFileId] = Field(min_length=1, max_length=100)


class CleanupResult(WireModel):
    cleanup_token: SecretStr
    deleted: bool


class OutputPrepare(WireModel):
    operation_id: uuid.UUID
    request_id: str = Field(max_length=255)
    attempt_id: str = Field(max_length=255)
    generation_id: uuid.UUID


class OutputRegister(WireModel):
    operation_id: uuid.UUID
    metadata: FileMetadata


class CleanupClaim(WireModel):
    limit: int = Field(default=20, ge=1, le=20)


class CleanupItem(WireModel):
    binding_id: uuid.UUID
    file_id: str


class CleanupLease(WireModel):
    id: uuid.UUID
    token: SecretStr
    deadline: datetime
    account: FileAccount
    items: list[CleanupItem]


class LeaseResult(WireModel):
    token: SecretStr
    results: dict[uuid.UUID, bool] = Field(max_length=20)


class OutputCleanup(WireModel):
    operation_id: uuid.UUID
    cleanup_token: SecretStr
