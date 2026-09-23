"""Request and response models of the files domain.

The same five routes serve OpenAI's Files API and Anthropic's GA Files API.
The two share their paths and verbs and differ only in the JSON they carry, so
every answer has a pair of shapes here and the caller's headers pick one.
"""

from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import BaseModel

from gateway.models.files import FileObject


def _as_utc(value: datetime) -> datetime:
    """Read a stored datetime as the UTC it was written as.

    SQLite hands datetimes back naive, and both conversions below would then
    read them as local time and skew the answer by the server's UTC offset.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _epoch_seconds(value: datetime) -> int:
    """Return a UTC epoch from a stored datetime."""
    return int(_as_utc(value).timestamp())


def _rfc3339(value: datetime) -> str:
    """Return an RFC 3339 timestamp from a stored datetime."""
    return _as_utc(value).isoformat().replace("+00:00", "Z")


class OpenAIFileObject(BaseModel):
    """One file in OpenAI's file object shape."""

    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    expires_at: int | None
    filename: str
    purpose: str

    @classmethod
    def of(cls, record: FileObject) -> Self:
        """Build the shape from a stored file's row."""
        return cls(
            id=record.id,
            bytes=record.bytes,
            created_at=_epoch_seconds(record.created_at),
            expires_at=None if record.expires_at is None else _epoch_seconds(record.expires_at),
            filename=record.filename,
            purpose=record.purpose,
        )


class OpenAIFileList(BaseModel):
    """A page of files in OpenAI's list shape, whose cursor is the last entry's ID."""

    object: Literal["list"] = "list"
    data: list[OpenAIFileObject]
    has_more: bool
    first_id: str | None
    last_id: str | None


class OpenAIFileDeleted(BaseModel):
    """OpenAI's answer to a delete."""

    id: str
    object: Literal["file"] = "file"
    deleted: Literal[True] = True


class AnthropicFileMetadata(BaseModel):
    """One file in the ``FileMetadata`` shape of Anthropic's GA Files API.

    ``expires_at`` is always present and ``None`` for a file kept indefinitely.
    ``downloadable`` is always true, because the gateway serves every stored file's bytes back.
    """

    id: str
    type: Literal["file"] = "file"
    filename: str
    mime_type: str
    size_bytes: int
    created_at: str
    expires_at: str | None
    downloadable: Literal[True] = True

    @classmethod
    def of(cls, record: FileObject) -> Self:
        """Build the shape from a stored file's row."""
        return cls(
            id=record.id,
            filename=record.filename,
            mime_type=record.mime_type,
            size_bytes=record.bytes,
            created_at=_rfc3339(record.created_at),
            expires_at=None if record.expires_at is None else _rfc3339(record.expires_at),
        )


class AnthropicFileList(BaseModel):
    """A page of files in Anthropic's list shape, whose cursor is an opaque token."""

    data: list[AnthropicFileMetadata]
    next_page: str | None


class AnthropicFileDeleted(BaseModel):
    """Anthropic's answer to a delete."""

    id: str
    type: Literal["file_deleted"] = "file_deleted"
