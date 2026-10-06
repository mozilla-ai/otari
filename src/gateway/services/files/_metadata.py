"""What a stored file's metadata is derived from: its media type and when it stops being served."""

import mimetypes
from datetime import UTC, datetime, timedelta

from gateway.core.config import GatewayConfig


def guess_mime_type(filename: str | None, declared: str | None = None) -> str:
    """The media type for ``filename``: the declared one when it says something, else by extension."""
    if declared and declared != "application/octet-stream":
        return declared
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return declared or "application/octet-stream"


def expiry_for(config: GatewayConfig, now: datetime | None = None) -> datetime | None:
    """When a file stored now stops being served, or ``None`` when files are kept indefinitely."""
    if config.files_retention_hours is None:
        return None
    return (now or datetime.now(UTC)) + timedelta(hours=config.files_retention_hours)
