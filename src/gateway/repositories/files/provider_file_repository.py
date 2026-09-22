"""Data access for files a provider's own sandbox produced.

A row normally carries a ``storage_ref``, because the bytes are copied into
Otari's store as the run is recorded: a provider reclaims its container long
before the caller is finished with the chart it drew. ``storage_ref`` is
``None`` only for a copy that could not be made, and for rows written before
copying existed; those still name the provider, so
``GET /v1/files/{id}/content`` can stream them on demand for as long as the
provider keeps them. See ``services/files/provider_files.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select

from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.tools import FileObject


@dataclass(frozen=True)
class ProviderFileRow:
    """One file to record, already resolved to the columns it lands in."""

    file_id: str
    user_id: str
    workspace_id: uuid.UUID
    filename: str
    mime_type: str
    purpose: str
    provider: str
    provider_instance: str | None
    container_id: str | None
    expires_at: datetime | None
    # Where Otari put the bytes, and how many. ``None`` leaves the row serving
    # by proxy from the provider, which is all a failed copy can still offer.
    storage_ref: str | None = None
    size_bytes: int = 0


async def existing_file_ids(uow: UnitOfWork, file_ids: Collection[str]) -> set[str]:
    """Which of ``file_ids`` already have a row, recorded or uploaded."""
    if not file_ids:
        return set()
    result = await session_for(uow).execute(select(FileObject.id).where(FileObject.id.in_(list(file_ids))))
    return set(result.scalars())


async def record_provider_file_rows(uow: UnitOfWork, rows: Collection[ProviderFileRow]) -> None:
    """Stage one row per provider-held file. Flushes; the caller's unit of work commits."""
    db = session_for(uow)
    for row in rows:
        db.add(
            FileObject(
                id=row.file_id,
                user_id=row.user_id,
                workspace_id=row.workspace_id,
                filename=row.filename,
                mime_type=row.mime_type,
                #0 for a row still served by proxy: the provider does not say
                # how many bytes it holds until they are read.
                bytes=row.size_bytes,
                purpose=row.purpose,
                storage_ref=row.storage_ref,
                provider=row.provider,
                provider_instance=row.provider_instance,
                provider_container_id=row.container_id,
                expires_at=row.expires_at,
            )
        )
    await db.flush()
