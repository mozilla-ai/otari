"""Data access for ``sandbox_containers``, the leases a caller resumes a sandbox by.

The registry that decides ownership, lifetime and exclusivity lives in the
service layer; this module is the only place its statements are spelled.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, or_, select, update

from gateway.core.unit_of_work import UnitOfWork, session_for
from gateway.models.tools import SandboxContainer


@dataclass(frozen=True)
class SandboxContainerRow:
    """One lease, resolved to its columns, with every datetime UTC-aware."""

    id: str
    user_id: str
    workspace_id: uuid.UUID
    provider: str
    provider_session_id: str
    created_at: datetime
    last_used_at: datetime
    expires_at: datetime
    hard_expires_at: datetime
    in_use_until: datetime | None = None


def _as_utc(value: datetime) -> datetime:
    # SQLite hands datetimes back naive; they were stored as UTC.
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _row(model: SandboxContainer) -> SandboxContainerRow:
    return SandboxContainerRow(
        id=model.id,
        user_id=model.user_id,
        workspace_id=model.workspace_id,
        provider=model.provider,
        provider_session_id=model.provider_session_id,
        created_at=_as_utc(model.created_at),
        last_used_at=_as_utc(model.last_used_at),
        expires_at=_as_utc(model.expires_at),
        hard_expires_at=_as_utc(model.hard_expires_at),
        in_use_until=_as_utc(model.in_use_until) if model.in_use_until is not None else None,
    )


async def get_container_row(uow: UnitOfWork, container_id: str) -> SandboxContainerRow | None:
    """The lease behind ``container_id``, whoever owns it; the caller decides who may use it."""
    db = session_for(uow)
    model = (await db.execute(select(SandboxContainer).where(SandboxContainer.id == container_id))).scalar_one_or_none()
    return _row(model) if model is not None else None


async def upsert_container_row(uow: UnitOfWork, row: SandboxContainerRow) -> None:
    """Write the lease: a new row for a first lease, a refreshed one for a resume.

    ``created_at`` and ``hard_expires_at`` are kept from the existing row on a
    resume, so the hard clock never restarts. Flushes; the caller's block commits.
    """
    db = session_for(uow)
    existing = (await db.execute(select(SandboxContainer).where(SandboxContainer.id == row.id))).scalar_one_or_none()
    if existing is None:
        db.add(
            SandboxContainer(
                id=row.id,
                user_id=row.user_id,
                workspace_id=row.workspace_id,
                provider=row.provider,
                provider_session_id=row.provider_session_id,
                created_at=row.created_at,
                last_used_at=row.last_used_at,
                expires_at=row.expires_at,
                hard_expires_at=row.hard_expires_at,
                in_use_until=None,
            )
        )
    else:
        existing.provider_session_id = row.provider_session_id
        existing.last_used_at = row.last_used_at
        existing.expires_at = row.expires_at
        # The request that held the claim is the one writing this, so recording
        # the lease is also how it gives the sandbox back to the next request.
        existing.in_use_until = None
    await db.flush()


async def claim_container_row(uow: UnitOfWork, container_id: str, *, now: datetime, until: datetime) -> bool:
    """Take the exclusive claim on a lease, returning whether this caller got it.

    One conditional update, so two requests racing for the same container cannot
    both win: the loser sees no row updated. A claim whose holder never gave it
    back is retaken once ``in_use_until`` is past. Flushes; the caller's block
    commits.
    """
    db = session_for(uow)
    result = cast(
        "CursorResult[Any]",
        await db.execute(
            update(SandboxContainer)
            .where(
                SandboxContainer.id == container_id,
                or_(SandboxContainer.in_use_until.is_(None), SandboxContainer.in_use_until <= now),
            )
            .values(in_use_until=until)
        ),
    )
    await db.flush()
    return bool(result.rowcount)


async def release_container_claim(uow: UnitOfWork, container_id: str) -> None:
    """Give the claim back without touching the clocks. Flushes; the caller's block commits."""
    db = session_for(uow)
    await db.execute(update(SandboxContainer).where(SandboxContainer.id == container_id).values(in_use_until=None))
    await db.flush()


async def delete_container_rows(uow: UnitOfWork, container_ids: Sequence[str]) -> None:
    """Drop the named leases. Flushes; the caller's block commits."""
    if not container_ids:
        return
    db = session_for(uow)
    await db.execute(delete(SandboxContainer).where(SandboxContainer.id.in_(list(container_ids))))
    await db.flush()


async def expired_container_ids(uow: UnitOfWork, *, now: datetime, batch_size: int) -> list[str]:
    """Ids of up to ``batch_size`` leases past either clock, oldest expiry first."""
    db = session_for(uow)
    result = await db.execute(
        select(SandboxContainer.id)
        .where(or_(SandboxContainer.expires_at <= now, SandboxContainer.hard_expires_at <= now))
        .order_by(SandboxContainer.expires_at)
        .limit(batch_size)
    )
    return list(result.scalars().all())
