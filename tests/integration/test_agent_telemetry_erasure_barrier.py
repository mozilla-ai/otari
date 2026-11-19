"""The telemetry erasure barrier on PostgreSQL: a write and an erasure of its user never interleave.

Each test pauses a write after it has taken the barrier, the moment the issue's
race needed: the export was admitted, its user's erasure starts, and the write
has not inserted yet.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import col

import gateway.adapters.telemetry_storage_adapter as adapter_module
from gateway.adapters.telemetry_storage_adapter import DatabaseTelemetryStorageAdapter
from gateway.core.sql import insert_ignoring_duplicates
from gateway.models.api_keys import APIKey
from gateway.models.tenancy import Workspace
from gateway.models.usage import AgentTelemetry
from gateway.models.users import User
from gateway.ports.telemetry_storage_port import TelemetryRecord

from .conftest import _to_async_url

_KEY_ID = "erasure-barrier-key"
# Long enough that a write which did not wait would have finished.
_SETTLE_SECONDS = 0.5


def _record(dedup_key: str) -> TelemetryRecord:
    return TelemetryRecord(name="tool_result", timestamp=datetime.now(UTC), source="claude_code", dedup_key=dedup_key)


@pytest_asyncio.fixture
async def sessions(postgres_url: str, clean_database: None) -> AsyncIterator[Callable[[], AsyncSession]]:
    """A factory of independent sessions, as two requests on two replicas would hold, and alice with a key."""
    engine = create_async_engine(_to_async_url(postgres_url))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        workspace_id = (await session.execute(select(col(Workspace.id)))).scalars().first()
        assert workspace_id is not None
        session.add(User(user_id="alice"))
        session.add(APIKey(id=_KEY_ID, key_hash="erasure-barrier", workspace_id=workspace_id, user_id="alice"))
        await session.commit()
    try:
        yield factory
    finally:
        await engine.dispose()


def _pause_first_insert() -> tuple[asyncio.Event, asyncio.Event, Callable[..., Awaitable[Any]]]:
    """Return (holding, release, insert): the first insert sets ``holding`` and waits for ``release``."""
    holding, release = asyncio.Event(), asyncio.Event()

    async def insert(*args: Any, **kwargs: Any) -> Any:
        if not holding.is_set():
            holding.set()
            await release.wait()
        return await insert_ignoring_duplicates(*args, **kwargs)

    return holding, release, insert


async def _write_earlier(sessions: Callable[[], AsyncSession]) -> None:
    """Store one record for alice, so her barrier row exists and is not created under the test."""
    async with sessions() as session:
        await DatabaseTelemetryStorageAdapter(session).record(
            admitted_at=datetime.now(UTC), api_key_id=_KEY_ID, user_id="alice", records=(_record("earlier-0"),)
        )


async def _stored(sessions: Callable[[], AsyncSession]) -> int:
    async with sessions() as session:
        return (await session.execute(select(func.count()).select_from(AgentTelemetry))).scalar_one()


@pytest.mark.asyncio
async def test_an_erasure_waits_for_a_write_holding_the_barrier_and_then_deletes_it(
    sessions: Callable[[], AsyncSession],
) -> None:
    await _write_earlier(sessions)
    holding, release, insert = _pause_first_insert()
    async with sessions() as writer, sessions() as eraser:
        with patch.object(adapter_module, "insert_ignoring_duplicates", insert):
            write = asyncio.create_task(
                DatabaseTelemetryStorageAdapter(writer).record(
                    admitted_at=datetime.now(UTC), api_key_id=_KEY_ID, user_id="alice", records=(_record("held-0"),)
                )
            )
            await holding.wait()
            erasure = asyncio.create_task(DatabaseTelemetryStorageAdapter(eraser).purge_user(user_id="alice"))
            await asyncio.sleep(_SETTLE_SECONDS)
            assert not erasure.done(), "the erasure must wait for the write holding the barrier"

            release.set()
            assert (await write).accepted == 1
            assert await erasure == 2

    assert await _stored(sessions) == 0


@pytest.mark.asyncio
async def test_a_users_concurrent_writes_do_not_wait_on_each_other(sessions: Callable[[], AsyncSession]) -> None:
    """Once the user has a barrier row. Their first two writes ever do wait: each creates the row."""
    await _write_earlier(sessions)
    holding, release, insert = _pause_first_insert()
    async with sessions() as first, sessions() as second:
        with patch.object(adapter_module, "insert_ignoring_duplicates", insert):
            held = asyncio.create_task(
                DatabaseTelemetryStorageAdapter(first).record(
                    admitted_at=datetime.now(UTC), api_key_id=_KEY_ID, user_id="alice", records=(_record("first-0"),)
                )
            )
            await holding.wait()
            other = await asyncio.wait_for(
                DatabaseTelemetryStorageAdapter(second).record(
                    admitted_at=datetime.now(UTC), api_key_id=_KEY_ID, user_id="alice", records=(_record("second-0"),)
                ),
                timeout=_SETTLE_SECONDS * 4,
            )
            release.set()
            assert (await held).accepted == 1

    assert other.accepted == 1
    assert await _stored(sessions) == 3


@pytest.mark.asyncio
async def test_a_write_for_a_hard_deleted_user_is_rejected(sessions: Callable[[], AsyncSession]) -> None:
    """Account deletion removes the user row, and the barrier row with it; a late write is refused, not miscounted."""
    admitted_at = datetime.now(UTC)
    async with sessions() as session:
        await DatabaseTelemetryStorageAdapter(session).purge_user(user_id="alice")
        user = await session.get(User, "alice")
        assert user is not None
        await session.delete(user)
        await session.commit()

    async with sessions() as session:
        result = await DatabaseTelemetryStorageAdapter(session).record(
            admitted_at=admitted_at, api_key_id=_KEY_ID, user_id="alice", records=(_record("late-0"), _record("late-1"))
        )

    assert (result.accepted, result.duplicate, result.rejected) == (0, 0, 2)
    assert await _stored(sessions) == 0
