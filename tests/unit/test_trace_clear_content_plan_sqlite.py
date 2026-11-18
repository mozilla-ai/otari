"""Clearing a workspace's content reads the partial index, not every span it already cleared.

The cost is the point: the delete runs a batch at a time, so a plan that scans the
workspace's spans makes each batch step over every row the batches before it cleared.
Asserted on SQLite, which names the index it chose in a query plan and needs no server.
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

import gateway.models  # noqa: F401  (registers every table on the metadata)
from gateway.repositories.traces.span_repository import holding_content

pytestmark = pytest.mark.asyncio

_INDEX = "ix_trace_spans_workspace_sealed"


async def _plan() -> str:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
            compiled = holding_content(uuid.uuid4(), 5_000).compile(
                engine.sync_engine, compile_kwargs={"literal_binds": True}
            )
            rows = await conn.execute(text(f"EXPLAIN QUERY PLAN {compiled}"))
            return " ".join(str(row.detail) for row in rows)
    finally:
        await engine.dispose()


async def test_the_spans_still_holding_content_are_found_through_the_partial_index() -> None:
    assert _INDEX in await _plan()
