"""Unit tests for handing the request's connection back before the provider call.

The preamble reads the API key, the billed user, the organization's guardrails
and the workspace's tool rows on the request-scoped session, which holds its
connection until something ends the transaction. Nothing in the preamble
commits, so before this the connection stayed checked out for the whole
upstream call and ``db_pool_size + db_max_overflow`` became a ceiling on
concurrent provider calls. Past it a request waits ``db_pool_timeout`` and is
then refused by the auth dependency, which reports a pool timeout as
"Authentication temporarily unavailable, please retry".
"""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from gateway.core.database import release_session


@pytest.mark.asyncio
async def test_commits_so_the_connection_returns_to_the_pool() -> None:
    db = AsyncMock()
    assert await release_session(db) is True
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_mode_has_no_session_to_release() -> None:
    # Hybrid mode resolves credentials over HTTP and is handed no session at
    # all, so this must not assume one.
    assert await release_session(None) is False


@pytest.mark.asyncio
async def test_a_failed_commit_does_not_fail_the_request() -> None:
    # The request is about to reach the provider. Losing it because the
    # connection could not be handed back early would be a worse outcome than
    # holding the connection.
    db = AsyncMock()
    db.commit.side_effect = OperationalError("SELECT 1", {}, Exception("gone"))
    assert await release_session(db) is False
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_commit_rolls_the_session_back() -> None:
    # A failed commit leaves the session unusable, and settlement writes on it
    # after dispatch: without the rollback those raise PendingRollbackError and
    # the budget hold leaks.
    db = AsyncMock()
    db.commit.side_effect = OperationalError("SELECT 1", {}, Exception("gone"))
    await release_session(db)
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_command_timeout_is_caught_like_any_database_failure() -> None:
    # `db_command_timeout` makes a bare TimeoutError reachable here. Catching
    # only SQLAlchemyError would kill a request that passed every gate.
    db = AsyncMock()
    db.commit.side_effect = TimeoutError("command timed out")
    assert await release_session(db) is False
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_rollback_is_swallowed_too() -> None:
    db = AsyncMock()
    db.commit.side_effect = TimeoutError("command timed out")
    db.rollback.side_effect = OperationalError("ROLLBACK", {}, Exception("gone"))
    assert await release_session(db) is False
