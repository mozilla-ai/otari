"""Unit tests for handing back the request's connection before the provider call.

The preamble reads the API key, the billed user, the organization's guardrails
and the workspace's tool rows on the request-scoped session, which holds its
connection until something ends the transaction. Nothing in the preamble
commits, so before this the connection stayed checked out for the whole
upstream call and ``db_pool_size + db_max_overflow`` became a ceiling on
concurrent provider calls. Past it, a request waited ``db_pool_timeout`` and
then failed in the auth dependency, whose ``SQLAlchemyError`` arm reports a pool
timeout as "Authentication temporarily unavailable, please retry".
"""

import time
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import OperationalError

from gateway.api.routes._pipeline import RequestContext, release_request_connection
from gateway.core.config import GatewayConfig


def _ctx(db: Any) -> RequestContext:
    return RequestContext(
        config=GatewayConfig(),
        db=db,
        log_writer=cast(Any, None),
        hybrid_mode=False,
        route=None,
        user_token=None,
        api_key_id=None,
        user_id=None,
        rate_limit_info=None,
        reservation=None,
        started_at=time.monotonic(),
    )


@pytest.mark.asyncio
async def test_commits_so_the_connection_returns_to_the_pool() -> None:
    db = AsyncMock()
    await release_request_connection(_ctx(db))
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_hybrid_mode_has_no_session_to_release() -> None:
    # Hybrid mode resolves credentials over HTTP and is handed no session at
    # all, so this must not assume one.
    await release_request_connection(_ctx(None))


@pytest.mark.asyncio
async def test_a_failed_commit_does_not_fail_the_request() -> None:
    # The request is about to reach the provider. Losing it because the
    # connection could not be handed back early would be a worse outcome than
    # holding the connection.
    db = AsyncMock()
    db.commit.side_effect = OperationalError("SELECT 1", {}, Exception("gone"))
    await release_request_connection(_ctx(db))
    db.commit.assert_awaited_once()
