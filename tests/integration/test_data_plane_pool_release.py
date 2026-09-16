"""The data plane hands every pooled connection back.

A leak here is what mozilla-ai/otari#1236 was: ``get_db_if_needed`` abandoned the
session it borrowed, so each inference request kept a pooled connection checked
out until the garbage collector got to it, and sustained traffic drained the
pool and turned every request into a 503.

Nothing caught it because the suite cannot see this path from the outside.
``build_test_client`` overrides ``get_db``, but the data-plane routes take their
session from ``get_db_if_needed``, which calls ``get_db`` as a plain function
rather than through ``Depends``, so the override never reaches them: the
inference routes run on the engine ``init_db`` built while the management routes
run on the test engine. Overriding ``get_db_if_needed`` too would make the suite
tidier and this class of bug permanently invisible, because the connection the
production dependency leaks is one the test engine would never have lent. So
these tests assert against the real engine's pool instead, which is also the
only place the count exists: ``checkedout`` belongs to ``AsyncAdaptedQueuePool``
and SQLite runs on ``NullPool``, so this is PostgreSQL-only and lives here
rather than in ``tests/unit``.
"""

from typing import Any
from unittest.mock import patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from fastapi.testclient import TestClient
from sqlalchemy import event, text
from sqlalchemy.pool import AsyncAdaptedQueuePool

from gateway.api.deps import get_db_if_needed
from gateway.core import database
from gateway.core.config import API_KEY_HEADER, API_ROOT, GatewayConfig
from gateway.core.database import dispose_db, init_db

from .conftest import MODEL_NAME

_REQUESTS = 8


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-pool",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.mark.asyncio
async def test_tearing_down_the_data_plane_dependency_returns_its_connection(
    test_config: GatewayConfig,
    clean_database: None,
) -> None:
    """Closing the dependency puts the connection back, in the caller's own task.

    Measured on a real pool rather than a fake session, because the count that
    matters is the pool's: a session whose ``__aexit__`` never runs is a
    connection the next request cannot have. Nothing here waits for the garbage
    collector, which is the point. The abandoned generator does get finalized
    eventually, so an assertion made a moment later passes whether the
    dependency closes its session or not, and the leak stays invisible.

    The engine is built and disposed here rather than borrowed from a booted
    app: a pool belongs to the event loop that opened its connections, and the
    test client's loop is not this test's.
    """
    init_db(test_config)
    try:
        engine = database._engine
        assert engine is not None
        assert isinstance(engine.pool, AsyncAdaptedQueuePool), "checkedout() needs a queue pool, so PostgreSQL"

        dependency = get_db_if_needed(test_config)
        session = await anext(dependency)
        assert session is not None
        # The session opens its transaction on the first statement, which is
        # when a connection is actually checked out.
        await session.execute(text("SELECT 1"))
        assert engine.pool.checkedout() == 1

        await dependency.aclose()

        assert engine.pool.checkedout() == 0, "the request's session was abandoned with its connection checked out"
    finally:
        await dispose_db()


def test_chat_completions_returns_every_connection_it_checks_out(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """A run of inference requests leaves the pool where it found it.

    The checkout count is asserted as well as the balance, because it is what
    says the requests reached this engine at all. Should the data-plane routes
    ever be pointed at a test engine, this stops observing the production pool
    and fails here rather than passing on an engine nothing used.
    """
    key = client.post(f"{API_ROOT}/keys", json={"key_name": "pool"}, headers=master_key_header)
    assert key.status_code == 200, key.text
    headers = {API_KEY_HEADER: f"Bearer {key.json()['key']}"}

    engine = database._engine
    assert engine is not None
    pool = engine.pool
    assert isinstance(pool, AsyncAdaptedQueuePool), "checkedout() needs a queue pool, so PostgreSQL"
    baseline = pool.checkedout()
    checkouts = 0

    def _count_checkout(*_args: Any) -> None:
        nonlocal checkouts
        checkouts += 1

    async def _acompletion(**_kwargs: Any) -> ChatCompletion:
        return _completion()

    event.listen(engine.sync_engine, "checkout", _count_checkout)
    try:
        with patch("gateway.api.routes.chat.acompletion") as mock:
            mock.side_effect = _acompletion
            for _ in range(_REQUESTS):
                response = client.post(
                    f"{API_ROOT}/chat/completions",
                    json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "hi"}]},
                    headers=headers,
                )
                assert response.status_code == 200, response.text
    finally:
        event.remove(engine.sync_engine, "checkout", _count_checkout)

    assert checkouts >= _REQUESTS, "the inference route did not use the engine this deployment runs on"
    assert pool.checkedout() == baseline
