"""A retried completion sent with an ``Idempotency-Key`` is answered once and billed once.

The case this exists for is a non-streaming request that succeeded upstream
while its response was lost on the way back (a dropped connection, a client
timeout): without the key, the retry calls the provider again and bills again.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, Choice, CompletionUsage
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from gateway.api.routes._idempotency import (
    _REQUEST_SHAPING_HEADERS,
    IDEMPOTENCY_KEY_HEADER,
    IDEMPOTENT_REPLAYED_HEADER,
    IdempotencyGuard,
)
from gateway.core.config import API_KEY_HEADER, API_ROOT, REQUEST_ID_HEADER, GatewayConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.models.inference import IdempotencyRecord, IdempotencyState
from gateway.models.usage import UsageLog
from gateway.models.users import User
from gateway.repositories.inference import InferenceRepositories
from gateway.services.inference import (
    Claimed,
    IdempotencyService,
    IdempotentRequest,
    Replay,
    StillInFlight,
    keep_claim_alive,
)
from gateway.services.inference import _idempotency as service_module
from gateway.services.inference import _lease as lease_module
from gateway.services.secret_box import generate_secret_key

from .conftest import MODEL_NAME, _to_async_url, build_test_client

_USER = "idempotency-user"
_CHAT_ENDPOINT = "/v1/chat/completions"


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-idem",
        object="chat.completion",
        created=0,
        model=MODEL_NAME,
        choices=[Choice(index=0, message=ChatCompletionMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=CompletionUsage(prompt_tokens=100_000, completion_tokens=200_000, total_tokens=300_000),
    )


def _chat_body(content: str = "hi", **extra: Any) -> dict[str, Any]:
    return {"model": MODEL_NAME, "messages": [{"role": "user", "content": content}], "user": _USER, **extra}


def _post_chat(
    client: TestClient, headers: dict[str, str], provider: AsyncMock, body: dict[str, Any] | None = None
) -> Any:
    with patch("gateway.api.routes.chat.acompletion", new=provider):
        return client.post(f"{API_ROOT}/chat/completions", json=body or _chat_body(), headers=headers)


def _usage_rows(make_session: Callable[[], Session]) -> int:
    with make_session() as db:
        return db.execute(select(func.count()).select_from(UsageLog).where(UsageLog.user_id == _USER)).scalar_one()


def _wait_for_usage_rows(make_session: Callable[[], Session], expected: int, *, timeout: float = 3.0) -> None:
    """Poll with fresh sessions, since the usage row may be written by a background writer."""
    deadline = time.monotonic() + timeout
    while _usage_rows(make_session) < expected:
        assert time.monotonic() < deadline, "the usage row was never written"
        time.sleep(0.05)


@pytest.fixture(autouse=True)
def secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored responses are encrypted, so a deployment needs a key to store any."""
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())


@pytest.fixture
def user(client: TestClient, master_key_header: dict[str, str]) -> None:
    response = client.post(f"{API_ROOT}/users", json={"user_id": _USER}, headers=master_key_header)
    assert response.status_code == 200, response.text
    response = client.post(
        f"{API_ROOT}/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 1.0, "output_price_per_million": 1.0},
        headers=master_key_header,
    )
    assert response.status_code == 200, response.text


def _keyed(headers: dict[str, str], key: str) -> dict[str, str]:
    return {**headers, IDEMPOTENCY_KEY_HEADER: key}


def test_retry_replays_the_original_response_without_billing_again(
    client: TestClient,
    master_key_header: dict[str, str],
    user: None,
    db_session_factory: Callable[[], Session],
) -> None:
    provider = AsyncMock(return_value=_completion())
    headers = _keyed(master_key_header, "retry-1")

    first = _post_chat(client, headers, provider)
    assert first.status_code == 200, first.text
    _wait_for_usage_rows(db_session_factory, 1)
    second = _post_chat(client, headers, provider)

    assert second.status_code == 200, second.text
    assert provider.await_count == 1
    assert second.json() == first.json()
    assert second.json()["usage"]["cost_usd"] == first.json()["usage"]["cost_usd"]
    assert second.headers[REQUEST_ID_HEADER] == first.headers[REQUEST_ID_HEADER]
    assert second.headers[IDEMPOTENT_REPLAYED_HEADER] == "true"
    assert IDEMPOTENT_REPLAYED_HEADER not in first.headers
    assert _usage_rows(db_session_factory) == 1
    with db_session_factory() as db:
        stored = db.execute(select(IdempotencyRecord.response_body)).scalar_one()
    assert stored is not None
    assert "chatcmpl-idem" not in stored


def test_without_a_secret_key_nothing_is_stored(
    client: TestClient,
    master_key_header: dict[str, str],
    user: None,
    db_session_factory: Callable[[], Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OTARI_SECRET_KEY")
    provider = AsyncMock(return_value=_completion())
    headers = _keyed(master_key_header, "no-secret")

    for _ in range(2):
        assert _post_chat(client, headers, provider).status_code == 200

    assert provider.await_count == 2
    with db_session_factory() as db:
        assert db.execute(select(func.count()).select_from(IdempotencyRecord)).scalar_one() == 0


def test_a_body_no_key_can_decrypt_runs_again(
    client: TestClient,
    master_key_header: dict[str, str],
    user: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AsyncMock(return_value=_completion())
    headers = _keyed(master_key_header, "rotated")

    assert _post_chat(client, headers, provider).status_code == 200
    monkeypatch.setenv("OTARI_SECRET_KEY", generate_secret_key())
    retried = _post_chat(client, headers, provider)

    assert retried.status_code == 200, retried.text
    assert IDEMPOTENT_REPLAYED_HEADER not in retried.headers
    assert provider.await_count == 2


def test_key_order_and_spacing_do_not_make_a_different_request(
    client: TestClient, master_key_header: dict[str, str], user: None
) -> None:
    provider = AsyncMock(return_value=_completion())
    headers = {**_keyed(master_key_header, "canonical"), "Content-Type": "application/json"}

    with patch("gateway.api.routes.chat.acompletion", new=provider):
        first = client.post(
            f"{API_ROOT}/chat/completions",
            content=(
                f'{{"user": "{_USER}", "model": "{MODEL_NAME}", "messages": [{{"role": "user", "content": "hi"}}]}}'
            ),
            headers=headers,
        )
        second = client.post(
            f"{API_ROOT}/chat/completions",
            content=f'{{"messages":[{{"content":"hi","role":"user"}}],"model":"{MODEL_NAME}","user":"{_USER}"}}',
            headers=headers,
        )

    assert first.status_code == second.status_code == 200
    assert provider.await_count == 1


def test_reusing_a_key_for_a_different_request_is_refused(
    client: TestClient, master_key_header: dict[str, str], user: None
) -> None:
    provider = AsyncMock(return_value=_completion())
    headers = _keyed(master_key_header, "reused")

    assert _post_chat(client, headers, provider).status_code == 200
    refused = _post_chat(client, headers, provider, _chat_body("something else"))

    assert refused.status_code == 422, refused.text
    assert IDEMPOTENCY_KEY_HEADER in refused.text
    assert provider.await_count == 1


def test_reusing_a_key_with_different_tool_headers_is_refused(
    client: TestClient, master_key_header: dict[str, str], user: None
) -> None:
    provider = AsyncMock(return_value=_completion())
    headers = _keyed(master_key_header, "reused-header")

    assert _post_chat(client, headers, provider).status_code == 200
    refused = _post_chat(client, {**headers, "Otari-Web-Search": "otari"}, provider)

    assert refused.status_code == 422, refused.text
    assert provider.await_count == 1


def test_a_failed_request_releases_its_key_so_the_retry_runs(
    client: TestClient,
    master_key_header: dict[str, str],
    user: None,
    db_session_factory: Callable[[], Session],
) -> None:
    provider = AsyncMock(side_effect=[RuntimeError("upstream overloaded"), _completion()])
    headers = _keyed(master_key_header, "after-failure")

    failed = _post_chat(client, headers, provider)
    retried = _post_chat(client, headers, provider)

    assert failed.status_code >= 500
    assert retried.status_code == 200, retried.text
    assert IDEMPOTENT_REPLAYED_HEADER not in retried.headers
    assert provider.await_count == 2
    with db_session_factory() as db:
        record = db.execute(select(IdempotencyRecord)).scalar_one()
    assert record.state == IdempotencyState.COMPLETED


def test_keys_are_scoped_to_the_caller(client: TestClient, master_key_header: dict[str, str], user: None) -> None:
    def _key_header() -> dict[str, str]:
        response = client.post(
            f"{API_ROOT}/keys", json={"key_name": f"k-{uuid.uuid4()}", "user_id": _USER}, headers=master_key_header
        )
        assert response.status_code == 200, response.text
        return {API_KEY_HEADER: f"Bearer {response.json()['key']}"}

    provider = AsyncMock(return_value=_completion())
    first, second = _key_header(), _key_header()

    assert _post_chat(client, _keyed(first, "shared"), provider).status_code == 200
    other = _post_chat(client, _keyed(second, "shared"), provider)

    assert other.status_code == 200, other.text
    assert IDEMPOTENT_REPLAYED_HEADER not in other.headers
    assert provider.await_count == 2


def test_an_unknown_user_is_refused_before_the_key_is_claimed(
    client: TestClient, master_key_header: dict[str, str], db_session_factory: Callable[[], Session]
) -> None:
    provider = AsyncMock(return_value=_completion())

    response = _post_chat(client, _keyed(master_key_header, "unknown-user"), provider, _chat_body(user="nobody"))

    assert response.status_code == 404, response.text
    provider.assert_not_awaited()
    with db_session_factory() as db:
        assert db.execute(select(func.count()).select_from(IdempotencyRecord)).scalar_one() == 0


def test_a_malformed_key_is_refused_before_the_user_is_looked_up(
    client: TestClient, master_key_header: dict[str, str]
) -> None:
    provider = AsyncMock(return_value=_completion())

    response = _post_chat(client, _keyed(master_key_header, " padded"), provider, _chat_body(user="nobody"))

    assert response.status_code == 400, response.text
    provider.assert_not_awaited()


def test_a_blocked_user_is_not_given_the_stored_response(
    client: TestClient, master_key_header: dict[str, str], user: None
) -> None:
    provider = AsyncMock(return_value=_completion())
    headers = _keyed(master_key_header, "blocked-later")
    assert _post_chat(client, headers, provider).status_code == 200

    blocked = client.patch(f"{API_ROOT}/users/{_USER}", json={"blocked": True}, headers=master_key_header)
    assert blocked.status_code == 200, blocked.text
    retried = _post_chat(client, headers, provider)

    assert retried.status_code == 403, retried.text
    assert IDEMPOTENT_REPLAYED_HEADER not in retried.headers


def test_streaming_requests_ignore_the_key(client: TestClient, master_key_header: dict[str, str], user: None) -> None:
    async def _stream(**_kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def _chunks() -> AsyncIterator[ChatCompletionChunk]:
            yield ChatCompletionChunk(
                id="chunk", object="chat.completion.chunk", created=0, model=MODEL_NAME, choices=[]
            )

        return _chunks()

    provider = AsyncMock(side_effect=_stream)
    headers = _keyed(master_key_header, "streamed")

    for _ in range(2):
        response = _post_chat(client, headers, provider, _chat_body(stream=True))
        assert response.status_code == 200, response.text

    assert provider.await_count == 2


@pytest.mark.parametrize("key", ["", "x" * 256])
def test_an_unusable_key_is_refused(
    client: TestClient, master_key_header: dict[str, str], user: None, key: str
) -> None:
    provider = AsyncMock(return_value=_completion())

    response = _post_chat(client, _keyed(master_key_header, key), provider)

    assert response.status_code == 400, response.text
    provider.assert_not_awaited()


def test_messages_retry_is_replayed(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    messages_request_body["metadata"] = {"user_id": test_user["user_id"]}
    result = MessageResponse(
        id="msg_idem",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="Hello!")],
        model="claude-3-5-sonnet",
        stop_reason="end_turn",
        usage=MessageUsage(input_tokens=10, output_tokens=5),
    )
    provider = AsyncMock(return_value=result)
    headers = _keyed(master_key_header, "messages-1")

    with patch("gateway.api.routes.messages.amessages", new=provider):
        first = client.post(f"{API_ROOT}/messages", json=messages_request_body, headers=headers)
        second = client.post(f"{API_ROOT}/messages", json=messages_request_body, headers=headers)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert second.headers[IDEMPOTENT_REPLAYED_HEADER] == "true"
    assert provider.await_count == 1


def test_responses_retry_is_replayed(
    client: TestClient, master_key_header: dict[str, str], responses_request_body: dict[str, Any]
) -> None:
    class _Result:
        usage = None

        def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
            return {"id": "resp_idem", "output": [{"type": "message", "content": "Hello"}]}

    provider = AsyncMock(return_value=_Result())
    headers = _keyed(master_key_header, "responses-1")

    with patch("gateway.api.routes.responses.aresponses", new=provider):
        first = client.post(f"{API_ROOT}/responses", json=responses_request_body, headers=headers)
        second = client.post(f"{API_ROOT}/responses", json=responses_request_body, headers=headers)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert second.headers[IDEMPOTENT_REPLAYED_HEADER] == "true"
    assert provider.await_count == 1


def _hold_key(
    make_session: Callable[[], Session], key: str, *, locked_until: datetime, body: dict[str, Any] | None = None
) -> None:
    """Leave a claim on ``key`` as a request still in flight on another worker would."""
    now = datetime.now(UTC)
    request = IdempotentRequest.of(
        key,
        endpoint=_CHAT_ENDPOINT,
        body=_json_bytes(body or _chat_body()),
        options=[None] * len(_REQUEST_SHAPING_HEADERS),
        user_id=_USER,
        api_key_id=None,
    )
    assert isinstance(request, IdempotentRequest)
    with make_session() as db:
        db.add(
            IdempotencyRecord(
                scope=f"master:{_USER}",
                idempotency_key=key,
                request_hash=request.request_hash,
                claim_token=str(uuid.uuid4()),
                state=IdempotencyState.IN_PROGRESS,
                user_id=_USER,
                api_key_id=None,
                created_at=now,
                locked_until=locked_until,
                expires_at=now + timedelta(days=1),
            )
        )
        db.commit()


def _json_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body).encode()


@pytest.fixture
def no_wait_client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(test_config.model_copy(update={"idempotency_wait_sec": 0}))


def test_a_retry_while_the_original_is_in_flight_is_told_to_come_back(
    no_wait_client: TestClient,
    master_key_header: dict[str, str],
    db_session_factory: Callable[[], Session],
) -> None:
    response = no_wait_client.post(f"{API_ROOT}/users", json={"user_id": _USER}, headers=master_key_header)
    assert response.status_code == 200
    _hold_key(db_session_factory, "in-flight", locked_until=datetime.now(UTC) + timedelta(minutes=10))
    provider = AsyncMock(return_value=_completion())

    response = _post_chat(no_wait_client, _keyed(master_key_header, "in-flight"), provider)

    assert response.status_code == 409, response.text
    assert response.headers["Retry-After"]
    provider.assert_not_awaited()


def test_an_abandoned_claim_is_taken_over_once_its_lease_passes(
    client: TestClient,
    master_key_header: dict[str, str],
    user: None,
    db_session_factory: Callable[[], Session],
) -> None:
    _hold_key(db_session_factory, "abandoned", locked_until=datetime.now(UTC) - timedelta(seconds=1))
    provider = AsyncMock(return_value=_completion())

    response = _post_chat(client, _keyed(master_key_header, "abandoned"), provider)

    assert response.status_code == 200, response.text
    assert provider.await_count == 1


@pytest.fixture
def ignoring_client(test_config: GatewayConfig, clean_database: None) -> Generator[TestClient]:
    yield from build_test_client(test_config.model_copy(update={"idempotency_retention_sec": 0}))


def test_a_deployment_can_turn_the_header_off(ignoring_client: TestClient, master_key_header: dict[str, str]) -> None:
    response = ignoring_client.post(f"{API_ROOT}/users", json={"user_id": _USER}, headers=master_key_header)
    assert response.status_code == 200
    provider = AsyncMock(return_value=_completion())
    headers = _keyed(master_key_header, "ignored")

    for _ in range(2):
        assert _post_chat(ignoring_client, headers, provider).status_code == 200

    assert provider.await_count == 2


@pytest.mark.asyncio
async def test_the_sweep_deletes_only_expired_records(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    async_db.add(User(user_id=_USER))
    await async_db.commit()
    now = datetime.now(UTC)

    def _record(key: str, *, state: str, expires_at: datetime, locked_until: datetime) -> IdempotencyRecord:
        return IdempotencyRecord(
            scope=f"master:{_USER}",
            idempotency_key=key,
            request_hash="0" * 64,
            claim_token=str(uuid.uuid4()),
            state=state,
            user_id=_USER,
            created_at=now,
            locked_until=locked_until,
            expires_at=expires_at,
        )

    past, future = now - timedelta(seconds=1), now + timedelta(hours=1)
    async_db.add_all(
        [
            _record("expired", state=IdempotencyState.COMPLETED, expires_at=past, locked_until=past),
            _record("kept", state=IdempotencyState.COMPLETED, expires_at=future, locked_until=past),
            _record("live-claim", state=IdempotencyState.IN_PROGRESS, expires_at=past, locked_until=future),
        ]
    )
    await async_db.commit()

    uow = UnitOfWork(async_db)
    deleted = await IdempotencyService(uow, InferenceRepositories.on(uow), test_config).sweep()

    assert deleted == 1
    remaining = (await async_db.execute(select(IdempotencyRecord.idempotency_key))).scalars().all()
    assert sorted(remaining) == ["kept", "live-claim"]


@pytest.mark.asyncio
@pytest.mark.parametrize("original_succeeds", [True, False])
async def test_a_retry_waits_for_the_original_request(
    postgres_url: str, clean_database: None, test_config: GatewayConfig, original_succeeds: bool
) -> None:
    """The retry joins the in-flight original: its response if it succeeds, the key if it fails."""
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    request = IdempotentRequest(
        scope=f"master:{_USER}", key="joined", request_hash="0" * 64, user_id=_USER, api_key_id=None
    )
    try:
        async with sessions() as original_db, sessions() as retry_db:
            original_db.add(User(user_id=_USER))
            await original_db.commit()
            original_uow, retry_uow = UnitOfWork(original_db), UnitOfWork(retry_db)
            original = IdempotencyService(original_uow, InferenceRepositories.on(original_uow), test_config)
            retry = IdempotencyService(retry_uow, InferenceRepositories.on(retry_uow), test_config)

            claimed = await original.admit(request)
            assert isinstance(claimed, Claimed)
            waiting = asyncio.create_task(retry.admit(request))
            await asyncio.sleep(0.5)
            assert not waiting.done()

            if original_succeeds:
                await original.complete(request, claimed, status_code=200, body='{"ok":true}', headers={})
            else:
                await original.release(request, claimed)
            outcome = await asyncio.wait_for(waiting, timeout=5)

        if original_succeeds:
            assert outcome == Replay(status_code=200, body='{"ok":true}', headers={})
        else:
            assert isinstance(outcome, Claimed)
            assert outcome.token != claimed.token
    finally:
        await engine.dispose()


class _StubRequest:
    headers: dict[str, str] = {}

    async def body(self) -> bytes:
        return b"{}"


@pytest.mark.asyncio
async def test_a_running_request_keeps_its_claim_past_the_lease(
    postgres_url: str, clean_database: None, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request slower than the lease renews its claim, so a retry cannot take it over and run it again."""
    config = test_config.model_copy(update={"idempotency_lease_sec": 3})
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def worker_unit_of_work() -> AsyncIterator[UnitOfWork]:
        async with sessions() as db:
            yield UnitOfWork(db)

    monkeypatch.setattr(lease_module, "create_unit_of_work", worker_unit_of_work)

    def build(uow: UnitOfWork) -> IdempotencyService:
        return IdempotencyService(uow, InferenceRepositories.on(uow), config)

    async def locked_until() -> datetime:
        async with sessions() as db:
            return (await db.execute(select(IdempotencyRecord.locked_until))).scalar_one()

    async def keep_alive(request: IdempotentRequest, claimed: Claimed) -> None:
        await keep_claim_alive(request, claimed, config.idempotency_lease_sec, build)

    try:
        async with sessions() as request_db:
            request_db.add(User(user_id=_USER))
            await request_db.commit()
            guard = IdempotencyGuard(
                _StubRequest(),  # type: ignore[arg-type]
                build(UnitOfWork(request_db)),
                "slow-request",
                keep_alive=keep_alive,
            )
            assert isinstance(await guard.admit(endpoint=_CHAT_ENDPOINT, user_id=_USER, api_key_id=None), Claimed)
            first = await locked_until()
            await asyncio.sleep(1.5)
            assert await locked_until() > first

            await guard.release()

        async with sessions() as db:
            assert (await db.execute(select(func.count()).select_from(IdempotencyRecord))).scalar_one() == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_running_claim_outlives_its_retention(
    postgres_url: str, clean_database: None, test_config: GatewayConfig
) -> None:
    """Only a lapsed lease frees a running claim, so a request slower than the retention is not run twice."""
    config = test_config.model_copy(
        update={"idempotency_retention_sec": 1, "idempotency_lease_sec": 60, "idempotency_wait_sec": 0}
    )
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    request = IdempotentRequest(
        scope=f"master:{_USER}", key="outlives", request_hash="0" * 64, user_id=_USER, api_key_id=None
    )
    try:
        async with sessions() as original_db, sessions() as retry_db:
            original_db.add(User(user_id=_USER))
            await original_db.commit()
            original_uow, retry_uow = UnitOfWork(original_db), UnitOfWork(retry_db)
            original = IdempotencyService(original_uow, InferenceRepositories.on(original_uow), config)
            retry = IdempotencyService(retry_uow, InferenceRepositories.on(retry_uow), config)

            assert isinstance(await original.admit(request), Claimed)
            # Past the lease-or-retention expiry an in-progress claim is stamped with, but inside its lease.
            async with sessions() as db:
                await db.execute(update(IdempotencyRecord).values(expires_at=datetime.now(UTC) - timedelta(seconds=1)))
                await db.commit()

            assert isinstance(await retry.admit(request), StillInFlight)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_gateway_whose_clock_runs_ahead_does_not_take_over_a_live_claim(
    postgres_url: str, clean_database: None, test_config: GatewayConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leases are timed by the database, so clock skew between gateways cannot free a running claim."""
    config = test_config.model_copy(update={"idempotency_wait_sec": 0})
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    request = IdempotentRequest(
        scope=f"master:{_USER}", key="skewed", request_hash="0" * 64, user_id=_USER, api_key_id=None
    )
    try:
        async with sessions() as original_db, sessions() as retry_db:
            original_db.add(User(user_id=_USER))
            await original_db.commit()
            original_uow, retry_uow = UnitOfWork(original_db), UnitOfWork(retry_db)
            original = IdempotencyService(original_uow, InferenceRepositories.on(original_uow), config)
            retry = IdempotencyService(retry_uow, InferenceRepositories.on(retry_uow), config)
            assert isinstance(await original.admit(request), Claimed)

            class _FastClock(datetime):
                @classmethod
                def now(cls, tz: Any = None) -> "_FastClock":
                    return cls.fromtimestamp(time.time() + 600, tz)

            monkeypatch.setattr(service_module, "datetime", _FastClock)

            assert isinstance(await retry.admit(request), StillInFlight)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_the_sweep_deletes_in_bounded_batches(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    async_db.add(User(user_id=_USER))
    await async_db.commit()
    past = datetime.now(UTC) - timedelta(seconds=1)
    async_db.add_all(
        IdempotencyRecord(
            scope=f"master:{_USER}",
            idempotency_key=f"expired-{index}",
            request_hash="0" * 64,
            claim_token=str(uuid.uuid4()),
            state=IdempotencyState.COMPLETED,
            user_id=_USER,
            created_at=past,
            locked_until=past,
            expires_at=past,
        )
        for index in range(5)
    )
    await async_db.commit()
    uow = UnitOfWork(async_db)
    repositories = InferenceRepositories.on(uow)
    keys = repositories.idempotency

    with patch.object(keys, "delete_expired", wraps=keys.delete_expired) as delete_expired:
        deleted = await IdempotencyService(uow, repositories, test_config).sweep(batch_size=2)

    assert deleted == 5
    assert delete_expired.await_count == 3
    assert (await async_db.execute(select(func.count()).select_from(IdempotencyRecord))).scalar_one() == 0


@pytest.mark.asyncio
async def test_the_database_clock_is_read_when_asked_not_when_the_transaction_began(
    postgres_url: str, clean_database: None
) -> None:
    engine = create_async_engine(_to_async_url(postgres_url))
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as db:
            uow = UnitOfWork(db)
            keys = InferenceRepositories.on(uow).idempotency
            async with uow:
                began = (await db.execute(select(func.now()))).scalar_one()
                await asyncio.sleep(1.0)
                read = await keys.get_database_time()

        assert read - began >= timedelta(seconds=0.9)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_one_sweep_deletes_at_most_its_batch_limit(async_db: AsyncSession, test_config: GatewayConfig) -> None:
    async_db.add(User(user_id=_USER))
    await async_db.commit()
    past = datetime.now(UTC) - timedelta(seconds=1)
    async_db.add_all(
        IdempotencyRecord(
            scope=f"master:{_USER}",
            idempotency_key=f"backlog-{index}",
            request_hash="0" * 64,
            claim_token=str(uuid.uuid4()),
            state=IdempotencyState.COMPLETED,
            user_id=_USER,
            created_at=past,
            locked_until=past,
            expires_at=past,
        )
        for index in range(5)
    )
    await async_db.commit()
    uow = UnitOfWork(async_db)

    deleted = await IdempotencyService(uow, InferenceRepositories.on(uow), test_config).sweep(
        batch_size=2, max_batches=2
    )

    assert deleted == 4
