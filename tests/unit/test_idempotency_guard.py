"""A request's hold on its idempotency key never costs it a response the provider already billed."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response
from starlette.datastructures import Headers

from gateway.api.routes._helpers import GUARDRAILS_RESULT_HEADER
from gateway.api.routes._idempotency import IdempotencyGuard
from gateway.services.inference import Claimed, IdempotentRequest


def _raw_request() -> MagicMock:
    raw = MagicMock()
    raw.body = AsyncMock(return_value=b"{}")
    raw.headers = Headers()
    return raw


@pytest.mark.asyncio
async def test_a_failed_keep_alive_does_not_lose_the_response() -> None:
    service = MagicMock()
    service.admit = AsyncMock(return_value=Claimed("token"))
    service.complete = AsyncMock(return_value=True)

    async def keep_alive(request: IdempotentRequest, claimed: Claimed) -> None:
        raise OSError("connection refused")

    guard = IdempotencyGuard(_raw_request(), service, "k", keep_alive=keep_alive)
    await guard.admit(endpoint="/v1/chat/completions", user_id="u", api_key_id=None)
    await asyncio.sleep(0)

    await guard.complete({"ok": True}, Response())

    service.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_claim_is_kept_alive_until_the_response_is_stored() -> None:
    ticks = 0

    async def keep_alive(request: IdempotentRequest, claimed: Claimed) -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    ticks_during_store: list[int] = []

    async def slow_store(*_args: object, **_kwargs: object) -> bool:
        before = ticks
        await asyncio.sleep(0.05)
        ticks_during_store.append(ticks - before)
        return True

    service = MagicMock()
    service.admit = AsyncMock(return_value=Claimed("token"))
    service.complete = AsyncMock(side_effect=slow_store)
    guard = IdempotencyGuard(_raw_request(), service, "k", keep_alive=keep_alive)
    await guard.admit(endpoint="/v1/chat/completions", user_id="u", api_key_id=None)

    await guard.complete({"ok": True}, Response())

    assert ticks_during_store[0] > 0


@pytest.mark.asyncio
async def test_an_unexpected_store_error_does_not_lose_the_response() -> None:
    service = MagicMock()
    service.admit = AsyncMock(return_value=Claimed("token"))
    service.complete = AsyncMock(side_effect=RuntimeError("unexpected"))
    guard = IdempotencyGuard(_raw_request(), service, "k")
    await guard.admit(endpoint="/v1/chat/completions", user_id="u", api_key_id=None)

    await guard.complete({"ok": True}, Response())

    service.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_unexpected_release_error_does_not_replace_the_request_outcome() -> None:
    service = MagicMock()
    service.admit = AsyncMock(return_value=Claimed("token"))
    service.release = AsyncMock(side_effect=RuntimeError("unexpected"))
    guard = IdempotencyGuard(_raw_request(), service, "k")
    await guard.admit(endpoint="/v1/chat/completions", user_id="u", api_key_id=None)

    await guard.release()

    service.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_guardrail_verdict_is_stored_for_a_replay() -> None:
    service = MagicMock()
    service.admit = AsyncMock(return_value=Claimed("token"))
    service.complete = AsyncMock(return_value=True)
    guard = IdempotencyGuard(_raw_request(), service, "k")
    await guard.admit(endpoint="/v1/chat/completions", user_id="u", api_key_id=None)
    response = Response()
    response.headers[GUARDRAILS_RESULT_HEADER] = '[{"profile":"p","mode":"monitor","valid":true,"score":null}]'

    await guard.complete({"ok": True}, response)

    assert GUARDRAILS_RESULT_HEADER in service.complete.await_args.kwargs["headers"]
