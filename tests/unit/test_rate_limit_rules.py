"""The ``rate_limits`` rules: their config, admission, settlement, and the middleware that gives slots back."""

from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from gateway.adapters.rate_limit_store_adapter import InMemoryRateLimitStore
from gateway.core.config import GatewayConfig, RateLimitRule
from gateway.main import _validate_rate_limit_store
from gateway.rate_limit import RateLimitGrantMiddleware, RateLimitRules


def _request() -> Request:
    return Request({"type": "http", "headers": []})


def _rules(store: InMemoryRateLimitStore, *rules: dict[str, Any]) -> RateLimitRules:
    return RateLimitRules(store, [RateLimitRule(**rule) for rule in rules])


async def _admit(
    rules: RateLimitRules, *, key_id: str | None = "k1", user_id: str | None = "u1", tokens: int = 10
) -> Any:
    return await rules.admit(_request(), key_id=key_id, user_id=user_id, estimated_tokens=tokens)


def test_a_rule_must_set_a_limit() -> None:
    with pytest.raises(ValidationError, match="sets none of rpm, tpm or max_concurrent"):
        RateLimitRule(name="empty", per="key")


def test_a_rule_rejects_an_unknown_field() -> None:
    """A misspelled limit is an error, not a rule that silently limits nothing."""
    with pytest.raises(ValidationError):
        RateLimitRule(name="typo", per="key", rpn=10)  # type: ignore[call-arg]


def test_a_rule_name_is_limited_to_key_safe_characters() -> None:
    with pytest.raises(ValidationError):
        RateLimitRule(name="has space", per="key", rpm=1)


def test_rule_names_are_unique() -> None:
    with pytest.raises(ValidationError, match="repeated: twice"):
        GatewayConfig(
            rate_limits=[
                RateLimitRule(name="twice", per="key", rpm=1),
                RateLimitRule(name="twice", per="user", rpm=1),
            ]
        )


def test_hybrid_mode_refuses_to_start_with_rules() -> None:
    config = GatewayConfig(mode="hybrid", rate_limits=[RateLimitRule(name="keys", per="key", rpm=1)])

    with pytest.raises(ValueError, match="hybrid mode"):
        _validate_rate_limit_store(config)


@pytest.mark.asyncio
async def test_per_key_counts_each_key_on_its_own_and_skips_keyless_requests() -> None:
    rules = _rules(InMemoryRateLimitStore(), {"name": "keys", "per": "key", "rpm": 1})

    await _admit(rules, key_id="k1")
    await _admit(rules, key_id="k2")
    await _admit(rules, key_id=None)
    await _admit(rules, key_id=None)
    with pytest.raises(HTTPException) as exc_info:
        await _admit(rules, key_id="k1")

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == "Rate limit 'keys' exceeded"
    assert exc_info.value.headers is not None
    assert 1 <= int(exc_info.value.headers["Retry-After"]) <= 60


@pytest.mark.asyncio
async def test_per_user_and_per_deployment_count_what_they_name() -> None:
    rules = _rules(
        InMemoryRateLimitStore(),
        {"name": "users", "per": "user", "rpm": 1},
        {"name": "all", "per": "deployment", "rpm": 2},
    )

    await _admit(rules, user_id="alice")
    with pytest.raises(HTTPException, match="'users'"):
        await _admit(rules, user_id="alice")
    await _admit(rules, user_id="bob")
    with pytest.raises(HTTPException, match="'all'"):
        await _admit(rules, user_id="carol")


@pytest.mark.asyncio
async def test_a_refused_request_is_counted_by_no_rule() -> None:
    """Rules are all-or-nothing, so a retry does not use up the limits the refused attempt fit."""
    store = InMemoryRateLimitStore()
    rules = _rules(store, {"name": "wide", "per": "deployment", "rpm": 5}, {"name": "narrow", "per": "key", "rpm": 1})

    await _admit(rules)
    with pytest.raises(HTTPException, match="'narrow'"):
        await _admit(rules)

    assert (await store.hit("rule:wide:all:rpm", 5, 60)).count == 2


@pytest.mark.asyncio
async def test_tokens_are_admitted_on_the_estimate_and_charged_what_was_used() -> None:
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "key", "tpm": 1000})

    grant = await _admit(rules, tokens=600)
    with pytest.raises(HTTPException, match="'tpm'"):
        await _admit(rules, tokens=600)

    await grant.settle(50)
    await _admit(rules, tokens=600)


@pytest.mark.asyncio
async def test_a_request_larger_than_the_limit_is_not_told_to_retry() -> None:
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "key", "tpm": 1000})

    with pytest.raises(HTTPException) as exc_info:
        await _admit(rules, tokens=1001)

    assert exc_info.value.headers is None


@pytest.mark.asyncio
async def test_only_the_first_settlement_counts() -> None:
    """A request settled on success and again on a failure path is charged once."""
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "key", "tpm": 1000})

    grant = await _admit(rules, tokens=600)
    await grant.settle(900)
    await grant.settle(0)

    with pytest.raises(HTTPException):
        await _admit(rules, tokens=200)


@pytest.mark.asyncio
async def test_a_full_concurrency_limit_refuses_until_a_slot_is_given_back() -> None:
    rules = _rules(InMemoryRateLimitStore(), {"name": "inflight", "per": "key", "max_concurrent": 1})

    grant = await _admit(rules)
    with pytest.raises(HTTPException) as exc_info:
        await _admit(rules)
    assert exc_info.value.headers == {"Retry-After": "1"}

    await grant.release()
    await _admit(rules)


@pytest.mark.asyncio
async def test_a_refusal_gives_back_the_slots_it_took() -> None:
    rules = _rules(
        InMemoryRateLimitStore(),
        {"name": "inflight", "per": "deployment", "max_concurrent": 1},
        {"name": "keys", "per": "key", "rpm": 1},
    )

    grant = await _admit(rules, key_id="k1")
    await grant.release()
    with pytest.raises(HTTPException, match="'keys'"):
        await _admit(rules, key_id="k1")

    await _admit(rules, key_id="k2")


async def _receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


async def _send(message: Message) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_the_middleware_gives_slots_back_however_the_request_ends(fails: bool) -> None:
    rules = _rules(InMemoryRateLimitStore(), {"name": "inflight", "per": "deployment", "max_concurrent": 1})

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await rules.admit(Request(scope), key_id=None, user_id=None, estimated_tokens=1)
        if fails:
            raise RuntimeError("handler failed")

    scope: Scope = {"type": "http", "headers": []}
    if fails:
        with pytest.raises(RuntimeError):
            await RateLimitGrantMiddleware(app)(scope, _receive, _send)
    else:
        await RateLimitGrantMiddleware(app)(scope, _receive, _send)

    await _admit(rules)


@pytest.mark.asyncio
@pytest.mark.parametrize("handed_over", [False, True])
async def test_the_middleware_charges_no_tokens_to_a_request_refused_before_dispatch(handed_over: bool) -> None:
    """A budget refusal after admission frees its estimate; a dispatched request settles its own."""
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "deployment", "tpm": 1000})

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        grant = await rules.admit(Request(scope), key_id=None, user_id=None, estimated_tokens=600)
        if handed_over:
            grant.hand_over()

    await RateLimitGrantMiddleware(app)({"type": "http", "headers": []}, _receive, _send)

    if handed_over:
        with pytest.raises(HTTPException):
            await _admit(rules, tokens=600)
    else:
        await _admit(rules, tokens=600)
