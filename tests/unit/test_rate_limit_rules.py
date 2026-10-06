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
    return RateLimitRules(store, GatewayConfig(rate_limits=[RateLimitRule(**rule) for rule in rules]))


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
    assert exc_info.value.detail == "Rate limit 'keys' exceeded: 1 request per minute"
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
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "key", "tpm": 1000, "tpm_admission": "estimate"})

    grant = await _admit(rules, tokens=600)
    with pytest.raises(HTTPException, match="'tpm' exceeded: 1,000 tokens per minute"):
        await _admit(rules, tokens=600)

    await grant.settle(50)
    await _admit(rules, tokens=600)


@pytest.mark.asyncio
async def test_a_request_larger_than_the_limit_is_not_told_to_retry() -> None:
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "key", "tpm": 1000, "tpm_admission": "estimate"})

    with pytest.raises(HTTPException) as exc_info:
        await _admit(rules, tokens=1001)

    assert exc_info.value.headers is not None
    assert "Retry-After" not in exc_info.value.headers
    assert exc_info.value.detail == "Request needs an estimated 1,001 tokens; rate limit 'tpm' allows 1,000 per minute"


@pytest.mark.asyncio
async def test_only_the_first_settlement_counts() -> None:
    """A request settled on success and again on a failure path is charged once."""
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "key", "tpm": 1000, "tpm_admission": "estimate"})

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
    assert exc_info.value.headers == {
        "Retry-After": "1",
        "Otari-Error-Code": "rate_limited",
        "Otari-Rate-Limit-Rule": "inflight",
    }
    assert exc_info.value.detail == "Rate limit 'inflight' exceeded: 1 request in flight"

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
    rules = _rules(
        InMemoryRateLimitStore(), {"name": "tpm", "per": "deployment", "tpm": 1000, "tpm_admission": "estimate"}
    )

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


def test_a_per_model_rule_must_name_its_models() -> None:
    with pytest.raises(ValidationError, match="names no models"):
        RateLimitRule(name="cap", per="model", rpm=1)


def test_only_a_per_model_rule_names_models() -> None:
    with pytest.raises(ValidationError, match="only a per: model rule reads"):
        RateLimitRule(name="cap", per="key", models=["openai:gpt-4o"], rpm=1)


def test_a_per_model_rule_spells_each_model_as_instance_and_model() -> None:
    rule = RateLimitRule(
        name="cap",
        per="model",
        models=["openai:gpt-4o", "together/meta-llama/Llama-3.3-70B", "ollama:llama3:8b"],
        rpm=1,
    )

    assert rule.models == ["openai:gpt-4o", "together:meta-llama/Llama-3.3-70B", "ollama:llama3:8b"]
    with pytest.raises(ValidationError, match="write a model as instance:model"):
        RateLimitRule(name="cap", per="model", models=["gpt-4o"], rpm=1)


def test_a_slash_selector_with_a_tagged_model_splits_at_its_first_delimiter() -> None:
    """``ollama/llama3:latest`` is instance ``ollama`` and model ``llama3:latest``, or the rule never matches."""
    rule = RateLimitRule(name="cap", per="model", models=["ollama/llama3:latest"], rpm=1)

    assert rule.models == ["ollama:llama3:latest"]


@pytest.mark.asyncio
async def test_admission_leaves_per_model_rules_to_the_attempt() -> None:
    rules = _rules(InMemoryRateLimitStore(), {"name": "cap", "per": "model", "models": ["openai:gpt-4o"], "rpm": 1})

    await _admit(rules)
    grant = await _admit(rules)

    assert await grant.admit_model("openai", "gpt-4o") is not None


@pytest.mark.asyncio
async def test_each_model_a_rule_names_is_counted_on_its_own() -> None:
    rules = _rules(
        InMemoryRateLimitStore(),
        {"name": "cap", "per": "model", "models": ["openai:gpt-4o", "openai:gpt-4o-mini"], "rpm": 1},
    )

    await (await _admit(rules)).admit_model("openai", "gpt-4o")
    grant = await _admit(rules)
    await grant.admit_model("openai", "gpt-4o-mini")
    with pytest.raises(HTTPException) as exc_info:
        await grant.admit_model("openai", "gpt-4o")

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail == "Rate limit 'cap' for openai:gpt-4o exceeded: 1 request per minute"
    assert await grant.admit_model("anthropic", "claude") is None


@pytest.mark.asyncio
async def test_a_dropped_attempt_gives_back_its_slot_and_tokens_but_keeps_its_request() -> None:
    """The provider was sent the request, so it stays counted; nothing else does."""
    store = InMemoryRateLimitStore()
    rules = _rules(
        store, {"name": "cap", "per": "model", "models": ["openai:gpt-4o"], "rpm": 5, "tpm": 1000, "max_concurrent": 1}
    )
    grant = await _admit(rules, tokens=600)

    hold = await grant.admit_model("openai", "gpt-4o")
    assert hold is not None
    await hold.drop()
    await grant.admit_model("openai", "gpt-4o")

    assert (await store.hit("rule:cap:openai:gpt-4o:rpm", 5, 60)).count == 3


@pytest.mark.asyncio
async def test_an_attempt_dropped_before_it_was_sent_keeps_nothing() -> None:
    store = InMemoryRateLimitStore()
    rules = _rules(store, {"name": "cap", "per": "model", "models": ["openai:gpt-4o"], "rpm": 5, "max_concurrent": 1})
    grant = await _admit(rules)

    hold = await grant.admit_model("openai", "gpt-4o")
    assert hold is not None
    await hold.drop(sent=False)
    await grant.admit_model("openai", "gpt-4o")

    assert (await store.hit("rule:cap:openai:gpt-4o:rpm", 5, 60)).count == 2


class _BrokenStore(InMemoryRateLimitStore):
    """A store whose window for ``broken_key`` fails, as a lost Redis connection would."""

    def __init__(self, broken_key: str) -> None:
        super().__init__()
        self._broken_key = broken_key

    async def hit(self, key: str, limit: int, window_sec: float, cost: int = 1) -> Any:
        if key == self._broken_key:
            raise ConnectionError("store unreachable")
        return await super().hit(key, limit, window_sec, cost=cost)


@pytest.mark.asyncio
async def test_an_attempt_whose_admission_breaks_gives_back_what_it_took() -> None:
    store = _BrokenStore("rule:second:openai:gpt-4o:rpm")
    rules = _rules(
        store,
        {"name": "first", "per": "model", "models": ["openai:gpt-4o"], "max_concurrent": 1},
        {"name": "second", "per": "model", "models": ["openai:gpt-4o"], "rpm": 5},
    )
    grant = await _admit(rules)

    with pytest.raises(ConnectionError):
        await grant.admit_model("openai", "gpt-4o")

    lease = await store.acquire("rule:first:openai:gpt-4o:concurrent", 1, 900)
    assert lease is not None, "the slot the first rule took was kept"


@pytest.mark.asyncio
async def test_a_served_attempt_is_settled_and_released_with_its_request() -> None:
    rules = _rules(
        InMemoryRateLimitStore(),
        {
            "name": "cap",
            "per": "model",
            "models": ["openai:gpt-4o"],
            "tpm": 1000,
            "tpm_admission": "estimate",
            "max_concurrent": 1,
        },
    )
    grant = await _admit(rules, tokens=600)
    await grant.admit_model("openai", "gpt-4o")
    other = await _admit(rules, tokens=600)
    with pytest.raises(HTTPException, match="1,000 tokens per minute"):
        await other.admit_model("openai", "gpt-4o")

    await grant.settle(50)
    with pytest.raises(HTTPException, match="1 request in flight"):
        await other.admit_model("openai", "gpt-4o")
    await grant.release()

    await other.admit_model("openai", "gpt-4o")


@pytest.mark.asyncio
async def test_a_refused_attempt_is_counted_by_no_model_rule() -> None:
    store = InMemoryRateLimitStore()
    rules = _rules(
        store,
        {"name": "wide", "per": "model", "models": ["openai:gpt-4o"], "rpm": 5},
        {"name": "narrow", "per": "model", "models": ["openai:gpt-4o"], "max_concurrent": 1},
    )
    await (await _admit(rules)).admit_model("openai", "gpt-4o")

    with pytest.raises(HTTPException, match="'narrow'"):
        await (await _admit(rules)).admit_model("openai", "gpt-4o")

    assert (await store.hit("rule:wide:openai:gpt-4o:rpm", 5, 60)).count == 2


@pytest.mark.asyncio
async def test_used_admission_admits_past_an_estimate_and_counts_what_was_used() -> None:
    """MLPA sends max_tokens 8192 against a 2,000 tpm: only what a request used may count."""
    rules = _rules(InMemoryRateLimitStore(), {"name": "tpm", "per": "user", "tpm": 2000})

    first = await _admit(rules, tokens=8192)
    await first.settle(1500)
    second = await _admit(rules, tokens=8192)
    await second.settle(600)
    with pytest.raises(HTTPException) as exc_info:
        await _admit(rules, tokens=8192)

    assert exc_info.value.detail == "Rate limit 'tpm' exceeded: 2,000 tokens per minute"


def test_a_rule_counts_what_was_used_unless_it_asks_for_the_estimate() -> None:
    assert RateLimitRule(name="t", per="user", tpm=1).tpm_admission == "used"
    assert RateLimitRule(name="t", per="user", tpm=1, tpm_admission="estimate").tpm_admission == "estimate"
