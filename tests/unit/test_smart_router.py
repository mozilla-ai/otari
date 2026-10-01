"""The ``smart_router`` backend: what it asks the service, what it does with the answer, and its policy fields.

The service is faked with ``httpx.MockTransport``, so each test states exactly
what the service answered. Every failure of the route call is a decline, because
a router is an optimization and the policy's default target is always safe.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from gateway.core.config import GatewayConfig
from gateway.exceptions.routing_exceptions import RouterFeedbackFailedError
from gateway.models.routing import PolicySpec
from gateway.ports.routing_port import LearningRouterBackend, RoutingContext, RoutingMessage, RoutingOutcome
from gateway.services.routing.backends import (
    backend_pool_is_teachable,
    backend_requires_pricing,
    clear_router_backend_cache,
    close_router_backends,
    get_router_backend,
    known_backends,
)
from gateway.services.routing.decide import ROUTER_DEADLINE_SECONDS
from gateway.services.routing.smart_router import SmartRouterBackend

BASE = "http://smart-router.test"
POOL = ["openai:gpt-5-mini", "openai:gpt-5"]
SAMPLE = "7d3c1f0e-5b0a-4f4e-9f1e-0c6f8f2b9a11"


class _Service:
    """Records each request and answers with the next scripted response (or raises it)."""

    def __init__(self, *answers: httpx.Response | Exception) -> None:
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else httpx.Response(204)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def body(self, index: int = 0) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(self.requests[index].content)
        return loaded


def _backend(service: _Service) -> SmartRouterBackend:
    return SmartRouterBackend(BASE, 1.0, client=httpx.AsyncClient(transport=httpx.MockTransport(service)))


def _ctx(**overrides: Any) -> RoutingContext:
    values: dict[str, Any] = {
        "user_id": "u",
        "default_model": "openai:gpt-5",
        "candidate_pool": list(POOL),
        "messages": (RoutingMessage("system", "be brief"), RoutingMessage("user", "reverse a string")),
        "policy_name": "smart",
    }
    values.update(overrides)
    return RoutingContext(**values)


def _route(model_id: str | None, **decision: Any) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "decision": {"cluster_id": 3, "model_id": model_id, "utility_score": 0.42, "explored": False, **decision},
            "sample_id": SAMPLE,
        },
    )


# -- rank ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_route_call_carries_the_messages_and_the_candidates() -> None:
    service = _Service(_route("openai:gpt-5"))

    decision = await _backend(service).rank(_ctx())

    request = service.requests[0]
    assert (request.method, str(request.url)) == ("POST", f"{BASE}/v1/route")
    assert service.body() == {
        "application_id": "smart",
        "inputs": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "reverse a string"}],
        "lambda": 1.0,
        "allowed_model_ids": POOL,
    }
    # The pick leads; the rest of the pool follows in the policy's order.
    assert decision.ordered_models == ["openai:gpt-5", "openai:gpt-5-mini"]
    assert decision.decision_id == SAMPLE
    assert "cluster 3" in decision.rationale


@pytest.mark.asyncio
async def test_the_policy_names_the_application_and_the_cost_weight() -> None:
    service = _Service(_route("openai:gpt-5-mini"))

    await _backend(service).rank(_ctx(application_id="checkout-bot", cost_weight=0.25))

    assert service.body()["application_id"] == "checkout-bot"
    assert service.body()["lambda"] == 0.25


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (_route(None), "no opinion"),
        (_route("anthropic:claude-haiku-4-5"), "not offered"),
        (httpx.Response(503, json={"detail": "Router is not loaded yet."}), "503"),
        (httpx.Response(200, json={"unexpected": True}), "not a route decision"),
        (httpx.Response(200, content=b"<html>"), "not a route decision"),
        (httpx.ReadTimeout("slow"), "timed out"),
        (httpx.ConnectError("refused"), "unreachable"),
    ],
)
@pytest.mark.asyncio
async def test_every_failure_of_the_route_call_declines(answer: httpx.Response | Exception, reason: str) -> None:
    decision = await _backend(_Service(answer)).rank(_ctx())

    assert decision.ordered_models == []
    assert decision.decision_id is None
    assert reason in decision.rationale


@pytest.mark.asyncio
async def test_a_request_with_no_text_is_not_sent() -> None:
    service = _Service()

    decision = await _backend(service).rank(_ctx(messages=()))

    assert decision.ordered_models == []
    assert service.requests == []


@pytest.mark.asyncio
async def test_every_call_carries_the_configured_timeout() -> None:
    """The service is cut off on its own clock, under the router deadline, rather than the deadline's."""
    service = _Service(_route("openai:gpt-5"))
    backend = SmartRouterBackend(BASE, 0.25, client=httpx.AsyncClient(transport=httpx.MockTransport(service)))

    await backend.rank(_ctx())

    assert service.requests[0].extensions["timeout"] == {"connect": 0.25, "read": 0.25, "write": 0.25, "pool": 0.25}


# -- outcome and feedback --------------------------------------------------


def _outcome() -> RoutingOutcome:
    return RoutingOutcome(
        model="openai:gpt-5-mini",
        success=True,
        error=None,
        started_at=datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 10, 1, 12, 0, 2, tzinfo=UTC),
        prompt_tokens=20,
        completion_tokens=50,
        total_tokens=70,
        prompt_cost_usd=Decimal("0.001"),
        completion_cost_usd=Decimal("0.002"),
        total_cost_usd=Decimal("0.003"),
    )


@pytest.mark.asyncio
async def test_the_outcome_is_posted_as_a_completion() -> None:
    service = _Service(httpx.Response(204))

    await _backend(service).record_outcome(SAMPLE, _outcome())

    assert str(service.requests[0].url) == f"{BASE}/v1/completion"
    assert service.body() == {
        "sample_id": SAMPLE,
        "model_id": "openai:gpt-5-mini",
        "success": True,
        "error": None,
        "request_started_at": "2026-10-01T12:00:00+00:00",
        "request_completed_at": "2026-10-01T12:00:02+00:00",
        "prompt_tokens": 20,
        "completion_tokens": 50,
        "total_tokens": 70,
        "prompt_cost_usd": 0.001,
        "completion_cost_usd": 0.002,
        "total_cost_usd": 0.003,
    }


@pytest.mark.asyncio
async def test_a_rating_is_posted_as_feedback() -> None:
    service = _Service(httpx.Response(204))

    await _backend(service).record_feedback(SAMPLE, 0.8)

    assert str(service.requests[0].url) == f"{BASE}/v1/feedback"
    assert service.body() == {"sample_id": SAMPLE, "score": 0.8}


@pytest.mark.parametrize("answer", [httpx.Response(500), httpx.ConnectError("refused")])
@pytest.mark.asyncio
async def test_a_rating_the_service_does_not_take_is_an_error(answer: httpx.Response | Exception) -> None:
    with pytest.raises(RouterFeedbackFailedError):
        await _backend(_Service(answer)).record_feedback(SAMPLE, 0.8)


@pytest.mark.asyncio
async def test_a_rating_waits_for_its_outcome_to_land() -> None:
    """The service counts a rating only for a sample with a completion, so the completion goes first."""
    order: list[str] = []
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/completion":
            await release.wait()
        order.append(request.url.path)
        return httpx.Response(204)

    backend = SmartRouterBackend(BASE, 1.0, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    report = asyncio.create_task(backend.record_outcome(SAMPLE, _outcome()))
    await asyncio.sleep(0)
    rating = asyncio.create_task(backend.record_feedback(SAMPLE, 1.0))
    await asyncio.sleep(0.01)
    release.set()
    await asyncio.gather(report, rating)

    assert order == ["/v1/completion", "/v1/feedback"]


def test_the_backend_learns() -> None:
    assert isinstance(SmartRouterBackend(BASE, 1.0), LearningRouterBackend)


# -- registration ----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_backend_is_unavailable_until_its_url_is_set() -> None:
    clear_router_backend_cache()
    assert get_router_backend(GatewayConfig(), "smart_router") is None

    config = GatewayConfig(smart_router_url=f"{BASE}/", smart_router_timeout_seconds=2)
    backend = get_router_backend(config, " Smart_Router ")

    assert isinstance(backend, SmartRouterBackend)
    assert backend is get_router_backend(config, "smart_router")
    assert backend._base_url == BASE
    assert "smart_router" in known_backends()
    await close_router_backends()
    assert get_router_backend(config, "smart_router") is not backend
    clear_router_backend_cache()


def test_the_smart_router_pool_is_not_taught_through_routing_memory() -> None:
    assert not backend_pool_is_teachable("smart_router")
    assert not backend_requires_pricing("smart_router")


@pytest.mark.parametrize("url", ["smart-router:5056", "ftp://host", "http://host?x=1"])
def test_a_smart_router_url_must_be_an_http_base(url: str) -> None:
    with pytest.raises(ValidationError, match="smart_router_url"):
        GatewayConfig(smart_router_url=url)


def test_the_timeout_fits_under_the_router_deadline() -> None:
    with pytest.raises(ValidationError):
        GatewayConfig(smart_router_timeout_seconds=ROUTER_DEADLINE_SECONDS)
    assert GatewayConfig().smart_router_timeout_seconds < ROUTER_DEADLINE_SECONDS


# -- policy fields ---------------------------------------------------------


def _policy(entry: dict[str, Any]) -> PolicySpec:
    return PolicySpec.model_validate({"select": [{"candidates": POOL, **entry}, {"default": "openai:gpt-5"}]})


def test_a_smart_router_entry_takes_an_application_and_a_cost_weight() -> None:
    spec = _policy({"router": "smart_router", "application_id": "checkout-bot", "cost_weight": 0.5})

    assert spec.router_application_id == "checkout-bot"
    assert spec.router_cost_weight == 0.5
    dumped = spec.model_dump(mode="json", exclude_none=True)
    assert dumped["select"][0]["application_id"] == "checkout-bot"
    assert dumped["select"][0]["cost_weight"] == 0.5


def test_the_parameters_default_when_omitted() -> None:
    spec = _policy({"router": "smart_router"})

    assert spec.router_application_id is None
    assert spec.router_cost_weight is None


@pytest.mark.parametrize("field", [{"application_id": "x"}, {"cost_weight": 1.0}])
def test_the_parameters_are_refused_on_another_router(field: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="only applies to a `router: smart_router` entry"):
        _policy({"router": "knn", **field})


@pytest.mark.parametrize("cost_weight", [-0.1, float("inf"), float("nan")])
def test_a_cost_weight_must_be_finite_and_non_negative(cost_weight: float) -> None:
    with pytest.raises(ValidationError, match="cost_weight"):
        _policy({"router": "smart_router", "cost_weight": cost_weight})


def test_a_blank_application_is_refused() -> None:
    with pytest.raises(ValidationError, match="application_id"):
        _policy({"router": "smart_router", "application_id": "   "})
