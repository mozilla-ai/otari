"""What a learning router backend is told after a request it decided, and when.

The outcome is built from the usage row the request settled, so the backend sees
the same tokens and cost the activity log does. It is reported once per request,
from the row that settled it, never from an attempt the request recovered from.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from gateway.api.routes import _pipeline
from gateway.api.routes._pipeline import RoutingAttribution, _report_routing_outcome
from gateway.core.config import GatewayConfig
from gateway.models.routing import PolicySpec
from gateway.models.usage import UsageLog
from gateway.ports.routing_port import RoutingContext, RoutingDecision, RoutingOutcome
from gateway.services.routing.decide import RoutingSignal, decide_ordering
from gateway.services.routing.outcome import outcome_from_usage_row

STARTED = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
COMPLETED = STARTED + timedelta(seconds=2)


def _row(**overrides: object) -> UsageLog:
    values: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "workspace_id": uuid.uuid4(),
        "timestamp": COMPLETED,
        "model": "gpt-5-mini",
        "provider": "openai",
        "endpoint": "/v1/chat/completions",
        "status": "success",
        "prompt_tokens": 20,
        "completion_tokens": 50,
        "total_tokens": 70,
        "cost": Decimal("0.0031"),
        "pricing_breakdown": [
            {"meter": "input", "units": 20, "rate_per_million": 50.0, "cost": 0.001},
            {"meter": "output", "units": 50, "rate_per_million": 40.0, "cost": 0.002},
            {"meter": "web_search_calls", "units": 1, "rate_per_million": 0.0, "cost": 0.0001},
        ],
    }
    values.update(overrides)
    return UsageLog(**values)


class _Learner:
    """A learning backend that records what it is told."""

    def __init__(self, decision_id: str | None = "sample-1") -> None:
        self.decision_id = decision_id
        self.outcomes: list[tuple[str, RoutingOutcome]] = []
        self.feedback: list[tuple[str, float]] = []

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        return RoutingDecision(
            ordered_models=list(reversed(ctx.candidate_pool)),
            confidence=1.0,
            rationale="learned",
            decision_id=self.decision_id,
        )

    async def record_outcome(self, decision_id: str, outcome: RoutingOutcome) -> None:
        self.outcomes.append((decision_id, outcome))

    async def record_feedback(self, decision_id: str, score: float) -> None:
        self.feedback.append((decision_id, score))


class _Ranker:
    """A backend that ranks and learns nothing, but still hands out a token."""

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        return RoutingDecision(ordered_models=list(ctx.candidate_pool), confidence=1.0, rationale="x", decision_id="t")


class _Port:
    def __init__(self, backend: object) -> None:
        self._backend = backend

    def backend(self, name: str) -> object:
        return self._backend

    def known_backends(self) -> tuple[str, ...]:
        return ("custom",)

    def traits(self, name: str | None) -> object:
        raise NotImplementedError


def _attribution(learner: _Learner, **overrides: object) -> RoutingAttribution:
    values: dict[str, object] = {
        "policy_name": "smart",
        "selection_reason": "router:custom",
        "position": 1,
        "attempt_count": 2,
        "request_group_id": "group",
        "router_backend": "custom",
        "decision_id": learner.decision_id,
        "observer": learner,
        "served_selector": "openai:gpt-5-mini",
        "request_started_at": STARTED,
    }
    values.update(overrides)
    return RoutingAttribution(**values)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _forget_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_pipeline, "_REPORTED_DECISIONS", type(_pipeline._REPORTED_DECISIONS)())


# -- the payload -----------------------------------------------------------


def test_a_served_row_reports_its_tokens_and_split_cost() -> None:
    outcome = outcome_from_usage_row(_row(), model="openai:gpt-5-mini", started_at=STARTED)

    assert outcome.model == "openai:gpt-5-mini"
    assert outcome.success is True
    assert outcome.error is None
    assert (outcome.started_at, outcome.completed_at) == (STARTED, COMPLETED)
    assert (outcome.prompt_tokens, outcome.completion_tokens, outcome.total_tokens) == (20, 50, 70)
    assert outcome.prompt_cost_usd == Decimal("0.001")
    assert outcome.completion_cost_usd == Decimal("0.002")
    # The total is the row's, tool charge included, which neither side's split carries.
    assert outcome.total_cost_usd == Decimal("0.0031")


def test_a_failed_row_reports_the_failure_and_no_usage() -> None:
    row = _row(
        status="error",
        error_message="Provider request failed",
        status_code=502,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        cost=None,
        pricing_breakdown=None,
    )

    outcome = outcome_from_usage_row(row, model="openai:gpt-5", started_at=STARTED)

    assert outcome.success is False
    assert outcome.error == {"message": "Provider request failed", "status_code": 502}
    assert outcome.prompt_cost_usd is None
    assert outcome.completion_cost_usd is None
    assert outcome.total_cost_usd is None


def test_an_unpriced_row_with_only_tool_charges_has_no_token_split() -> None:
    row = _row(pricing_breakdown=[{"meter": "web_search_calls", "units": 1, "rate_per_million": 0.0, "cost": 0.01}])

    outcome = outcome_from_usage_row(row, model="openai:gpt-5", started_at=STARTED)

    assert outcome.prompt_cost_usd is None
    assert outcome.completion_cost_usd is None


# -- when it is reported ---------------------------------------------------


@pytest.mark.asyncio
async def test_the_settling_row_reports_once() -> None:
    learner = _Learner()
    attribution = _attribution(learner)

    _report_routing_outcome(attribution, _row())
    # A second non-absorbed row for the same request (an unusual failure path) is not a second outcome.
    _report_routing_outcome(attribution, _row(status="error"))
    await asyncio.sleep(0)

    assert [(decision, outcome.model, outcome.success) for decision, outcome in learner.outcomes] == [
        ("sample-1", "openai:gpt-5-mini", True)
    ]


@pytest.mark.asyncio
async def test_an_absorbed_attempt_is_not_an_outcome() -> None:
    learner = _Learner()

    _report_routing_outcome(_attribution(learner, absorbed=True), _row(status="absorbed"))
    await asyncio.sleep(0)
    assert learner.outcomes == []

    # The attempt that then served is.
    _report_routing_outcome(_attribution(learner, position=2, served_selector="openai:gpt-5"), _row())
    await asyncio.sleep(0)
    assert [outcome.model for _, outcome in learner.outcomes] == ["openai:gpt-5"]


@pytest.mark.asyncio
async def test_a_row_with_no_learning_decision_reports_nothing() -> None:
    learner = _Learner()

    _report_routing_outcome(None, _row())
    _report_routing_outcome(_attribution(learner, observer=None, decision_id=None), _row())
    await asyncio.sleep(0)

    assert learner.outcomes == []


@pytest.mark.asyncio
async def test_a_failing_backend_does_not_reach_the_caller() -> None:
    class _Broken(_Learner):
        async def record_outcome(self, decision_id: str, outcome: RoutingOutcome) -> None:
            raise RuntimeError("router down")

    _report_routing_outcome(_attribution(_Broken()), _row())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not _pipeline._ROUTING_OUTCOME_TASKS


# -- the token on the ordering ---------------------------------------------


def _spec() -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "select": [
                {"router": "custom", "candidates": ["openai:gpt-5-mini", "openai:gpt-5"]},
                {"default": "openai:gpt-5"},
            ]
        }
    )


def _config() -> GatewayConfig:
    return GatewayConfig(master_key="k", model_discovery=False, providers={"openai": {"api_key": "sk"}})


async def _decide(backend: object) -> object:
    return await decide_ordering(
        _config(),
        _spec(),
        policy_name="smart",
        user_id="u",
        allowlist=None,
        signal=RoutingSignal(task_signal="hi"),
        routing=_Port(backend),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_a_learning_backend_keeps_its_token_on_the_ordering() -> None:
    learner = _Learner()

    ordering = await _decide(learner)

    assert ordering.decision_id == "sample-1"  # type: ignore[attr-defined]
    assert ordering.observer is learner  # type: ignore[attr-defined]
    assert ordering.backend == "custom"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_backend_that_cannot_learn_has_its_token_dropped() -> None:
    # Nothing could be told about the decision, so nothing claims it is rateable.
    ordering = await _decide(_Ranker())

    assert ordering.decision_id is None  # type: ignore[attr-defined]
    assert ordering.observer is None  # type: ignore[attr-defined]
