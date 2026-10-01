"""The ``smart_router`` backend: ask an external smart-router service which candidate to serve.

The service picks one of the candidates it is allowed (``POST /v1/route``), is
told what serving it cost and whether it worked (``POST /v1/completion``), and
is told how good the answer was when the caller rates it (``POST /v1/feedback``).
The three calls are tied together by the ``sample_id`` the route call returns,
which is this backend's decision id.

A router is an optimization, so every way the route call can fail declines:
the service has no opinion (``model_id: null``), names a model it was not
offered, answers with an error, is slow or is unreachable. The policy's default
target then serves. "No opinion" is the one decline that keeps its sample: the
service asked to see the default serve, so it still gets the outcome and any
rating, and the cluster starts accumulating statistics. The outcome and feedback
calls never affect the response.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from decimal import Decimal
from typing import Any

import httpx

from gateway.exceptions.routing_exceptions import RatingAlreadyRecordedError, RouterFeedbackFailedError
from gateway.log_config import logger
from gateway.ports.routing_port import RoutingContext, RoutingDecision, RoutingOutcome

__all__ = ["COMPLETION_PATH", "DEFAULT_COST_WEIGHT", "FEEDBACK_PATH", "ROUTE_PATH", "SmartRouterBackend"]

# The service's own paths, under its base URL. Its wire, not this gateway's API.
ROUTE_PATH = "/v1/route"
COMPLETION_PATH = "/v1/completion"
FEEDBACK_PATH = "/v1/feedback"
DEFAULT_COST_WEIGHT = 1.0
# How long a rating waits for the same decision's outcome report to land. The
# service counts feedback only for a sample that has a completion, and a caller
# can rate the moment its response arrives, while the report is still in flight.
_OUTCOME_GRACE_SECONDS = 2.0
# Outcome reports this process has in flight, by decision, so a rating can wait
# for its own. Bounded, because entries are dropped as reports finish and an
# abandoned one is worth nothing after the grace above.
_PENDING_MAX = 1024


def _usd(value: Decimal | None) -> float | None:
    """A cost for the wire, which speaks float."""
    return float(value) if value is not None else None


class SmartRouterBackend:
    """Route through a smart-router service at ``base_url``, on one pooled HTTP client.

    Learns from outcomes and ratings (``LearningRouterBackend``). The client is
    created on first use and closed by :meth:`aclose`, which the app's shutdown
    calls through ``close_router_backends``.
    """

    def __init__(self, base_url: str, timeout_seconds: float, *, client: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._client = client
        self._pending: OrderedDict[str, asyncio.Task[Any]] = OrderedDict()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the pooled client. A no-op when nothing was ever sent."""
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            await client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        return await self._http().post(f"{self._base_url}{path}", json=body, timeout=self._timeout)

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        if not ctx.messages:
            return RoutingDecision.decline("smart_router: the request has no text to route on")
        body = {
            "application_id": ctx.application_id or ctx.policy_name or ctx.default_model,
            "inputs": [{"role": message.role, "content": message.content} for message in ctx.messages],
            "lambda": ctx.cost_weight if ctx.cost_weight is not None else DEFAULT_COST_WEIGHT,
            "allowed_model_ids": list(ctx.candidate_pool),
        }
        try:
            response = await self._post(ROUTE_PATH, body)
        except httpx.TimeoutException:
            return RoutingDecision.decline("smart_router: the route call timed out")
        except httpx.HTTPError as exc:
            return RoutingDecision.decline(f"smart_router: the service is unreachable ({type(exc).__name__})")
        if not response.is_success:
            return RoutingDecision.decline(f"smart_router: the service answered {response.status_code}")
        try:
            payload = response.json()
            decision = payload["decision"]
            model_id = decision.get("model_id")
            sample_id = str(payload["sample_id"])
        except (ValueError, KeyError, TypeError, AttributeError):
            return RoutingDecision.decline("smart_router: the service's answer was not a route decision")
        if model_id is None:
            # No opinion yet (a cluster with no data for these models) is the service
            # asking to see the default serve: it keeps the sample, so the outcome and
            # any rating of the default go to it and the cluster starts learning.
            return RoutingDecision.decline("smart_router: no opinion for this request", decision_id=sample_id)
        if model_id not in ctx.candidate_pool:
            return RoutingDecision.decline(f"smart_router: picked '{model_id}', which it was not offered")
        utility = decision.get("utility_score")
        explored = bool(decision.get("explored"))
        # The pick leads and the rest follow in the policy's order, so a failed
        # pick falls over to another candidate before the failure chain.
        ordered = [model_id, *(candidate for candidate in ctx.candidate_pool if candidate != model_id)]
        return RoutingDecision(
            ordered_models=ordered,
            confidence=1.0,
            rationale=(
                f"smart_router cluster {decision.get('cluster_id')}"
                f"{', explored' if explored else ''}"
                f"{f', utility {utility:.3f}' if isinstance(utility, int | float) else ''}"
            ),
            decision_id=sample_id,
        )

    async def record_outcome(self, decision_id: str, outcome: RoutingOutcome) -> None:
        current = asyncio.current_task()
        if current is not None:
            self._pending[decision_id] = current
            while len(self._pending) > _PENDING_MAX:
                self._pending.popitem(last=False)
        try:
            response = await self._post(
                COMPLETION_PATH,
                {
                    "sample_id": decision_id,
                    "model_id": outcome.model,
                    "success": outcome.success,
                    "error": outcome.error,
                    "request_started_at": outcome.started_at.isoformat(),
                    "request_completed_at": outcome.completed_at.isoformat(),
                    "prompt_tokens": outcome.prompt_tokens,
                    "completion_tokens": outcome.completion_tokens,
                    "total_tokens": outcome.total_tokens,
                    "prompt_cost_usd": _usd(outcome.prompt_cost_usd),
                    "completion_cost_usd": _usd(outcome.completion_cost_usd),
                    "total_cost_usd": _usd(outcome.total_cost_usd),
                },
            )
        finally:
            if self._pending.get(decision_id) is current:
                self._pending.pop(decision_id, None)
        if not response.is_success:
            logger.warning("Smart router refused the outcome for sample %s with %d", decision_id, response.status_code)

    async def record_feedback(self, decision_id: str, score: float) -> None:
        pending = self._pending.get(decision_id)
        if pending is not None and not pending.done():
            # Let the outcome land first; a rating for a sample with no completion is not counted.
            await asyncio.wait({pending}, timeout=_OUTCOME_GRACE_SECONDS)
        try:
            response = await self._post(FEEDBACK_PATH, {"sample_id": decision_id, "score": score})
        except httpx.HTTPError as exc:
            raise RouterFeedbackFailedError(
                f"The smart router could not be reached to record the rating ({type(exc).__name__})."
            ) from exc
        if response.status_code == 409:
            # The service keeps one rating per sample; a second one is the caller's
            # conflict to see, not a router failure to retry.
            raise RatingAlreadyRecordedError("This response has already been rated.")
        if not response.is_success:
            raise RouterFeedbackFailedError(f"The smart router refused the rating with {response.status_code}.")
