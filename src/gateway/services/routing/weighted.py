"""The ``weighted`` router backend: split traffic across providers by weight.

A load balancer expressed as a routing policy. ``{router: weighted, candidates:
[...], weights: {...}}`` sends each request to one candidate drawn at random in
proportion to its weight, so two providers weighted 70 and 30 serve roughly that
share of the traffic. What it is for: spreading load across providers, holding a
second provider warm so a failover has somewhere to go, moving traffic onto a new
provider a few percent at a time, and draining one to zero without deleting it.

Three properties are worth stating because they are the reasons this is the
simplest correct design and not a placeholder for a smarter one:

* **Stateless.** Each request is an independent draw, so the split is exactly as
  correct behind twenty replicas as behind one. Unlike learned trace stickiness,
  it needs no cross-request decision store. The cost is that the ratio converges
  statistically: a ten-request burst is not necessarily seven and three.
* **No pricing needed by the router.** Unlike the kNN router, nothing here scores
  cost, so an unpriced candidate is neither refused at policy-write time nor a
  reason to decline the draw. Weight is the operator's statement about capacity,
  not a number the gateway derives. The gateway's own ``require_pricing`` billing
  gate is separate and unchanged: it still 402s a metered caller drawn onto an
  unpriced candidate, exactly as it would for that model named directly.
* **The whole ordering is the plan.** The draw continues without replacement over
  the candidates that were not picked, so a provider that fails before it has
  responded hands the request to another weighted provider (itself chosen by
  weight) before the policy's ``on_failure`` chain is reached. A provider that is
  failing therefore sheds its share to the others for free, without any health
  tracking.

Zero-weight candidates are not excluded from the plan, only from the draw: they
sit at its tail in declared order. That is what makes ``{a: 100, b: 0}`` a drain
rather than a deletion, and it is why the tail keeps a stable order instead of
being shuffled.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from gateway.models.routing import WEIGHTED_BACKEND
from gateway.services.routing.backends import RoutingContext, RoutingDecision

__all__ = [
    "WeightedRouterBackend",
    "declared_shares",
    "describe_split",
    "explain_ordering",
    "weighted_ordering",
]

# One stream for the whole process, because the registry builds a backend per
# request: a fresh `random.Random()` each time would re-seed from OS entropy on
# every request, and successive draws would be independent seeds rather than
# successive values from one sequence. Not locked, deliberately. A worker runs one
# request at a time under asyncio, and the worst a torn read could do is skew a
# single draw, which the next request corrects.
_DEFAULT_RNG = random.Random()


def declared_shares(weights: Mapping[str, float], pool: Sequence[str]) -> dict[str, float]:
    """Percentage of traffic each candidate in ``pool`` receives.

    Normalized over ``pool`` rather than over the declared weights, because the
    pool is what survived this caller's allow-list: with a 70/30 policy whose 70
    is not permitted, the 30 candidate really does serve every request, and
    reporting it as 30% would describe a split that is not happening.

    An all-zero pool reports the head taking everything, because that is what the
    draw does with it: :func:`weighted_ordering` has nothing to sample and keeps
    declared order, so the first candidate serves every request. Policy validation
    guarantees at least one positive weight, so this only happens when every
    weighted candidate was filtered out and the drained ones are all that is left.
    An even split would read as a balanced policy that is not balancing, and these
    numbers exist to say what is running.
    """
    if not pool:
        return {}
    raw = {selector: max(0.0, float(weights.get(selector, 0.0))) for selector in pool}
    total = sum(raw.values())
    if total <= 0:
        return {selector: 100.0 if index == 0 else 0.0 for index, selector in enumerate(pool)}
    return {selector: weight * 100.0 / total for selector, weight in raw.items()}


def weighted_ordering(
    pool: Sequence[str], weights: Mapping[str, float], rng: random.Random
) -> list[str]:
    """Order ``pool`` by repeated weighted draw without replacement.

    The head is the request's provider; the rest is the order a failure walks, each
    step drawn by weight among what is left. Candidates whose weight is zero are
    never drawn, so they land at the tail in declared order.
    """
    remaining = list(pool)
    ordered: list[str] = []
    while remaining:
        weighted = [(selector, max(0.0, float(weights.get(selector, 0.0)))) for selector in remaining]
        total = sum(weight for _, weight in weighted)
        if total <= 0:
            # Nothing left has a share. Keeping declared order (rather than
            # shuffling) is what makes a drained provider a predictable last resort.
            ordered.extend(remaining)
            break
        draw = rng.random() * total
        cumulative = 0.0
        # Falls back to the last candidate with a positive weight, which only comes
        # up when floating-point drift leaves `draw` past the final boundary. A
        # zero-weight candidate must not win the draw that way.
        picked = next(selector for selector, weight in reversed(weighted) if weight > 0)
        for selector, weight in weighted:
            if weight <= 0:
                continue
            cumulative += weight
            if draw < cumulative:
                picked = selector
                break
        ordered.append(picked)
        remaining.remove(picked)
    return ordered


def explain_ordering(pool: Sequence[str], weights: Mapping[str, float]) -> list[str]:
    """The ordering to show when there is no request to draw for.

    ``explain`` dispatches nothing, so there is nothing to sample; showing one
    sampled outcome would also make the command's output change between runs.
    Heaviest share first, ties in declared order, which is the split itself rather
    than one roll of it.
    """
    shares = declared_shares(weights, pool)
    positions = {selector: index for index, selector in enumerate(pool)}
    return sorted(pool, key=lambda selector: (-shares[selector], positions[selector]))


def describe_split(pool: Sequence[str], weights: Mapping[str, float]) -> str:
    """Operator-facing text for a split, e.g. ``openai:gpt-5 70%, anthropic:x 30%``."""
    shares = declared_shares(weights, pool)
    return ", ".join(f"{selector} {shares[selector]:.0f}%" for selector in explain_ordering(pool, weights))


class WeightedRouterBackend:
    """Draws one candidate per request in proportion to its weight.

    Holds no per-request or per-conversation state, so a single instance is safe
    across requests and across policies: everything it reads about the policy
    arrives on the :class:`RoutingContext`. ``rng`` is injectable so a test can
    assert an exact sequence rather than a distribution; left out, every instance
    shares the process-wide stream.
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng if rng is not None else _DEFAULT_RNG

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        pool = list(ctx.candidate_pool)
        if not pool:
            return RoutingDecision.decline("no candidate in the pool is usable by this caller")
        if not ctx.weights:
            # Unreachable through a validated policy (a weighted entry must carry
            # weights), so this is the "policy document written by an older build"
            # case. Declining serves the default target, which is the safe reading.
            return RoutingDecision.decline(f"policy declares no weights for router '{WEIGHTED_BACKEND}'")
        ordered = weighted_ordering(pool, ctx.weights, self._rng)
        shares = declared_shares(ctx.weights, pool)
        return RoutingDecision(
            ordered_models=ordered,
            # The head's own share, so a decision log line reads as "this provider
            # was due about 70% of the traffic" rather than borrowing the kNN
            # meaning of confidence (neighbor support), which has no analogue here.
            confidence=shares[ordered[0]] / 100.0,
            rationale=f"weighted split ({describe_split(pool, ctx.weights)})",
            # One line per request is one line per request forever for a load
            # balancer, and the usage row already records which model served with
            # `selection_reason: router:weighted`.
            log_decision=False,
        )
