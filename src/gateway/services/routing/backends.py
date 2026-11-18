"""Router backends: the pluggable half of a routing policy's ``select``.

A policy entry ``{router: weighted, candidates: [...]}`` names a backend here. The
backend ranks the candidates for one request and the compiler turns that ranking
into the plan; everything else about the policy (guardrails, ``on_failure``, the
allow-list, the caps) is unchanged, so a router is one decision inside a policy
rather than a second routing system.

There is deliberately no global on/off switch. The policy naming a backend is the
switch: routing cannot be turned on for a gateway behind an operator's back, and
two policies cannot disagree about whether it is on. An *unknown* backend name is
not an error either, on the same principle the compiler already applies to a
router that declines: routing is an optimization, so a policy naming a backend
this build does not have serves its default target and warns once.

* ``weighted`` → :class:`gateway.services.routing.weighted.WeightedRouterBackend`,
  a load balancer: one candidate per request, drawn in proportion to the weights
  the policy declares.
* ``priority`` → :class:`PriorityRouterBackend`, which keeps the candidates in
  declared order. The walker skips a candidate a ``per: model`` rate limit has
  no room for, so traffic stays on the first candidate until its limit is
  reached and then spills to the next.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from gateway.models.routing import WEIGHTED_BACKEND

__all__ = [
    "PRIORITY_BACKEND",
    "WEIGHTED_BACKEND",
    "PriorityRouterBackend",
    "RouterBackend",
    "RoutingContext",
    "RoutingDecision",
    "backend_is_priority",
    "backend_is_weighted",
    "clear_router_backend_cache",
    "get_router_backend",
    "known_backends",
    "owes_missing_backend_warning",
]

PRIORITY_BACKEND = "priority"


@dataclass
class RoutingContext:
    """Inputs a backend may use to rank candidates for a single request.

    Plain values only, so a backend never has to know which endpoint it is serving.
    """

    user_id: str
    default_model: str
    """The policy's default target: what serves if the backend declines, and the
    safe choice a low-confidence decision leads with."""
    candidate_pool: list[str]
    """The policy's candidates, already filtered to what this caller may use."""
    workspace_id: uuid.UUID | None = None
    """The workspace this request bills to, when there is one. A backend that
    reads stored state partitions on it as well as on ``user_id``, so a user
    holding keys in two workspaces does not have one steer the other. ``None``
    only where there is no request (a synchronous surface), which is also where
    no backend is asked to rank."""
    weights: dict[str, float] = field(default_factory=dict)
    """Per-candidate traffic weights the policy declared, for a backend that takes
    its parameters from the policy document rather than from the environment. Empty
    for every other backend. Kept as declared: normalizing needs ``candidate_pool``,
    which is already filtered to this caller."""


@dataclass
class RoutingDecision:
    """A backend's ranking for one request, best first.

    An empty ``ordered_models`` is a decline: a normal outcome (no usable
    candidate, no weights) that leaves the policy's default target to serve.
    ``rationale`` is operator-facing text saying why.
    """

    ordered_models: list[str]
    confidence: float
    rationale: str
    log_decision: bool = True
    """Whether this decision earns its INFO line. A load balancer's draw is one
    line per request forever, and the usage row already records what served. The
    backend decides, because only it knows how often it is asked."""

    @classmethod
    def decline(cls, rationale: str) -> RoutingDecision:
        return cls(ordered_models=[], confidence=0.0, rationale=rationale)


@runtime_checkable
class RouterBackend(Protocol):
    """Contract a router backend implements."""

    async def rank(self, ctx: RoutingContext) -> RoutingDecision: ...


class PriorityRouterBackend:
    """Backend that keeps the pool in declared order: the first candidate with room serves.

    Room is not decided here. The walker admits each candidate under the
    ``per: model`` rate limits as it reaches it, so a full candidate is skipped
    the same way in every policy, alias and fallback chain that names it.
    """

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        pool = list(ctx.candidate_pool)
        if not pool:
            return RoutingDecision.decline("no candidate in the pool is usable by this caller")
        return RoutingDecision(
            ordered_models=pool,
            confidence=1.0,
            rationale=f"priority order ({', '.join(pool)})",
            # Every request decides the same thing; the usage row records which candidate served.
            log_decision=False,
        )


# (policy, backend name) pairs already warned about. Unbounded in principle,
# bounded in practice: the pairs come from policy documents, so the set is the
# size of the config rather than of the traffic.
_warned_missing: set[tuple[uuid.UUID | None, str, str]] = set()


def clear_router_backend_cache() -> None:
    """Forget which missing backends were warned about (test isolation; called from reset_config)."""
    _warned_missing.clear()


def known_backends() -> tuple[str, ...]:
    """Backend names this build resolves, for an error message that lists them."""
    return (PRIORITY_BACKEND, WEIGHTED_BACKEND)


def backend_is_weighted(name: str | None) -> bool:
    """Whether this name selects the weighted load balancer.

    A named check because the weighted backend is the one whose decision needs no
    request state, so several synchronous surfaces (``explain``, the CLI) special
    case it, and each of them would otherwise repeat the same normalization.
    """
    return name is not None and name.strip().lower() == WEIGHTED_BACKEND


def backend_is_priority(name: str | None) -> bool:
    """Whether this name selects the priority router, whose order is the declared one."""
    return name is not None and name.strip().lower() == PRIORITY_BACKEND


def owes_missing_backend_warning(workspace_id: uuid.UUID | None, policy_name: str, name: str) -> bool:
    """Whether this policy and backend still owe their one warning.

    A policy naming a backend this build does not have is a misconfiguration worth
    saying once. Once, because the condition is static config and the policy
    compiles on every request through it: an unconditional warning would be one log
    line per request forever, which buries the real ones. Keyed by workspace too,
    since two workspaces may each store a policy of one name.
    """
    key = (workspace_id, policy_name, name)
    if key in _warned_missing:
        return False
    _warned_missing.add(key)
    return True


def get_router_backend(name: str) -> RouterBackend | None:
    """Resolve the backend a policy named, or ``None`` if this build has no such backend.

    ``None`` is not an error: the caller compiles the policy without a router
    ordering, which serves the default target. Warned once per (policy, name) by
    the caller rather than here, because the policy name is what makes the warning
    actionable and this function does not know it.
    """
    backend = name.strip().lower()
    if backend == PRIORITY_BACKEND:
        return PriorityRouterBackend()
    if backend == WEIGHTED_BACKEND:
        # Instantiated per call rather than cached, because the backend is stateless:
        # everything it reads about the policy arrives on the RoutingContext, and the
        # draw comes from a stream shared across instances. Imported inside the
        # function only to keep this module free of intra-package imports (``decide``
        # imports the split helpers at module level, so nothing is deferred by it).
        from gateway.services.routing.weighted import WeightedRouterBackend

        return WeightedRouterBackend()
    return None
