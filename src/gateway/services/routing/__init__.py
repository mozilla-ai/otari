"""Routing policies: turning a policy name into an ordered plan of attempts.

The decision half of routing. Something here decides which candidates to try and
in what order; the API layer's attempt walker executes the result and makes no
choices of its own.

Plain core code, with no port. ``ARCHITECTURE.md`` names a ``RoutingPort`` and
marks the routing capability line provisional, and whether that port should exist
at all is an open maintainer decision, so this does not presume one.

A policy's ``select`` may hand the ordering to a *router backend*
(``backends.py``), which is where the learned kNN router (``knn.py``), the
weighted load balancer (``weighted.py``) and the priority order plug in. The split is deliberate: the
compiler stays pure and synchronous, and a backend's asynchronous work
(embedding, reading stored examples) happens in the request pipeline, which
passes the resulting order in as a value.

``recommend.py`` answers a different question: which model a coding agent's
subagent should start on. That is put to a decision model, not compiled from
a policy, and the harness acts on the answer itself.
"""

from gateway.services.routing.backends import (
    KNN_BACKEND,
    NOOP_BACKEND,
    PRIORITY_BACKEND,
    WEIGHTED_BACKEND,
    RouterBackend,
    RoutingContext,
    RoutingDecision,
    backend_is_priority,
    backend_is_weighted,
    backend_pool_is_teachable,
    backend_requires_pricing,
    clear_router_backend_cache,
    get_router_backend,
    known_backends,
)
from gateway.services.routing.compiler import (
    CompiledPlan,
    DroppedCandidate,
    NoEligibleCandidatesError,
    RouterOrdering,
    compile_policy,
    needs_budget_state,
    selection_consults_router,
)
from gateway.services.routing.decide import RoutingSignal, decide_ordering, explain_router_ordering
from gateway.services.routing.knn import KnnRoutingMemory, unpriced_router_candidates
from gateway.services.routing.recommend import (
    UnreadableRecommendationError,
    build_decision_request,
    recommendation_from_decision,
)
from gateway.types.budget_state import BudgetState

__all__ = [
    "KNN_BACKEND",
    "NOOP_BACKEND",
    "PRIORITY_BACKEND",
    "WEIGHTED_BACKEND",
    "BudgetState",
    "CompiledPlan",
    "DroppedCandidate",
    "KnnRoutingMemory",
    "NoEligibleCandidatesError",
    "RouterBackend",
    "RouterOrdering",
    "RoutingContext",
    "RoutingDecision",
    "RoutingSignal",
    "UnreadableRecommendationError",
    "backend_is_priority",
    "backend_is_weighted",
    "backend_pool_is_teachable",
    "backend_requires_pricing",
    "build_decision_request",
    "clear_router_backend_cache",
    "compile_policy",
    "decide_ordering",
    "explain_router_ordering",
    "get_router_backend",
    "known_backends",
    "needs_budget_state",
    "recommendation_from_decision",
    "selection_consults_router",
    "unpriced_router_candidates",
]
