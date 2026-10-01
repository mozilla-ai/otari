"""Routing policies: turning a policy name into an ordered plan of attempts.

The decision half of routing. Something here decides which candidates to try and
in what order; the API layer's attempt walker executes the result and makes no
choices of its own.

A policy's ``select`` may hand the ordering to a *router backend*
(``backends.py``), which is where the learned kNN router (``knn.py``) and the
weighted load balancer (``weighted.py``) plug in. The request path reaches a
backend through ``RoutingPort``, so an overlay can add one; the compiler and
the conditions stay plain core code. The split is deliberate: the
compiler stays pure and synchronous, and a backend's asynchronous work
(embedding, reading stored examples) happens in the request pipeline, which
passes the resulting order in as a value.
"""

from gateway.services.routing._feedback import RoutingFeedbackService
from gateway.services.routing.backends import (
    KNN_BACKEND,
    NOOP_BACKEND,
    WEIGHTED_BACKEND,
    RouterBackend,
    RoutingContext,
    RoutingDecision,
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
from gateway.types.budget_state import BudgetState

__all__ = [
    "KNN_BACKEND",
    "NOOP_BACKEND",
    "WEIGHTED_BACKEND",
    "BudgetState",
    "CompiledPlan",
    "DroppedCandidate",
    "NoEligibleCandidatesError",
    "RouterBackend",
    "RouterOrdering",
    "RoutingContext",
    "RoutingDecision",
    "RoutingFeedbackService",
    "backend_pool_is_teachable",
    "backend_requires_pricing",
    "clear_router_backend_cache",
    "compile_policy",
    "get_router_backend",
    "known_backends",
    "needs_budget_state",
    "selection_consults_router",
]
