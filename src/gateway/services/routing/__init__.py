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

``RoutingService`` reads what routing gives an organization's callers: the
automatic fallbacks a catalog ID gets with no policy, and the traffic that used them.

``recommend.py`` answers a different question: which model a coding agent's
subagent should start on. That is put to a decision model, not compiled from
a policy, and the harness acts on the answer itself. The decision model it is
put to is the stored subagent router switched on (``_agent_recommenders.py``
holds the rows in memory, ``_agent_recommender_service.py`` administers them).
"""

from gateway.exceptions.routing_exceptions import UnreadableRecommendationError
from gateway.services.routing._agent_recommender_price import HOLD_INPUT_TOKENS, HOLD_OUTPUT_TOKENS, RecommenderPrice
from gateway.services.routing._agent_recommender_rules import check_recommender
from gateway.services.routing._agent_recommender_service import AgentRecommenderService
from gateway.services.routing._agent_recommenders import (
    StoredRecommender,
    active_agent_recommender,
    apply_agent_recommenders,
    load_agent_recommenders_at_startup,
    refresh_agent_recommenders,
    reset_agent_recommenders,
    run_agent_recommender_refresher,
)
from gateway.services.routing._service import FALLBACK_WINDOW, AutomaticFallback, FallbackOffering, RoutingService
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
    compile_catalog_plan,
    compile_policy,
    needs_budget_state,
    selection_consults_router,
)
from gateway.services.routing.decide import RoutingSignal, decide_ordering, explain_router_ordering
from gateway.services.routing.knn import KnnRoutingMemory, unpriced_router_candidates
from gateway.services.routing.recommend import (
    ROUTER_LABEL,
    ROUTER_LABEL_MODEL,
    ROUTER_LABEL_PROVIDER,
    build_decision_request,
    candidates_for,
    recommendation_from_decision,
)
from gateway.types.budget_state import BudgetState

__all__ = [
    "FALLBACK_WINDOW",
    "HOLD_INPUT_TOKENS",
    "HOLD_OUTPUT_TOKENS",
    "KNN_BACKEND",
    "NOOP_BACKEND",
    "PRIORITY_BACKEND",
    "ROUTER_LABEL",
    "ROUTER_LABEL_MODEL",
    "ROUTER_LABEL_PROVIDER",
    "WEIGHTED_BACKEND",
    "AgentRecommenderService",
    "BudgetState",
    "AutomaticFallback",
    "CompiledPlan",
    "DroppedCandidate",
    "FallbackOffering",
    "KnnRoutingMemory",
    "NoEligibleCandidatesError",
    "RecommenderPrice",
    "RouterBackend",
    "RouterOrdering",
    "RoutingContext",
    "RoutingDecision",
    "RoutingService",
    "RoutingSignal",
    "StoredRecommender",
    "UnreadableRecommendationError",
    "active_agent_recommender",
    "apply_agent_recommenders",
    "backend_is_priority",
    "backend_is_weighted",
    "backend_pool_is_teachable",
    "backend_requires_pricing",
    "build_decision_request",
    "candidates_for",
    "check_recommender",
    "clear_router_backend_cache",
    "compile_catalog_plan",
    "compile_policy",
    "decide_ordering",
    "explain_router_ordering",
    "get_router_backend",
    "known_backends",
    "load_agent_recommenders_at_startup",
    "needs_budget_state",
    "recommendation_from_decision",
    "refresh_agent_recommenders",
    "reset_agent_recommenders",
    "run_agent_recommender_refresher",
    "selection_consults_router",
    "unpriced_router_candidates",
]
