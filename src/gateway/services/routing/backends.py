"""Router backends: the pluggable half of a routing policy's ``select``.

A policy entry ``{router: knn, candidates: [...]}`` names a backend here. The
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

* ``noop`` → :class:`NoOpRouterBackend`, which declines every request. Useful to
  hold a policy's shape while its pool is still being taught.
* ``knn`` → :class:`gateway.services.routing.knn.KnnRoutingMemory`, imported
  lazily so a gateway with no learned policy never loads the embedding path.
* ``weighted`` → :class:`gateway.services.routing.weighted.WeightedRouterBackend`,
  a load balancer: one candidate per request, drawn in proportion to the weights
  the policy declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from gateway.models.routing import WEIGHTED_BACKEND

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig

__all__ = [
    "KNN_BACKEND",
    "NOOP_BACKEND",
    "WEIGHTED_BACKEND",
    "NoOpRouterBackend",
    "RouterBackend",
    "RoutingContext",
    "RoutingDecision",
    "backend_is_weighted",
    "backend_pool_is_teachable",
    "backend_requires_pricing",
    "clear_router_backend_cache",
    "get_router_backend",
    "known_backends",
    "owes_missing_backend_warning",
]

KNN_BACKEND = "knn"
NOOP_BACKEND = "noop"


@dataclass
class RoutingContext:
    """Inputs a backend may use to rank candidates for a single request.

    The prompt arrives as already-flattened text rather than as wire messages.
    Flattening is format-specific (chat, Anthropic messages, and responses all
    shape content differently) and the API layer already does it for guardrails,
    so a backend never has to know which endpoint it is serving.
    """

    user_id: str
    default_model: str
    """The policy's default target: what serves if the backend declines, and the
    safe choice a low-confidence decision leads with."""
    candidate_pool: list[str]
    """The policy's candidates, already filtered to what this caller may use."""
    task_signal: str = ""
    """This turn's prompt text. What ``step`` granularity routes on."""
    trace_signal: str = ""
    """The conversation's opening prompt text. What ``trace_sticky`` routes on, so
    every turn of one conversation embeds the same thing."""
    trace_anchor: str = ""
    """Stable text identifying the conversation when the client sends no id."""
    task_id: str | None = None
    has_tools: bool = False
    is_trace_continuation: bool = False
    trace_key: str | None = None
    weights: dict[str, float] = field(default_factory=dict)
    """Per-candidate traffic weights the policy declared, for a backend that takes
    its parameters from the policy document rather than from the environment. Empty
    for every other backend. Kept as declared: normalizing needs ``candidate_pool``,
    which is already filtered to this caller."""


@dataclass
class RoutingDecision:
    """A backend's ranking for one request, best first.

    An empty ``ordered_models`` is a decline: a normal outcome (cold pool, sparse
    neighborhood, no embeddable signal) that leaves the policy's default target to
    serve. ``rationale`` is operator-facing text saying which of those it was.
    """

    ordered_models: list[str]
    confidence: float
    rationale: str
    log_decision: bool = True
    """Whether this decision earns its INFO line. A learned router's pick is
    unreconstructable after the fact and worth one; a load balancer's draw is one
    line per request forever, and the usage row already records what served. The
    backend decides, because only it knows how often it is asked."""

    @classmethod
    def decline(cls, rationale: str) -> RoutingDecision:
        return cls(ordered_models=[], confidence=0.0, rationale=rationale)


@runtime_checkable
class RouterBackend(Protocol):
    """Contract a router backend implements."""

    async def rank(self, ctx: RoutingContext) -> RoutingDecision: ...


class NoOpRouterBackend:
    """Backend that always declines, so the policy's default target serves."""

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        return RoutingDecision.decline("noop backend: always defers to the policy default")


# The kNN backend carries per-process mutable state (the trace-sticky decision
# cache), so a fresh instance per request would reset that cache and break
# stickiness across the turns of one conversation. Cached per backend-config
# signature; cleared by clear_router_backend_cache().
_KNN_CACHE: dict[tuple[Any, ...], RouterBackend] = {}
# (policy, backend name) pairs already warned about. Unbounded in principle,
# bounded in practice: the pairs come from policy documents, so the set is the
# size of the config rather than of the traffic.
_warned_missing: set[tuple[str, str]] = set()


def _knn_signature(config: GatewayConfig) -> tuple[Any, ...]:
    return (
        config.router_alpha,
        config.router_k,
        config.router_embedding_model,
        config.router_confidence_floor,
        config.router_seed_count,
        config.router_granularity,
        config.router_max_records_per_user,
    )


def clear_router_backend_cache() -> None:
    """Drop cached backend instances (test isolation; called from reset_config)."""
    _KNN_CACHE.clear()
    _warned_missing.clear()


def known_backends() -> tuple[str, ...]:
    """Backend names this build resolves, for an error message that lists them."""
    return (KNN_BACKEND, NOOP_BACKEND, WEIGHTED_BACKEND)


def backend_is_weighted(name: str | None) -> bool:
    """Whether this name selects the weighted load balancer.

    A named check because the weighted backend is the one whose decision needs no
    request state, so several synchronous surfaces (``explain``, the CLI) special
    case it, and each of them would otherwise repeat the same normalization.
    """
    return name is not None and name.strip().lower() == WEIGHTED_BACKEND


def backend_pool_is_teachable(name: str | None) -> bool:
    """Whether this policy's candidates are a pool routing memory is taught about.

    True for ``knn``, which reads the examples, and deliberately also for ``noop``
    and for a name this build does not know. ``noop`` exists to hold a policy's
    shape *while its pool is being taught*, so its candidates are the very ones an
    operator is seeding; an unknown name is most likely a backend from a newer
    build, and treating its pool as teachable keeps the typo guard on rather than
    silently widening what ``POST /v1/routing/preferences/rank`` accepts.

    False only for ``weighted``, whose split is written in the policy document. It
    reads no examples and has no warmth to report, so counting it would report a
    pool it never consults and would let its candidates decide which score keys a
    user may teach.
    """
    return name is not None and not backend_is_weighted(name)


def backend_requires_pricing(name: str | None) -> bool:
    """Whether a policy naming this backend must have every candidate priced.

    Only the kNN router does: it scores quality against cost, so one unpriced
    candidate makes it decline every request. The weighted router balances on
    operator-declared capacity and never reads a price, so demanding pricing there
    would refuse a working policy.
    """
    return name is not None and name.strip().lower() == KNN_BACKEND


def owes_missing_backend_warning(policy_name: str, name: str) -> bool:
    """Whether this ``(policy, backend)`` pair still owes its one warning.

    A policy naming a backend this build does not have is a misconfiguration worth
    saying once. Once, because the condition is static config and the policy
    compiles on every request through it: an unconditional warning would be one log
    line per request forever, which buries the real ones.
    """
    key = (policy_name, name)
    if key in _warned_missing:
        return False
    _warned_missing.add(key)
    return True


def get_router_backend(config: GatewayConfig, name: str) -> RouterBackend | None:
    """Resolve the backend a policy named, or ``None`` if this build has no such backend.

    ``None`` is not an error: the caller compiles the policy without a router
    ordering, which serves the default target. Warned once per (policy, name) by
    the caller rather than here, because the policy name is what makes the warning
    actionable and this function does not know it.
    """
    backend = name.strip().lower()
    if backend == NOOP_BACKEND:
        return NoOpRouterBackend()
    if backend == WEIGHTED_BACKEND:
        # Instantiated per call rather than cached, because the backend is stateless:
        # everything it reads about the policy arrives on the RoutingContext, and the
        # draw comes from a stream shared across instances. Imported inside the
        # function only to keep this module free of intra-package imports (``decide``
        # imports the split helpers at module level, so nothing is deferred by it).
        from gateway.services.routing.weighted import WeightedRouterBackend

        return WeightedRouterBackend()
    if backend == KNN_BACKEND:
        # Imported lazily: the kNN backend pulls in any_llm embeddings and the
        # example store, neither of which a gateway without a learned policy needs.
        from gateway.services.routing.knn import KnnRoutingMemory

        signature = _knn_signature(config)
        cached = _KNN_CACHE.get(signature)
        if cached is None:
            cached = KnnRoutingMemory(config)
            _KNN_CACHE[signature] = cached
        return cached
    return None
