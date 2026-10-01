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
* ``smart_router`` → :class:`gateway.services.routing.smart_router.SmartRouterBackend`,
  which asks an external smart-router service at ``smart_router_url`` and learns
  from outcomes and ratings. Unavailable while that URL is unset.

The request path reaches these through ``RoutingPort`` (``gateway.ports``), whose
core adapter is the switch in :func:`get_router_backend`, so an overlay can add or
replace a backend. The contract types live on the port and are re-exported here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gateway.models.routing import SMART_ROUTER_BACKEND, WEIGHTED_BACKEND
from gateway.ports.routing_port import RouterBackend, RouterTraits, RoutingContext, RoutingDecision

if TYPE_CHECKING:
    from gateway.core.config import GatewayConfig
    from gateway.services.routing.smart_router import SmartRouterBackend

__all__ = [
    "KNN_BACKEND",
    "NOOP_BACKEND",
    "SMART_ROUTER_BACKEND",
    "WEIGHTED_BACKEND",
    "NoOpRouterBackend",
    "RouterBackend",
    "RouterTraits",
    "RoutingContext",
    "RoutingDecision",
    "backend_is_weighted",
    "backend_pool_is_teachable",
    "backend_requires_pricing",
    "clear_router_backend_cache",
    "close_router_backends",
    "get_router_backend",
    "known_backends",
    "owes_missing_backend_warning",
    "router_traits",
]

KNN_BACKEND = "knn"
NOOP_BACKEND = "noop"


class NoOpRouterBackend:
    """Backend that always declines, so the policy's default target serves."""

    async def rank(self, ctx: RoutingContext) -> RoutingDecision:
        return RoutingDecision.decline("noop backend: always defers to the policy default")


# The kNN backend carries per-process mutable state (the trace-sticky decision
# cache), so a fresh instance per request would reset that cache and break
# stickiness across the turns of one conversation. Cached per backend-config
# signature; cleared by clear_router_backend_cache().
_KNN_CACHE: dict[tuple[Any, ...], RouterBackend] = {}
# The smart router holds one pooled HTTP client and the outcome reports in
# flight, so it is cached for the same reason, per (url, timeout).
_SMART_ROUTER_CACHE: dict[tuple[str, float], SmartRouterBackend] = {}
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
    _SMART_ROUTER_CACHE.clear()
    _warned_missing.clear()


async def close_router_backends() -> None:
    """Close the network clients cached backends hold, at shutdown. A no-op when none was built."""
    backends = list(_SMART_ROUTER_CACHE.values())
    _SMART_ROUTER_CACHE.clear()
    for backend in backends:
        await backend.aclose()


def known_backends() -> tuple[str, ...]:
    """Backend names this build resolves, for an error message that lists them."""
    return (KNN_BACKEND, NOOP_BACKEND, SMART_ROUTER_BACKEND, WEIGHTED_BACKEND)


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

    False for ``weighted``, whose split is written in the policy document, and for
    ``smart_router``, which learns in its own service. Neither reads the examples
    or has warmth to report, so counting them would report a pool nothing consults
    and would let their candidates decide which score keys a user may teach.
    """
    return name is not None and name.strip().lower() not in {WEIGHTED_BACKEND, SMART_ROUTER_BACKEND}


def backend_requires_pricing(name: str | None) -> bool:
    """Whether a policy naming this backend must have every candidate priced.

    Only the kNN router does: it scores quality against cost, so one unpriced
    candidate makes it decline every request. The weighted router balances on
    operator-declared capacity and never reads a price, so demanding pricing there
    would refuse a working policy.
    """
    return name is not None and name.strip().lower() == KNN_BACKEND


def router_traits(name: str | None) -> RouterTraits:
    """What a policy naming ``name`` asks of the rest of the gateway, for this build's backends."""
    return RouterTraits(teachable=backend_pool_is_teachable(name), requires_pricing=backend_requires_pricing(name))


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
    ordering, which serves the default target. ``smart_router`` resolves to
    ``None`` while ``smart_router_url`` is unset. Warned once per (policy, name) by
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
    if backend == SMART_ROUTER_BACKEND:
        if config.smart_router_url is None:
            return None
        from gateway.services.routing.smart_router import SmartRouterBackend

        key = (config.smart_router_url, config.smart_router_timeout_seconds)
        smart = _SMART_ROUTER_CACHE.get(key)
        if smart is None:
            smart = SmartRouterBackend(config.smart_router_url, config.smart_router_timeout_seconds)
            _SMART_ROUTER_CACHE[key] = smart
        return smart
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
