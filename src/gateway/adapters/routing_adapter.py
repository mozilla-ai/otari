"""The router backends this build ships, behind ``RoutingPort``.

``noop``, ``weighted`` and ``knn``, resolved by the switch in
``gateway.services.routing.backends``. An overlay binds an adapter of its own to
add a backend or replace one, and usually delegates the names it does not
handle to this one.
"""

from gateway.core.config import GatewayConfig
from gateway.ports.routing_port import RouterBackend, RouterTraits
from gateway.services.routing.backends import get_router_backend, known_backends, router_traits


class CoreRoutingAdapter:
    """``RoutingPort`` over the core backend switch, closed over this app's config.

    Stateless itself: a backend that keeps state across requests (the kNN
    router's trace-sticky cache) is cached by the switch, so every adapter built
    for one config hands out the same instance.
    """

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config

    def backend(self, name: str) -> RouterBackend | None:
        return get_router_backend(self._config, name)

    def known_backends(self) -> tuple[str, ...]:
        return known_backends()

    def traits(self, name: str | None) -> RouterTraits:
        return router_traits(name)
