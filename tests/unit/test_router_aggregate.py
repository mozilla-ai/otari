"""Every route Otari serves is mounted through one aggregate router.

The mount prefix therefore has a single owner, and setting it moves the whole
API at once. A router mounted anywhere else would keep its old path and split
the served surface in two.

Operations are counted rather than collected into a set, so a router mounted
twice is a failure rather than a silent no-op.
"""

from collections import Counter

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute

from gateway.api.main import _register_contributed_routers, _register_core_routers, register_routers
from gateway.container import build_container
from gateway.core.config import GatewayConfig

Operations = Counter[tuple[str, str]]


def _operations(app: FastAPI) -> Operations:
    return Counter(
        (route.path, method) for route in app.routes if isinstance(route, APIRoute) for method in route.methods
    )


def _mounted_on(aggregate: APIRouter) -> Operations:
    """Mount the gateway onto ``aggregate`` and report the operations that reach an app."""
    config = GatewayConfig()
    _register_core_routers(aggregate, config)
    _register_contributed_routers(aggregate, build_container(config.bootstrap))

    app = FastAPI()
    app.include_router(aggregate)
    return _operations(app)


def test_a_prefix_on_the_aggregate_moves_every_route() -> None:
    plain = _mounted_on(APIRouter())
    prefixed = _mounted_on(APIRouter(prefix="/probe"))

    assert plain, "nothing was mounted, so the comparison below would hold vacuously"
    assert prefixed == Counter({(f"/probe{path}", method): count for (path, method), count in plain.items()})


def test_register_routers_hands_the_app_the_aggregate_and_nothing_else() -> None:
    app = FastAPI()
    app.state.container = build_container(None)

    register_routers(app, GatewayConfig())

    assert _operations(app) == _mounted_on(APIRouter())
