"""Every route Otari serves is mounted through one aggregate router, or the OTLP sibling.

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
from gateway.api.routes import hosted_mode, otlp
from gateway.container import RouterContribution, build_container
from gateway.core.config import API_ROOT, OTLP_ROOT, GatewayConfig

Operations = Counter[tuple[str, str]]


def _operations(app: FastAPI) -> Operations:
    return Counter(
        (route.path, method) for route in app.routes if isinstance(route, APIRoute) for method in route.methods
    )


def _standalone() -> GatewayConfig:
    """A config that does not read the machine it runs on.

    A bare ``GatewayConfig()`` takes four routing decisions from the ambient
    environment: the mode (three routers or sixty), whether a search provider
    and a gateway token mount the web-search backend, and whether a bootstrap
    contributes routers. Both sides of the comparisons below shift together
    under each of those, so the test stays green while covering a different
    surface than it names, which is worse than failing.
    """
    return GatewayConfig(
        mode="standalone",
        bootstrap=None,
        web_search_provider=None,
        web_search_backend_token=None,
    )


def _mounted_on(aggregate: APIRouter) -> Operations:
    """Mount the gateway onto ``aggregate`` and report the operations that reach an app."""
    config = _standalone()
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


def test_register_routers_hands_the_app_the_aggregate_and_the_otlp_sibling() -> None:
    app = FastAPI()
    app.state.container = build_container(None)

    register_routers(app, _standalone())

    otlp_only = FastAPI()
    otlp_only.include_router(otlp.router, prefix=OTLP_ROOT)
    assert _operations(app) == _mounted_on(APIRouter(prefix=API_ROOT)) + _operations(otlp_only)


def _mount_order(config: GatewayConfig) -> list[str]:
    """Every operation the app serves, in the order Starlette will try to match it."""
    app = FastAPI()
    app.state.container = build_container(config.bootstrap)
    register_routers(app, config)
    return [route.path for route in app.routes if isinstance(route, APIRoute)]


def test_a_fixed_route_is_matched_before_the_catch_all_that_would_swallow_it() -> None:
    """Starlette serves the first route that matches, so mount order is behavior.

    Three routers here end in a greedy path parameter, and the fixed routes
    that live under the same prefix are mounted ahead of them on purpose. Swap
    the pair and ``/models/discoverable`` starts resolving as a model id: a 200
    with the wrong body, no error anywhere, and a counted-operations check
    cannot see it because the set of routes did not change.
    """
    order = _mount_order(_standalone())

    def first(path: str) -> int:
        assert path in order, f"{path} is not mounted, so the check below would hold vacuously"
        return order.index(path)

    assert first(f"{API_ROOT}/models/discoverable") < first(f"{API_ROOT}/models/{{model_id:path}}")
    assert first(f"{API_ROOT}/models/metadata") < first(f"{API_ROOT}/models/{{model_id:path}}")
    assert first(f"{API_ROOT}/pricing/refresh") < first(f"{API_ROOT}/pricing/{{model_key:path}}")


def test_a_contributed_route_is_matched_before_a_mode_stub() -> None:
    """The mode stubs are catch-alls over whole prefixes, so they are mounted last.

    An overlay that contributes a route under a stubbed prefix has made a
    choice, and a fallback must not overrule it. Move the stub above the
    contributed routers and every such route answers "manage this via the
    platform" instead, with the suite still green.
    """
    contributed = APIRouter()

    @contributed.get("/chat/overlay-probe")
    async def overlay_probe() -> dict[str, str]:
        return {"ok": "yes"}

    container = build_container(None)
    container.contribute_router(RouterContribution(capability="probe", router=contributed))

    app = FastAPI()
    app.state.container = container
    register_routers(app, GatewayConfig(mode="hosted", bootstrap=None))
    routes = [route for route in app.routes if isinstance(route, APIRoute)]

    # The probe sits under /chat, which the hosted stub claims with a catch-all.
    probe = next(i for i, route in enumerate(routes) if route.path.endswith("/chat/overlay-probe"))

    stubs = [i for i, route in enumerate(routes) if route.endpoint.__module__ == hosted_mode.__name__]
    served = [
        i
        for i, route in enumerate(routes)
        if route.endpoint.__module__ != hosted_mode.__name__ and route.path.startswith(API_ROOT)
    ]
    assert stubs, "no mode stub was mounted, so this check would hold vacuously"
    assert served, "nothing else was mounted, so this check would hold vacuously"
    assert min(stubs) > max(served), (
        "a mode stub is mounted ahead of a route the deployment actually serves, "
        "so the stub will answer for it"
    )
    assert probe < min(stubs), "the stub is ahead of a contributed route, so it will answer for it"
