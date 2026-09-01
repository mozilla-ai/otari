"""The deployment-wide routers declare the operator gate on themselves.

`tests/integration/test_deployment_operator_gate.py` asserts what the gate does
over HTTP, and why it exists at all (otari-ai#1880). This file asserts where it
is declared, which is the other half: the gate used to be a decorator on each
route, every one of them correct, and the hazard was the next route someone
added. One forgotten `dependencies=[...]` and a deployment-wide handler is
reachable with no credential at all, which no roster of probes can notice
because the probes name paths and the missing path is the one nobody wrote
(otari-ai#1937).

Read from the routers rather than from a running app: this is a claim about the
declarations, it needs no database, and a route that never reached the app is
still a route somebody will mount.
"""

import importlib
import pkgutil
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi import APIRouter
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

import gateway.api.routes
from gateway.api.deps import (
    require_deployment_operator,
    verify_api_key_or_master_key,
    verify_catalog_reader,
    verify_master_key,
)
from gateway.api.routes import (
    agent_telemetry,
    aliases,
    budgets,
    keys,
    mail,
    maintenance_mode,
    models,
    pricing,
    providers,
    routing,
    routing_memory,
    scoped_budgets,
    search_tools,
    settings,
    tool_settings,
    tools,
    usage,
    users,
)

# Every router whose whole surface is deployment-wide: a read or write over
# every tenant's rows, or over the deployment's own configuration.
_DEPLOYMENT_WIDE_ROUTERS: list[tuple[str, APIRouter]] = [
    ("agent_telemetry", agent_telemetry.router),
    ("aliases", aliases.router),
    ("budgets", budgets.router),
    ("keys", keys.router),
    ("mail", mail.router),
    ("maintenance_mode", maintenance_mode.router),
    ("models", models.operator_router),
    ("pricing", pricing.operator_router),
    ("providers", providers.router),
    ("routing", routing.router),
    ("routing_memory", routing_memory.router),
    ("scoped_budgets", scoped_budgets.router),
    ("search_tools", search_tools.router),
    ("settings", settings.router),
    ("tool_settings", tool_settings.router),
    ("usage", usage.operator_router),
    ("users", users.router),
]

# The routers a caller reaches without operator standing, and the dependency
# each admits them with instead. Listed so the gate can be asserted *off* them:
# putting it on the whole of `models.py`, `pricing.py` or `usage.py` is the
# plausible wrong fix, and it would take the dashboard's Models and Pricing
# pages and a data-plane gateway's usage report with it.
_NON_OPERATOR_ROUTERS: list[tuple[str, APIRouter, Callable[..., Any]]] = [
    ("models.catalog", models.catalog_router, verify_catalog_reader),
    ("pricing.catalog", pricing.catalog_router, verify_catalog_reader),
    ("tools", tools.router, verify_catalog_reader),
    ("usage.ingest", usage.ingest_router, verify_api_key_or_master_key),
]


# Why each remaining router declares nothing at the router level. Everything not
# listed has to carry a router-level dependency, which is what extends "a new
# route starts gated" to "a new router does too": the partial-gating case below
# cannot see a router with no gate anywhere on it, because it has no gated route
# to compare against.
_DATA_PLANE = "data plane: an API key or the master key, checked per route or inside the handler"
_PUBLIC_AUTH = "public auth: reached before the caller holds any credential"

_UNGATED_ROUTERS: dict[str, str] = {
    "audio.router": _DATA_PLANE,
    "batches.router": _DATA_PLANE,
    "chat.router": _DATA_PLANE,
    "embeddings.router": _DATA_PLANE,
    "files.router": _DATA_PLANE,
    "images.router": _DATA_PLANE,
    "messages.router": _DATA_PLANE,
    "moderations.router": _DATA_PLANE,
    "otlp.router": _DATA_PLANE,
    "rerank.router": _DATA_PLANE,
    "responses.router": _DATA_PLANE,
    "search.router": _DATA_PLANE,
    "auth_oauth.router": _PUBLIC_AUTH,
    "auth_password_reset.router": _PUBLIC_AUTH,
    "auth_session.router": _PUBLIC_AUTH,
    "auth_signup.router": _PUBLIC_AUTH,
    "auth_webauthn.router": _PUBLIC_AUTH,
    "invitations.router": "the invitation token is the credential, and the invitee has no account yet",
    "bootstrap.router": "unauthenticated on purpose: how a browser learns which mode it reached",
    "health.router": "unauthenticated liveness and readiness",
    "web_search_backend.router": "its own X-Gateway-Token, checked in the handler",
    "hosted_mode.router": "mode stub: a 404 naming why the prefix is absent on this deployment",
    "hybrid_mode.router": "mode stub, as above",
}


# What counts as a gate when a router declares it. Named rather than taken as
# "declares any dependency at all": a router-level ``Depends(get_db)`` is a
# convenience, and reading it as authorization would let a router satisfy the
# classification below while every route on it stayed open.
# ``verify_master_key`` is here because the tenant-scoped routers declare it and
# then authorize the caller against the organization or workspace themselves.
_ROUTER_LEVEL_GATES: frozenset[Callable[..., Any]] = frozenset(
    {
        require_deployment_operator,
        verify_api_key_or_master_key,
        verify_catalog_reader,
        verify_master_key,
    }
)


def _exposed_routers() -> dict[str, APIRouter]:
    """Every router the routes package exposes, keyed ``module.attribute``."""
    found: dict[str, APIRouter] = {}
    for module_info in pkgutil.iter_modules(gateway.api.routes.__path__):
        module = importlib.import_module(f"gateway.api.routes.{module_info.name}")
        for attribute, value in vars(module).items():
            if isinstance(value, APIRouter):
                found[f"{module_info.name}.{attribute}"] = value
    return found


def _resolves(dependant: Dependant, gate: Callable[..., Any]) -> bool:
    """Whether a route reaches ``gate``, by way of its router or its own signature."""
    return any(sub.call is gate or _resolves(sub, gate) for sub in dependant.dependencies)


def _route_id(name: str, route: APIRoute) -> str:
    return f"{name} {'/'.join(sorted(route.methods))} {route.path or '/'}"


def _routes(routers: list[tuple[str, APIRouter]]) -> Iterator[Any]:
    for name, router in routers:
        for route in router.routes:
            if isinstance(route, APIRoute):
                yield pytest.param(route, id=_route_id(name, route))


def _non_operator_routes() -> Iterator[Any]:
    for name, router, gate in _NON_OPERATOR_ROUTERS:
        for route in router.routes:
            if isinstance(route, APIRoute):
                yield pytest.param(route, gate, id=_route_id(name, route))


@pytest.mark.parametrize(("name", "router"), _DEPLOYMENT_WIDE_ROUTERS, ids=[n for n, _ in _DEPLOYMENT_WIDE_ROUTERS])
def test_each_deployment_wide_router_declares_the_gate_itself(name: str, router: APIRouter) -> None:
    """On the router, so a route added to it later inherits the gate."""
    assert any(dependency.dependency is require_deployment_operator for dependency in router.dependencies), name


@pytest.mark.parametrize("route", _routes(_DEPLOYMENT_WIDE_ROUTERS))
def test_every_route_on_a_deployment_wide_router_resolves_the_gate(route: APIRoute) -> None:
    """The declaration above, read back per route as FastAPI resolves it."""
    assert _resolves(route.dependant, require_deployment_operator)


@pytest.mark.parametrize(("route", "gate"), _non_operator_routes())
def test_the_non_operator_routers_admit_their_own_caller_instead(route: APIRoute, gate: Callable[..., Any]) -> None:
    """Both directions: the route keeps the dependency it takes, and gains no operator gate."""
    assert _resolves(route.dependant, gate)
    assert not _resolves(route.dependant, require_deployment_operator)


def test_every_router_declares_a_gate_or_says_why_it_does_not() -> None:
    """The hazard this change moved rather than removed: the next *router*.

    Declaring the gate on the router makes a new route inherit it, so the gap
    walks up one level. A wholly ungated new router is invisible to every other
    case in this file, including the partial-gating one below, which needs a
    gated route on the same router to compare against.

    So every router is classified: it declares a router-level dependency, or it
    is named in ``_UNGATED_ROUTERS`` with the reason. A new management router
    fails this until it does one or the other, which is the decision being
    forced. What counts as a gate is the roster in ``_ROUTER_LEVEL_GATES``, not
    any dependency at all, so a router-level ``Depends(get_db)`` does not read as
    one. The second assertion keeps the list from going stale once a listed
    router gains a gate.
    """
    routers = _exposed_routers()
    declares_a_gate = {
        name
        for name, router in routers.items()
        if any(dependency.dependency in _ROUTER_LEVEL_GATES for dependency in router.dependencies)
    }

    unclassified = sorted(set(routers) - declares_a_gate - set(_UNGATED_ROUTERS))
    listed_but_gated = sorted(declares_a_gate & set(_UNGATED_ROUTERS))

    assert unclassified == []
    assert listed_but_gated == []


def test_no_router_gates_some_of_its_routes_and_not_the_rest() -> None:
    """The old shape: a gate on some routes of a router and not the others.

    If any route on a router resolves the gate then every route on it must,
    which is what declaring it on the router gets you and what a per-route
    decorator gives up. A router with a genuine exception splits it off (see
    ``_NON_OPERATOR_ROUTERS``) rather than mixing the two. This says nothing
    about a router with no gate anywhere; that is the case above.
    """
    partly_gated: list[str] = []
    for name, router in _exposed_routers().items():
        routes = [route for route in router.routes if isinstance(route, APIRoute)]
        gated = [route for route in routes if _resolves(route.dependant, require_deployment_operator)]
        if gated and len(gated) != len(routes):
            ungated = sorted({route.path for route in routes if route not in gated})
            partly_gated.append(f"{name}: ungated {ungated}")

    assert partly_gated == []
