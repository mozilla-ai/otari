"""Every route an API key can reach declares the scope it needs, and no other route declares one.

The credential check refuses a key on a route that declares nothing, so a route
added without a declaration fails closed at runtime. These tests move that failure
into CI, and pin the two lists a declaration cannot express by itself: the routes
that authenticate a key inside their handler, where no dependency shows it, and
the routes open to every key, which a new entry should only join on purpose.

Read from the app ``create_app`` builds, with every optional router mounted the
way ``scripts/generate_openapi.py`` mounts them, so a feature-gated route is
covered too.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute, iter_route_contexts

from gateway.api.deps import (
    verify_api_key,
    verify_api_key_or_master_key,
    verify_catalog_reader,
    verify_catalog_reader_or_public,
    verify_service_key,
)
from gateway.api.key_scopes import KEY_SCOPE_EXTENSION, UndeclaredKeyScope, declared_key_scope
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.settings.api_keys import FederationIssuer
from gateway.main import create_app
from gateway.services.secret_box import generate_secret_key
from gateway.types.key_scope import KeyScope

_KEY_GATES: tuple[Callable[..., Any], ...] = (
    verify_api_key,
    verify_api_key_or_master_key,
    verify_service_key,
    verify_catalog_reader,
    verify_catalog_reader_or_public,
)

# Routes whose handler calls ``verify_api_key_or_master_key`` itself, after reading
# what it needs from the raw request, so no dependency says a key reaches them.
SELF_AUTHENTICATING = {
    ("POST", f"{API_ROOT}/chat/completions"),
    ("POST", f"{API_ROOT}/responses"),
    ("POST", f"{API_ROOT}/messages"),
    ("POST", f"{API_ROOT}/messages/count_tokens"),
    ("GET", f"{API_ROOT}/mcp/servers"),
    ("GET", f"{API_ROOT}/mcp/servers/{{mcp_server_id}}/tools"),
    ("POST", f"{API_ROOT}/mcp/execute"),
}

# Read-only discovery a client needs before it can do anything, which every key may reach.
OPEN_TO_EVERY_KEY = {
    ("GET", f"{API_ROOT}/catalog/models"),
    ("GET", f"{API_ROOT}/catalog/models/{{model_id:path}}"),
    ("GET", f"{API_ROOT}/models"),
    ("GET", f"{API_ROOT}/models/{{model_id:path}}"),
    ("GET", f"{API_ROOT}/pricing"),
    ("GET", f"{API_ROOT}/pricing/current"),
    ("GET", f"{API_ROOT}/pricing/{{model_key:path}}"),
    ("GET", f"{API_ROOT}/pricing/{{model_key:path}}/history"),
    ("GET", f"{API_ROOT}/providers/catalog"),
    ("GET", f"{API_ROOT}/providers/catalog/{{provider_id}}"),
    ("GET", f"{API_ROOT}/search-tools/providers"),
    ("GET", f"{API_ROOT}/tool-settings/guardrails/catalog"),
    ("GET", f"{API_ROOT}/tool-settings/guardrails/profiles"),
    ("GET", f"{API_ROOT}/tools"),
}

Operation = tuple[str, str]


@pytest.fixture(scope="module")
def app(tmp_path_factory: pytest.TempPathFactory, monkeypatch_module: pytest.MonkeyPatch) -> FastAPI:
    database = Path(tmp_path_factory.mktemp("scopes")) / "routes.db"
    monkeypatch_module.setenv("OTARI_SECRET_KEY", generate_secret_key())
    config = GatewayConfig(
        database_url=f"sqlite:///{database}",
        bootstrap=None,
        bootstrap_api_key=False,
        federation_issuers=[FederationIssuer(issuer="https://scopes.invalid")],
        federation_audiences=["https://scopes.invalid"],
        feedback_enabled=True,
    )
    return create_app(config)


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    with pytest.MonkeyPatch.context() as patch:
        yield patch


def _routes(app: FastAPI) -> Iterator[tuple[Operation, APIRoute]]:
    for context in iter_route_contexts(app.routes):
        route = context.original_route
        if isinstance(route, APIRoute):
            for method in sorted(route.methods or ()):
                yield (method, context.path or ""), route


def _resolves_a_key_gate(dependant: Dependant) -> bool:
    return any(sub.call in _KEY_GATES or _resolves_a_key_gate(sub) for sub in dependant.dependencies)


def _reaches_keys(operation: Operation, route: APIRoute) -> bool:
    return operation in SELF_AUTHENTICATING or _resolves_a_key_gate(route.dependant)


def test_every_route_a_key_reaches_declares_a_scope(app: FastAPI) -> None:
    undeclared = sorted(
        operation
        for operation, route in _routes(app)
        if _reaches_keys(operation, route) and isinstance(declared_key_scope(route), UndeclaredKeyScope)
    )
    assert not undeclared, (
        "Each route an API key can reach passes openapi_extra=key_scope(...) naming the scope it needs "
        f"(gateway/api/key_scopes.py); until it does, every key is refused there: {undeclared}"
    )


def test_no_route_a_key_cannot_reach_declares_a_scope(app: FastAPI) -> None:
    stray = sorted(
        operation
        for operation, route in _routes(app)
        if not _reaches_keys(operation, route) and not isinstance(declared_key_scope(route), UndeclaredKeyScope)
    )
    assert not stray, f"These routes declare a key scope but take no API key, so the declaration misleads: {stray}"


def test_each_self_authenticating_route_is_served(app: FastAPI) -> None:
    served = {operation for operation, _ in _routes(app)}
    assert SELF_AUTHENTICATING <= served, sorted(SELF_AUTHENTICATING - served)


def test_only_the_catalog_reads_are_open_to_every_key(app: FastAPI) -> None:
    open_routes = {operation for operation, route in _routes(app) if declared_key_scope(route) is None}
    assert open_routes == OPEN_TO_EVERY_KEY


def test_every_scope_guards_a_route(app: FastAPI) -> None:
    guarded = {declared_key_scope(route) for _, route in _routes(app)}
    assert set(KeyScope) <= guarded, sorted(set(KeyScope) - guarded)


def test_the_openapi_document_publishes_each_declaration(app: FastAPI) -> None:
    paths: dict[str, Any] = app.openapi()["paths"]
    for (method, path), route in _routes(app):
        declared = declared_key_scope(route)
        if isinstance(declared, UndeclaredKeyScope):
            continue
        published = paths[path.replace(":path", "")][method.lower()]
        assert published[KEY_SCOPE_EXTENSION] == ([] if declared is None else [declared.value]), (method, path)
