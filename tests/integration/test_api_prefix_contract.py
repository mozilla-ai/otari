"""The published API lives under one prefix, and the security stamps say so.

Two things ``make openapi-check`` cannot catch. First, that every path in the
generated document sits under ``API_ROOT`` or ``OTLP_ROOT``. Second, that the
allowlists in ``gateway.main`` name routes that exist: the generator reads
those same lists, so the committed document and the generator agree even when
both are stale. The routing table is the independent source of truth here.

The stamp checks are weaker on purpose. ``custom_openapi`` reads the same
lists, so a mounted path wrongly added to ``_UNAUTHENTICATED_PATHS`` stamps
itself consistent and passes; only the gateway-token check asserts an exact
value. Whether a listed path really answers without a credential is a runtime
question, and not this test's.

On the worker's PostgreSQL like the rest of ``tests/integration``, because
``create_app`` is the honest way to get a routing table.
"""

from typing import Any

import pytest
from fastapi import FastAPI

from gateway.core.config import API_ROOT, OTLP_ROOT, GatewayConfig
from gateway.main import (
    _COOKIE_AUTH_PREFIXES,
    _GATEWAY_TOKEN_PATHS,
    _PUBLIC_PREFIXES,
    _UNAUTHENTICATED_PATHS,
    create_app,
)

# App-level paths that deliberately stay at the origin root. Each has a reason
# outside this repository's control; see the spec's D8.
FROZEN_ROOT_PATHS = frozenset({"/auth/{provider}/callback", "/metrics"})
# The SPA shell and its assets. Not API, never versioned.
SHELL_PATHS = frozenset({"/", "/welcome", "/favicon.svg", "/dashboard-build.json"})
SHELL_MOUNTS = ("/assets", "/pwa", "/fonts")


def _config(postgres_url: str, mode: str | None, **overrides: Any) -> GatewayConfig:
    return GatewayConfig(
        mode=mode,
        database_url=postgres_url,
        master_key="test-master-key",
        auto_migrate=False,
        require_pricing=False,
        model_discovery=False,
        bootstrap_api_key=False,
        **overrides,
    )


def _mounted(app: FastAPI) -> set[str]:
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    # A newer FastAPI may stop flattening nested routers into app.routes; an
    # empty set here would make every assertion below pass vacuously.
    assert paths, "app.routes carried no paths"
    return paths


@pytest.fixture(scope="module")
def standalone(postgres_url: str) -> FastAPI:
    # Web search configured, so web_search_backend.router mounts and the
    # gateway-token path exists to be checked.
    return create_app(
        _config(
            postgres_url,
            "standalone",
            enable_metrics=True,
            web_search_provider="tavily",
            web_search_provider_api_key="test-search-key",
            web_search_backend_token="test-gateway-token",
        )
    )


@pytest.fixture(scope="module")
def hosted(postgres_url: str) -> FastAPI:
    return create_app(_config(postgres_url, "hosted"))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root}/")


def test_every_mounted_route_is_under_a_root_or_frozen(standalone: FastAPI, hosted: FastAPI) -> None:
    for app in (standalone, hosted):
        stray = {
            p
            for p in _mounted(app)
            if not (
                _under(p, API_ROOT)
                or _under(p, OTLP_ROOT)
                or p in FROZEN_ROOT_PATHS
                or p in SHELL_PATHS
                or p.startswith(SHELL_MOUNTS)
            )
        }
        assert not stray, f"routes outside {API_ROOT} and {OTLP_ROOT}: {sorted(stray)}"


def test_every_published_path_is_under_a_root(standalone: FastAPI) -> None:
    paths = standalone.openapi()["paths"]
    assert paths, "the generated document has no paths"
    stray = {p for p in paths if not (_under(p, API_ROOT) or _under(p, OTLP_ROOT))}
    assert not stray, f"published outside {API_ROOT} and {OTLP_ROOT}: {sorted(stray)}"


def test_frozen_root_paths_still_exist(standalone: FastAPI) -> None:
    mounted = _mounted(standalone)
    missing = FROZEN_ROOT_PATHS - mounted
    assert not missing, f"frozen root paths were moved or dropped: {sorted(missing)}"


def test_exact_match_allowlists_name_mounted_routes(standalone: FastAPI) -> None:
    mounted = _mounted(standalone)
    for name, paths in (
        ("_UNAUTHENTICATED_PATHS", _UNAUTHENTICATED_PATHS),
        ("_GATEWAY_TOKEN_PATHS", _GATEWAY_TOKEN_PATHS),
    ):
        assert paths, f"{name} is empty"
        missing = set(paths) - mounted
        assert not missing, f"{name} names routes that are not mounted: {sorted(missing)}"


def test_prefix_allowlists_cover_at_least_one_route(standalone: FastAPI) -> None:
    mounted = _mounted(standalone)
    for name, prefixes in (
        ("_PUBLIC_PREFIXES", _PUBLIC_PREFIXES),
        ("_COOKIE_AUTH_PREFIXES", _COOKIE_AUTH_PREFIXES),
    ):
        assert prefixes, f"{name} is empty"
        for prefix in prefixes:
            assert any(_under(p, prefix) for p in mounted), f"{name}: nothing mounted under {prefix}"


def test_generated_document_stamps_security_as_the_allowlists_say(standalone: FastAPI) -> None:
    paths = standalone.openapi()["paths"]
    for path in _UNAUTHENTICATED_PATHS:
        for method, operation in paths[path].items():
            if isinstance(operation, dict):
                assert "security" not in operation, f"{method.upper()} {path} is stamped but listed as unauthenticated"
    for path in _GATEWAY_TOKEN_PATHS:
        for method, operation in paths[path].items():
            if isinstance(operation, dict):
                assert operation.get("security") == [{"GatewayTokenAuth": []}], (
                    f"{method.upper()} {path} must be stamped GatewayTokenAuth only"
                )
