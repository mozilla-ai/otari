"""``organization_oidc_enabled`` decides whether this deployment serves organization SSO.

Where it is off, both route groups are absent rather than refusing, because a route
that answers a refusal tells an unauthenticated caller that the surface exists here.
"""

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute, iter_route_contexts

from gateway import features
from gateway.api.main import _register_core_routers
from gateway.api.routes.bootstrap import published_surfaces
from gateway.core.config import API_ROOT, GatewayConfig
from gateway.core.feature import CoreFeature

# Every path of the two SSO route groups, by the prefix each router declares.
_SIGN_IN_PREFIX = f"{API_ROOT}/auth/sso"
_SETTINGS_PREFIX = f"{API_ROOT}/organizations/me/sso"


def _config(*, organization_oidc_enabled: bool) -> GatewayConfig:
    """A config that takes no routing decision from the machine the test runs on."""
    return GatewayConfig(bootstrap=None, web_search_provider=None, organization_oidc_enabled=organization_oidc_enabled)


def _enabled_features(config: GatewayConfig) -> tuple[CoreFeature, ...]:
    """The features this config switches on, as the app factory resolves them."""
    return tuple(feature for feature in features.CORE_FEATURES if feature.enabled(config))


def _served_paths(config: GatewayConfig) -> set[str]:
    aggregate = APIRouter(prefix=API_ROOT)
    _register_core_routers(aggregate, config, _enabled_features(config))
    app = FastAPI()
    app.include_router(aggregate)
    return {
        context.path or ""
        for context in iter_route_contexts(app.routes)
        if isinstance(context.original_route, APIRoute)
    }


def _sso_paths(config: GatewayConfig) -> set[str]:
    return {path for path in _served_paths(config) if path.startswith((_SIGN_IN_PREFIX, _SETTINGS_PREFIX))}


def test_sso_is_served_by_default() -> None:
    """A deployment that configures nothing serves SSO, so the switch cannot turn it off by omission."""
    assert GatewayConfig(bootstrap=None, web_search_provider=None).organization_oidc_enabled is True


def test_switching_sso_on_serves_both_halves() -> None:
    paths = _sso_paths(_config(organization_oidc_enabled=True))

    assert any(path.startswith(_SIGN_IN_PREFIX) for path in paths), paths
    assert any(path.startswith(_SETTINGS_PREFIX) for path in paths), paths


def test_switching_sso_off_serves_neither_half() -> None:
    """Both groups go, so a direct call reaches no route rather than a refusal."""
    assert _sso_paths(_config(organization_oidc_enabled=False)) == set()


def test_switching_sso_off_leaves_every_other_route_alone() -> None:
    """The switch removes the SSO routes and no others."""
    on = _served_paths(_config(organization_oidc_enabled=True))
    off = _served_paths(_config(organization_oidc_enabled=False))

    assert off == on - _sso_paths(_config(organization_oidc_enabled=True))


def test_the_dashboard_surface_follows_the_switch() -> None:
    """The dashboard draws the page and its sidebar row only where the routes behind them answer."""
    on = published_surfaces(_enabled_features(_config(organization_oidc_enabled=True)))
    off = published_surfaces(_enabled_features(_config(organization_oidc_enabled=False)))

    assert "organization_sso" in on
    assert "organization_sso" not in off
