"""Surface declarations and the published surface lists."""

import importlib
import pkgutil

import gateway.api.routes
from gateway.api.routes.bootstrap import _DECLARED_SURFACES, SURFACES
from gateway.core.surface import Surface


def test_every_route_module_surface_is_listed_in_bootstrap() -> None:
    """An unlisted ``SURFACE`` is never published, and nothing else would notice."""
    declared = set()
    for module_info in pkgutil.iter_modules(gateway.api.routes.__path__):
        module = importlib.import_module(f"gateway.api.routes.{module_info.name}")
        surface = getattr(module, "SURFACE", None)
        if isinstance(surface, Surface):
            declared.add(surface)

    assert declared == set(_DECLARED_SURFACES)


def test_the_surface_lists_are_spelled_out() -> None:
    """Literal lists, so adding or dropping a surface anywhere fails here."""
    assert sorted(SURFACES) == [
        "admin",
        "budgets",
        # The deployment's own upstream credentials.
        "hosted_providers",
        "keys",
        "models",
        "organization_guardrails",
        "organization_providers",
        # Where the organization is narrower than the deployment.
        "organization_usage",
        "organizations",
        "playground",
        "pricing",
        "routing",
        "settings",
        "tools",
        "usage",
        "users",
        "workspaces",
    ]
