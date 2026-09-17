"""Surface declarations and the published surface lists."""

import importlib
import pkgutil

import pytest

import gateway.api.routes
from gateway.api.routes.bootstrap import _DECLARED_SURFACES, HOSTED_SURFACES, STANDALONE_SURFACES
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
    assert sorted(STANDALONE_SURFACES) == [
        "admin",
        "budgets",
        "keys",
        "models",
        "organizations",
        "playground",
        "pricing",
        "providers",
        "routing",
        "settings",
        "tools",
        "usage",
        "users",
        "workspaces",
    ]
    assert sorted(HOSTED_SURFACES) == [
        "admin",
        "budgets",
        "keys",
        "models",
        "organization_providers",
        "organization_usage",
        "organizations",
        # Declared on both, and withheld at publish time by a hosted deployment
        # that has no ``data_plane_url`` to forward a completion to. This roster
        # is the topology axis, which is why the configuration one is not in it.
        "playground",
        "pricing",
        "routing",
        "settings",
        "tools",
        "usage",
        "users",
        "workspaces",
    ]


def test_a_surface_no_deployment_publishes_is_refused() -> None:
    with pytest.raises(ValueError, match="published by no deployment"):
        Surface("orphan", standalone=False, hosted=False)
