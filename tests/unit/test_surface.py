"""Surface declarations and the published surface lists."""

import pytest

from gateway.api.routes.bootstrap import HOSTED_SURFACES, STANDALONE_SURFACES
from gateway.core.surface import Surface


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
