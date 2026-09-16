"""The published surface lists."""

from gateway.api.routes.bootstrap import HOSTED_SURFACES, STANDALONE_SURFACES


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

