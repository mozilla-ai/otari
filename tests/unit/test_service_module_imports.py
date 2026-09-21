"""Every listed service module imports first in a fresh interpreter.

The app imports its packages in an order that hides an import cycle between services.
A subprocess that imports one module first is the only way to see the cycle.
"""

import subprocess
import sys

import pytest

# The service modules that participate in the tenancy import graph, plus the two
# that closed a cycle. Not every module in the package: this is a regression pin
# for a specific shape, not an inventory to keep in step with the directory.
_MODULES = [
    "gateway.services.workspace_scope",
    "gateway.services.budgets",
    "gateway.services.budgets._scoped_enforcement",
    "gateway.services.budgets._reservations",
    "gateway.services.tenancy",
    "gateway.services.tenancy.workspace_service",
    "gateway.services.tenancy.organization_service",
    "gateway.services.tenancy.provisioning_service",
    # _member_policies reaches organization_service and authorization, and no organizations module reaches it.
    "gateway.services.budgets._member_policies",
    "gateway.services.tenancy.authorization",
    "gateway.services.budgets._organization_surface",
    "gateway.services.budgets._retiming",
    # workspace_mcp_server_service reaches authorization and organization_service
    # the same way, and is additionally imported from the request pipeline, which
    # is a second entry point into the graph.
    "gateway.services.tenancy.workspace_mcp_server_service",
    # workspace_web_search_service is here for the same two reasons, plus a
    # third: it reaches `services.web_search_backend` for the result ceiling it
    # validates against, which is the first edge from the tenancy graph into the
    # tool backends.
    "gateway.services.tenancy.workspace_web_search_service",
]


@pytest.mark.parametrize("module", _MODULES)
def test_module_imports_first(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"{module} cannot be imported first:\n{result.stderr}"


# Budgets depends on organizations, never the reverse, so an organizations
# module must not load a budget module.
_BUDGETS_PACKAGE = "gateway.services.budgets"


@pytest.mark.parametrize(
    "module",
    [
        "gateway.services.tenancy.workspace_service",
        "gateway.services.tenancy.organization_service",
        "gateway.services.tenancy.provisioning_service",
    ],
)
def test_organizations_modules_load_no_budget_module(module: str) -> None:
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        f"loaded = sorted(m for m in sys.modules if m.startswith({_BUDGETS_PACKAGE!r}))\n"
        "print(','.join(loaded))\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "", f"{module} loaded budget modules: {result.stdout.strip()}"
