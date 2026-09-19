"""Every service module imports on its own, in a fresh interpreter.

Import cycles between services are invisible in normal use, because the app
imports its packages in an order that happens to resolve them: `api/main.py`
pulls in `services.tenancy` before anything reaches `services.workspace_scope`,
so the half-initialized module is already complete by the time it is read. The
first thing to import one of them *first* is what breaks, and that is usually a
new script, a migration helper, or a test, which is a poor place to discover it.

Each module is therefore imported as the very first thing a subprocess does,
which is the only way to see the cycle. `workspace_scope` and
`scoped_budget_service` are here because both did fail this way:
`workspace_scope` imports `tenancy.provisioning_service`, which runs
`tenancy/__init__`, which imports `workspace_service`, which imported back into
`workspace_scope` at module scope.
"""

import subprocess
import sys

import pytest

# The service modules that participate in the tenancy import graph, plus the two
# that closed a cycle. Not every module in the package: this is a regression pin
# for a specific shape, not an inventory to keep in step with the directory.
_MODULES = [
    "gateway.services.workspace_scope",
    "gateway.services.scoped_budget_service",
    "gateway.services.budget_service",
    "gateway.services.tenancy",
    "gateway.services.tenancy.workspace_service",
    "gateway.services.tenancy.organization_service",
    "gateway.services.tenancy.provisioning_service",
    # workspace_budget_default_service reaches organization_service and
    # authorization, and nothing in organizations reaches it. authorization sits
    # between workspace_service and organization_service. Pinned here for the
    # same reason as the two above them.
    "gateway.services.tenancy.workspace_budget_default_service",
    "gateway.services.tenancy.authorization",
    "gateway.services.tenancy.organization_budget_service",
    "gateway.services.budget_retiming",
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
_BUDGET_MODULE_PREFIXES = (
    "gateway.services.budget",
    "gateway.services.scoped_budget_service",
    "gateway.services.tenancy.organization_budget_service",
    "gateway.services.tenancy.workspace_budget_default_service",
    "gateway.services.budgets",
)


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
        f"loaded = sorted(m for m in sys.modules if m.startswith({_BUDGET_MODULE_PREFIXES!r}))\n"
        "print(','.join(loaded))\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "", f"{module} loaded budget modules: {result.stdout.strip()}"
