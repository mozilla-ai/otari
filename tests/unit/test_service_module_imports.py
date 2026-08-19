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
