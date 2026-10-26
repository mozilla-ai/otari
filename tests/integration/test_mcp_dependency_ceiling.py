"""mcp's pyproject.toml constraint must keep a breaking mcp release out of a
fresh resolution, not just satisfy the committed uv.lock.

uv.lock pins one mcp, so `uv sync` here never re-resolves and never
notices a missing (or misplaced) ceiling: CI stays green even though a
fresh, unlocked resolution elsewhere (a fresh clone before the first `uv
lock`, a consuming superset build resolving otari's tree alongside its own)
can pick up a release that removed a name the gateway imports. mcp 2.0 did,
removing `mcp.client.streamable_http.streamablehttp_client`, which
`gateway/services/mcp_client.py` imported at module scope, so every gateway
command died with ImportError. See #689.

This resolves mcp into a throwaway venv using only the constraint declared
in pyproject.toml - uv.lock is never consulted - and proves the actual
import that broke still works against whatever version that constraint
allows today. If the ceiling is ever loosened, dropped, or simply placed
too high (e.g. "<3.0.0"), this installs the newest matching mcp and fails
the same way production did; the static shape check in
tests/unit/test_mcp_dependency_ceiling.py only catches a ceiling missing
entirely.

Real network I/O and subprocess/venv creation, not a pure-function check -
hence tests/integration rather than tests/unit.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The mcp names the gateway imports at module scope.
_IMPORTS = (
    "from mcp.client.streamable_http import streamable_http_client; "
    "from mcp.shared.exceptions import MCPError; "
    "from mcp.types import PaginatedRequestParams"
)


def _mcp_requirement() -> Requirement:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    for raw in pyproject["project"]["dependencies"]:
        requirement = Requirement(raw)
        if requirement.name == "mcp":
            return requirement
    pytest.fail("mcp is no longer declared as a direct dependency in pyproject.toml")


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_mcp_constraint_resolves_to_an_importable_version(tmp_path: Path) -> None:
    """Reruns cover a transient PyPI/index failure during the real install below,
    not the assertions themselves."""
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH")

    requirement = _mcp_requirement()

    # with_pip=False: `uv pip install --python` does the installing, so the venv's
    # own pip is never used, but bootstrapping it needs ensurepip, which some
    # distro interpreters ship separately from the venv module (e.g. Debian's
    # python3-venv without python3-pip). Skipping it keeps this failing only on
    # what it's actually checking, not on an unrelated ensurepip gap.
    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=False)
    venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    # 60s here plus 30s on the import check below keeps both subprocesses inside
    # pytest.ini's global 120s bound, so a stuck `uv` fails with a message naming
    # the install it hung on rather than as a bare global timeout.
    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(venv_python), str(requirement)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert install.returncode == 0, f"fresh install of '{requirement}' failed:\n{install.stderr}"

    imported = subprocess.run(
        [str(venv_python), "-c", _IMPORTS],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert imported.returncode == 0, (
        f"a fresh install of '{requirement}' (ignoring uv.lock) cannot import "
        f"what the gateway imports, the same break as #689:\n{imported.stderr}"
    )
