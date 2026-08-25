"""Unit tests for the Alembic head check (the revision graph stays linear)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_alembic_heads.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_alembic_heads", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


def test_the_revision_graph_has_exactly_one_head() -> None:
    """Two revisions claiming one parent is a rebase artifact, not a design.

    Asserted here as well as in `make lint` because of how the failure presents
    otherwise: every schema-building test errors at once with the same
    MultipleHeads, which reads like a broken branch rather than a chain that
    needs re-pointing.
    """
    heads = check.find_heads(_REPO_ROOT)

    assert len(heads) == 1, (
        f"{len(heads)} heads: {sorted(heads)}. Re-point the newer revision's down_revision at the older head."
    )


def test_the_guard_passes_on_a_linear_graph(capsys: pytest.CaptureFixture[str]) -> None:
    assert check.main() == 0
    assert "Single head" in capsys.readouterr().out


def test_a_head_is_described_with_its_file_and_parent() -> None:
    """The report names the file, since a revision id alone is not greppable enough."""
    head = check.find_heads(_REPO_ROOT)[0]

    described = check.describe(head, _REPO_ROOT)

    assert head in described
    assert "down_revision=" in described
    assert described.strip().endswith(".py")
