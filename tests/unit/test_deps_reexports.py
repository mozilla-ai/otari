"""Gate on the public surface of ``gateway.api.deps``: one ``__all__`` that resolves.

The package is being split into a module per concern. Each step moves a group of
builders into a submodule and re-exports it from the root, so that no caller
outside the package changes, and ``__all__`` is where the root records that
surface.

Nothing else reaches the rule. A second ``__all__`` assignment replaces the first
instead of extending it, and every name on the replaced list leaves the public
surface while still resolving as a module attribute, so no import breaks and no
lint rule, type check or other test fails. Only a star import and the tooling
that reads ``__all__`` see the gap.

A name that leaves the root without a re-export fails loudly elsewhere, because
the imports that reach for it stop resolving, so the silent case is the one
gated here.
"""

from __future__ import annotations

import ast
from pathlib import Path

from gateway.api import deps

DEPS_ROOT = Path(__file__).resolve().parents[2] / "src" / "gateway" / "api" / "deps" / "__init__.py"


def _public_surface_lines() -> list[int]:
    """The line of each ``__all__`` assignment at the root's module scope."""
    tree = ast.parse(DEPS_ROOT.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    ]


def test_the_root_declares_one_public_surface() -> None:
    lines = _public_surface_lines()
    assert len(lines) == 1, (
        f"gateway/api/deps/__init__.py assigns __all__ at lines {lines}. "
        "A later assignment replaces an earlier one, so every name on the earlier list "
        "leaves the public surface without breaking an import. Merge them into one list."
    )


def test_every_public_name_resolves() -> None:
    unresolved = sorted(name for name in deps.__all__ if not hasattr(deps, name))
    assert not unresolved, (
        f"gateway.api.deps names {unresolved} in __all__ and neither defines nor imports them. "
        "Drop each name, or import it from the submodule that now holds it."
    )
