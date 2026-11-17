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

A name that moves into a submodule and is not re-exported is just as quiet. It
only breaks an import that some caller already makes, so a name nothing happens
to import today leaves the root and nothing fails, and the next caller to want it
finds it gone. The root is the package's facade, so every public name a submodule
defines is required here to be reachable from it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from gateway.api import deps

DEPS_PACKAGE = Path(__file__).resolve().parents[2] / "src" / "gateway" / "api" / "deps"
DEPS_ROOT = DEPS_PACKAGE / "__init__.py"

# Names a submodule offers only to the rest of the package. Each entry says who uses it.
PACKAGE_INTERNAL = {
    "federated_token_verifier": "Read by api_keys.py to build the token exchange; no caller outside the package.",
}


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


def _submodule_public_names() -> dict[str, set[str]]:
    """The public names each submodule of the package defines at module scope."""
    public: dict[str, set[str]] = {}
    for module in sorted(DEPS_PACKAGE.glob("*.py")):
        if module.name == "__init__.py":
            continue
        names: set[str] = set()
        for node in ast.parse(module.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        public[module.name] = {name for name in names if not name.startswith("_") and name != "__all__"}
    return public


def test_the_root_reaches_every_public_submodule_name() -> None:
    unreachable = {
        module: sorted(name for name in names if name not in PACKAGE_INTERNAL and not hasattr(deps, name))
        for module, names in _submodule_public_names().items()
    }
    unreachable = {module: names for module, names in unreachable.items() if names}
    assert not unreachable, (
        f"gateway.api.deps does not reach {unreachable}. The root is the package's facade, so import each "
        "name there, or record it in PACKAGE_INTERNAL with the package member that uses it."
    )


def test_no_retired_name_sits_on_the_internal_list() -> None:
    defined = {name for names in _submodule_public_names().values() for name in names}
    retired = sorted(name for name in PACKAGE_INTERNAL if name not in defined)
    assert not retired, (
        f"PACKAGE_INTERNAL names {retired}, which no submodule defines any more. "
        "Remove each entry, so the list cannot pre-authorize a future name."
    )
