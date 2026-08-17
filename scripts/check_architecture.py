#!/usr/bin/env python3
"""Check gateway architectural boundaries.

Enforces:
1. Service layer boundaries: services must not import the API layer.
2. API route purity: routes must not use the sync ORM layer (sqlalchemy.orm).
3. Repository boundaries: repositories must not import services or the API layer.
4. Naming conventions: repository modules end in _repository.py.
5. OSS/enterprise boundary: OSS code must not import the enterprise overlay.

Usage:
    uv run python scripts/check_architecture.py

Exit codes:
    0 - No violations
    1 - Violations found
"""

import ast
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
GATEWAY_ROOT = SRC_ROOT / "gateway"
TESTS_ROOT = REPO_ROOT / "tests"


class LayerRule(TypedDict):
    """Import rules for one gateway layer."""

    allowed: list[str]
    forbidden: list[str]
    description: str


# Rules are keyed by the layer's path below src/. Only "forbidden" is enforced;
# "allowed" documents the layer contract for reviewers. Cross-cutting top-level
# modules (gateway.log_config, gateway.metrics, ...) are always importable and
# are not listed. Restrictions accumulate down the tree, so a file under
# gateway/api/routes answers to a gateway/api entry as well as its own; a nested
# layer can add restrictions but cannot opt out of an enclosing layer's.
RULES: dict[str, LayerRule] = {
    # OSS -> enterprise boundary. Keyed at the gateway root so it covers every
    # layer, ports and adapters included; nothing legitimately in this tree
    # matches. The overlay (e.g. otari.ai's enterprise adapters) is imported
    # as "overlay.*" today; a build that composes it into the gateway
    # namespace instead would spell it "gateway.overlay.*". Both are
    # forbidden, so an OSS file that reaches for an enterprise concept fails
    # the build rather than waiting on review, whichever way the overlay is
    # composed.
    "gateway": {
        "allowed": [],
        "forbidden": ["gateway.overlay", "overlay"],
        "description": "OSS base",
    },
    # The OSS test suite answers to the same boundary: a test of overlay
    # behavior belongs in the overlay's own suite, not here.
    "tests": {
        "allowed": [],
        "forbidden": ["gateway.overlay", "overlay"],
        "description": "OSS test suite",
    },
    "gateway/services": {
        "allowed": ["gateway.repositories", "gateway.models", "gateway.core", "gateway.auth"],
        "forbidden": ["gateway.api"],
        "description": "Services",
    },
    "gateway/api/routes": {
        # Routes reuse repository helpers (e.g. get_active_user) per the
        # repository conventions in AGENTS.md, so gateway.repositories stays
        # allowed here.
        "allowed": [
            "gateway.api",
            "gateway.services",
            "gateway.repositories",
            "gateway.models",
            "gateway.core",
            "gateway.auth",
        ],
        "forbidden": ["sqlalchemy.orm"],
        "description": "API routes",
    },
    "gateway/repositories": {
        "allowed": ["gateway.models"],
        "forbidden": ["gateway.services", "gateway.api"],
        "description": "Repositories",
    },
    # Leaf data types shared across layers (e.g. the routing Attempt, which
    # services build and the API layer executes). They sit below everything, so
    # they may not import any other gateway layer: a type that reaches back into
    # services or the API would smuggle a dependency edge into every module that
    # merely wants the shape.
    "gateway/types": {
        "allowed": [],
        "forbidden": ["gateway.api", "gateway.services", "gateway.repositories", "gateway.core"],
        "description": "Shared types",
    },
    # Open-core boundary scaffolding. Ports (domain-named interfaces) and their
    # adapters get their own layers so future boundary rules have a place to
    # land (see ARCHITECTURE.md). Intentionally no-op until the first port
    # arrives: with empty "forbidden" lists nothing is enforced and the check
    # stays green.
    "gateway/ports": {
        "allowed": [],
        "forbidden": [],
        "description": "Ports",
    },
    "gateway/adapters": {
        "allowed": [],
        "forbidden": [],
        "description": "Adapters",
    },
}


def _matches(module: str, prefix: str) -> bool:
    """Return whether a module path is the prefix module itself or lives inside it."""
    return module == prefix or module.startswith(prefix + ".")


def _resolve_relative(node: ast.ImportFrom, file_path: Path, src_root: Path) -> str | None:
    """Resolve a relative import to an absolute module path, or None if it escapes src_root."""
    package_parts = file_path.parent.relative_to(src_root).parts
    drop = node.level - 1
    if drop >= len(package_parts):
        return None
    base = ".".join(package_parts[: len(package_parts) - drop])
    if node.module:
        return f"{base}.{node.module}"
    return base


def _imported_modules(node: ast.Import | ast.ImportFrom, file_path: Path, src_root: Path) -> list[str]:
    """Return the absolute module paths an import statement pulls in."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module if node.level == 0 else _resolve_relative(node, file_path, src_root)
    if base is None:
        return []
    # `from pkg import name` may bind the submodule pkg.name, so check it too.
    return [base] + [f"{base}.{alias.name}" for alias in node.names]


def check_file(file_path: Path, src_root: Path) -> list[tuple[int, str, str]]:
    """Check one Python file below src_root against the layer rules for its location.

    Returns:
        One (line number, module, message) tuple per offending import statement.

    """
    relative_path = file_path.relative_to(src_root).as_posix()
    # Restrictions accumulate: a file answers to its own layer's rules and to
    # every enclosing layer's, so declaration order cannot silently shadow
    # either a nested rule or a broader one. Most specific first, so a violation
    # is attributed to the closest layer that forbids it.
    matches = sorted(
        ((fragment, layer_rule) for fragment, layer_rule in RULES.items() if relative_path.startswith(fragment + "/")),
        key=lambda match: len(match[0]),
        reverse=True,
    )
    forbidden = [(prefix, layer_rule["description"]) for _, layer_rule in matches for prefix in layer_rule["forbidden"]]
    if not forbidden:
        return []

    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    except SyntaxError as exc:
        # Unparseable files cannot be checked; ruff fails the same lint run on
        # them, so warn here rather than duplicating the failure.
        print(f"  ⚠ Syntax error in {file_path}: {exc}")
        return []

    violations: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        for module in _imported_modules(node, file_path, src_root):
            offended = next((description for prefix, description in forbidden if _matches(module, prefix)), None)
            if offended is not None:
                violations.append((node.lineno, module, f"Forbidden import in {offended}"))
                break
    return violations


# Service modules are purpose-named (guardrails.py, url_safety.py, ...), so
# there is no *_service.py naming rule to enforce.
def check_naming_conventions(src_root: Path) -> list[str]:
    """Check that repository modules follow the *_repository.py convention."""
    violations: list[str] = []
    repositories_path = src_root / "gateway" / "repositories"
    if not repositories_path.is_dir():
        return violations
    for repository_file in sorted(repositories_path.rglob("*.py")):
        if repository_file.name == "__init__.py":
            continue
        if not repository_file.name.endswith("_repository.py"):
            violations.append(f"Repository file {repository_file.relative_to(src_root)} must end with '_repository.py'")
    return violations


def main() -> int:
    """Run the architecture checks over the gateway package and the OSS test suite."""
    # Both must exist: silently skipping either would let its rules (including
    # the OSS/enterprise boundary) stop enforcing while the check stays green.
    for required_root in (GATEWAY_ROOT, TESTS_ROOT):
        if not required_root.is_dir():
            print(f"❌ Expected directory not found at {required_root}")
            return 1

    import_violations: list[tuple[Path, int, str, str]] = []
    for py_file in sorted(GATEWAY_ROOT.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        import_violations.extend(
            (py_file, lineno, module, message) for lineno, module, message in check_file(py_file, SRC_ROOT)
        )
    # tests/ sits beside src/, not under it, so its relative paths (and the
    # "tests" rule key above) are rooted at the repo root instead.
    for py_file in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        import_violations.extend(
            (py_file, lineno, module, message) for lineno, module, message in check_file(py_file, REPO_ROOT)
        )

    naming_violations = check_naming_conventions(SRC_ROOT)

    if import_violations:
        print("❌ Architecture violations found:\n")
        for file_path, lineno, module, message in import_violations:
            print(f"  {file_path.relative_to(REPO_ROOT)}:{lineno}")
            print(f"    {message}: {module}\n")
        print(f"Total import violations: {len(import_violations)}")

    if naming_violations:
        print("\n❌ Naming convention violations:\n")
        for violation in naming_violations:
            print(f"  {violation}")
        print(f"\nTotal naming violations: {len(naming_violations)}")

    if import_violations or naming_violations:
        print("\n💡 See ARCHITECTURE.md for the intended layering")
        return 1

    print("✅ No architecture violations found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
