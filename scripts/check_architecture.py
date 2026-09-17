#!/usr/bin/env python3
"""Check gateway architectural boundaries.

Enforces:
1. Service layer boundaries: services must not import the API layer.
2. API route purity: routes must not use the sync ORM layer (sqlalchemy.orm).
3. Repository boundaries: repositories must not import services or the API layer.
4. Naming conventions: repository modules end in _repository.py.
5. OSS/enterprise boundary: OSS code must not import the enterprise overlay.
6. Port boundaries: a port may describe the domain but not import a caller or an adapter.
7. Composition root: only gateway/container.py may name a concrete adapter.
8. Entrypoint purity: gateway/main.py may not import a route module.
9. Registry: only the app wiring reads gateway/features.py, so a service or a
   route may not import it; and nothing under gateway/ imports
   importlib.metadata, importlib_metadata or pkg_resources, so nothing is
   discovered.
10. Top-level packages: src/ holds only the packages on an explicit list, so a
    feature cannot sit beside gateway/, outside every rule above.
11. Query layering: a route or a service builds no query, because a query
    belongs in a repository. Modules that still do are named on a baseline,
    and the baseline only shrinks.

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
        # gateway.adapters is banned at the root, not only in the layers below,
        # because rule 7 is "only the composition root may name a concrete
        # adapter" and a layer-by-layer ban leaves every unlayered module
        # (gateway/main.py, gateway/cli.py, gateway/core, gateway/auth, ...)
        # free to shortcut past the seam. COMPOSITION_ROOT and the adapters
        # package itself are the two exemptions; see check_file.
        "forbidden": ["gateway.overlay", "overlay", "gateway.adapters"],
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
        "allowed": ["gateway.repositories", "gateway.models", "gateway.core", "gateway.auth", "gateway.ports"],
        # A service depends on the port and gets its adapter from the container;
        # naming a concrete adapter would pin the capability to one
        # implementation and defeat the seam (ARCHITECTURE.md, rule 5).
        # gateway.features is the registry the app wiring reads; a service
        # that imported it could register itself, which is discovery by
        # another name.
        "forbidden": ["gateway.api", "gateway.adapters", "gateway.features"],
        "description": "Services",
    },
    # The API layer resolves a port through the container in deps.py; only the
    # composition root (gateway/container.py) may name a concrete adapter.
    "gateway/api": {
        "allowed": [
            "gateway.api",
            "gateway.container",
            "gateway.ports",
            "gateway.services",
            "gateway.repositories",
            "gateway.models",
            "gateway.core",
            "gateway.auth",
        ],
        "forbidden": ["gateway.adapters"],
        "description": "API layer",
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
        # gateway.features for the reason services forbid it.
        "forbidden": ["sqlalchemy.orm", "gateway.features"],
        "description": "API routes",
    },
    "gateway/repositories": {
        "allowed": ["gateway.models"],
        "forbidden": ["gateway.services", "gateway.api", "gateway.adapters"],
        "description": "Repositories",
    },
    # Leaf data types shared across layers (e.g. the routing Attempt, which
    # services build and the API layer executes). They sit below everything, so
    # they may not import any other gateway layer: a type that reaches back into
    # services or the API would smuggle a dependency edge into every module that
    # merely wants the shape.
    "gateway/types": {
        "allowed": [],
        "forbidden": ["gateway.api", "gateway.services", "gateway.repositories", "gateway.core", "gateway.adapters"],
        "description": "Shared types",
    },
    # Open-core boundary. A port is a domain-named interface the core depends
    # on, so it sits below every layer that resolves one: it may describe the
    # domain (models, exceptions, core) and nothing else. Reaching into services
    # or the API would make the interface depend on one of its own callers, and
    # reaching for an adapter would name the implementation the port exists to
    # keep unnamed.
    "gateway/ports": {
        "allowed": ["gateway.models", "gateway.exceptions", "gateway.core"],
        "forbidden": ["gateway.api", "gateway.services", "gateway.repositories", "gateway.adapters"],
        "description": "Ports",
    },
    # An adapter implements a port and may use the layers below it to do so, but
    # it is driven, never driving: the API layer reaches it through the
    # container, not the other way round.
    "gateway/adapters": {
        "allowed": ["gateway.ports", "gateway.services", "gateway.repositories", "gateway.models", "gateway.core"],
        "forbidden": ["gateway.api"],
        "description": "Adapters",
    },
}


# Rules for one file rather than a layer. ``gateway/main.py`` is the process
# entrypoint and composes the app, so no directory rule covers it, yet a
# background task or a piece of domain logic it reaches for belongs in
# ``services/`` exactly as it does everywhere else. Without this, a route
# module imported by the lifespan (the selector index refresher, otari#1015)
# passes every other check.
FILE_RULES: dict[str, LayerRule] = {
    "gateway/main.py": {
        "allowed": ["gateway.api.main", "gateway.api.deps", "gateway.services", "gateway.core"],
        "forbidden": ["gateway.api.routes"],
        "description": "Application entrypoint",
    },
}


# The one file allowed to name a concrete adapter, and the package the adapters
# themselves live in (an adapter may of course refer to its siblings). Everything
# else under gateway/ answers to the root rule's ban above.
COMPOSITION_ROOT = "gateway/container.py"
ADAPTERS_PACKAGE = "gateway/adapters/"
ADAPTER_IMPORT = "gateway.adapters"

# Entry-point discovery is banned everywhere under gateway/, with a message of
# its own because "OSS base" would not say why: the feature registry in
# gateway/features.py is a literal tuple on purpose (ARCHITECTURE.md), and these
# are the modules discovery is written with.
DISCOVERY_SCOPE = "gateway/"
DISCOVERY_IMPORTS = ("importlib.metadata", "importlib_metadata", "pkg_resources")
DISCOVERY_RULE = "OSS base (no entry-point discovery; the feature registry is a literal tuple)"

ALLOWED_TOP_LEVEL_PACKAGES = ("gateway",)


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
    file_rule = FILE_RULES.get(relative_path)
    if file_rule is not None:
        forbidden = [(prefix, file_rule["description"]) for prefix in file_rule["forbidden"]] + forbidden
    if relative_path == COMPOSITION_ROOT or relative_path.startswith(ADAPTERS_PACKAGE):
        forbidden = [entry for entry in forbidden if entry[0] != ADAPTER_IMPORT]
    if relative_path.startswith(DISCOVERY_SCOPE):
        forbidden += [(prefix, DISCOVERY_RULE) for prefix in DISCOVERY_IMPORTS]
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


QUERY_SCOPES = ("gateway/api/routes", "gateway/services")
QUERY_LIBRARIES = ("sqlalchemy", "sqlmodel")
QUERY_PRIMITIVES = ("select", "insert", "update", "delete", "text")
# Modules that built a query when the rule landed. An entry that stops building
# one fails the check until it is removed, so the list only shrinks.
QUERY_BASELINE = (
    "gateway/api/routes/_helpers.py",
    "gateway/api/routes/agent_telemetry.py",
    "gateway/api/routes/aliases.py",
    "gateway/api/routes/budgets.py",
    "gateway/api/routes/catalog.py",
    "gateway/api/routes/files.py",
    "gateway/api/routes/health.py",
    "gateway/api/routes/keys.py",
    "gateway/api/routes/models.py",
    "gateway/api/routes/organization_keys.py",
    "gateway/api/routes/organization_routing.py",
    "gateway/api/routes/organization_usage.py",
    "gateway/api/routes/pricing.py",
    "gateway/api/routes/routing.py",
    "gateway/api/routes/routing_memory.py",
    "gateway/api/routes/scoped_budgets.py",
    "gateway/api/routes/usage.py",
    "gateway/api/routes/users.py",
    "gateway/services/alias_service.py",
    "gateway/services/batch_service.py",
    "gateway/services/bootstrap_service.py",
    "gateway/services/budget_reservation_ledger.py",
    "gateway/services/budget_retiming.py",
    "gateway/services/budget_service.py",
    "gateway/services/dashboard_session_service.py",
    "gateway/services/external_usage_service.py",
    "gateway/services/file_service.py",
    "gateway/services/maintenance_mode_service.py",
    "gateway/services/master_key_service.py",
    "gateway/services/merged_catalog_service.py",
    "gateway/services/model_access.py",
    "gateway/services/oauth_service.py",
    "gateway/services/organization_pricing_service.py",
    "gateway/services/playground_dispatch.py",
    "gateway/services/playground_service.py",
    "gateway/services/policy_store.py",
    "gateway/services/pricing_init_service.py",
    "gateway/services/pricing_refresh_service.py",
    "gateway/services/pricing_service.py",
    "gateway/services/provider_store_service.py",
    "gateway/services/routing/knn.py",
    "gateway/services/runtime_settings_service.py",
    "gateway/services/scoped_budget_service.py",
    "gateway/services/search_tool_store_service.py",
    "gateway/services/tenancy/org_provider_key_service.py",
    "gateway/services/tenancy/organization_budget_service.py",
    "gateway/services/tenancy/organization_guardrail_service.py",
    "gateway/services/tenancy/organization_model_access.py",
    "gateway/services/tenancy/provisioning_service.py",
    "gateway/services/tenancy/webauthn_service.py",
    "gateway/services/tenancy/workspace_activation_service.py",
    "gateway/services/tenancy/workspace_budget_default_service.py",
    "gateway/services/tenancy/workspace_mcp_server_service.py",
    "gateway/services/tenancy/workspace_service.py",
    "gateway/services/tool_settings_service.py",
    "gateway/services/usage_admin_service.py",
    "gateway/services/workspace_scope.py",
)


def _root_name(node: ast.expr) -> str | None:
    """Return the name an attribute chain starts from, or None when it starts from an expression."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _query_primitives(tree: ast.Module) -> list[tuple[int, str]]:
    """Return the line and name of each query-building primitive a module imports or references."""
    # An unaliased `import sqlalchemy.sql` binds the name `sqlalchemy`.
    library_names = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.split(".")[0] in QUERY_LIBRARIES
    } | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None
        if node.module.split(".")[0] in QUERY_LIBRARIES
        for alias in node.names
    }
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            if node.module.split(".")[0] in QUERY_LIBRARIES:
                found.extend((node.lineno, alias.name) for alias in node.names if alias.name in QUERY_PRIMITIVES)
        elif isinstance(node, ast.Attribute) and node.attr in QUERY_PRIMITIVES and _root_name(node) in library_names:
            found.append((node.lineno, node.attr))
    return sorted(found)


def check_query_layering(src_root: Path) -> list[str]:
    """Check that no route or service off the baseline builds a query, and that every baseline entry still does."""
    violations: list[str] = []
    querying: set[str] = set()
    for scope in QUERY_SCOPES:
        for py_file in sorted((src_root / scope).rglob("*.py")):
            relative_path = py_file.relative_to(src_root).as_posix()
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            except SyntaxError:
                continue  # check_file already reports an unparseable file.
            primitives = _query_primitives(tree)
            if not primitives:
                continue
            querying.add(relative_path)
            if relative_path not in QUERY_BASELINE:
                violations.extend(
                    f"{relative_path}:{line} builds a query with {name}; a query belongs in a repository"
                    for line, name in primitives
                )
    violations.extend(
        f"{relative_path} is on the query baseline but builds no query; remove it from the baseline"
        for relative_path in sorted(set(QUERY_BASELINE) - querying)
    )
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


def check_top_level_packages(src_root: Path) -> list[str]:
    """Check that src/ holds no importable package or module outside the allowed list."""
    violations: list[str] = []
    for entry in sorted(src_root.iterdir()):
        is_module = entry.is_file() and entry.suffix == ".py"
        is_package = entry.is_dir() and any(entry.rglob("*.py"))
        name = entry.stem if is_module else entry.name
        if (is_module or is_package) and name not in ALLOWED_TOP_LEVEL_PACKAGES:
            violations.append(
                f"Top-level package src/{entry.name} is not allowed; "
                "a feature in this repository belongs under src/gateway and in its feature registry"
            )
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
    package_violations = check_top_level_packages(SRC_ROOT)
    query_violations = check_query_layering(SRC_ROOT)

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

    if package_violations:
        print("\n❌ Top-level package violations:\n")
        for violation in package_violations:
            print(f"  {violation}")
        print(f"\nTotal top-level package violations: {len(package_violations)}")

    if query_violations:
        print("\n❌ Query layering violations:\n")
        for violation in query_violations:
            print(f"  {violation}")
        print(f"\nTotal query layering violations: {len(query_violations)}")

    if import_violations or naming_violations or package_violations or query_violations:
        print("\n💡 See ARCHITECTURE.md for the intended layering")
        return 1

    print("✅ No architecture violations found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
