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
11. Service database access: nothing under services/ imports sqlalchemy or
    sqlmodel, so a service can neither build a query nor name the session type.
12. Route database access: nothing under api/routes/ imports sqlalchemy or
    sqlmodel, so a route reaches the database through a service. Rules 11 and
    12 each name the modules that still import one on a baseline, and each
    baseline only shrinks.
13. Domain packages: services/ and repositories/ gain no top-level module, so
    new code goes in its domain's package, and every directory of modules in
    either layer has an __init__.py. The flat modules that exist are named on
    a baseline, and the baseline only shrinks.
14. Transaction control: only the Unit of Work calls commit or rollback, so a
    transaction ends where its block does. Modules that still call either are
    named on a baseline, and the baseline only shrinks.
15. Session accessor: only a repository imports session_for, so every query
    stays in the repository layer.
16. Schema boundaries: a schema does not import the API, service, repository or
    exception layer, or the composition root, so a request or response model
    carries no behavior from another layer.

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


# session_for hands out the Unit of Work's session, so the repositories package is
# the one place that may import it. The rule sits on the gateway root so that
# every layer answers to it; see check_file for the exemption.
SESSION_ACCESSOR_IMPORT = "gateway.core.unit_of_work.session_for"
SESSION_ACCESSOR_PACKAGE = "gateway/repositories/"
SESSION_ACCESSOR_RULE = "Repositories only (a query, and the session it runs on, stays in the repository layer)"


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
        "forbidden": ["gateway.overlay", "overlay", "gateway.adapters", SESSION_ACCESSOR_IMPORT],
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
        # Allowed only for the routes still in the old shape, which import
        # repository helpers such as get_active_user. A route in the target
        # shape calls its domain's service and imports no repository.
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
    "gateway/schemas": {
        "allowed": ["gateway.models"],
        "forbidden": [
            "gateway.api",
            "gateway.services",
            "gateway.repositories",
            "gateway.exceptions",
            "gateway.container",
        ],
        "description": "Schemas",
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
    forbidden = [
        (prefix, SESSION_ACCESSOR_RULE if prefix == SESSION_ACCESSOR_IMPORT else description)
        for prefix, description in forbidden
        if not (prefix == SESSION_ACCESSOR_IMPORT and relative_path.startswith(SESSION_ACCESSOR_PACKAGE))
    ]
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


DATABASE_LIBRARIES = ("sqlalchemy", "sqlmodel")
SERVICE_SCOPE = "gateway/services"
ROUTE_SCOPE = "gateway/api/routes"
# Modules that imported a database library when rules 11 and 12 landed. An
# entry that stops importing one fails the check until it is removed, so each
# list only shrinks.
SERVICE_DATABASE_IMPORT_BASELINE = (
    "gateway/services/agent_telemetry_service.py",
    "gateway/services/alias_service.py",
    "gateway/services/batch_service.py",
    "gateway/services/bootstrap_service.py",
    "gateway/services/budgets/_ledger.py",
    "gateway/services/budgets/_member_policies.py",
    "gateway/services/budgets/_reservations.py",
    "gateway/services/budgets/_retiming.py",
    "gateway/services/budgets/_scoped_enforcement.py",
    "gateway/services/content_normalizer.py",
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
    "gateway/services/search_tool_store_service.py",
    "gateway/services/selector_index_service.py",
    "gateway/services/tenancy/authorization.py",
    "gateway/services/tenancy/deployment_user_service.py",
    "gateway/services/tenancy/org_provider_key_service.py",
    "gateway/services/tenancy/organization_domain_service.py",
    "gateway/services/tenancy/organization_guardrail_service.py",
    "gateway/services/tenancy/organization_model_access.py",
    "gateway/services/tenancy/organization_service.py",
    "gateway/services/tenancy/provisioning_service.py",
    "gateway/services/tenancy/user_service.py",
    "gateway/services/tenancy/webauthn_service.py",
    "gateway/services/tenancy/workspace_activation_service.py",
    "gateway/services/tenancy/workspace_code_execution_policy_service.py",
    "gateway/services/tenancy/workspace_mcp_server_service.py",
    "gateway/services/tenancy/workspace_service.py",
    "gateway/services/tenancy/workspace_web_search_service.py",
    "gateway/services/tool_settings_service.py",
    "gateway/services/usage_admin_service.py",
    "gateway/services/workspace_scope.py",
)
ROUTE_DATABASE_IMPORT_BASELINE = (
    "gateway/api/routes/_helpers.py",
    "gateway/api/routes/_normalize.py",
    "gateway/api/routes/_passthrough.py",
    "gateway/api/routes/_pipeline.py",
    "gateway/api/routes/admin.py",
    "gateway/api/routes/agent_telemetry.py",
    "gateway/api/routes/aliases.py",
    "gateway/api/routes/audio.py",
    "gateway/api/routes/auth_oauth.py",
    "gateway/api/routes/auth_password.py",
    "gateway/api/routes/auth_password_reset.py",
    "gateway/api/routes/auth_profile.py",
    "gateway/api/routes/auth_session.py",
    "gateway/api/routes/auth_signup.py",
    "gateway/api/routes/auth_webauthn.py",
    "gateway/api/routes/batches.py",
    "gateway/api/routes/bootstrap.py",
    "gateway/api/routes/budgets.py",
    "gateway/api/routes/catalog.py",
    "gateway/api/routes/chat.py",
    "gateway/api/routes/embeddings.py",
    "gateway/api/routes/files.py",
    "gateway/api/routes/health.py",
    "gateway/api/routes/hooks.py",
    "gateway/api/routes/images.py",
    "gateway/api/routes/invitations.py",
    "gateway/api/routes/keys.py",
    "gateway/api/routes/maintenance_mode.py",
    "gateway/api/routes/mcp.py",
    "gateway/api/routes/messages.py",
    "gateway/api/routes/models.py",
    "gateway/api/routes/moderations.py",
    "gateway/api/routes/org_provider_keys.py",
    "gateway/api/routes/organization_guardrails.py",
    "gateway/api/routes/organization_keys.py",
    "gateway/api/routes/organization_pricing.py",
    "gateway/api/routes/organization_routing.py",
    "gateway/api/routes/organization_usage.py",
    "gateway/api/routes/organizations.py",
    "gateway/api/routes/otlp.py",
    "gateway/api/routes/playground.py",
    "gateway/api/routes/pricing.py",
    "gateway/api/routes/providers.py",
    "gateway/api/routes/rerank.py",
    "gateway/api/routes/responses.py",
    "gateway/api/routes/routing.py",
    "gateway/api/routes/routing_memory.py",
    "gateway/api/routes/scoped_budgets.py",
    "gateway/api/routes/search.py",
    "gateway/api/routes/search_tools.py",
    "gateway/api/routes/settings.py",
    "gateway/api/routes/tool_settings.py",
    "gateway/api/routes/usage.py",
    "gateway/api/routes/users.py",
    "gateway/api/routes/workspace_activation.py",
    "gateway/api/routes/workspace_code_execution_policy.py",
    "gateway/api/routes/workspace_mcp_servers.py",
    "gateway/api/routes/workspace_member_budget_policies.py",
    "gateway/api/routes/workspace_web_search.py",
    "gateway/api/routes/workspaces.py",
)


def _database_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return the line and module of each import of a database library in a module."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            modules = [node.module]
        else:
            continue
        found.extend(
            (node.lineno, module)
            for module in modules
            if any(_matches(module, library) for library in DATABASE_LIBRARIES)
        )
    return sorted(found)


def _check_layer_database_imports(src_root: Path, scope: str, baseline: tuple[str, ...], remedy: str) -> list[str]:
    """Check one layer against its database import baseline, reporting new importers and stale entries."""
    violations: list[str] = []
    importing: set[str] = set()
    for py_file in sorted((src_root / scope).rglob("*.py")):
        relative_path = py_file.relative_to(src_root).as_posix()
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue  # check_file already reports an unparseable file.
        imports = _database_imports(tree)
        if not imports:
            continue
        importing.add(relative_path)
        if relative_path not in baseline:
            violations.extend(f"{relative_path}:{line} imports {module}; {remedy}" for line, module in imports)
    violations.extend(
        f"{relative_path} is on the database import baseline but imports no database library; "
        "remove it from the baseline"
        for relative_path in sorted(set(baseline) - importing)
    )
    return violations


def check_database_imports(src_root: Path) -> list[str]:
    """Check that no service or route off its layer's baseline imports a database library."""
    return [
        *_check_layer_database_imports(
            src_root,
            SERVICE_SCOPE,
            SERVICE_DATABASE_IMPORT_BASELINE,
            "a service reaches the database through its repositories and the Unit of Work",
        ),
        *_check_layer_database_imports(
            src_root, ROUTE_SCOPE, ROUTE_DATABASE_IMPORT_BASELINE, "a route reaches the database through a service"
        ),
    ]


TRANSACTION_CALLS = ("commit", "rollback")
UNIT_OF_WORK = "gateway/core/unit_of_work.py"
# Modules that ended a transaction themselves when rule 14 landed. An entry
# that stops calling commit and rollback fails the check until it is removed,
# so the list only shrinks.
TRANSACTION_CONTROL_BASELINE = (
    "gateway/adapters/telemetry_storage_adapter.py",
    "gateway/api/deps.py",
    "gateway/api/routes/_passthrough.py",
    "gateway/api/routes/_pipeline.py",
    "gateway/api/routes/aliases.py",
    "gateway/api/routes/auth_oauth.py",
    "gateway/api/routes/auth_session.py",
    "gateway/api/routes/auth_webauthn.py",
    "gateway/api/routes/batches.py",
    "gateway/api/routes/budgets.py",
    "gateway/api/routes/files.py",
    "gateway/api/routes/keys.py",
    "gateway/api/routes/maintenance_mode.py",
    "gateway/api/routes/organization_keys.py",
    "gateway/api/routes/organization_pricing.py",
    "gateway/api/routes/pricing.py",
    "gateway/api/routes/providers.py",
    "gateway/api/routes/routing.py",
    "gateway/api/routes/routing_memory.py",
    "gateway/api/routes/scoped_budgets.py",
    "gateway/api/routes/search_tools.py",
    "gateway/api/routes/settings.py",
    "gateway/api/routes/tool_settings.py",
    "gateway/api/routes/users.py",
    "gateway/core/database.py",
    "gateway/services/batch_service.py",
    "gateway/services/bootstrap_service.py",
    "gateway/services/budgets/_ledger.py",
    "gateway/services/budgets/_member_policies.py",
    "gateway/services/budgets/_reservations.py",
    "gateway/services/budgets/_scoped_enforcement.py",
    "gateway/services/dashboard_session_service.py",
    "gateway/services/external_usage_service.py",
    "gateway/services/log_writer.py",
    "gateway/services/master_key_service.py",
    "gateway/services/organization_pricing_service.py",
    "gateway/services/playground_dispatch.py",
    "gateway/services/playground_service.py",
    "gateway/services/pricing_init_service.py",
    "gateway/services/pricing_refresh_service.py",
    "gateway/services/routing/knn.py",
    "gateway/services/tenancy/deployment_user_service.py",
    "gateway/services/tenancy/org_provider_key_service.py",
    "gateway/services/tenancy/organization_domain_service.py",
    "gateway/services/tenancy/organization_guardrail_service.py",
    "gateway/services/tenancy/organization_service.py",
    "gateway/services/tenancy/provisioning_service.py",
    "gateway/services/tenancy/user_service.py",
    "gateway/services/tenancy/webauthn_service.py",
    "gateway/services/tenancy/workspace_activation_service.py",
    "gateway/services/tenancy/workspace_code_execution_policy_service.py",
    "gateway/services/tenancy/workspace_mcp_server_service.py",
    "gateway/services/tenancy/workspace_service.py",
    "gateway/services/tenancy/workspace_web_search_service.py",
    "gateway/services/usage_admin_service.py",
)


def _transaction_calls(tree: ast.Module) -> list[tuple[int, str]]:
    """Return the line and name of each commit or rollback call in a module."""
    return sorted(
        (node.lineno, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in TRANSACTION_CALLS
    )


def check_transaction_control(src_root: Path) -> list[str]:
    """Check that no module off the baseline ends a transaction itself, and that every baseline entry still does."""
    violations: list[str] = []
    ending: set[str] = set()
    for py_file in sorted((src_root / "gateway").rglob("*.py")):
        relative_path = py_file.relative_to(src_root).as_posix()
        if relative_path == UNIT_OF_WORK or "__pycache__" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue  # check_file already reports an unparseable file.
        calls = _transaction_calls(tree)
        if not calls:
            continue
        ending.add(relative_path)
        if relative_path not in TRANSACTION_CONTROL_BASELINE:
            violations.extend(
                f"{relative_path}:{line} calls {call}; only a Unit of Work block ends a transaction"
                for line, call in calls
            )
    violations.extend(
        f"{relative_path} is on the transaction control baseline but calls neither commit nor rollback; "
        "remove it from the baseline"
        for relative_path in sorted(set(TRANSACTION_CONTROL_BASELINE) - ending)
    )
    return violations


# Service modules are purpose-named (guardrails.py, url_safety.py, ...), so
# there is no *_service.py naming rule to enforce.
def check_naming_conventions(src_root: Path) -> list[str]:
    """Check that repository modules follow the *_repository.py convention.

    A module that bundles a domain's repositories ends in _repositories.py instead.
    """
    violations: list[str] = []
    repositories_path = src_root / "gateway" / "repositories"
    if not repositories_path.is_dir():
        return violations
    for repository_file in sorted(repositories_path.rglob("*.py")):
        if repository_file.name == "__init__.py":
            continue
        if not repository_file.name.endswith(("_repository.py", "_repositories.py")):
            violations.append(
                f"Repository file {repository_file.relative_to(src_root)} must end with '_repository.py'"
                " or, for a bundle of repositories, '_repositories.py'"
            )
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


DOMAIN_PACKAGE_LAYERS = ("gateway/services", "gateway/repositories")
# The flat modules the domain layers held when rule 13 landed. An entry whose
# module no longer exists fails the check until it is removed.
FLAT_MODULE_BASELINE = (
    "gateway/repositories/base_repository.py",
    "gateway/repositories/users_repository.py",
    "gateway/services/_tool_loop.py",
    "gateway/services/agent_telemetry_admin_service.py",
    "gateway/services/agent_telemetry_service.py",
    "gateway/services/alias_service.py",
    "gateway/services/batch_service.py",
    "gateway/services/bedrock_gateway_auth.py",
    "gateway/services/bootstrap_service.py",
    "gateway/services/catalog_selectors.py",
    "gateway/services/claude_code_import.py",
    "gateway/services/content_normalizer.py",
    "gateway/services/dashboard_session_service.py",
    "gateway/services/external_usage_service.py",
    "gateway/services/file_extractors.py",
    "gateway/services/file_service.py",
    "gateway/services/file_store.py",
    "gateway/services/guardrail_catalog.py",
    "gateway/services/guardrails.py",
    "gateway/services/log_writer.py",
    "gateway/services/maintenance_mode_service.py",
    "gateway/services/master_key_service.py",
    "gateway/services/mcp_client.py",
    "gateway/services/mcp_loop.py",
    "gateway/services/mcp_loop_messages.py",
    "gateway/services/mcp_loop_responses.py",
    "gateway/services/mcp_stateless.py",
    "gateway/services/merged_catalog_service.py",
    "gateway/services/model_access.py",
    "gateway/services/model_capabilities.py",
    "gateway/services/model_catalog_service.py",
    "gateway/services/model_discovery_service.py",
    "gateway/services/model_identity.py",
    "gateway/services/oauth_service.py",
    "gateway/services/organization_pricing_service.py",
    "gateway/services/password_service.py",
    "gateway/services/playground_dispatch.py",
    "gateway/services/playground_service.py",
    "gateway/services/policy_store.py",
    "gateway/services/pricing_init_service.py",
    "gateway/services/pricing_refresh_service.py",
    "gateway/services/pricing_service.py",
    "gateway/services/provider_health_service.py",
    "gateway/services/provider_kwargs.py",
    "gateway/services/provider_metadata_service.py",
    "gateway/services/provider_store_service.py",
    "gateway/services/runtime_settings_service.py",
    "gateway/services/sandbox_backend.py",
    "gateway/services/search_backend.py",
    "gateway/services/search_tool_store_service.py",
    "gateway/services/secret_box.py",
    "gateway/services/selector_index_service.py",
    "gateway/services/tool_format.py",
    "gateway/services/tool_settings_service.py",
    "gateway/services/tool_usage.py",
    "gateway/services/upstream_redaction.py",
    "gateway/services/url_safety.py",
    "gateway/services/usage_admin_service.py",
    "gateway/services/vision.py",
    "gateway/services/web_extraction.py",
    "gateway/services/web_fetch_service.py",
    "gateway/services/web_retrieval_backend.py",
    "gateway/services/web_retrieval_network.py",
    "gateway/services/web_retrieval_policy.py",
    "gateway/services/web_search_backend.py",
    "gateway/services/web_search_budget.py",
    "gateway/services/web_search_providers.py",
    "gateway/services/workspace_scope.py",
)


def check_flat_modules(src_root: Path) -> list[str]:
    """Check that the domain layers hold no new top-level module and no directory of modules without an __init__.py."""
    violations: list[str] = []
    for layer in DOMAIN_PACKAGE_LAYERS:
        layer_root = src_root / layer
        if not layer_root.is_dir():
            continue
        for entry in sorted(layer_root.iterdir()):
            relative_path = entry.relative_to(src_root).as_posix()
            is_module = entry.is_file() and entry.suffix == ".py" and entry.name != "__init__.py"
            if is_module and relative_path not in FLAT_MODULE_BASELINE:
                violations.append(
                    f"{relative_path} is a new top-level module; put it in its domain's package under {layer}/"
                )
        directories = sorted(
            path for path in layer_root.rglob("*") if path.is_dir() and "__pycache__" not in path.parts
        )
        violations.extend(
            f"{directory.relative_to(src_root).as_posix()} has no __init__.py; a domain package needs one"
            for directory in directories
            if not (directory / "__init__.py").is_file() and any(directory.rglob("*.py"))
        )
    violations.extend(
        f"{relative_path} is on the flat module baseline but no longer exists; remove it from the baseline"
        for relative_path in FLAT_MODULE_BASELINE
        if not (src_root / relative_path).is_file()
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
    flat_module_violations = check_flat_modules(SRC_ROOT)
    database_violations = check_database_imports(SRC_ROOT)
    transaction_violations = check_transaction_control(SRC_ROOT)

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

    if database_violations:
        print("\n❌ Database import violations:\n")
        for violation in database_violations:
            print(f"  {violation}")
        print(f"\nTotal database import violations: {len(database_violations)}")

    if transaction_violations:
        print("\n❌ Transaction control violations:\n")
        for violation in transaction_violations:
            print(f"  {violation}")
        print(f"\nTotal transaction control violations: {len(transaction_violations)}")

    if flat_module_violations:
        print("\n❌ Flat module violations:\n")
        for violation in flat_module_violations:
            print(f"  {violation}")
        print(f"\nTotal flat module violations: {len(flat_module_violations)}")

    if (
        import_violations
        or naming_violations
        or package_violations
        or database_violations
        or transaction_violations
        or flat_module_violations
    ):
        print("\n💡 See ARCHITECTURE.md for the intended layering")
        return 1

    print("✅ No architecture violations found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
