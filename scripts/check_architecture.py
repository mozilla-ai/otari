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
   discovered. The same ban holds under otari_agent/ and the library.
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
17. Unit of Work construction: only the request's factory builds one, outside
    the module that defines it and holds the worker factories, so a scope has
    exactly one and an inner block still joins the outer one.
18. Light CLI: nothing under cli/src/otari_agent imports the gateway or the
    server stack it drags in (uvicorn, any-llm, SQLAlchemy, pydantic,
    FastAPI), so the `otari` command ships alone with a handful of pure-Python
    dependencies. The one exemption is otari_agent/cli.py, which names
    gateway.cli inside a find_spec guard to attach the server commands when
    the gateway is installed too. The member lives outside src/ on purpose:
    rule 10 keeps src/ to the gateway, and this one keeps the CLI out of it.
19. Domain names: a package in services/ or repositories/, a module in
    schemas/ and a module in exceptions/ take their domain from their name, so
    each name is a domain that DOMAINS.md gives a section. An exceptions
    module is named <domain>_exceptions.py. The names that do not match yet are
    on a baseline, and the baseline only shrinks.
    A package on it holds only the modules the baseline lists, so new code goes in its domain's package.
20. Repository imports: only a domain's own service package, its own
    repository package and the builders in gateway/api/deps.py import a
    domain's repository package, so a domain's queries stay behind its
    service. A domain is one that DOMAINS.md gives a section. Service code
    outside every domain package is not checked, because its path does not
    say which domain owns it.
21. Service package imports: code outside a domain's service package
    imports only its root, so the root's exports are the domain's whole
    public API.
22. Mode branches: nothing under services/ reads the deployment's mode,
    so the behavior for each mode is chosen by a binding, not by a branch in a service.
    A read is any use of is_hosted_mode, is_hybrid_mode, effective_mode or configured_mode, or a call to deployment_for.
    The raw mode field and the platform token stay with review, because their names do not say that a mode is read.
    The modules that still read one are named on a baseline, and the baseline only shrinks.
23. Model access: each ORM model has one repository module that may construct or query it,
    so a table has one writer and its queries stay in one place.
    A model has none until a repository holds its queries.
    The other modules that use a model are named on its baseline, and the baseline only shrinks.
    A class in models/ is an ORM model when it passes table=True or assigns __tablename__ or __table__.
24. Listener defaults: no parameter or class field typed as a listener has a default,
    so a caller cannot skip a listener by leaving it out.
    A listener type is one whose name contains Listener.
    Containing it, rather than ending in it, also catches an alias such as WorkspaceListenerDep.
    A name in a Callable's parameter list or in Annotated's metadata is not the parameter's type, so it is skipped.
    A FastAPI dependency declares a listener as Annotated[Listener, Depends(...)], since a Depends(...) default counts.
    A field kept out of the constructor, such as field(init=False), is not a parameter, so the rule skips it.
    The parameters and fields that still have one are named on a baseline, and the baseline only shrinks.
25. Libraries: nothing under any-search/, tests and scripts included,
    imports the gateway, SQLAlchemy, SQLModel, FastAPI, uvicorn or the
    fetch library, so it runs on its own with httpx and pydantic. A
    library keeps its tests and scripts inside its directory rather than
    under tests/ or src/, so each directory is a root of its own, mapped to
    its rule. Entry-point discovery is banned there as under gateway/.

Usage:
    uv run python scripts/check_architecture.py

Exit codes:
    0 - No violations
    1 - Violations found
"""

import ast
import re
import sys
from collections.abc import Callable, Container, Iterator
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
GATEWAY_ROOT = SRC_ROOT / "gateway"
TESTS_ROOT = REPO_ROOT / "tests"
# The otari-agent workspace member (cli/pyproject.toml); its files are checked
# relative to this root so the "otari_agent" rule key matches them.
CLI_ROOT = REPO_ROOT / "cli" / "src"
# The library workspace members, each checked whole against the rule it maps to.
LIBRARY_ROOTS: dict[str, Path] = {
    "any_search": REPO_ROOT / "any-search",
}


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
    # The laptop CLI ships on its own (Homebrew) with a dozen pure-Python
    # dependencies, so it may not reach the server stack. gateway is the whole
    # reason the split exists; the rest are what gateway.core.config drags in.
    "otari_agent": {
        "allowed": [],
        "forbidden": [
            "gateway",
            "uvicorn",
            "any_llm",
            "sqlalchemy",
            "sqlmodel",
            "pydantic",
            "pydantic_settings",
            "fastapi",
        ],
        "description": "Light CLI (otari-agent)",
    },
    # The web search library is built to run without otari: import it, give it
    # a key, call it. It does not import the fetch library either, so each
    # stays self-contained. These keys match through LIBRARY_ROOTS, not a path.
    "any_search": {
        "allowed": [],
        "forbidden": ["gateway", "sqlalchemy", "sqlmodel", "fastapi", "uvicorn", "any_fetch"],
        "description": "Library (any-search)",
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

# The one light-CLI module that may name gateway.cli, and nothing else of the
# gateway: it attaches the server commands when
# `importlib.util.find_spec("gateway")` finds one installed.
LIGHT_CLI_ATTACH_POINT = "otari_agent/cli.py"
LIGHT_CLI_ATTACH_IMPORT = "gateway.cli"

# Entry-point discovery is banned everywhere under gateway/, otari_agent/ and
# the library, with a message of its own because "OSS base" would not say
# why: the feature registry in gateway/features.py is a literal tuple on purpose
# (ARCHITECTURE.md), as is each library's provider table, and these are the
# modules discovery is written with.
DISCOVERY_SCOPE = ("gateway/", "otari_agent/", "any_search/")
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


def check_file(file_path: Path, src_root: Path, rule_key: str | None = None) -> list[tuple[int, str, str]]:
    """Check one Python file below src_root against the layer rules for its location.

    A library's files pass the rule key their root maps to (LIBRARY_ROOTS), and
    are checked as if their path below the root started with it.

    Returns:
        One (line number, module, message) tuple per offending import statement.

    """
    relative_path = file_path.relative_to(src_root).as_posix()
    if rule_key is not None:
        relative_path = f"{rule_key}/{relative_path}"
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
            if relative_path == LIGHT_CLI_ATTACH_POINT and _matches(module, LIGHT_CLI_ATTACH_IMPORT):
                continue
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
    "gateway/services/budgets/_reservations.py",
    "gateway/services/budgets/_retiming.py",
    "gateway/services/budgets/_scoped_enforcement.py",
    "gateway/services/dashboard_session_service.py",
    "gateway/services/external_usage_service.py",
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
    "gateway/services/tenancy/workspace_mcp_server_service.py",
    "gateway/services/tenancy/workspace_service.py",
    "gateway/services/tenancy/workspace_web_search_service.py",
    "gateway/services/tool_settings_service.py",
    "gateway/services/usage_admin_service.py",
    "gateway/services/workspace_scope.py",
)
ROUTE_DATABASE_IMPORT_BASELINE = (
    "gateway/api/routes/_helpers.py",
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
    "gateway/api/routes/workspace_mcp_servers.py",
    "gateway/api/routes/workspace_web_search.py",
    "gateway/api/routes/workspaces.py",
)


def _parsed_modules(src_root: Path, scope: str, exempt: tuple[str, ...] = ()) -> Iterator[tuple[str, ast.Module]]:
    """Yield the path relative to src_root and the tree of each parseable module under scope, in path order."""
    for py_file in sorted((src_root / scope).rglob("*.py")):
        relative_path = py_file.relative_to(src_root).as_posix()
        if relative_path in exempt or "__pycache__" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue  # check_file already reports an unparseable file.
        yield relative_path, tree


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
    for relative_path, tree in _parsed_modules(src_root, scope):
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
    "gateway/services/tenancy/user_service.py",
    "gateway/services/tenancy/webauthn_service.py",
    "gateway/services/tenancy/workspace_activation_service.py",
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
    for relative_path, tree in _parsed_modules(src_root, "gateway", exempt=(UNIT_OF_WORK,)):
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


UNIT_OF_WORK_TYPE = "UnitOfWork"
UNIT_OF_WORK_MODULE = "gateway.core.unit_of_work"
UNIT_OF_WORK_FACTORY = "gateway/api/deps.py"
UNIT_OF_WORK_FACTORY_FUNCTION = "get_unit_of_work"
UNIT_OF_WORK_WORKER_FACTORIES = "create_unit_of_work or create_log_unit_of_work"


def _called_name(func: ast.expr) -> str | None:
    """Return the name being called, whether it was imported or reached through its module."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _unit_of_work_constructions(node: ast.AST) -> list[ast.Call]:
    """Return each Unit of Work construction under a node, in source order."""
    calls = (
        call for call in ast.walk(node) if isinstance(call, ast.Call) and _called_name(call.func) == UNIT_OF_WORK_TYPE
    )
    return sorted(calls, key=lambda call: call.lineno)


def _factory_constructions(tree: ast.Module) -> set[ast.Call]:
    """Return the constructions inside the module-level function that is the request's factory."""
    return {
        call
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == UNIT_OF_WORK_FACTORY_FUNCTION
        for call in _unit_of_work_constructions(node)
    }


def _renamed_unit_of_work_imports(tree: ast.Module, file_path: Path, src_root: Path) -> list[tuple[int, str]]:
    """Return the line and new name of each import that renames the Unit of Work.

    A relative import is resolved first, because it names the module by a path relative to its own package.
    """
    return sorted(
        (node.lineno, alias.asname)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module if node.level == 0 else _resolve_relative(node, file_path, src_root)) == UNIT_OF_WORK_MODULE
        for alias in node.names
        if alias.name == UNIT_OF_WORK_TYPE and alias.asname is not None
    )


def check_unit_of_work_construction(src_root: Path) -> list[str]:
    """Check that only the request's factory constructs a Unit of Work.

    The module that defines the Unit of Work is exempt, because the worker factories live there.
    A Unit of Work counts its open blocks on itself.
    A second one over the same session therefore cannot see a block the first has open,
    and the rule that an inner block joins the outer one no longer holds between them.
    Renaming or moving the factory is caught as well, because its own construction then has no exemption.

    Gotcha: the rule reads the name at the call site.
    An import that renames the Unit of Work would hide a construction from it, so such an import is refused.
    """
    violations: list[str] = []
    for relative_path, tree in _parsed_modules(src_root, "gateway", exempt=(UNIT_OF_WORK,)):
        allowed = _factory_constructions(tree) if relative_path == UNIT_OF_WORK_FACTORY else set()
        violations.extend(
            f"{relative_path}:{call.lineno} constructs a {UNIT_OF_WORK_TYPE}; a request takes one from "
            f"{UNIT_OF_WORK_FACTORY_FUNCTION} and a worker job from {UNIT_OF_WORK_WORKER_FACTORIES}"
            for call in _unit_of_work_constructions(tree)
            if call not in allowed
        )
        violations.extend(
            f"{relative_path}:{line} imports {UNIT_OF_WORK_TYPE} as {name}; the rule reads the name at the "
            "call site, so import it under its own name"
            for line, name in _renamed_unit_of_work_imports(tree, src_root / relative_path, src_root)
        )
    return violations


MODE_READS = ("configured_mode", "effective_mode", "is_hosted_mode", "is_hybrid_mode")
MODE_FUNCTION = "deployment_for"
# The services that still read the deployment's mode.
# An entry that stops reading it fails the check until it is removed, so the list only shrinks.
SERVICE_MODE_READ_BASELINE = ("gateway/services/provider_kwargs.py",)


def _mode_reads(tree: ast.Module) -> list[tuple[int, str]]:
    """Return the line and name of each read of the deployment's mode in a module, including one through getattr."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in MODE_READS:
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.Call) and _called_name(node.func) == MODE_FUNCTION:
            found.append((node.lineno, MODE_FUNCTION))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) > 1
        ):
            name = node.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str) and name.value in MODE_READS:
                found.append((node.lineno, name.value))
    return sorted(found)


def check_service_mode_reads(src_root: Path) -> list[str]:
    """Check that no service off the baseline reads the deployment's mode, and that every baseline entry still does."""
    violations: list[str] = []
    reading: set[str] = set()
    for relative_path, tree in _parsed_modules(src_root, SERVICE_SCOPE):
        reads = _mode_reads(tree)
        if not reads:
            continue
        reading.add(relative_path)
        if relative_path not in SERVICE_MODE_READ_BASELINE:
            violations.extend(
                f"{relative_path}:{line} reads {name}; a service gets the behavior for each mode from a binding, "
                "not from a branch on the mode"
                for line, name in reads
            )
    violations.extend(
        f"{relative_path} is on the mode read baseline but reads no mode; remove it from the baseline"
        for relative_path in sorted(set(SERVICE_MODE_READ_BASELINE) - reading)
    )
    return violations


LISTENER_TYPE_MARKER = "Listener"
# A Literal holds values rather than types, and a ClassVar is not a constructor parameter.
LISTENER_SKIPPED_SUBSCRIPTS = ("ClassVar", "Literal")
# Only the first argument of Annotated is a type. The rest is metadata.
LISTENER_METADATA_SUBSCRIPTS = ("Annotated",)
# A Callable that returns a listener builds one, and a Callable that takes one only consumes it.
LISTENER_CALLABLE_SUBSCRIPTS = ("Callable",)
# These constructors match by name, because no module under gateway/ binds the names to anything else.
DATACLASS_FIELD_CONSTRUCTOR = "field"
PYDANTIC_FIELD_CONSTRUCTOR = "Field"
PYDANTIC_PRIVATE_ATTRIBUTE_CONSTRUCTOR = "PrivateAttr"
FIELD_DEFAULT_KEYWORDS = ("default", "default_factory")
# The listener parameters and fields that still have a default, as (module, path, name).
# The path is the dotted chain of enclosing class and function names, not Python's __qualname__.
# An entry whose default is gone fails the check until it is removed, so the list only shrinks.
LISTENER_DEFAULT_BASELINE: tuple[tuple[str, str, str], ...] = (
    ("gateway/container.py", "build_container", "membership_listener"),
)


def _names_a_listener(annotation: ast.expr) -> bool:
    """Return whether a type annotation names a listener type, including in a quoted forward reference."""
    if isinstance(annotation, ast.Constant):
        if not isinstance(annotation.value, str):
            return False
        try:
            return _names_a_listener(ast.parse(annotation.value, mode="eval").body)
        except (SyntaxError, ValueError):
            return False
    if isinstance(annotation, ast.Subscript):
        subscript = _called_name(annotation.value)
        if subscript in LISTENER_SKIPPED_SUBSCRIPTS:
            return False
        if subscript in LISTENER_METADATA_SUBSCRIPTS:
            arguments = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            return bool(arguments) and _names_a_listener(arguments[0])
        if subscript in LISTENER_CALLABLE_SUBSCRIPTS:
            arguments = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
            return bool(arguments) and _names_a_listener(arguments[-1])
    name = _called_name(annotation)
    if name is not None and LISTENER_TYPE_MARKER in name:
        return True
    # NOTE: An attribute chain names the type by its last name, so a type nested in a listener is not one.
    if isinstance(annotation, ast.Attribute):
        return False
    return any(_names_a_listener(child) for child in ast.iter_child_nodes(annotation) if isinstance(child, ast.expr))


def _defaulted_parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.arg]:
    """Yield each parameter of a function that has a default."""
    positional = [*function.args.posonlyargs, *function.args.args]
    yield from positional[len(positional) - len(function.args.defaults) :]
    yield from (
        parameter
        for parameter, default in zip(function.args.kwonlyargs, function.args.kw_defaults, strict=True)
        if default is not None
    )


def _is_pydantic_missing(value: ast.expr, keyword: str) -> bool:
    """Return whether a default argument to Pydantic's Field leaves the field required."""
    # Pydantic marks a required field with a default of ..., and a default_factory of None means no factory.
    missing = None if keyword == "default_factory" else Ellipsis
    return isinstance(value, ast.Constant) and value.value is missing


def _supplies_default(value: ast.expr) -> bool:
    """Return whether a class field's value gives the constructor a default.

    A call to a dataclass or Pydantic field constructor supplies one only through a default argument,
    and never when the field is left out of the constructor.
    An unpacked mapping counts as a default argument, because it may carry one.
    """
    if not isinstance(value, ast.Call):
        return True
    constructor = _called_name(value.func)
    if constructor == PYDANTIC_PRIVATE_ATTRIBUTE_CONSTRUCTOR:
        return False
    if constructor == DATACLASS_FIELD_CONSTRUCTOR:
        if any(
            keyword.arg == "init" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            for keyword in value.keywords
        ):
            return False
        return any(keyword.arg is None or keyword.arg in FIELD_DEFAULT_KEYWORDS for keyword in value.keywords)
    if constructor == PYDANTIC_FIELD_CONSTRUCTOR:
        if any(
            keyword.arg is None
            or (keyword.arg in FIELD_DEFAULT_KEYWORDS and not _is_pydantic_missing(keyword.value, keyword.arg))
            for keyword in value.keywords
        ):
            return True
        return bool(value.args) and not _is_pydantic_missing(value.args[0], "default")
    return True


def _listener_defaults(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Return the line, path and name of each listener parameter or class field with a default in a module.

    A class field counts because a dataclass or a Pydantic model turns its default into a constructor default.
    """
    found: list[tuple[int, str, str]] = []

    def visit(node: ast.AST, path: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                found.extend(
                    (field.lineno, f"{path}{child.name}", field.target.id)
                    for field in child.body
                    if isinstance(field, ast.AnnAssign)
                    and isinstance(field.target, ast.Name)
                    and field.value is not None
                    and _supplies_default(field.value)
                    and _names_a_listener(field.annotation)
                )
                visit(child, f"{path}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                function = f"{path}{child.name}"
                found.extend(
                    (parameter.lineno, function, parameter.arg)
                    for parameter in _defaulted_parameters(child)
                    if parameter.annotation is not None and _names_a_listener(parameter.annotation)
                )
                visit(child, f"{function}.")
            else:
                visit(child, path)

    visit(tree, "")
    return sorted(found)


def check_listener_defaults(src_root: Path) -> list[str]:
    """Check that no listener off the baseline has a default, and that every baseline entry still has one."""
    violations: list[str] = []
    defaulted: set[tuple[str, str, str]] = set()
    for relative_path, tree in _parsed_modules(src_root, "gateway"):
        for line, path, name in _listener_defaults(tree):
            defaulted.add((relative_path, path, name))
            if (relative_path, path, name) not in LISTENER_DEFAULT_BASELINE:
                violations.append(
                    f"{relative_path}:{line} gives the listener {name} of {path} a default; "
                    "a caller that leaves a listener out skips it with no error, so every caller passes one"
                )
    violations.extend(
        f"{relative_path} is on the listener default baseline for {name} of {path}, "
        "but no such listener has a default; remove it from the baseline"
        for relative_path, path, name in sorted(set(LISTENER_DEFAULT_BASELINE) - defaulted)
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
    "gateway/services/content_normalizer.py",
    "gateway/services/dashboard_session_service.py",
    "gateway/services/external_usage_service.py",
    "gateway/services/file_extractors.py",
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


DOMAINS_DOC = "DOMAINS.md"
DOMAINS_SECTION = "## The domains"
DOMAIN_HEADING = re.compile(r"^### (.*)$", re.MULTILINE)
DOMAIN_NAME = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
SHARED_HEADING = "Shared"
# These names do not match a domain yet. The baseline only shrinks, so a reviewer refuses a new entry.
# A package's entry names every module it holds, so a new module in it is refused.
DOMAIN_NAME_BASELINE: dict[str, tuple[str, ...]] = {
    "gateway/exceptions/budget_exceptions.py": (),
    "gateway/exceptions/control_plane_exceptions.py": (),
    "gateway/repositories/code_execution/": (
        "__init__.py",
        "sandbox_container_repository.py",
    ),
    "gateway/repositories/tenancy/": (
        "__init__.py",
        "invitation_repository.py",
        "org_provider_key_repository.py",
        "organization_domain_repository.py",
        "organization_guardrail_definition_repository.py",
        "organization_member_repository.py",
        "organization_repository.py",
        "user_repository.py",
        "workspace_repository.py",
    ),
    "gateway/services/code_execution/": (
        "__init__.py",
        "container_sweeper.py",
        "containers.py",
    ),
    "gateway/services/control_plane/": (
        "__init__.py",
        "_resolve.py",
        "transport.py",
    ),
    "gateway/services/mail/": (
        "__init__.py",
        "mailer.py",
        "message.py",
        "templates.py",
        "transports.py",
    ),
    "gateway/services/tenancy/": (
        "__init__.py",
        "authorization.py",
        "deployment_user_service.py",
        "domain_verification.py",
        "email_address.py",
        "invitation_email.py",
        "membership_listener.py",
        "org_provider_key_service.py",
        "organization_domain_service.py",
        "organization_guardrail_definition_service.py",
        "organization_guardrail_runner.py",
        "organization_guardrail_service.py",
        "organization_model_access.py",
        "organization_service.py",
        "password_policy.py",
        "password_reset_email.py",
        "provisioning_service.py",
        "tokens.py",
        "user_service.py",
        "verification_email.py",
        "webauthn_service.py",
        "workspace_activation_service.py",
        "workspace_listener.py",
        "workspace_mcp_server_service.py",
        "workspace_service.py",
        "workspace_web_search_service.py",
    ),
}
# These modules belong to no domain. The set grows when the shared set does.
SHARED_EXCEPTION_MODULES = ("gateway/exceptions/_base.py", "gateway/exceptions/shared_exceptions.py")


def documented_domains(doc_text: str) -> tuple[set[str], list[str]]:
    """Return the domains the page gives a section, and each problem with its headings."""
    start = re.search(rf"^{DOMAINS_SECTION}$", doc_text, re.MULTILINE)
    if start is None:
        return set(), [f"{DOMAINS_DOC} has no '{DOMAINS_SECTION}' section"]
    end = re.compile(r"^## ", re.MULTILINE).search(doc_text, start.end())
    section = doc_text[start.end() : end.start() if end else len(doc_text)]
    domains: set[str] = set()
    violations: list[str] = []
    for heading in DOMAIN_HEADING.findall(section):
        if heading.casefold() == SHARED_HEADING.casefold():
            continue
        name = heading.replace("-", "_")
        if not DOMAIN_NAME.fullmatch(heading):
            violations.append(f"{DOMAINS_DOC} heading '### {heading}' is not a domain name in lower case with hyphens")
        elif name in domains:
            violations.append(f"{DOMAINS_DOC} names the domain '{heading}' twice")
        domains.add(name)
    if not domains and not violations:
        violations.append(f"{DOMAINS_DOC} gives no domain a section")
    return domains, violations


def _domain_named_locations(src_root: Path) -> tuple[dict[str, str], list[str]]:
    """Return the domain each domain package and domain module is named for, and each misnamed exceptions module."""
    locations: dict[str, str] = {}
    misnamed: list[str] = []
    for layer in DOMAIN_PACKAGE_LAYERS:
        layer_root = src_root / layer
        if layer_root.is_dir():
            for package in sorted(layer_root.iterdir()):
                if (package / "__init__.py").is_file():
                    locations[f"{layer}/{package.name}/"] = package.name
    schemas_root = src_root / "gateway" / "schemas"
    for module in sorted(schemas_root.glob("*.py")):
        if module.name != "__init__.py":
            locations[f"gateway/schemas/{module.name}"] = module.stem
    exceptions_root = src_root / "gateway" / "exceptions"
    for module in sorted(exceptions_root.glob("*.py")):
        relative_path = f"gateway/exceptions/{module.name}"
        if module.name == "__init__.py" or relative_path in SHARED_EXCEPTION_MODULES:
            continue
        if module.stem.endswith("_exceptions"):
            locations[relative_path] = module.stem.removesuffix("_exceptions")
        elif relative_path not in DOMAIN_NAME_BASELINE:
            misnamed.append(f"{relative_path} is not named <domain>_exceptions.py")
    return locations, misnamed


def read_domains_page(doc_path: Path) -> tuple[set[str], list[str]]:
    """Return the domains the domains page gives a section, and each problem with the page."""
    if not doc_path.is_file():
        return set(), [f"{DOMAINS_DOC} not found; the domain names are read from its '{DOMAINS_SECTION}' section"]
    return documented_domains(doc_path.read_text(encoding="utf-8"))


def _baseline_package_modules(src_root: Path) -> list[str]:
    """Return each module a package on the domain name baseline holds but does not list, and each it lists but lacks."""
    violations: list[str] = []
    for package, listed in DOMAIN_NAME_BASELINE.items():
        package_root = src_root / package
        if not package.endswith("/"):
            if listed:
                violations.append(f"{package} is a module, so its domain name baseline entry lists no modules")
            continue
        if not package_root.is_dir():
            continue
        held = {
            module.relative_to(package_root).as_posix()
            for module in package_root.rglob("*.py")
            if "__pycache__" not in module.parts
        }
        violations.extend(
            f"{package}{module} is a new module in a package named for no domain; put it in its domain's package"
            for module in sorted(held - set(listed))
        )
        violations.extend(
            f"{package}{module} is on the domain name baseline but no longer exists; remove it from the baseline"
            for module in sorted(set(listed) - held)
        )
    return violations


def check_domain_names(src_root: Path, domains: set[str]) -> list[str]:
    """Check that each domain package and domain module names one of the documented domains.

    A package whose name is not a domain yet holds only the modules its baseline entry lists.
    """
    locations, misnamed = _domain_named_locations(src_root)
    violations = list(misnamed)
    violations.extend(
        f"{relative_path} names no domain in {DOMAINS_DOC}; "
        "name it for a domain there, or give the new domain a section"
        for relative_path, name in locations.items()
        if name not in domains and relative_path not in DOMAIN_NAME_BASELINE
    )
    violations.extend(
        f"{relative_path} is on the domain name baseline but no longer exists or now names a domain; "
        "remove it from the baseline"
        for relative_path in DOMAIN_NAME_BASELINE
        if not (src_root / relative_path).exists() or locations.get(relative_path) in domains
    )
    violations.extend(_baseline_package_modules(src_root))
    return violations


REPOSITORY_SCOPE = "gateway/repositories"
SERVICE_BUILDERS = "gateway/api/deps.py"


def _domain_packages(src_root: Path, layer: str, domains: set[str]) -> set[str]:
    """Return the name of each package in a layer that is named for a documented domain."""
    layer_root = src_root / layer
    if not layer_root.is_dir():
        return set()
    return {
        package.name
        for package in layer_root.iterdir()
        if package.name in domains and (package / "__init__.py").is_file()
    }


def _imported_domain(module: str, layer: str, packages: Container[str]) -> str | None:
    """Return the domain package of a layer that a module path lies in, or None if it lies in none."""
    prefix = layer.replace("/", ".") + "."
    if not module.startswith(prefix):
        return None
    package = module.removeprefix(prefix).split(".")[0]
    return package if package in packages else None


def _import_statements(tree: ast.Module, file_path: Path, src_root: Path) -> Iterator[tuple[int, list[str]]]:
    """Yield the line of each import statement in a module and the absolute module paths it pulls in."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            yield node.lineno, _imported_modules(node, file_path, src_root)


def _is_old_shape_service(relative_path: str, service_packages: set[str]) -> bool:
    """Return whether a module is service code outside every domain service package."""
    return relative_path.startswith(f"{SERVICE_SCOPE}/") and not any(
        relative_path.startswith(f"{SERVICE_SCOPE}/{package}/") for package in service_packages
    )


def _foreign_repository_import(modules: list[str], relative_path: str, repository_packages: set[str]) -> str | None:
    """Return the first of an import's modules that lies in another domain's repository package, or None."""
    for module in modules:
        domain = _imported_domain(module, REPOSITORY_SCOPE, repository_packages)
        if domain is not None and not relative_path.startswith(
            (f"{SERVICE_SCOPE}/{domain}/", f"{REPOSITORY_SCOPE}/{domain}/")
        ):
            return module
    return None


def check_repository_imports(src_root: Path, domains: set[str]) -> list[str]:
    """Check that only a domain's own packages and the service builders import its repositories.

    NOTE: Service code outside every domain package is skipped, because its path does not say which domain owns it.
    """
    repository_packages = _domain_packages(src_root, REPOSITORY_SCOPE, domains)
    service_packages = _domain_packages(src_root, SERVICE_SCOPE, domains)
    found: list[tuple[str, int, str]] = []
    for relative_path, tree in _parsed_modules(src_root, "gateway", exempt=(SERVICE_BUILDERS,)):
        if _is_old_shape_service(relative_path, service_packages):
            continue
        for line, modules in _import_statements(tree, src_root / relative_path, src_root):
            module = _foreign_repository_import(modules, relative_path, repository_packages)
            if module is not None:
                found.append((relative_path, line, module))
    return [
        f"{relative_path}:{line} imports {module}; only the domain's own service and repository packages and the "
        f"builders in {SERVICE_BUILDERS} import its repositories"
        for relative_path, line, module in sorted(found)
    ]


def _package_members(package_root: Path) -> set[str]:
    """Return the name of each module and subpackage directly inside a package, spelled as it is on disk.

    NOTE: A path test on a case-insensitive file system would match a class such as Mailer to mailer.py.
    """
    return {
        entry.stem if entry.is_file() else entry.name
        for entry in package_root.iterdir()
        if (entry.suffix == ".py" and entry.name != "__init__.py") or (entry / "__init__.py").is_file()
    }


def _below_root_service_import(modules: list[str], relative_path: str, members: dict[str, set[str]]) -> str | None:
    """Return the first of an import's modules that lies below another domain's service package root, or None."""
    for module in modules:
        domain = _imported_domain(module, SERVICE_SCOPE, members)
        if domain is None or relative_path.startswith(f"{SERVICE_SCOPE}/{domain}/"):
            continue
        package_root = f"{SERVICE_SCOPE.replace('/', '.')}.{domain}."
        if module.removeprefix(package_root).split(".")[0] in members[domain]:
            return module
    return None


def check_service_package_imports(src_root: Path, domains: set[str]) -> list[str]:
    """Check that code outside a domain service package imports only the package root."""
    members = {
        package: _package_members(src_root / SERVICE_SCOPE / package)
        for package in _domain_packages(src_root, SERVICE_SCOPE, domains)
    }
    found: list[tuple[str, int, str]] = []
    for relative_path, tree in _parsed_modules(src_root, "gateway"):
        for line, modules in _import_statements(tree, src_root / relative_path, src_root):
            module = _below_root_service_import(modules, relative_path, members)
            if module is not None:
                found.append((relative_path, line, module))
    return [
        f"{relative_path}:{line} imports {module}; code outside a domain imports what its service package root exports"
        for relative_path, line, module in sorted(found)
    ]


MODELS_SCOPE = "gateway/models"
TABLE_ATTRIBUTES = ("__table__", "__tablename__")
TYPE_ALIAS = ("TypeAlias", "typing.TypeAlias", "typing_extensions.TypeAlias")
TYPE_CHECKS = ("isinstance", "issubclass")
TYPING_CASTS = ("typing.cast", "typing_extensions.cast")


class ModelAccess(TypedDict):
    """Each field names modules that may use one ORM model: its repository, and the others that still do."""

    repository: str | None
    baseline: tuple[str, ...]


# The map has an entry for each ORM model, keyed by its import path.
# Its repository is the one module that may construct or query the model,
# or None where no repository holds its queries yet.
# Its baseline names the other modules that still use the model, and a baseline only shrinks.
MODEL_ACCESS: dict[str, ModelAccess] = {
    "gateway.models.api_keys.APIKey": {
        "repository": "gateway/repositories/api_keys/api_key_repository.py",
        "baseline": (
            "gateway/api/deps.py",
            "gateway/api/routes/batches.py",
            "gateway/api/routes/catalog.py",
            "gateway/api/routes/keys.py",
            "gateway/api/routes/organization_keys.py",
            "gateway/api/routes/organization_usage.py",
            "gateway/api/routes/scoped_budgets.py",
            "gateway/api/routes/usage.py",
            "gateway/api/routes/users.py",
            "gateway/repositories/overview/overview_repository.py",
            "gateway/repositories/users_repository.py",
            "gateway/services/bootstrap_service.py",
            "gateway/services/playground_dispatch.py",
            "gateway/services/tenancy/workspace_activation_service.py",
            "gateway/services/workspace_scope.py",
        ),
    },
    "gateway.models.budgets.Budget": {
        "repository": "gateway/repositories/budgets/budget_repository.py",
        "baseline": (
            "gateway/api/routes/budgets.py",
            "gateway/api/routes/scoped_budgets.py",
            "gateway/api/routes/users.py",
            "gateway/repositories/budgets/scoped_budget_repository.py",
            "gateway/repositories/overview/overview_repository.py",
            "gateway/repositories/tenancy/organization_member_repository.py",
            "gateway/services/budgets/_organization_surface.py",
            "gateway/services/budgets/_reservations.py",
            "gateway/services/budgets/_scoped_enforcement.py",
        ),
    },
    "gateway.models.budgets.BudgetReservation": {
        "repository": None,
        "baseline": ("gateway/services/budgets/_ledger.py",),
    },
    "gateway.models.budgets.BudgetReservationScope": {
        "repository": None,
        "baseline": ("gateway/services/budgets/_ledger.py",),
    },
    "gateway.models.budgets.BudgetResetLog": {
        "repository": "gateway/repositories/budgets/budget_repository.py",
        "baseline": (
            "gateway/api/routes/budgets.py",
            "gateway/services/budgets/_reservations.py",
        ),
    },
    "gateway.models.budgets.ScopedBudget": {
        "repository": "gateway/repositories/budgets/scoped_budget_repository.py",
        "baseline": (
            "gateway/api/routes/scoped_budgets.py",
            "gateway/repositories/overview/overview_repository.py",
            "gateway/repositories/tenancy/organization_member_repository.py",
            "gateway/services/budgets/_member_policies.py",
            "gateway/services/budgets/_organization_surface.py",
            "gateway/services/budgets/_retiming.py",
            "gateway/services/budgets/_scoped_enforcement.py",
        ),
    },
    "gateway.models.budgets.WorkspaceBudgetDefault": {
        "repository": "gateway/repositories/budgets/workspace_budget_default_repository.py",
        "baseline": ("gateway/services/budgets/_member_policies.py",),
    },
    "gateway.models.files.FileObject": {
        "repository": "gateway/repositories/files/file_repository.py",
        "baseline": ("gateway/services/files/_service.py",),
    },
    "gateway.models.files.FileProviderCopy": {
        "repository": "gateway/repositories/files/file_provider_copy_repository.py",
        "baseline": ("gateway/services/files/_provider_uploads.py",),
    },
    "gateway.models.guardrails.OrganizationGuardrail": {
        "repository": None,
        "baseline": (
            "gateway/repositories/tenancy/organization_guardrail_definition_repository.py",
            "gateway/services/tenancy/organization_guardrail_service.py",
        ),
    },
    "gateway.models.guardrails.OrganizationGuardrailDefinition": {
        "repository": "gateway/repositories/tenancy/organization_guardrail_definition_repository.py",
        "baseline": (
            "gateway/services/tenancy/organization_guardrail_definition_service.py",
            "gateway/services/tenancy/organization_guardrail_service.py",
        ),
    },
    "gateway.models.guardrails.OrganizationGuardrailWorkspace": {
        "repository": None,
        "baseline": ("gateway/services/tenancy/organization_guardrail_service.py",),
    },
    "gateway.models.inference.BatchRecord": {
        "repository": None,
        "baseline": ("gateway/services/batch_service.py",),
    },
    "gateway.models.inference.IdempotencyRecord": {
        "repository": "gateway/repositories/inference/idempotency_repository.py",
        "baseline": (),
    },
    "gateway.models.platform.RuntimeSetting": {
        "repository": None,
        "baseline": (
            "gateway/services/dashboard_session_service.py",
            "gateway/services/maintenance_mode_service.py",
            "gateway/services/master_key_service.py",
            "gateway/services/runtime_settings_service.py",
            "gateway/services/tenancy/provisioning_service.py",
            "gateway/services/tool_settings_service.py",
        ),
    },
    "gateway.models.playground.PlaygroundComparison": {
        "repository": None,
        "baseline": ("gateway/services/playground_service.py",),
    },
    "gateway.models.playground.PlaygroundConsent": {
        "repository": None,
        "baseline": ("gateway/services/playground_service.py",),
    },
    "gateway.models.playground.PlaygroundConversation": {
        "repository": None,
        "baseline": ("gateway/services/playground_service.py",),
    },
    "gateway.models.playground.PlaygroundFavoriteModel": {
        "repository": None,
        "baseline": ("gateway/services/playground_service.py",),
    },
    "gateway.models.playground.PlaygroundMessage": {
        "repository": None,
        "baseline": ("gateway/services/playground_service.py",),
    },
    "gateway.models.pricing.ModelPricing": {
        "repository": None,
        "baseline": (
            "gateway/api/routes/models.py",
            "gateway/api/routes/pricing.py",
            "gateway/repositories/pricing/organization_model_pricing_repository.py",
            "gateway/services/external_usage_service.py",
            "gateway/services/merged_catalog_service.py",
            "gateway/services/pricing_init_service.py",
            "gateway/services/pricing_service.py",
            "gateway/services/usage_admin_service.py",
        ),
    },
    "gateway.models.pricing.OrganizationModelPricing": {
        "repository": "gateway/repositories/pricing/organization_model_pricing_repository.py",
        "baseline": (
            "gateway/services/organization_pricing_service.py",
            "gateway/services/pricing_service.py",
            "gateway/services/providers/_org_provider_model_service.py",
        ),
    },
    "gateway.models.pricing.PricingSnapshot": {
        "repository": None,
        "baseline": (
            "gateway/api/routes/catalog.py",
            "gateway/services/pricing_refresh_service.py",
        ),
    },
    "gateway.models.pricing.PricingSnapshotHistory": {
        "repository": None,
        "baseline": ("gateway/services/pricing_refresh_service.py",),
    },
    "gateway.models.provider_keys.OrgProviderKey": {
        "repository": "gateway/repositories/tenancy/org_provider_key_repository.py",
        "baseline": ("gateway/services/tenancy/org_provider_key_service.py",),
    },
    "gateway.models.provider_keys.OrgProviderKeyModel": {
        "repository": "gateway/repositories/providers/org_provider_key_model_repository.py",
        "baseline": ("gateway/services/providers/_org_provider_model_service.py",),
    },
    "gateway.models.provider_keys.WorkspaceProviderKeyOverride": {
        "repository": "gateway/repositories/tenancy/org_provider_key_repository.py",
        "baseline": ("gateway/services/tenancy/org_provider_key_service.py",),
    },
    "gateway.models.provider_keys.WorkspaceProviderModelRestriction": {
        "repository": "gateway/repositories/tenancy/org_provider_key_repository.py",
        "baseline": ("gateway/services/tenancy/org_provider_key_service.py",),
    },
    "gateway.models.providers.ModelAlias": {
        "repository": None,
        "baseline": (
            "gateway/api/routes/aliases.py",
            "gateway/api/routes/organization_routing.py",
            "gateway/services/alias_service.py",
        ),
    },
    "gateway.models.providers.ProviderCredential": {
        "repository": None,
        "baseline": ("gateway/services/provider_store_service.py",),
    },
    "gateway.models.providers.ProviderEndpoint": {
        "repository": "gateway/repositories/providers/provider_endpoint_repository.py",
        "baseline": (),
    },
    "gateway.models.rate_limits.StoredRateLimitRule": {
        "repository": "gateway/repositories/rate_limits/rate_limit_rule_repository.py",
        "baseline": ("gateway/services/rate_limits/_service.py",),
    },
    "gateway.models.routing.RouterPreference": {
        "repository": None,
        "baseline": ("gateway/api/routes/routing_memory.py",),
    },
    "gateway.models.routing.RoutingMemory": {
        "repository": None,
        "baseline": (
            "gateway/api/routes/routing_memory.py",
            "gateway/services/routing/knn.py",
        ),
    },
    "gateway.models.routing.RoutingPolicy": {
        "repository": None,
        "baseline": (
            "gateway/api/routes/organization_routing.py",
            "gateway/api/routes/routing.py",
            "gateway/services/policy_store.py",
        ),
    },
    "gateway.models.tenancy.DashboardSession": {
        "repository": None,
        "baseline": ("gateway/services/dashboard_session_service.py",),
    },
    "gateway.models.tenancy.Invitation": {
        "repository": "gateway/repositories/tenancy/invitation_repository.py",
        "baseline": (),
    },
    "gateway.models.tenancy.OAuthPendingState": {
        "repository": None,
        "baseline": ("gateway/services/oauth_service.py",),
    },
    "gateway.models.tenancy.Organization": {
        "repository": "gateway/repositories/tenancy/organization_repository.py",
        "baseline": (
            "gateway/api/routes/scoped_budgets.py",
            "gateway/repositories/tenancy/invitation_repository.py",
            "gateway/repositories/tenancy/organization_member_repository.py",
            "gateway/services/tenancy/provisioning_service.py",
            "gateway/services/workspace_scope.py",
        ),
    },
    "gateway.models.tenancy.OrganizationDomain": {
        "repository": "gateway/repositories/tenancy/organization_domain_repository.py",
        "baseline": (),
    },
    "gateway.models.tenancy.OrganizationMember": {
        "repository": "gateway/repositories/tenancy/organization_member_repository.py",
        "baseline": (
            "gateway/api/routes/scoped_budgets.py",
            "gateway/repositories/tenancy/invitation_repository.py",
            "gateway/repositories/users_repository.py",
            "gateway/services/budgets/_scoped_enforcement.py",
        ),
    },
    "gateway.models.tenancy.User": {
        "repository": "gateway/repositories/tenancy/user_repository.py",
        "baseline": (
            "gateway/api/deps.py",
            "gateway/repositories/tenancy/organization_member_repository.py",
            "gateway/repositories/tenancy/workspace_repository.py",
            "gateway/services/dashboard_session_service.py",
            "gateway/services/tenancy/provisioning_service.py",
            "gateway/services/tenancy/webauthn_service.py",
        ),
    },
    "gateway.models.tenancy.WebAuthnChallenge": {
        "repository": None,
        "baseline": ("gateway/services/tenancy/webauthn_service.py",),
    },
    "gateway.models.tenancy.WebAuthnCredential": {
        "repository": None,
        "baseline": ("gateway/services/tenancy/webauthn_service.py",),
    },
    "gateway.models.tenancy.Workspace": {
        "repository": "gateway/repositories/tenancy/workspace_repository.py",
        "baseline": (
            "gateway/api/routes/_helpers.py",
            "gateway/api/routes/catalog.py",
            "gateway/api/routes/keys.py",
            "gateway/api/routes/organization_keys.py",
            "gateway/api/routes/organization_routing.py",
            "gateway/api/routes/organization_usage.py",
            "gateway/api/routes/scoped_budgets.py",
            "gateway/repositories/budgets/workspace_budget_default_repository.py",
            "gateway/repositories/overview/overview_repository.py",
            "gateway/repositories/tenancy/organization_member_repository.py",
            "gateway/repositories/users_repository.py",
            "gateway/services/budgets/_scoped_enforcement.py",
            "gateway/services/tenancy/org_provider_key_service.py",
            "gateway/services/tenancy/organization_model_access.py",
            "gateway/services/workspace_scope.py",
        ),
    },
    "gateway.models.tenancy.WorkspaceActivationState": {
        "repository": None,
        "baseline": ("gateway/services/tenancy/workspace_activation_service.py",),
    },
    "gateway.models.tenancy.WorkspaceMember": {
        "repository": "gateway/repositories/tenancy/workspace_repository.py",
        "baseline": (
            "gateway/api/routes/scoped_budgets.py",
            "gateway/repositories/overview/overview_repository.py",
            "gateway/repositories/tenancy/organization_member_repository.py",
            "gateway/services/budgets/_scoped_enforcement.py",
        ),
    },
    "gateway.models.tools.OrgWebSearchKey": {
        "repository": "gateway/repositories/tools/web_search_key_repository.py",
        "baseline": (),
    },
    "gateway.models.tools.SandboxContainer": {
        "repository": "gateway/repositories/code_execution/sandbox_container_repository.py",
        "baseline": (),
    },
    "gateway.models.tools.SearchToolCredential": {
        "repository": None,
        "baseline": ("gateway/services/search_tool_store_service.py",),
    },
    "gateway.models.tools.WorkspaceCodeExecutionPolicy": {
        "repository": "gateway/repositories/tools/workspace_code_execution_policy_repository.py",
        "baseline": (),
    },
    "gateway.models.tools.WorkspaceMcpServer": {
        "repository": None,
        "baseline": (
            "gateway/services/playground_service.py",
            "gateway/services/tenancy/workspace_mcp_server_service.py",
        ),
    },
    "gateway.models.tools.WorkspaceWebSearchConfig": {
        "repository": None,
        "baseline": (
            "gateway/services/playground_service.py",
            "gateway/services/tenancy/workspace_web_search_service.py",
        ),
    },
    "gateway.models.tools.WorkspaceWebSearchKeyOverride": {
        "repository": "gateway/repositories/tools/web_search_key_repository.py",
        "baseline": (),
    },
    "gateway.models.usage.AgentTelemetry": {
        "repository": None,
        "baseline": ("gateway/adapters/telemetry_storage_adapter.py",),
    },
    "gateway.models.usage.UsageLog": {
        "repository": None,
        "baseline": (
            "gateway/api/routes/_passthrough.py",
            "gateway/api/routes/_pipeline.py",
            "gateway/api/routes/agent_telemetry.py",
            "gateway/api/routes/batches.py",
            "gateway/api/routes/catalog.py",
            "gateway/api/routes/organization_usage.py",
            "gateway/api/routes/search.py",
            "gateway/api/routes/usage.py",
            "gateway/api/routes/users.py",
            "gateway/repositories/users_repository.py",
            "gateway/services/external_usage_service.py",
            "gateway/services/tenancy/workspace_activation_service.py",
            "gateway/services/usage_admin_service.py",
        ),
    },
    "gateway.models.users.User": {
        "repository": "gateway/repositories/users_repository.py",
        "baseline": (
            "gateway/api/routes/budgets.py",
            "gateway/api/routes/keys.py",
            "gateway/api/routes/organization_keys.py",
            "gateway/api/routes/organization_usage.py",
            "gateway/api/routes/usage.py",
            "gateway/api/routes/users.py",
            "gateway/repositories/budgets/budget_repository.py",
            "gateway/repositories/budgets/end_user_repository.py",
            "gateway/repositories/inference/idempotency_repository.py",
            "gateway/repositories/overview/overview_repository.py",
            "gateway/services/budgets/_end_users.py",
            "gateway/services/budgets/_ledger.py",
            "gateway/services/budgets/_reservations.py",
            "gateway/services/external_usage_service.py",
            "gateway/services/model_access.py",
            "gateway/services/playground_service.py",
        ),
    },
}


def _module_name(relative_path: str) -> str:
    """Return the import path of a module from its path below the source root."""
    return relative_path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _assigned_names(statement: ast.stmt) -> list[str]:
    """Return each plain name an assignment statement binds, or none for any other statement."""
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    else:
        return []
    return [target.id for target in targets if isinstance(target, ast.Name)]


def _declares_table(node: ast.ClassDef) -> bool:
    """Return whether a class is an ORM model, which it is when it passes table=True or assigns a table attribute."""
    if any(
        keyword.arg == "table" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        for keyword in node.keywords
    ):
        return True
    return any(name in TABLE_ATTRIBUTES for statement in node.body for name in _assigned_names(statement))


def _orm_models(src_root: Path) -> set[str]:
    """Return the import path of each ORM model in the models package."""
    return {
        f"{_module_name(relative_path)}.{node.name}"
        for relative_path, tree in _parsed_modules(src_root, MODELS_SCOPE)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and _declares_table(node)
    }


def _module_bindings(tree: ast.Module, file_path: Path, src_root: Path) -> dict[str, dict[str, int]]:
    """Return each import path a name in a module is bound to, with the first line that binds it.

    A name is bound by an import anywhere in the module, or by a module-level assignment from a bound name.
    """
    bindings: dict[str, dict[str, int]] = {}

    def bind(name: str, target: str, line: int) -> None:
        bindings.setdefault(name, {}).setdefault(target, line)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                bind(alias.asname or head, alias.name if alias.asname else head, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            base = node.module if node.level == 0 else _resolve_relative(node, file_path, src_root)
            for alias in node.names:
                if base is not None and alias.name != "*":
                    bind(alias.asname or alias.name, f"{base}.{alias.name}", node.lineno)
    for statement in tree.body:
        value = statement.value if isinstance(statement, ast.Assign | ast.AnnAssign) else None
        dotted = None if value is None else _dotted_name(value)
        head, _, rest = (dotted or "").partition(".")
        if dotted is None or len(bindings.get(head, {})) != 1:
            continue
        target = next(iter(bindings[head]))
        for name in _assigned_names(statement):
            bind(name, f"{target}.{rest}" if rest else target, statement.lineno)
    return bindings


def _star_imports(tree: ast.Module, file_path: Path, src_root: Path) -> list[tuple[int, str]]:
    """Return the line and module of each star import in a module."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or all(alias.name != "*" for alias in node.names):
            continue
        base = node.module if node.level == 0 else _resolve_relative(node, file_path, src_root)
        if base is not None:
            found.append((node.lineno, base))
    return sorted(found)


def _resolve_reexports(path: str, reexports: dict[str, str]) -> str:
    """Follow an import path through the modules that re-export it to where the name is defined."""
    seen: set[str] = set()
    while path not in seen:
        seen.add(path)
        parts = path.split(".")
        prefixes = (".".join(parts[:end]) for end in range(len(parts), 0, -1))
        prefix = next((candidate for candidate in prefixes if candidate in reexports), None)
        if prefix is None:
            break
        path = reexports[prefix] + path.removeprefix(prefix)
    return path


def _dotted_name(node: ast.expr) -> str | None:
    """Return the dotted name an expression spells, such as tools.Workspace, or None if it is not one.

    A getattr call with a literal name spells the attribute it reads.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        base = _dotted_name(node.args[0])
        return None if base is None else f"{base}.{node.args[1].value}"
    return None


def _type_position_nodes(tree: ast.Module, bindings: dict[str, str]) -> set[int]:
    """Return the identity of every node in a module that names a type without constructing or querying it.

    Such a node sits in an annotation, a type alias, the target type of a typing cast, the bound of a TypeVar,
    or the types an isinstance or issubclass call checks against.
    A SQL cast takes a column, so its arguments stay uses.
    """
    positions: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            positions.append(node.annotation)
        elif isinstance(node, ast.AnnAssign):
            positions.append(node.annotation)
            if node.value is not None and _dotted_name(node.annotation) in TYPE_ALIAS:
                positions.append(node.value)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            positions.append(node.returns)
        elif isinstance(node, ast.TypeAlias):
            positions.append(node.value)
        elif isinstance(node, ast.Call):
            called = _called_name(node.func)
            dotted = _dotted_name(node.func) or ""
            head, _, rest = dotted.partition(".")
            target = bindings.get(head)
            if target is not None and (f"{target}.{rest}" if rest else target) in TYPING_CASTS and node.args:
                positions.append(node.args[0])
            elif called == "TypeVar":
                positions.extend(keyword.value for keyword in node.keywords if keyword.arg == "bound")
            elif called in TYPE_CHECKS and len(node.args) > 1:
                positions.append(node.args[1])
    return {id(inner) for position in positions for inner in ast.walk(position)}


def _model_uses(
    tree: ast.Module, bindings: dict[str, str], reexports: dict[str, str], models: set[str]
) -> dict[str, int]:
    """Return the first line on which a module names each ORM model in an expression."""
    type_positions = _type_position_nodes(tree, bindings)
    uses: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name | ast.Attribute | ast.Call) or id(node) in type_positions:
            continue
        if isinstance(node, ast.Name | ast.Attribute) and not isinstance(node.ctx, ast.Load):
            continue
        dotted = _dotted_name(node)
        if dotted is None:
            continue
        head, _, rest = dotted.partition(".")
        if head not in bindings:
            continue
        model = _resolve_reexports(f"{bindings[head]}.{rest}" if rest else bindings[head], reexports)
        if model in models:
            uses[model] = min(uses.get(model, node.lineno), node.lineno)
    return uses


def _model_access_violations(model: str, access: ModelAccess, users: dict[str, int], src_root: Path) -> list[str]:
    """Check one ORM model's entry in the model access map against the modules that use it."""
    repository = access["repository"]
    violations: list[str] = []
    if repository is not None and not repository.startswith(f"{REPOSITORY_SCOPE}/"):
        violations.append(f"{model} names {repository} as its repository, which is not under {REPOSITORY_SCOPE}/")
    elif repository is not None and not (src_root / repository).is_file():
        violations.append(f"{model} names {repository} as its repository, which does not exist")
    elif repository is not None and repository not in users:
        violations.append(f"{model} names {repository} as its repository, which does not use it")
    if repository is not None and repository in access["baseline"]:
        violations.append(f"{model} has its repository {repository} on its baseline; remove it from the baseline")
    remedy = (
        f"only {repository} constructs or queries it"
        if repository is not None
        else "its queries belong in a repository, which the model access map names"
    )
    violations.extend(
        f"{module}:{line} uses {model}; {remedy}"
        for module, line in sorted(users.items())
        if module != repository and module not in access["baseline"]
    )
    violations.extend(
        f"{module} is on the baseline for {model} but does not use it; remove it from the baseline"
        for module in access["baseline"]
        if module not in users and module != repository
    )
    return violations


def _ambiguous_model_names(
    relative_path: str,
    bindings: dict[str, dict[str, int]],
    resolve: Callable[[str], str],
    holds_model: Callable[[str], bool],
) -> list[str]:
    """Return a violation for each name a module binds to more than one target when one of them holds an ORM model."""
    return [
        f"{relative_path}:{sorted(targets.values())[1]} binds {name} to more than one target, one of them an ORM "
        "model or a module that holds one; import each under a name of its own"
        for name, targets in sorted(bindings.items())
        if len({resolve(target) for target in targets}) > 1 and any(holds_model(target) for target in targets)
    ]


def check_model_access(src_root: Path) -> list[str]:
    """Check that only an ORM model's repository and the modules on its baseline use it.

    A module uses a model when it names it in an expression, which is how a module constructs or queries one.
    Naming a type without constructing or querying it, as an annotation or a cast does, is not a use.
    The module that declares a model may name it.
    A star import from a module that holds a model is refused, because it would hide a use.
    So is a name bound both to a model or a module that holds one and to something else.

    NOTE: Names are resolved per module, not per scope, so a parameter that shadows an imported model counts as a use.
    """
    models = _orm_models(src_root)
    parsed = list(_parsed_modules(src_root, "gateway"))
    bindings = {
        relative_path: _module_bindings(tree, src_root / relative_path, src_root) for relative_path, tree in parsed
    }
    reexports = {
        f"{_module_name(relative_path)}.{name}": next(iter(targets))
        for relative_path, names in bindings.items()
        for name, targets in names.items()
        if len(targets) == 1 and f"{_module_name(relative_path)}.{name}" not in targets
    }

    def resolve(path: str) -> str:
        return _resolve_reexports(path, reexports)

    model_holders = {model.rpartition(".")[0] for model in models} | {
        name.rpartition(".")[0] for name, target in reexports.items() if resolve(target) in models
    }

    def holds_model(path: str) -> bool:
        resolved = resolve(path)
        return resolved in models or any(
            holder == resolved or holder.startswith(f"{resolved}.") for holder in model_holders
        )

    violations: list[str] = []
    uses: dict[str, dict[str, int]] = {model: {} for model in models}
    for relative_path, tree in parsed:
        names = bindings[relative_path]
        violations.extend(_ambiguous_model_names(relative_path, names, resolve, holds_model))
        violations.extend(
            f"{relative_path}:{line} imports * from {module}, which holds an ORM model; import the names it uses"
            for line, module in _star_imports(tree, src_root / relative_path, src_root)
            if holds_model(module)
        )
        resolved = {name: {resolve(target) for target in targets} for name, targets in names.items()}
        unambiguous = {name: next(iter(paths)) for name, paths in resolved.items() if len(paths) == 1}
        for model, line in _model_uses(tree, unambiguous, reexports, models).items():
            uses[model][relative_path] = line
    violations.extend(
        f"{model} is an ORM model with no entry in the model access map; name the repository that may use it"
        for model in sorted(models - MODEL_ACCESS.keys())
    )
    violations.extend(
        f"{model} is in the model access map but is not an ORM model; remove it from the map"
        for model in sorted(MODEL_ACCESS.keys() - models)
    )
    for model, access in sorted(MODEL_ACCESS.items()):
        if model in models:
            violations.extend(_model_access_violations(model, access, uses[model], src_root))
    return violations


def main() -> int:
    """Run the architecture checks over the gateway package, the light CLI, the libraries and the OSS test suite."""
    # All must exist: silently skipping one would let its rules (including
    # the OSS/enterprise boundary) stop enforcing while the check stays green.
    for required_root in (GATEWAY_ROOT, TESTS_ROOT, CLI_ROOT, *LIBRARY_ROOTS.values()):
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
    for py_file in sorted(CLI_ROOT.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        import_violations.extend(
            (py_file, lineno, module, message) for lineno, module, message in check_file(py_file, CLI_ROOT)
        )
    for rule_key, library_root in LIBRARY_ROOTS.items():
        for py_file in sorted(library_root.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            import_violations.extend(
                (py_file, lineno, module, message)
                for lineno, module, message in check_file(py_file, library_root, rule_key)
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
    unit_of_work_violations = check_unit_of_work_construction(SRC_ROOT)
    mode_read_violations = check_service_mode_reads(SRC_ROOT)
    listener_default_violations = check_listener_defaults(SRC_ROOT)
    # NOTE: A page with no domains fails here, so a rule that reads them cannot pass by checking nothing.
    domains, domains_page_violations = read_domains_page(REPO_ROOT / DOMAINS_DOC)
    domain_name_violations = domains_page_violations + (check_domain_names(SRC_ROOT, domains) if domains else [])
    repository_import_violations = check_repository_imports(SRC_ROOT, domains)
    service_package_import_violations = check_service_package_imports(SRC_ROOT, domains)
    model_access_violations = check_model_access(SRC_ROOT)

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

    if unit_of_work_violations:
        print("\n❌ Unit of Work construction violations:\n")
        for violation in unit_of_work_violations:
            print(f"  {violation}")
        print(f"\nTotal Unit of Work construction violations: {len(unit_of_work_violations)}")

    if mode_read_violations:
        print("\n❌ Mode read violations:\n")
        for violation in mode_read_violations:
            print(f"  {violation}")
        print(f"\nTotal mode read violations: {len(mode_read_violations)}")

    if listener_default_violations:
        print("\n❌ Listener default violations:\n")
        for violation in listener_default_violations:
            print(f"  {violation}")
        print(f"\nTotal listener default violations: {len(listener_default_violations)}")

    if flat_module_violations:
        print("\n❌ Flat module violations:\n")
        for violation in flat_module_violations:
            print(f"  {violation}")
        print(f"\nTotal flat module violations: {len(flat_module_violations)}")

    if domain_name_violations:
        print("\n❌ Domain name violations:\n")
        for violation in domain_name_violations:
            print(f"  {violation}")
        print(f"\nTotal domain name violations: {len(domain_name_violations)}")

    if repository_import_violations:
        print("\n❌ Repository import violations:\n")
        for violation in repository_import_violations:
            print(f"  {violation}")
        print(f"\nTotal repository import violations: {len(repository_import_violations)}")

    if service_package_import_violations:
        print("\n❌ Service package import violations:\n")
        for violation in service_package_import_violations:
            print(f"  {violation}")
        print(f"\nTotal service package import violations: {len(service_package_import_violations)}")

    if model_access_violations:
        print("\n❌ Model access violations:\n")
        for violation in model_access_violations:
            print(f"  {violation}")
        print(f"\nTotal model access violations: {len(model_access_violations)}")

    if (
        import_violations
        or naming_violations
        or package_violations
        or database_violations
        or transaction_violations
        or unit_of_work_violations
        or mode_read_violations
        or listener_default_violations
        or flat_module_violations
        or domain_name_violations
        or repository_import_violations
        or service_package_import_violations
        or model_access_violations
    ):
        print("\n💡 See ARCHITECTURE.md for the intended layering")
        return 1

    print("✅ No architecture violations found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
