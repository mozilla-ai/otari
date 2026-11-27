"""Unit tests for the architecture check (layer rules over src/gateway, cli/src/otari_agent and the libraries)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_architecture.py"
_DISCOVERY_MESSAGE = "Forbidden import in OSS base (no entry-point discovery; the feature registry is a literal tuple)"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_architecture", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load()


def _write(src_root: Path, relative_path: str, content: str) -> Path:
    file_path = src_root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def _point_main_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Every baseline and the model access map are emptied, so a stale entry cannot fail main() on its own.
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(check, "GATEWAY_ROOT", tmp_path / "src" / "gateway")
    monkeypatch.setattr(check, "TESTS_ROOT", tmp_path / "tests")
    monkeypatch.setattr(check, "CLI_ROOT", tmp_path / "cli" / "src")
    monkeypatch.setattr(
        check, "LIBRARY_ROOTS", {"any_search": tmp_path / "any-search", "any_fetch": tmp_path / "any-fetch"}
    )
    # main() refuses to run without the light CLI's root and the libraries', so every temporary tree gets empty ones.
    _write(tmp_path, "cli/src/otari_agent/__init__.py", "")
    _write(tmp_path, "any-search/src/any_search/__init__.py", "")
    _write(tmp_path, "any-fetch/src/any_fetch/__init__.py", "")
    _write(tmp_path, "DOMAINS.md", "## The domains\n\n### things\n")
    for name in [name for name in vars(check) if name.endswith("_BASELINE")]:
        monkeypatch.setattr(check, name, type(getattr(check, name))())
    monkeypatch.setattr(check, "MODEL_ACCESS", {})


def test_service_importing_models_is_clean(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway.models.users import User\n")
    assert check.check_file(file_path, tmp_path) == []


def test_service_importing_api_is_flagged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway.api.routes import chat\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.api.routes", "Forbidden import in Services")]


def test_service_importing_api_via_from_gateway_is_flagged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway import api\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.api", "Forbidden import in Services")]


def test_service_relative_import_of_api_is_flagged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "from ..api import deps\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.api", "Forbidden import in Services")]


def test_relative_import_above_src_root_is_ignored(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "from ... import something\n")
    assert check.check_file(file_path, tmp_path) == []


def test_nested_and_enclosing_layer_rules_both_apply(tmp_path: Path) -> None:
    # The real rules already have the shape this guards: gateway/api is declared
    # before the nested gateway/api/routes, and a route file must answer to
    # both. Declaring one first must neither shadow the nested rule nor stop
    # applying to files under it.
    file_path = _write(
        tmp_path,
        "gateway/api/routes/users.py",
        "from gateway.adapters.billing_adapter import NullBillingAdapter\nfrom sqlalchemy.orm import Session\n",
    )
    assert check.check_file(file_path, tmp_path) == [
        (1, "gateway.adapters.billing_adapter", "Forbidden import in API layer"),
        (2, "sqlalchemy.orm", "Forbidden import in API routes"),
    ]


def test_violation_is_attributed_to_the_closest_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Both layers forbid it; the nested layer owns the message.
    broadened = {
        "gateway/api": {"allowed": [], "forbidden": ["sqlalchemy.orm"], "description": "API"},
        **check.RULES,
    }
    monkeypatch.setattr(check, "RULES", broadened)
    file_path = _write(tmp_path, "gateway/api/routes/users.py", "from sqlalchemy.orm import Session\n")
    assert check.check_file(file_path, tmp_path) == [(1, "sqlalchemy.orm", "Forbidden import in API routes")]


def test_forbidden_prefix_requires_a_module_boundary(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "import gateway.apilike\n")
    assert check.check_file(file_path, tmp_path) == []


def test_repository_importing_service_is_flagged(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "gateway/repositories/users_repository.py",
        "from gateway.services.budgets import reserve\n",
    )
    violations = check.check_file(file_path, tmp_path)
    assert violations == [(1, "gateway.services.budgets", "Forbidden import in Repositories")]


@pytest.mark.parametrize(
    "forbidden",
    [
        "gateway.api.deps",
        "gateway.services.budgets",
        "gateway.repositories.base_repository",
        "gateway.exceptions.budget_exceptions",
        "gateway.container",
    ],
)
def test_schema_importing_a_layer_beside_or_above_it_is_flagged(tmp_path: Path, forbidden: str) -> None:
    file_path = _write(tmp_path, "gateway/schemas/budgets.py", f"import {forbidden}\n")
    assert check.check_file(file_path, tmp_path) == [(1, forbidden, "Forbidden import in Schemas")]


def test_schema_importing_models_is_clean(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/schemas/budgets.py", "from gateway.models.budgets import Budget\n")
    assert check.check_file(file_path, tmp_path) == []


def test_api_route_importing_sqlalchemy_orm_is_flagged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/api/routes/users.py", "from sqlalchemy.orm import Session\n")
    assert check.check_file(file_path, tmp_path) == [(1, "sqlalchemy.orm", "Forbidden import in API routes")]


def test_api_route_may_import_repositories(tmp_path: Path) -> None:
    # Routes reuse repository helpers (AGENTS.md), so gateway.repositories is
    # deliberately not forbidden here.
    file_path = _write(
        tmp_path,
        "gateway/api/routes/users.py",
        "from gateway.repositories.users_repository import get_active_user\n",
    )
    assert check.check_file(file_path, tmp_path) == []


@pytest.mark.parametrize(
    "forbidden",
    ["gateway.api.deps", "gateway.services.budgets", "gateway.adapters.billing_adapter"],
)
def test_port_may_not_import_a_caller_or_an_adapter(tmp_path: Path, forbidden: str) -> None:
    # A port is the interface its callers depend on, so it sits below them, and
    # naming an adapter would name the implementation it exists to keep unnamed.
    file_path = _write(tmp_path, "gateway/ports/billing_port.py", f"from {forbidden} import thing\n")
    assert check.check_file(file_path, tmp_path) == [(1, forbidden, "Forbidden import in Ports")]


def test_port_may_describe_the_domain(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "gateway/ports/billing_port.py",
        "from gateway.models.money import USD\nfrom gateway.core.config import GatewayConfig\n",
    )
    assert check.check_file(file_path, tmp_path) == []


def test_adapter_may_not_import_the_api_layer(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/adapters/billing_adapter.py", "from gateway.api.deps import get_db\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.api.deps", "Forbidden import in Adapters")]


def test_adapter_may_use_the_layers_below_it(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "gateway/adapters/billing_adapter.py",
        "from gateway.ports.billing_port import BillingPort\nfrom gateway.services.budgets import reserve\n",
    )
    assert check.check_file(file_path, tmp_path) == []


def test_service_may_not_name_a_concrete_adapter(tmp_path: Path) -> None:
    # Only the composition root binds a concrete adapter; a service depends on
    # the port and takes whatever the container resolved.
    file_path = _write(
        tmp_path,
        "gateway/services/thing.py",
        "from gateway.adapters.billing_adapter import NullBillingAdapter\n",
    )
    assert check.check_file(file_path, tmp_path) == [
        (1, "gateway.adapters.billing_adapter", "Forbidden import in Services")
    ]


def test_service_may_import_a_port(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway.ports.billing_port import BillingPort\n")
    assert check.check_file(file_path, tmp_path) == []


def test_the_composition_root_may_name_a_concrete_adapter(tmp_path: Path) -> None:
    # The one file exempted from the root rule's ban on gateway.adapters, which
    # is what leaves it the one file allowed to name an adapter.
    file_path = _write(
        tmp_path,
        "gateway/container.py",
        "from gateway.adapters.billing_adapter import NullBillingAdapter\n",
    )
    assert check.check_file(file_path, tmp_path) == []


def test_an_adapter_may_name_its_siblings(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "gateway/adapters/billing_adapter.py",
        "from gateway.adapters.entitlement_adapter import BaseEntitlementAdapter\n",
    )
    assert check.check_file(file_path, tmp_path) == []


@pytest.mark.parametrize(
    "relative_path",
    ["gateway/main.py", "gateway/cli.py", "gateway/core/config.py", "gateway/auth/models.py", "gateway/db/base.py"],
)
def test_an_unlayered_module_may_not_name_a_concrete_adapter(tmp_path: Path, relative_path: str) -> None:
    # The gap a layer-by-layer ban leaves: these answer to no gateway/<layer>
    # rule, so without the root rule's ban they could shortcut past the
    # container and pin a capability to one implementation. gateway/main.py is
    # where that shortcut would most naturally be written, since it is already
    # the file that builds the container.
    file_path = _write(tmp_path, relative_path, "from gateway.adapters.billing_adapter import NullBillingAdapter\n")
    assert check.check_file(file_path, tmp_path) == [
        (1, "gateway.adapters.billing_adapter", "Forbidden import in OSS base")
    ]


def test_shared_types_may_not_import_other_gateway_layers(tmp_path: Path) -> None:
    # gateway/types holds leaf data shapes that every layer may depend on (the
    # routing Attempt is built by services and executed by the API layer), so it
    # must not reach back into any of them.
    file_path = _write(
        tmp_path,
        "gateway/types/attempt.py",
        "from gateway.services.provider_kwargs import ResolvedProvider\n",
    )
    assert check.check_file(file_path, tmp_path) == [
        (1, "gateway.services.provider_kwargs", "Forbidden import in Shared types")
    ]


def test_shared_types_may_import_third_party(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/types/attempt.py", "from any_llm import LLMProvider\n")
    assert check.check_file(file_path, tmp_path) == []


def test_repository_naming_convention(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/repositories/__init__.py", "")
    _write(tmp_path, "gateway/repositories/users_repository.py", "")
    _write(tmp_path, "gateway/repositories/budgets/budget_repositories.py", "")
    _write(tmp_path, "gateway/repositories/helpers.py", "")
    violations = check.check_naming_conventions(tmp_path)
    assert violations == [
        "Repository file gateway/repositories/helpers.py must end with '_repository.py'"
        " or, for a bundle of repositories, '_repositories.py'"
    ]


def test_service_importing_the_overlay_is_flagged(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway.overlay.billing import charge\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.overlay.billing", "Forbidden import in OSS base")]


def test_adapter_importing_the_overlay_is_flagged(tmp_path: Path) -> None:
    # The issue's example: an OSS adapter must not reference an enterprise one.
    file_path = _write(
        tmp_path, "gateway/adapters/thing_adapter.py", "from gateway.overlay.adapters import Enterprise\n"
    )
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.overlay.adapters", "Forbidden import in OSS base")]


def test_file_outside_named_layers_importing_the_overlay_is_flagged(tmp_path: Path) -> None:
    # The boundary covers the whole gateway tree, not only the named layers.
    file_path = _write(tmp_path, "gateway/main.py", "from gateway.overlay import register\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.overlay", "Forbidden import in OSS base")]


def test_overlay_boundary_via_from_gateway_import_is_flagged(tmp_path: Path) -> None:
    # `from gateway import overlay` binds the submodule gateway.overlay, which the resolver flags.
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway import overlay\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.overlay", "Forbidden import in OSS base")]


def test_overlay_prefix_requires_a_module_boundary(tmp_path: Path) -> None:
    # A sibling module whose name merely starts with "overlay" is not the overlay.
    file_path = _write(tmp_path, "gateway/services/thing.py", "import gateway.overlaything\n")
    assert check.check_file(file_path, tmp_path) == []


def test_service_importing_the_top_level_overlay_is_flagged(tmp_path: Path) -> None:
    # The overlay is imported as "overlay.*" today; a build that composes it
    # into the gateway namespace instead would spell it "gateway.overlay.*".
    # Both spellings are the boundary.
    file_path = _write(tmp_path, "gateway/services/thing.py", "from overlay.adapters import Enterprise\n")
    assert check.check_file(file_path, tmp_path) == [(1, "overlay.adapters", "Forbidden import in OSS base")]


def test_top_level_overlay_prefix_requires_a_module_boundary(tmp_path: Path) -> None:
    # An unrelated module whose name merely starts with "overlay" is not the overlay.
    file_path = _write(tmp_path, "gateway/services/thing.py", "import overlaything\n")
    assert check.check_file(file_path, tmp_path) == []


def test_test_suite_importing_the_overlay_is_flagged(tmp_path: Path) -> None:
    # The OSS test suite answers to the same boundary as the gateway package.
    file_path = _write(
        tmp_path,
        "tests/unit/services/test_thing.py",
        "from overlay.adapters.billing_adapter import WalletBillingAdapter\n",
    )
    assert check.check_file(file_path, tmp_path) == [
        (1, "overlay.adapters.billing_adapter", "Forbidden import in OSS test suite")
    ]


def test_main_discovers_tests_root_and_fails_on_overlay_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Unlike the check_file() tests above, this exercises main() itself: that
    # it walks TESTS_ROOT (not just GATEWAY_ROOT) and resolves those paths
    # against REPO_ROOT, using the gateway-composed overlay spelling.
    _write(tmp_path, "src/gateway/__init__.py", "")
    _write(tmp_path, "tests/unit/test_thing.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "tests/unit/test_thing.py", "from gateway.overlay.billing import charge\n")
    assert check.main() == 1


@pytest.mark.parametrize(
    "forbidden",
    ["gateway.core.config", "uvicorn", "any_llm", "sqlalchemy", "sqlmodel", "pydantic", "pydantic_settings", "fastapi"],
)
def test_light_cli_importing_the_server_stack_is_flagged(tmp_path: Path, forbidden: str) -> None:
    file_path = _write(tmp_path, "otari_agent/hook.py", f"from {forbidden} import thing\n")
    assert check.check_file(file_path, tmp_path) == [(1, forbidden, "Forbidden import in Light CLI (otari-agent)")]


def test_light_cli_may_import_its_own_dependencies(tmp_path: Path) -> None:
    imports = "import click\nimport httpx\nimport yaml\nfrom dotenv import load_dotenv\n"
    file_path = _write(tmp_path, "otari_agent/hook.py", imports)
    assert check.check_file(file_path, tmp_path) == []


def test_the_light_cli_attach_point_may_name_the_gateway_cli(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "otari_agent/cli.py", "from gateway.cli import register\n")
    assert check.check_file(file_path, tmp_path) == []


@pytest.mark.parametrize(
    ("statement", "module"),
    [("from gateway.core.config import load_config", "gateway.core.config"), ("import gateway", "gateway")],
)
def test_the_attach_point_may_not_import_the_rest_of_the_gateway(tmp_path: Path, statement: str, module: str) -> None:
    file_path = _write(tmp_path, "otari_agent/cli.py", f"{statement}\n")
    assert check.check_file(file_path, tmp_path) == [(1, module, "Forbidden import in Light CLI (otari-agent)")]


def test_the_light_cli_attach_point_still_may_not_import_the_server_stack(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "otari_agent/cli.py", "import uvicorn\n")
    assert check.check_file(file_path, tmp_path) == [(1, "uvicorn", "Forbidden import in Light CLI (otari-agent)")]


def test_light_cli_may_not_discover_entry_points(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "otari_agent/cli.py", "from importlib.metadata import version\n")
    assert check.check_file(file_path, tmp_path) == [(1, "importlib.metadata", _DISCOVERY_MESSAGE)]


def test_main_walks_the_light_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/__init__.py", "")
    _write(tmp_path, "tests/unit/test_thing.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "cli/src/otari_agent/hook.py", "from gateway.core.config import load_config\n")
    assert check.main() == 1


@pytest.mark.parametrize("relative_path", ["src/any_search/_api.py", "tests/test_fake.py", "scripts/record_fixture.py"])
@pytest.mark.parametrize(
    "forbidden", ["gateway", "gateway.core.config", "sqlalchemy", "sqlmodel", "fastapi", "uvicorn"]
)
def test_a_library_importing_the_gateway_or_the_server_stack_is_flagged(
    tmp_path: Path, relative_path: str, forbidden: str
) -> None:
    file_path = _write(tmp_path, relative_path, f"import {forbidden}\n")
    assert check.check_file(file_path, tmp_path, "any_search") == [
        (1, forbidden, "Forbidden import in Library (any-search)")
    ]


def test_a_library_may_import_httpx_pydantic_and_itself(tmp_path: Path) -> None:
    imports = "import httpx\nfrom pydantic import BaseModel\nfrom any_fetch._types import FetchedPage\n"
    file_path = _write(tmp_path, "tests/test_fake.py", imports)
    assert check.check_file(file_path, tmp_path, "any_fetch") == []


@pytest.mark.parametrize(("rule_key", "other"), [("any_search", "any_fetch"), ("any_fetch", "any_search")])
def test_the_libraries_may_not_import_each_other(tmp_path: Path, rule_key: str, other: str) -> None:
    file_path = _write(tmp_path, "src/module.py", f"from {other} import thing\n")
    [(_, module, _)] = check.check_file(file_path, tmp_path, rule_key)
    assert module == other


def test_a_library_may_not_discover_entry_points(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "src/any_fetch/_registry.py", "from importlib.metadata import entry_points\n")
    assert check.check_file(file_path, tmp_path, "any_fetch") == [(1, "importlib.metadata", _DISCOVERY_MESSAGE)]


def test_main_walks_the_libraries_tests_included(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/__init__.py", "")
    _write(tmp_path, "tests/unit/test_thing.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "any-fetch/tests/conformance/test_conformance.py", "import gateway\n")
    assert check.main() == 1


def test_main_refuses_to_run_without_a_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/__init__.py", "")
    _write(tmp_path, "tests/unit/test_thing.py", "")
    _point_main_at(tmp_path, monkeypatch)
    monkeypatch.setitem(check.LIBRARY_ROOTS, "any_search", tmp_path / "missing")
    assert check.main() == 1


def test_real_gateway_tree_is_clean() -> None:
    assert check.main() == 0


def test_the_gateway_package_is_an_allowed_top_level_package(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/__init__.py", "")
    assert check.check_top_level_packages(tmp_path) == []


@pytest.mark.parametrize("relative_path", ["otari_probe/__init__.py", "otari_probe/routes.py", "otari_probe.py"])
def test_a_new_top_level_package_is_refused(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, "gateway/__init__.py", "")
    _write(tmp_path, relative_path, "")
    assert check.check_top_level_packages(tmp_path) == [
        f"Top-level package src/{relative_path.split('/')[0]} is not allowed; "
        "a feature in this repository belongs under src/gateway and in its feature registry"
    ]


def test_a_directory_without_python_source_is_not_a_package(tmp_path: Path) -> None:
    # Installing the project in editable mode writes gateway.egg-info beside the package.
    _write(tmp_path, "gateway/__init__.py", "")
    _write(tmp_path, "gateway.egg-info/PKG-INFO", "")
    assert check.check_top_level_packages(tmp_path) == []


def test_main_fails_on_a_new_top_level_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "src/otari_probe/__init__.py", "")
    assert check.main() == 1


def test_a_service_may_not_import_the_feature_registry(tmp_path: Path) -> None:
    # Only the app wiring reads the registry; a service that imported it could
    # register itself, which is discovery by another name.
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway.features import CORE_FEATURES\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.features", "Forbidden import in Services")]


def test_a_route_may_not_import_the_feature_registry(tmp_path: Path) -> None:
    file_path = _write(tmp_path, "gateway/api/routes/alerts.py", "from gateway.features import CORE_FEATURES\n")
    assert check.check_file(file_path, tmp_path) == [(1, "gateway.features", "Forbidden import in API routes")]


@pytest.mark.parametrize("relative_path", ["gateway/features.py", "gateway/main.py", "gateway/services/thing.py"])
def test_entry_point_discovery_is_forbidden_anywhere_under_gateway(tmp_path: Path, relative_path: str) -> None:
    # The registry is a literal tuple on purpose; importlib.metadata is how the
    # alternative gets written, and the message says so.
    file_path = _write(tmp_path, relative_path, "from importlib.metadata import entry_points\n")
    assert check.check_file(file_path, tmp_path) == [(1, "importlib.metadata", _DISCOVERY_MESSAGE)]


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("from importlib import metadata\n", "importlib.metadata"),
        ("import importlib_metadata\n", "importlib_metadata"),
        ("import pkg_resources\n", "pkg_resources"),
    ],
)
def test_every_spelling_of_entry_point_discovery_is_forbidden(tmp_path: Path, source: str, module: str) -> None:
    file_path = _write(tmp_path, "gateway/core/plugins.py", source)
    assert check.check_file(file_path, tmp_path) == [(1, module, _DISCOVERY_MESSAGE)]


_SERVICE_REMEDY = "a service reaches the database through its repositories and the Unit of Work"
_ROUTE_REMEDY = "a route reaches the database through a service"


def _use_empty_database_baselines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "SERVICE_DATABASE_IMPORT_BASELINE", ())
    monkeypatch.setattr(check, "ROUTE_DATABASE_IMPORT_BASELINE", ())


@pytest.mark.parametrize(
    ("relative_path", "remedy"),
    [
        ("gateway/services/thing_service.py", _SERVICE_REMEDY),
        ("gateway/services/things/_store.py", _SERVICE_REMEDY),
        ("gateway/api/routes/things.py", _ROUTE_REMEDY),
    ],
)
@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("from sqlalchemy import select\n", "sqlalchemy"),
        ("from sqlmodel import col\n", "sqlmodel"),
        ("import sqlalchemy as sa\n", "sqlalchemy"),
        ("from sqlalchemy.ext.asyncio import AsyncSession\n", "sqlalchemy.ext.asyncio"),
        (
            "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import sqlmodel.sql.expression\n",
            "sqlmodel.sql.expression",
        ),
    ],
)
def test_a_route_or_service_that_imports_a_database_library_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, remedy: str, source: str, module: str
) -> None:
    _use_empty_database_baselines(monkeypatch)
    _write(tmp_path, relative_path, source)
    line = source.count("\n")
    assert check.check_database_imports(tmp_path) == [f"{relative_path}:{line} imports {module}; {remedy}"]


def test_a_service_class_that_takes_a_session_in_its_constructor_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_empty_database_baselines(monkeypatch)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from sqlalchemy.ext.asyncio import AsyncSession\n\n\n"
        "class ThingService:\n"
        "    def __init__(self, db: AsyncSession) -> None:\n"
        "        self._db = db\n",
    )
    assert check.check_database_imports(tmp_path) == [
        f"gateway/services/thing_service.py:1 imports sqlalchemy.ext.asyncio; {_SERVICE_REMEDY}"
    ]


@pytest.mark.parametrize(
    ("relative_path", "source"),
    [
        ("gateway/repositories/thing_repository.py", "from sqlalchemy import select\n"),
        ("gateway/services/thing_service.py", "import sqlalchemylike\nfrom gateway.repositories import things\n"),
    ],
)
def test_a_repository_or_an_unrelated_import_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, source: str
) -> None:
    _use_empty_database_baselines(monkeypatch)
    _write(tmp_path, relative_path, source)
    assert check.check_database_imports(tmp_path) == []


def test_a_module_on_its_layers_baseline_may_import_a_database_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "SERVICE_DATABASE_IMPORT_BASELINE", ("gateway/services/thing_service.py",))
    monkeypatch.setattr(check, "ROUTE_DATABASE_IMPORT_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", "from sqlalchemy import select\n")
    _write(tmp_path, "gateway/api/routes/things.py", "from sqlalchemy import select\n")
    assert check.check_database_imports(tmp_path) == [
        f"gateway/api/routes/things.py:1 imports sqlalchemy; {_ROUTE_REMEDY}"
    ]


@pytest.mark.parametrize("source", ["from gateway.repositories import things\n", None])
def test_a_database_baseline_entry_that_imports_no_database_library_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "SERVICE_DATABASE_IMPORT_BASELINE", ())
    monkeypatch.setattr(check, "ROUTE_DATABASE_IMPORT_BASELINE", ("gateway/api/routes/things.py",))
    _write(tmp_path, "gateway/api/routes/other.py", "")
    if source is not None:
        _write(tmp_path, "gateway/api/routes/things.py", source)
    assert check.check_database_imports(tmp_path) == [
        "gateway/api/routes/things.py is on the database import baseline but imports no database library; "
        "remove it from the baseline"
    ]


def test_main_fails_on_a_service_that_imports_a_database_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "src/gateway/services/thing_service.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    monkeypatch.setattr(check, "FLAT_MODULE_BASELINE", ("gateway/services/thing_service.py",))
    assert check.main() == 0
    _write(tmp_path, "src/gateway/services/thing_service.py", "from sqlalchemy import select\n")
    assert check.main() == 1


def _use_empty_flat_module_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "FLAT_MODULE_BASELINE", ())


@pytest.mark.parametrize("layer", ["services", "repositories"])
def test_a_new_top_level_module_in_a_domain_layer_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    _use_empty_flat_module_baseline(monkeypatch)
    _write(tmp_path, f"gateway/{layer}/__init__.py", "")
    _write(tmp_path, f"gateway/{layer}/things.py", "")
    assert check.check_flat_modules(tmp_path) == [
        f"gateway/{layer}/things.py is a new top-level module; put it in its domain's package under gateway/{layer}/"
    ]


@pytest.mark.parametrize("layer", ["services", "repositories"])
def test_a_domain_package_in_a_domain_layer_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    _use_empty_flat_module_baseline(monkeypatch)
    _write(tmp_path, f"gateway/{layer}/__init__.py", "")
    _write(tmp_path, f"gateway/{layer}/things/__init__.py", "")
    _write(tmp_path, f"gateway/{layer}/things/_store.py", "")
    _write(tmp_path, f"gateway/{layer}/things/nested/__init__.py", "")
    _write(tmp_path, f"gateway/{layer}/things/nested/deep.py", "")
    assert check.check_flat_modules(tmp_path) == []


def test_a_directory_of_modules_without_an_init_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_flat_module_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/things/store.py", "")
    _write(tmp_path, "gateway/services/cache/readme.txt", "")
    assert check.check_flat_modules(tmp_path) == [
        "gateway/services/things has no __init__.py; a domain package needs one"
    ]


@pytest.mark.parametrize("layer", ["services", "repositories"])
def test_a_nested_directory_of_modules_without_an_init_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str
) -> None:
    _use_empty_flat_module_baseline(monkeypatch)
    _write(tmp_path, f"gateway/{layer}/things/__init__.py", "")
    _write(tmp_path, f"gateway/{layer}/things/handlers/task.py", "")
    _write(tmp_path, f"gateway/{layer}/things/assets/readme.txt", "")
    assert check.check_flat_modules(tmp_path) == [
        f"gateway/{layer}/things/handlers has no __init__.py; a domain package needs one"
    ]


def test_a_module_on_the_flat_module_baseline_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "FLAT_MODULE_BASELINE", ("gateway/repositories/users_repository.py",))
    _write(tmp_path, "gateway/repositories/users_repository.py", "")
    assert check.check_flat_modules(tmp_path) == []


def test_a_flat_module_baseline_entry_that_no_longer_exists_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "FLAT_MODULE_BASELINE", ("gateway/services/things.py",))
    _write(tmp_path, "gateway/services/things/__init__.py", "")
    assert check.check_flat_modules(tmp_path) == [
        "gateway/services/things.py is on the flat module baseline but no longer exists; remove it from the baseline"
    ]


def test_main_fails_on_a_new_top_level_service_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "src/gateway/services/things.py", "")
    assert check.main() == 1


_TRANSACTION_REMEDY = "only a Unit of Work block ends a transaction"


@pytest.mark.parametrize(
    ("source", "call"),
    [
        ("async def save(db: object) -> None:\n    await db.commit()\n", "commit"),
        ("async def save(db: object) -> None:\n    await db.rollback()\n", "rollback"),
        ("class Store:\n    async def save(self) -> None:\n        await self.db.commit()\n", "commit"),
    ],
)
def test_a_commit_or_rollback_outside_the_unit_of_work_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, call: str
) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ())
    monkeypatch.setattr(check, "SAVEPOINT_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", source)
    line = source.count("\n")
    assert check.check_transaction_control(tmp_path) == [
        f"gateway/services/thing_service.py:{line} calls {call}; {_TRANSACTION_REMEDY}"
    ]


def test_the_unit_of_work_may_commit_and_roll_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ())
    monkeypatch.setattr(check, "SAVEPOINT_BASELINE", ())
    _write(
        tmp_path,
        "gateway/core/unit_of_work.py",
        "class UnitOfWork:\n    async def end(self) -> None:\n        await self._session.commit()\n",
    )
    assert check.check_transaction_control(tmp_path) == []


def test_a_module_on_the_transaction_baseline_may_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ("gateway/services/thing_service.py",))
    monkeypatch.setattr(check, "SAVEPOINT_BASELINE", ())
    _write(
        tmp_path, "gateway/services/thing_service.py", "async def save(db: object) -> None:\n    await db.commit()\n"
    )
    assert check.check_transaction_control(tmp_path) == []


@pytest.mark.parametrize(
    "source",
    [
        "async def save(db: object) -> None:\n    await db.flush()\n",
        # A savepoint alone does not keep a module on the baseline, or its later commits would pass unchecked.
        "async def save(db: object) -> None:\n    async with db.begin_nested():\n        pass\n",
        None,
    ],
)
def test_a_transaction_baseline_entry_that_ends_no_transaction_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ("gateway/services/thing_service.py",))
    monkeypatch.setattr(check, "SAVEPOINT_BASELINE", ())
    _write(tmp_path, "gateway/services/other_service.py", "")
    if source is not None:
        _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_transaction_control(tmp_path) == [
        "gateway/services/thing_service.py is on the transaction control baseline but calls neither commit nor "
        "rollback; remove it from the baseline"
    ]


_SAVEPOINT_REMEDY = "a repository opens a savepoint only to diagnose a refusal of its own write"
_SAVEPOINT_SOURCE = "async def add(db: object) -> None:\n    async with db.begin_nested():\n        pass\n"


def test_a_savepoint_off_the_savepoint_baseline_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ())
    monkeypatch.setattr(check, "SAVEPOINT_BASELINE", ())
    _write(tmp_path, "gateway/repositories/thing_repository.py", _SAVEPOINT_SOURCE)
    assert check.check_transaction_control(tmp_path) == [
        f"gateway/repositories/thing_repository.py:2 calls begin_nested; {_SAVEPOINT_REMEDY}"
    ]


def test_the_savepoint_baseline_allows_a_savepoint_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ())
    monkeypatch.setattr(check, "SAVEPOINT_BASELINE", ("gateway/repositories/thing_repository.py",))
    _write(tmp_path, "gateway/repositories/thing_repository.py", _SAVEPOINT_SOURCE + "    await db.commit()\n")
    assert check.check_transaction_control(tmp_path) == [
        f"gateway/repositories/thing_repository.py:4 calls commit; {_TRANSACTION_REMEDY}"
    ]


def test_a_savepoint_baseline_entry_that_opens_no_savepoint_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ())
    monkeypatch.setattr(check, "SAVEPOINT_BASELINE", ("gateway/repositories/thing_repository.py",))
    _write(tmp_path, "gateway/repositories/thing_repository.py", "async def add(db: object) -> None:\n    pass\n")
    assert check.check_transaction_control(tmp_path) == [
        "gateway/repositories/thing_repository.py is on the savepoint baseline but opens no savepoint; "
        "remove it from the baseline"
    ]


@pytest.mark.parametrize(
    "relative_path", ["gateway/services/thing_service.py", "gateway/api/routes/things.py", "gateway/core/thing.py"]
)
def test_importing_the_session_accessor_outside_repositories_is_flagged(tmp_path: Path, relative_path: str) -> None:
    file_path = _write(tmp_path, relative_path, "from gateway.core.unit_of_work import session_for\n")
    assert check.check_file(file_path, tmp_path) == [
        (1, "gateway.core.unit_of_work.session_for", f"Forbidden import in {check.SESSION_ACCESSOR_RULE}")
    ]


def test_a_repository_may_import_the_session_accessor(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path,
        "gateway/repositories/thing_repository.py",
        "from gateway.core.unit_of_work import UnitOfWork, session_for\n",
    )
    assert check.check_file(file_path, tmp_path) == []


def test_a_service_may_import_the_unit_of_work_itself(tmp_path: Path) -> None:
    file_path = _write(
        tmp_path, "gateway/services/thing_service.py", "from gateway.core.unit_of_work import UnitOfWork\n"
    )
    assert check.check_file(file_path, tmp_path) == []


def test_main_fails_on_a_commit_outside_the_unit_of_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "src/gateway/services/thing_service.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    monkeypatch.setattr(check, "FLAT_MODULE_BASELINE", ("gateway/services/thing_service.py",))
    assert check.main() == 0
    _write(
        tmp_path,
        "src/gateway/services/thing_service.py",
        "async def save(db: object) -> None:\n    await db.commit()\n",
    )
    assert check.main() == 1


_UNIT_OF_WORK_REMEDY = (
    "a request takes one from get_unit_of_work and a worker job from create_unit_of_work or create_log_unit_of_work"
)
_FACTORY_SOURCE = "def get_unit_of_work(db: object) -> UnitOfWork:\n    return UnitOfWork(db)\n"


def test_the_request_factory_may_construct_a_unit_of_work(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/_base.py", _FACTORY_SOURCE)
    assert check.check_unit_of_work_construction(tmp_path) == []


def test_the_unit_of_work_module_may_construct_one(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "gateway/core/unit_of_work.py",
        "def create_unit_of_work(session: object) -> UnitOfWork:\n    return UnitOfWork(session)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == []


@pytest.mark.parametrize(
    "relative_path", ["gateway/api/routes/things.py", "gateway/services/thing_service.py", "gateway/api/deps/_base.py"]
)
def test_constructing_a_unit_of_work_elsewhere_is_flagged(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path, "def build(db: object) -> UnitOfWork:\n    return UnitOfWork(db)\n")
    assert check.check_unit_of_work_construction(tmp_path) == [
        f"{relative_path}:2 constructs a UnitOfWork; {_UNIT_OF_WORK_REMEDY}"
    ]


def test_a_unit_of_work_reached_through_its_module_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/_base.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "def build(db: object) -> object:\n    return unit_of_work.UnitOfWork(db)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == [
        f"gateway/services/thing_service.py:2 constructs a UnitOfWork; {_UNIT_OF_WORK_REMEDY}"
    ]


def test_naming_the_unit_of_work_as_a_type_is_not_a_construction(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/_base.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "def save(uow: UnitOfWork | None) -> UnitOfWork | None:\n    return uow\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == []


def test_main_fails_on_a_unit_of_work_built_outside_the_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "src/gateway/services/thing_service.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    monkeypatch.setattr(check, "FLAT_MODULE_BASELINE", ("gateway/services/thing_service.py",))
    assert check.main() == 0
    _write(
        tmp_path,
        "src/gateway/services/thing_service.py",
        "def build(db: object) -> object:\n    return UnitOfWork(db)\n",
    )
    assert check.main() == 1


def test_only_the_module_level_factory_is_exempt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "gateway/api/deps/_base.py",
        "class Deps:\n    def get_unit_of_work(self, db: object) -> UnitOfWork:\n        return UnitOfWork(db)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == [
        f"gateway/api/deps/_base.py:3 constructs a UnitOfWork; {_UNIT_OF_WORK_REMEDY}"
    ]


def test_renaming_the_unit_of_work_on_import_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/_base.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from gateway.core.unit_of_work import UnitOfWork as UoW\n\n\n"
        "def build(db: object) -> UoW:\n    return UoW(db)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == [
        "gateway/services/thing_service.py:1 imports UnitOfWork as UoW; "
        "the rule reads the name at the call site, so import it under its own name"
    ]


def test_importing_the_unit_of_work_under_its_own_name_is_clean(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/_base.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from gateway.core.unit_of_work import UnitOfWork\n\n\n"
        "def save(uow: UnitOfWork) -> UnitOfWork:\n    return uow\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == []


def test_renaming_the_unit_of_work_on_a_relative_import_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/_base.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from ..core.unit_of_work import UnitOfWork as UoW\n\n\ndef build(db: object) -> UoW:\n    return UoW(db)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == [
        "gateway/services/thing_service.py:1 imports UnitOfWork as UoW; "
        "the rule reads the name at the call site, so import it under its own name"
    ]


_LISTENER_DEFAULT_REMEDY = "a caller that leaves a listener out skips it with no error, so every caller passes one"
_LISTENER_MODULE = "gateway/services/things/thing_service.py"


def _listener_default(line: int, name: str, path: str) -> str:
    return f"{_LISTENER_MODULE}:{line} gives the listener {name} of {path} a default; {_LISTENER_DEFAULT_REMEDY}"


@pytest.mark.parametrize(
    ("source", "line", "path"),
    [
        ("def wire(budget_listener: BudgetListener | None = None) -> None: ...\n", 1, "wire"),
        ("async def wire(budget_listener: BudgetListener | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(*, budget_listener: BudgetListener | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(budget_listener: BudgetListener | None = None, /) -> None: ...\n", 1, "wire"),
        ("def wire(budget_listener: BudgetListener = NoOpListener()) -> None: ...\n", 1, "wire"),
        ("def wire(budget_listener: BudgetListenerBuilder | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(budget_listener: BudgetListenerDep | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(budget_listener: tenancy.BudgetListener | None = None) -> None: ...\n", 1, "wire"),
        (
            "def wire(budget_listener: Callable[[UnitOfWork], BudgetListener] | None = None) -> None: ...\n",
            1,
            "wire",
        ),
        (
            "def wire(budget_listener: Annotated[BudgetListener, Depends(get)] | None = None) -> None: ...\n",
            1,
            "wire",
        ),
        ('def wire(budget_listener: "BudgetListener | None" = None) -> None: ...\n', 1, "wire"),
        ('def wire(budget_listener: Callable[[], "BudgetListener"] | None = None) -> None: ...\n', 1, "wire"),
        (
            "class Thing:\n    def __init__(self, budget_listener: BudgetListener | None = None) -> None: ...\n",
            2,
            "Thing.__init__",
        ),
        (
            "def outer() -> None:\n    def inner(budget_listener: BudgetListener | None = None) -> None: ...\n",
            2,
            "outer.inner",
        ),
        ("@dataclass\nclass Thing:\n    budget_listener: BudgetListener | None = None\n", 3, "Thing"),
        (
            "@dataclass\nclass Thing:\n    budget_listener: BudgetListener | None = field(default=None)\n",
            3,
            "Thing",
        ),
        ("class Thing(BaseModel):\n    budget_listener: BudgetListener | None = Field(None)\n", 2, "Thing"),
        (
            "class Thing(BaseModel):\n    budget_listener: BudgetListener | None = Field(None, init=False)\n",
            2,
            "Thing",
        ),
        (
            "class Thing(BaseModel):\n    budget_listener: BudgetListener = Field(default_factory=make)\n",
            2,
            "Thing",
        ),
        ("class Thing(BaseModel):\n    budget_listener: BudgetListener = Field(**options)\n", 2, "Thing"),
        ("@dataclass\nclass Thing:\n    budget_listener: BudgetListener = field(default=...)\n", 3, "Thing"),
        ("@dataclass\nclass Thing:\n    budget_listener: BudgetListener = field(**options)\n", 3, "Thing"),
    ],
)
def test_a_listener_with_a_default_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, line: int, path: str
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(tmp_path, _LISTENER_MODULE, source)
    assert check.check_listener_defaults(tmp_path) == [_listener_default(line, "budget_listener", path)]


def test_each_listener_with_a_default_in_one_signature_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(
        tmp_path,
        _LISTENER_MODULE,
        "def wire(\n"
        "    budget_listener: BudgetListener | None = None,\n"
        "    budget_listener: BudgetListener | None = None,\n"
        ") -> None: ...\n",
    )
    assert check.check_listener_defaults(tmp_path) == [
        _listener_default(2, "budget_listener", "wire"),
        _listener_default(3, "budget_listener", "wire"),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "def wire(budget_listener: BudgetListener | None) -> None: ...\n",
        "def wire(budget_listener: BudgetListener, /, limit: int = 3) -> None: ...\n",
        "def wire(*, budget_listener: BudgetListener) -> None: ...\n",
        "def wire(budget_listener: BudgetListener, limit: int = 3) -> None: ...\n",
        "def wire(*, budget_listener: BudgetListener, limit: int = 3) -> None: ...\n",
        "def wire(settings: Settings | None = None) -> None: ...\n",
        'def wire(kind: Literal["BudgetListener"] = "BudgetListener") -> None: ...\n',
        'def wire(label: Annotated[str, "BudgetListener"] = "") -> None: ...\n',
        'def wire(label: "\\x00" = "") -> None: ...\n',
        "class Thing:\n    budget_listener: BudgetListener\n",
        "@dataclass\nclass Thing:\n    budget_listener: BudgetListener = field(kw_only=True)\n",
        'class Thing(BaseModel):\n    budget_listener: BudgetListener = Field(description="x")\n',
        "class Thing(BaseModel):\n    budget_listener: BudgetListener = Field(...)\n",
        "class Thing(BaseModel):\n    budget_listener: BudgetListener = Field(default=...)\n",
        "class Thing(BaseModel):\n    budget_listener: BudgetListener = Field(default_factory=None)\n",
        "@dataclass\nclass Thing:\n    budget_listener: BudgetListener = field(init=False, default=None)\n",
        "class Thing(BaseModel):\n    _budget_listener: BudgetListener | None = PrivateAttr(default=None)\n",
        "@dataclass\nclass Thing:\n    budget_listener: ClassVar[BudgetListener] = NoOpListener()\n",
        "def wire(label: Annotated[()] = 0) -> None: ...\n",
        "def wire(make: Callable[[BudgetListener], Service] = build) -> None: ...\n",
        "def wire(make: Callable[()] = build) -> None: ...\n",
        "def wire(kind: BudgetListener.Kind = BudgetListener.Kind.A) -> None: ...\n",
    ],
)
def test_a_listener_without_a_default_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(tmp_path, _LISTENER_MODULE, source)
    assert check.check_listener_defaults(tmp_path) == []


def test_a_listener_with_a_default_outside_services_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(tmp_path, "gateway/container.py", "def wire(budget_listener: BudgetListener | None = None) -> None: ...\n")
    assert check.check_listener_defaults(tmp_path) == [
        f"gateway/container.py:1 gives the listener budget_listener of wire a default; {_LISTENER_DEFAULT_REMEDY}"
    ]


def test_a_listener_on_the_baseline_may_have_a_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ((_LISTENER_MODULE, "wire", "budget_listener"),))
    _write(tmp_path, _LISTENER_MODULE, "def wire(budget_listener: BudgetListener | None = None) -> None: ...\n")
    assert check.check_listener_defaults(tmp_path) == []


def test_a_second_listener_default_in_a_module_on_the_baseline_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ((_LISTENER_MODULE, "wire", "budget_listener"),))
    _write(
        tmp_path,
        _LISTENER_MODULE,
        "def wire(budget_listener: BudgetListener | None = None) -> None: ...\n"
        "def rewire(budget_listener: BudgetListener | None = None) -> None: ...\n",
    )
    assert check.check_listener_defaults(tmp_path) == [_listener_default(2, "budget_listener", "rewire")]


@pytest.mark.parametrize(
    "source",
    [
        "def wire(budget_listener: BudgetListener | None) -> None: ...\n",
        "def renamed(budget_listener: BudgetListener) -> None: ...\n",
        None,
    ],
)
def test_a_listener_default_baseline_entry_with_no_default_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ((_LISTENER_MODULE, "wire", "budget_listener"),))
    _write(tmp_path, "gateway/services/things/other.py", "")
    if source is not None:
        _write(tmp_path, _LISTENER_MODULE, source)
    assert check.check_listener_defaults(tmp_path) == [
        f"{_LISTENER_MODULE} is on the listener default baseline for budget_listener of wire, "
        "but no such listener has a default; remove it from the baseline"
    ]


def test_main_fails_on_a_listener_with_a_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "src/gateway/services/things/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(
        tmp_path,
        "src/gateway/services/things/thing_service.py",
        "def wire(budget_listener: BudgetListener | None = None) -> None: ...\n",
    )
    assert check.main() == 1


_DOMAINS_PAGE = """# Backend domains

## The target shape

### Not a domain

## The domains

### api-keys

### budgets

### Shared

## Order of work

### also-not-a-domain
"""


_PAGE_DOMAINS = {"api_keys", "budgets"}


def _domains_page(tmp_path: Path, text: str = _DOMAINS_PAGE) -> Path:
    return _write(tmp_path, "DOMAINS.md", text)


def test_documented_domains_reads_only_the_domains_section() -> None:
    assert check.documented_domains(_DOMAINS_PAGE) == ({"api_keys", "budgets"}, [])


@pytest.mark.parametrize(
    ("heading", "violation"),
    [
        (
            "agent guardrails",
            "DOMAINS.md heading '### agent guardrails' is not a domain name in lower case with hyphens",
        ),
        ("Budgets", "DOMAINS.md heading '### Budgets' is not a domain name in lower case with hyphens"),
        ("budgets", "DOMAINS.md names the domain 'budgets' twice"),
    ],
)
def test_documented_domains_refuses_a_heading_it_cannot_read(heading: str, violation: str) -> None:
    page = _DOMAINS_PAGE.replace("### Shared", f"### {heading}\n\n### Shared")
    assert violation in check.documented_domains(page)[1]


def test_documented_domains_refuses_a_page_with_no_domains_section() -> None:
    assert check.documented_domains("# Backend domains\n") == (
        set(),
        ["DOMAINS.md has no '## The domains' section"],
    )


def test_a_lower_case_shared_heading_is_not_a_domain() -> None:
    domains, violations = check.documented_domains(_DOMAINS_PAGE.replace("### Shared", "### shared"))
    assert (domains, violations) == ({"api_keys", "budgets"}, [])


@pytest.mark.parametrize(
    "relative_path",
    [
        "gateway/services/api_keys/__init__.py",
        "gateway/repositories/budgets/__init__.py",
        "gateway/schemas/budgets.py",
        "gateway/exceptions/api_keys_exceptions.py",
    ],
)
def test_a_location_named_for_a_documented_domain_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {})
    _write(tmp_path, relative_path, "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == []


@pytest.mark.parametrize(
    ("relative_path", "location"),
    [
        ("gateway/services/billing/__init__.py", "gateway/services/billing/"),
        ("gateway/repositories/billing/__init__.py", "gateway/repositories/billing/"),
        ("gateway/schemas/billing.py", "gateway/schemas/billing.py"),
        ("gateway/exceptions/billing_exceptions.py", "gateway/exceptions/billing_exceptions.py"),
    ],
)
def test_a_location_named_for_no_documented_domain_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, location: str
) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {})
    _write(tmp_path, relative_path, "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == [
        f"{location} names no domain in DOMAINS.md; name it for a domain there, or give the new domain a section"
    ]


def test_an_exceptions_module_without_the_suffix_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {})
    _write(tmp_path, "gateway/exceptions/budgets.py", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == [
        "gateway/exceptions/budgets.py is not named <domain>_exceptions.py"
    ]


def test_a_shared_exceptions_module_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {})
    _write(tmp_path, "gateway/exceptions/_base.py", "")
    _write(tmp_path, "gateway/exceptions/shared_exceptions.py", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == []


def test_a_directory_that_is_not_a_package_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {})
    _write(tmp_path, "gateway/services/.mypy_cache/cache.json", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == []


def test_a_location_on_the_domain_name_baseline_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        check,
        "DOMAIN_NAME_BASELINE",
        {"gateway/services/tenancy/": ("__init__.py", "handlers/__init__.py", "handlers/invite.py")},
    )
    _write(tmp_path, "gateway/services/tenancy/__init__.py", "")
    _write(tmp_path, "gateway/services/tenancy/handlers/__init__.py", "")
    _write(tmp_path, "gateway/services/tenancy/handlers/invite.py", "")
    _write(tmp_path, "gateway/services/tenancy/__pycache__/cached.py", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == []


@pytest.mark.parametrize("module", ["invite.py", "handlers/invite.py"])
def test_a_new_module_in_a_package_on_the_domain_name_baseline_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {"gateway/services/tenancy/": ("__init__.py",)})
    _write(tmp_path, "gateway/services/tenancy/__init__.py", "")
    _write(tmp_path, f"gateway/services/tenancy/{module}", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == [
        f"gateway/services/tenancy/{module} is a new module in a package named for no domain; "
        "put it in its domain's package"
    ]


def test_a_listed_module_that_no_longer_exists_must_leave_the_domain_name_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {"gateway/services/tenancy/": ("__init__.py", "invite.py")})
    _write(tmp_path, "gateway/services/tenancy/__init__.py", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == [
        "gateway/services/tenancy/invite.py is on the domain name baseline but no longer exists; "
        "remove it from the baseline"
    ]


def test_a_module_entry_on_the_domain_name_baseline_that_lists_modules_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {"gateway/exceptions/budget_exceptions.py": ("invite.py",)})
    _write(tmp_path, "gateway/exceptions/budget_exceptions.py", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == [
        "gateway/exceptions/budget_exceptions.py is a module, so its domain name baseline entry lists no modules"
    ]


@pytest.mark.parametrize("package", ["tenancy", "budgets"])
def test_a_domain_name_baseline_entry_that_is_gone_or_now_a_domain_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str
) -> None:
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {f"gateway/services/{package}/": ("__init__.py",)})
    if package == "budgets":
        _write(tmp_path, "gateway/services/budgets/__init__.py", "")
    assert check.check_domain_names(tmp_path, _PAGE_DOMAINS) == [
        f"gateway/services/{package}/ is on the domain name baseline but no longer exists or now names a domain; "
        "remove it from the baseline"
    ]


def test_a_missing_domains_page_is_flagged(tmp_path: Path) -> None:
    assert check.read_domains_page(tmp_path / "DOMAINS.md") == (
        set(),
        ["DOMAINS.md not found; the domain names are read from its '## The domains' section"],
    )


def test_the_domains_page_gives_its_domains(tmp_path: Path) -> None:
    assert check.read_domains_page(_domains_page(tmp_path)) == (_PAGE_DOMAINS, [])


def test_main_fails_on_a_package_named_for_no_domain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "src/gateway/services/billing/__init__.py", "")
    assert check.main() == 1


def test_main_fails_on_a_new_module_in_a_package_on_the_domain_name_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "src/gateway/services/tenancy/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    monkeypatch.setattr(check, "DOMAIN_NAME_BASELINE", {"gateway/services/tenancy/": ("__init__.py",)})
    assert check.main() == 0
    _write(tmp_path, "src/gateway/services/tenancy/invite.py", "")
    assert check.main() == 1


_DOMAINS = {"gadgets", "things", "widgets"}
_REPOSITORY_REMEDY = (
    "only the domain's own service and repository packages and the builders in gateway/api/deps/ import its "
    "repositories"
)


def _write_domain_packages(src_root: Path, layer: str, *packages: str) -> None:
    for package in packages:
        _write(src_root, f"gateway/{layer}/{package}/__init__.py", "")


@pytest.mark.parametrize(
    ("relative_path", "source", "module"),
    [
        (
            "gateway/services/widgets/_store.py",
            "from gateway.repositories.things import ThingRepository\n",
            "gateway.repositories.things",
        ),
        (
            "gateway/services/widgets/_store.py",
            "from ...repositories.things.thing_repository import ThingRepository\n",
            "gateway.repositories.things.thing_repository",
        ),
        (
            "gateway/repositories/widgets/widget_repository.py",
            "import gateway.repositories.things\n",
            "gateway.repositories.things",
        ),
        ("gateway/api/routes/things.py", "from gateway.repositories import things\n", "gateway.repositories.things"),
        (
            "gateway/adapters/thing_adapter.py",
            "from gateway.repositories.things import ThingRepository\n",
            "gateway.repositories.things",
        ),
    ],
)
def test_importing_another_domains_repositories_is_flagged(
    tmp_path: Path, relative_path: str, source: str, module: str
) -> None:
    _write_domain_packages(tmp_path, "repositories", "things", "widgets")
    _write_domain_packages(tmp_path, "services", "widgets")
    _write(tmp_path, relative_path, source)
    assert check.check_repository_imports(tmp_path, _DOMAINS) == [
        f"{relative_path}:1 imports {module}; {_REPOSITORY_REMEDY}"
    ]


@pytest.mark.parametrize(
    ("relative_path", "source"),
    [
        ("gateway/services/things/_service.py", "from gateway.repositories.things import ThingRepository\n"),
        ("gateway/repositories/things/other_repository.py", "from .thing_repository import ThingRepository\n"),
        ("gateway/api/deps/__init__.py", "from gateway.repositories.things import ThingRepository\n"),
        ("gateway/api/routes/things.py", "from gateway.repositories.base_repository import BaseRepository\n"),
        ("gateway/api/routes/things.py", "from gateway.repositories.legacy import LegacyRepository\n"),
    ],
)
def test_importing_a_domains_own_or_a_non_domain_repository_is_clean(
    tmp_path: Path, relative_path: str, source: str
) -> None:
    _write_domain_packages(tmp_path, "repositories", "things", "legacy")
    _write_domain_packages(tmp_path, "services", "things")
    _write(tmp_path, relative_path, source)
    assert check.check_repository_imports(tmp_path, _DOMAINS) == []


@pytest.mark.parametrize("relative_path", ["gateway/services/thing_service.py", "gateway/services/legacy/_store.py"])
def test_service_code_outside_every_domain_package_is_not_checked(tmp_path: Path, relative_path: str) -> None:
    _write_domain_packages(tmp_path, "repositories", "things")
    _write_domain_packages(tmp_path, "services", "legacy")
    _write(tmp_path, relative_path, "from gateway.repositories.things import ThingRepository\n")
    assert check.check_repository_imports(tmp_path, _DOMAINS) == []


def test_repository_import_violations_are_reported_in_line_order(tmp_path: Path) -> None:
    _write_domain_packages(tmp_path, "repositories", "things", "widgets")
    _write(
        tmp_path,
        "gateway/core/thing.py",
        "def load():\n    from gateway.repositories.things import T\nfrom gateway.repositories.widgets import W\n",
    )
    assert [violation.split(" imports ")[0] for violation in check.check_repository_imports(tmp_path, _DOMAINS)] == [
        "gateway/core/thing.py:2",
        "gateway/core/thing.py:3",
    ]


def test_main_fails_on_an_import_of_another_domains_repositories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "src/gateway/repositories/things/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "src/gateway/core/thing.py", "from gateway.repositories.things import ThingRepository\n")
    assert check.main() == 1


_SERVICE_PACKAGE_REMEDY = "code outside a domain imports what its service package root exports"


def _write_things_service(src_root: Path) -> None:
    _write(src_root, "gateway/services/things/__init__.py", "from ._store import Store\n")
    _write(src_root, "gateway/services/things/_store.py", "")
    _write(src_root, "gateway/services/things/store.py", "")


@pytest.mark.parametrize(
    ("relative_path", "source", "module"),
    [
        (
            "gateway/api/routes/things.py",
            "from gateway.services.things._store import Store\n",
            "gateway.services.things._store",
        ),
        (
            "gateway/api/routes/things.py",
            "from gateway.services.things import _store\n",
            "gateway.services.things._store",
        ),
        ("gateway/api/deps/_base.py", "import gateway.services.things.store\n", "gateway.services.things.store"),
        ("gateway/services/widgets/_service.py", "from ..things.store import Store\n", "gateway.services.things.store"),
        (
            "gateway/services/thing_service.py",
            "from gateway.services.things.store import Store\n",
            "gateway.services.things.store",
        ),
    ],
)
def test_importing_below_another_domains_service_package_root_is_flagged(
    tmp_path: Path, relative_path: str, source: str, module: str
) -> None:
    _write_things_service(tmp_path)
    _write(tmp_path, relative_path, source)
    assert check.check_service_package_imports(tmp_path, _DOMAINS) == [
        f"{relative_path}:1 imports {module}; {_SERVICE_PACKAGE_REMEDY}"
    ]


@pytest.mark.parametrize(
    ("relative_path", "source"),
    [
        ("gateway/api/routes/things.py", "from gateway.services.things import Store\n"),
        ("gateway/api/routes/things.py", "from gateway.services import things\n"),
        ("gateway/services/things/_service.py", "from gateway.services.things._store import Store\n"),
        ("gateway/services/things/_service.py", "from ._store import Store\n"),
        ("gateway/api/routes/things.py", "from gateway.services.thing_service import ThingService\n"),
        ("gateway/api/routes/things.py", "from gateway.services.legacy._store import Store\n"),
    ],
)
def test_importing_a_service_package_root_own_modules_or_a_non_domain_package_is_clean(
    tmp_path: Path, relative_path: str, source: str
) -> None:
    _write_things_service(tmp_path)
    _write(tmp_path, "gateway/services/thing_service.py", "")
    _write(tmp_path, "gateway/services/legacy/__init__.py", "")
    _write(tmp_path, "gateway/services/legacy/_store.py", "")
    _write(tmp_path, relative_path, source)
    assert check.check_service_package_imports(tmp_path, _DOMAINS) == []


def test_main_fails_on_an_import_below_a_service_package_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/things/__init__.py", "")
    _write(tmp_path, "src/gateway/services/things/_store.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "src/gateway/core/thing.py", "from gateway.services.things._store import Store\n")
    assert check.main() == 1


_THING = "gateway.models.things.Thing"
_THING_REPOSITORY = "gateway/repositories/things/thing_repository.py"
_THING_REMEDY = f"only {_THING_REPOSITORY} constructs or queries it"
_MODELS_SOURCE = (
    "from sqlmodel import Field, SQLModel\n\nfrom gateway.models.base import Base\n\n\n"
    'class Thing(Base):\n    __tablename__ = "things"\n\n\n'
    "class Gadget(SQLModel, table=True):\n    id: int = Field(primary_key=True)\n\n\n"
    "class ThingCreate(SQLModel):\n    name: str\n"
)


def _thing_access(
    repository: str | None = _THING_REPOSITORY, baseline: tuple[str, ...] = ()
) -> dict[str, dict[str, object]]:
    return {
        _THING: {"repository": repository, "baseline": baseline},
        "gateway.models.things.Gadget": {"repository": None, "baseline": ()},
    }


def _write_thing_model(src_root: Path) -> None:
    _write(src_root, "gateway/models/things.py", _MODELS_SOURCE)
    _write(src_root, "gateway/repositories/things/__init__.py", "")
    _write(src_root, _THING_REPOSITORY, "from gateway.models.things import Thing\n\nMODEL = Thing\n")


@pytest.mark.parametrize(
    "source",
    [
        "from gateway.models.things import Thing\n\n\ndef build():\n    return Thing()\n",
        "from gateway.models.things import Thing\n\n\ndef build():\n    return select(Thing)\n",
        "from gateway.models.things import Thing\n\n\ndef build(name):\n    return Thing.name == name\n",
        "from gateway.models.things import Thing as Row\n\n\ndef build():\n    return Row()\n",
        "from gateway.models import things\n\n\ndef build():\n    return things.Thing()\n",
        "import gateway.models.things as things\n\n\ndef build():\n    return things.Thing()\n",
        "import gateway.models.things\n\n\ndef build():\n    return gateway.models.things.Thing()\n",
        "from ...models.things import Thing\n\n\ndef build():\n    return Thing()\n",
        "from gateway.services.shared import Thing\n\n\ndef build():\n    return Thing()\n",
        "from gateway.services import shared\n\n\ndef build():\n    return shared.Thing()\n",
        "from gateway.models import things\n\n\ndef build():\n    return getattr(things, 'Thing')()\n",
        "from gateway.models.things import Thing\nfrom gateway.services.shared import Thing\n\n\nROW = Thing()\n",
        "from sqlalchemy import cast\nfrom gateway.models.things import Thing\n\n\nNAME = cast(Thing.name, str)\n",
    ],
)
def test_a_module_other_than_the_repository_that_uses_a_model_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access())
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/services/shared.py", "from gateway.models.things import Thing\n\n__all__ = []\n")
    _write(tmp_path, "gateway/services/things/_store.py", source)
    assert check.check_model_access(tmp_path) == [f"gateway/services/things/_store.py:5 uses {_THING}; {_THING_REMEDY}"]


@pytest.mark.parametrize(
    "source",
    [
        "from gateway.models.things import Thing\n\n\ndef keep(row: Thing) -> Thing:\n    return row\n",
        "from gateway.models.things import Thing\n\nrows: list[Thing] = []\n",
        "from gateway.models.things import Thing\n\ntype Rows = list[Thing]\n",
        "from typing import TypeAlias\n\nfrom gateway.models.things import Thing\n\nRows: TypeAlias = list[Thing]\n",
        "from typing import cast\n\nfrom gateway.models.things import Thing\n\n\ndef keep(row):\n"
        "    return cast(Thing, row)\n",
        "from typing import TypeVar\n\nfrom gateway.models.things import Thing\n\n"
        "RowT = TypeVar('RowT', bound=Thing)\n",
        "from gateway.models.things import Thing\n\n\ndef build(row) -> bool:\n    return isinstance(row, Thing)\n",
        "from gateway.services.unrelated import *\n\n\ndef build():\n    return 1\n",
        "from gateway.models.things import ThingCreate\n\n\ndef build():\n    return ThingCreate(name='a')\n",
        "def build(Thing):\n    return Thing()\n",
    ],
)
def test_annotating_with_a_model_or_naming_something_else_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access())
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/services/things/_store.py", source)
    assert check.check_model_access(tmp_path) == []


def test_the_declaring_module_and_a_module_on_the_baseline_may_use_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access(baseline=("gateway/services/thing_service.py",)))
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/models/things.py", _MODELS_SOURCE + "\n\nDEFAULT = Thing\n")
    _write(tmp_path, "gateway/services/thing_service.py", "from gateway.models.things import Thing\n\nROW = Thing()\n")
    assert check.check_model_access(tmp_path) == []


def test_another_models_module_that_uses_a_model_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access())
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/models/links.py", "from gateway.models.things import Thing\n\nLINKED = Thing\n")
    assert check.check_model_access(tmp_path) == [f"gateway/models/links.py:3 uses {_THING}; {_THING_REMEDY}"]


def test_a_model_reexported_by_assignment_is_followed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access(baseline=("gateway/services/shared.py",)))
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/services/shared.py", "from gateway.models.things import Thing\n\nRow = Thing\n")
    _write(tmp_path, "gateway/services/things/_store.py", "from gateway.services.shared import Row\n\nROW = Row()\n")
    assert check.check_model_access(tmp_path) == [f"gateway/services/things/_store.py:3 uses {_THING}; {_THING_REMEDY}"]


def test_a_name_bound_to_a_model_and_to_something_else_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access())
    _write_thing_model(tmp_path)
    _write(
        tmp_path,
        "gateway/services/things/_store.py",
        "from gateway.services.other import Thing\n\n\ndef build():\n"
        "    from gateway.models.things import Thing\n\n    return Thing()\n",
    )
    assert check.check_model_access(tmp_path) == [
        "gateway/services/things/_store.py:5 binds Thing to more than one target, one of them an ORM model or a "
        "module that holds one; import each under a name of its own"
    ]


def test_a_name_bound_to_a_module_that_holds_a_model_and_to_something_else_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access())
    _write_thing_model(tmp_path)
    _write(
        tmp_path,
        "gateway/services/things/_store.py",
        "from gateway.repositories import things\n\n\ndef build():\n"
        "    from gateway.models import things\n\n    return things.Thing()\n",
    )
    assert check.check_model_access(tmp_path) == [
        "gateway/services/things/_store.py:5 binds things to more than one target, one of them an ORM model or a "
        "module that holds one; import each under a name of its own"
    ]


@pytest.mark.parametrize(
    ("source", "module"),
    [
        ("from gateway.models.things import *\n", "gateway.models.things"),
        ("from gateway.models import *\n", "gateway.models"),
        ("from gateway.services.shared import *\n", "gateway.services.shared"),
    ],
)
def test_a_star_import_from_a_module_that_holds_a_model_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, module: str
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access())
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/services/shared.py", "from gateway.models.things import Thing\n\n__all__ = []\n")
    _write(tmp_path, "gateway/services/things/_store.py", source)
    assert check.check_model_access(tmp_path) == [
        f"gateway/services/things/_store.py:1 imports * from {module}, which holds an ORM model; "
        "import the names it uses"
    ]


def test_a_model_no_repository_holds_yet_is_flagged_wherever_it_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access(repository=None, baseline=(_THING_REPOSITORY,)))
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/api/routes/things.py", "from gateway.models.things import Thing\n\nROW = Thing()\n")
    assert check.check_model_access(tmp_path) == [
        f"gateway/api/routes/things.py:3 uses {_THING}; its queries belong in a repository, "
        "which the model access map names"
    ]


def test_two_models_with_one_class_name_are_told_apart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    access = _thing_access()
    access["gateway.models.legacy.Thing"] = {"repository": None, "baseline": ("gateway/services/thing_service.py",)}
    monkeypatch.setattr(check, "MODEL_ACCESS", access)
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/models/legacy.py", 'class Thing:\n    __tablename__ = "legacy_things"\n')
    _write(tmp_path, "gateway/services/thing_service.py", "from gateway.models.legacy import Thing\n\nROW = Thing()\n")
    assert check.check_model_access(tmp_path) == []


def test_a_model_missing_from_the_map_and_an_entry_that_is_no_model_are_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access = _thing_access()
    del access["gateway.models.things.Gadget"]
    access["gateway.models.things.ThingCreate"] = {"repository": None, "baseline": ()}
    monkeypatch.setattr(check, "MODEL_ACCESS", access)
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/models/legacy.py", "class Legacy:\n    __table__ = None\n")
    assert check.check_model_access(tmp_path) == [
        "gateway.models.legacy.Legacy is an ORM model with no entry in the model access map; "
        "name the repository that may use it",
        "gateway.models.things.Gadget is an ORM model with no entry in the model access map; "
        "name the repository that may use it",
        "gateway.models.things.ThingCreate is in the model access map but is not an ORM model; remove it from the map",
    ]


@pytest.mark.parametrize(
    ("repository", "message"),
    [
        (
            "gateway/services/thing_service.py",
            f"{_THING} names gateway/services/thing_service.py as its repository, "
            "which is not under gateway/repositories/",
        ),
        (
            "gateway/repositories/things/other_repository.py",
            f"{_THING} names gateway/repositories/things/other_repository.py as its repository, which does not exist",
        ),
        (
            "gateway/repositories/things/__init__.py",
            f"{_THING} names gateway/repositories/things/__init__.py as its repository, which does not use it",
        ),
    ],
)
def test_a_repository_entry_that_is_no_repository_or_does_not_use_the_model_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repository: str, message: str
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access(repository=repository, baseline=(_THING_REPOSITORY,)))
    _write_thing_model(tmp_path)
    _write(tmp_path, "gateway/services/thing_service.py", "from gateway.models.things import Thing\n\nROW = Thing()\n")
    assert check.check_model_access(tmp_path)[0] == message


def test_a_repository_on_its_own_models_baseline_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access(baseline=(_THING_REPOSITORY,)))
    _write_thing_model(tmp_path)
    assert check.check_model_access(tmp_path) == [
        f"{_THING} has its repository {_THING_REPOSITORY} on its baseline; remove it from the baseline"
    ]


@pytest.mark.parametrize(
    "source", ["from gateway.models.things import Thing\n\n\ndef keep(row: Thing) -> None: ...\n", None]
)
def test_a_model_baseline_entry_that_does_not_use_the_model_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access(baseline=("gateway/services/thing_service.py",)))
    _write_thing_model(tmp_path)
    if source is not None:
        _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_model_access(tmp_path) == [
        f"gateway/services/thing_service.py is on the baseline for {_THING} but does not use it; "
        "remove it from the baseline"
    ]


def test_main_fails_on_a_second_module_that_uses_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "src/gateway/services/things/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    _write_thing_model(tmp_path / "src")
    monkeypatch.setattr(check, "MODEL_ACCESS", _thing_access())
    assert check.main() == 0
    _write(
        tmp_path, "src/gateway/services/things/_store.py", "from gateway.models.things import Thing\n\nROW = Thing()\n"
    )
    assert check.main() == 1


def _service_root(tmp_path: Path, exports: str, body: str) -> None:
    """A one-package service layer whose root re-exports ``exports`` from a private module holding ``body``."""
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_thing.py", body)
    names = ", ".join(exports.split())
    listed = "".join(f'    "{name}",\n' for name in exports.split())
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        f"from gateway.services.things._thing import {names}\n\n__all__ = [\n{listed}]\n",
    )


def _use_empty_service_root_export_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "SERVICE_ROOT_EXPORT_BASELINE", {})


def test_a_service_root_exporting_the_service_and_its_types_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _service_root(tmp_path, "ThingService NewThing", "class ThingService:\n    pass\n\n\nclass NewThing:\n    pass\n")
    assert check.check_service_root_exports(tmp_path) == []


def test_a_service_root_exporting_a_free_function_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _service_root(tmp_path, "names_a_thing", "def names_a_thing(value: str) -> bool:\n    return True\n")
    assert check.check_service_root_exports(tmp_path) == [
        "gateway/services/things/__init__.py exports names_a_thing, which is neither the service nor a type its "
        "public methods use; make it a method on what owns it, move it below the domain, or stop exporting it"
    ]


def test_a_service_root_exporting_an_async_free_function_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _service_root(tmp_path, "run_things", "async def run_things() -> None:\n    return None\n")
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_service_root_exporting_a_constant_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _service_root(tmp_path, "MAX_THINGS", "MAX_THINGS = 10\n")
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_service_root_exporting_a_type_alias_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A type alias is a type rather than a constant, so its lower-case assignment is left alone."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _service_root(tmp_path, "ThingStore", "from collections.abc import Callable\n\nThingStore = Callable[[str], str]\n")
    assert check.check_service_root_exports(tmp_path) == []


def test_a_baselined_free_function_is_allowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _service_root(tmp_path, "names_a_thing", "def names_a_thing(value: str) -> bool:\n    return True\n")
    monkeypatch.setattr(check, "SERVICE_ROOT_EXPORT_BASELINE", {"gateway/services/things/": ("names_a_thing",)})
    assert check.check_service_root_exports(tmp_path) == []


def test_a_baseline_entry_that_is_no_longer_exported_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline only shrinks, so cleaning an export up without removing its entry fails."""
    _service_root(tmp_path, "ThingService", "class ThingService:\n    pass\n")
    monkeypatch.setattr(check, "SERVICE_ROOT_EXPORT_BASELINE", {"gateway/services/things/": ("names_a_thing",)})
    assert check.check_service_root_exports(tmp_path) == [
        "gateway/services/things/ is on the service root export baseline but no longer exports names_a_thing; "
        "remove it from the baseline"
    ]


def test_a_root_re_exporting_a_free_function_from_another_layer_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The source is followed, so moving the function out of the package does not hide it."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/types/things.py", "def names_a_thing(value: str) -> bool:\n    return True\n")
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        'from gateway.types.things import names_a_thing\n\n__all__ = ["names_a_thing"]\n',
    )
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_main_fails_on_a_service_root_that_exports_a_free_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "tests/__init__.py", "")
    _service_root(tmp_path / "src", "ThingService", "class ThingService:\n    pass\n")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _service_root(tmp_path / "src", "names_a_thing", "def names_a_thing(value: str) -> bool:\n    return True\n")
    assert check.main() == 1


_FREE_FUNCTION = "def names_a_thing(value: str) -> bool:\n    return True\n"


def test_a_root_with_no_dunder_all_is_still_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """__all__ governs a star import alone, so a root that declares none still offers what it binds."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_thing.py", _FREE_FUNCTION)
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        "from gateway.services.things._thing import names_a_thing\n",
    )
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_binding_left_out_of_dunder_all_is_still_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A name omitted from __all__ remains importable from the root, so omitting it hides nothing."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_thing.py", f"class ThingService:\n    pass\n\n\n{_FREE_FUNCTION}")
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        'from gateway.services.things._thing import ThingService, names_a_thing\n\n__all__ = ["ThingService"]\n',
    )
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_function_defined_in_the_root_itself_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/__init__.py", f'{_FREE_FUNCTION}\n__all__ = ["names_a_thing"]\n')
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_renamed_import_is_resolved_under_its_original_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The alias is what a caller imports, and the definition is under the original name."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_thing.py", "THING = 1\n")
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        'from gateway.services.things._thing import THING as THING_TOOL\n\n__all__ = ["THING_TOOL"]\n',
    )
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_source_that_is_a_package_is_resolved_through_its_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subpackage has no module file of its own, so the definition sits in its __init__.py."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_inner/__init__.py", _FREE_FUNCTION)
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        'from gateway.services.things._inner import names_a_thing\n\n__all__ = ["names_a_thing"]\n',
    )
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_an_intermediate_re_export_is_followed_to_the_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root imports from a subpackage that re-exports from a module, so neither hop holds the definition."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_inner/_leaf.py", _FREE_FUNCTION)
    _write(
        tmp_path,
        "gateway/services/things/_inner/__init__.py",
        "from gateway.services.things._inner._leaf import names_a_thing\n",
    )
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        'from gateway.services.things._inner import names_a_thing\n\n__all__ = ["names_a_thing"]\n',
    )
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_an_annotated_constant_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _service_root(tmp_path, "MAX_THINGS", "MAX_THINGS: int = 10\n")
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_private_binding_is_not_offered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An underscored name is private by convention, so a root that binds one for its own use is clean."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_thing.py", "def _names_a_thing(value: str) -> bool:\n    return True\n")
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        "from gateway.services.things._thing import _names_a_thing\n",
    )
    assert check.check_service_root_exports(tmp_path) == []


def test_a_circular_re_export_does_not_recurse_forever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_a.py", "from gateway.services.things._b import looping\n")
    _write(tmp_path, "gateway/services/things/_b.py", "from gateway.services.things._a import looping\n")
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        'from gateway.services.things._a import looping\n\n__all__ = ["looping"]\n',
    )
    assert check.check_service_root_exports(tmp_path) == []


def test_a_root_star_importing_a_free_function_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A star import binds every public name of its source, so it hides none of them."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_thing.py", f"class ThingService:\n    pass\n\n\n{_FREE_FUNCTION}")
    _write(tmp_path, "gateway/services/things/__init__.py", "from gateway.services.things._thing import *\n")
    assert check.check_service_root_exports(tmp_path) == [
        "gateway/services/things/__init__.py exports names_a_thing, which is neither the service nor a type its "
        "public methods use; make it a method on what owns it, move it below the domain, or stop exporting it"
    ]


def test_a_star_import_binds_only_what_the_source_lists_in_dunder_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the source declares __all__, a star import binds those names and no others."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(
        tmp_path,
        "gateway/services/things/_thing.py",
        f'class ThingService:\n    pass\n\n\n{_FREE_FUNCTION}\n__all__ = ["ThingService"]\n',
    )
    _write(tmp_path, "gateway/services/things/__init__.py", "from gateway.services.things._thing import *\n")
    assert check.check_service_root_exports(tmp_path) == []


def test_a_star_import_at_an_intermediate_hop_is_followed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The root names the function, and the module it names it from only star-imports it."""
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_inner/_leaf.py", _FREE_FUNCTION)
    _write(tmp_path, "gateway/services/things/_inner/__init__.py", "from ._leaf import *\n")
    _write(
        tmp_path,
        "gateway/services/things/__init__.py",
        'from gateway.services.things._inner import names_a_thing\n\n__all__ = ["names_a_thing"]\n',
    )
    assert len(check.check_service_root_exports(tmp_path)) == 1


def test_a_circular_star_import_does_not_recurse_forever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_a.py", "from gateway.services.things._b import *\n")
    _write(tmp_path, "gateway/services/things/_b.py", "from gateway.services.things._a import *\n")
    _write(tmp_path, "gateway/services/things/__init__.py", "from gateway.services.things._a import *\n")
    assert check.check_service_root_exports(tmp_path) == []


def test_a_relative_import_in_the_root_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_empty_service_root_export_baseline(monkeypatch)
    _write(tmp_path, "gateway/services/__init__.py", "")
    _write(tmp_path, "gateway/services/things/_thing.py", _FREE_FUNCTION)
    _write(tmp_path, "gateway/services/things/__init__.py", "from ._thing import names_a_thing\n")
    assert len(check.check_service_root_exports(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Rule 25: import cycles
# ---------------------------------------------------------------------------


def test_two_sibling_modules_importing_each_other_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/__init__.py", "")
    _write(tmp_path, "gateway/api/deps/one.py", "from gateway.api.deps.two import thing\n")
    _write(tmp_path, "gateway/api/deps/two.py", "from gateway.api.deps.one import other\n")

    violations = check.check_import_cycles(tmp_path)

    assert len(violations) == 1
    assert "gateway.api.deps.one, gateway.api.deps.two" in violations[0]


def test_a_member_reading_a_name_from_its_own_package_is_flagged(tmp_path: Path) -> None:
    """``from pkg import Name`` waits for the package body, which is itself waiting for the member."""
    _write(tmp_path, "gateway/services/widgets/__init__.py", "from gateway.services.widgets._store import Store\n")
    _write(tmp_path, "gateway/services/widgets/_store.py", "from gateway.services.widgets import Thing\n")

    violations = check.check_import_cycles(tmp_path)

    assert len(violations) == 1
    assert "gateway.services.widgets, gateway.services.widgets._store" in violations[0]


def test_a_member_importing_a_sibling_out_of_its_package_is_not_a_cycle(tmp_path: Path) -> None:
    """``from pkg import sibling`` loads the submodule, so it waits for the sibling and not the package."""
    _write(tmp_path, "gateway/services/widgets/__init__.py", "from gateway.services.widgets._store import Store\n")
    _write(tmp_path, "gateway/services/widgets/_store.py", "from gateway.services.widgets import _rows\n")
    _write(tmp_path, "gateway/services/widgets/_rows.py", "ROWS = ()\n")

    assert check.check_import_cycles(tmp_path) == []


def test_a_member_importing_a_sibling_by_its_full_path_is_not_a_cycle(tmp_path: Path) -> None:
    """A module's own package is already running when its body runs, so passing through it waits for nothing."""
    _write(tmp_path, "gateway/services/widgets/__init__.py", "from gateway.services.widgets._store import Store\n")
    _write(tmp_path, "gateway/services/widgets/_store.py", "from gateway.services.widgets._rows import ROWS\n")
    _write(tmp_path, "gateway/services/widgets/_rows.py", "ROWS = ()\n")

    assert check.check_import_cycles(tmp_path) == []


def test_a_cycle_running_through_two_package_roots_is_flagged(tmp_path: Path) -> None:
    """A root waits for the member it exports, so a member reaching another root closes the ring."""
    _write(tmp_path, "gateway/services/widgets/__init__.py", "from gateway.services.widgets._store import Store\n")
    _write(
        tmp_path,
        "gateway/services/widgets/_store.py",
        "from gateway.services.gadgets import Gadget\n\nStore = 1\n",
    )
    _write(tmp_path, "gateway/services/gadgets/__init__.py", "from gateway.services.gadgets._parts import Gadget\n")
    _write(
        tmp_path,
        "gateway/services/gadgets/_parts.py",
        "from gateway.services.widgets import Store\n\nGadget = 2\n",
    )

    violations = check.check_import_cycles(tmp_path)

    assert len(violations) == 1
    assert "gateway.services.gadgets, gateway.services.gadgets._parts" in violations[0]
    assert "gateway.services.widgets, gateway.services.widgets._store" in violations[0]


def test_a_cycle_through_three_siblings_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/services/widgets/__init__.py", "")
    _write(tmp_path, "gateway/services/widgets/a.py", "from gateway.services.widgets.b import b\n")
    _write(tmp_path, "gateway/services/widgets/b.py", "from gateway.services.widgets.c import c\n")
    _write(tmp_path, "gateway/services/widgets/c.py", "from gateway.services.widgets.a import a\n")

    violations = check.check_import_cycles(tmp_path)

    assert len(violations) == 1
    assert "widgets.a, gateway.services.widgets.b, gateway.services.widgets.c" in violations[0]


def test_an_import_that_does_not_run_at_import_time_is_not_a_cycle(tmp_path: Path) -> None:
    """A deferred import is how a cycle is broken, so reading one as an edge would refuse the fix."""
    _write(tmp_path, "gateway/api/deps/__init__.py", "")
    _write(tmp_path, "gateway/api/deps/one.py", "from gateway.api.deps.two import thing\n")
    _write(
        tmp_path,
        "gateway/api/deps/two.py",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from gateway.api.deps.one import other\n",
    )

    assert check.check_import_cycles(tmp_path) == []


def test_a_module_importing_itself_is_not_reported(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps/__init__.py", "")
    _write(tmp_path, "gateway/api/deps/one.py", "from gateway.api.deps.one import thing\n")

    assert check.check_import_cycles(tmp_path) == []


def test_a_cycle_through_a_package_initializer_is_flagged(tmp_path: Path) -> None:
    """Importing a.b runs a's body first, so a package a module reaches through is on the cycle."""
    _write(tmp_path, "gateway/__init__.py", "")
    _write(tmp_path, "gateway/a.py", "from gateway.pkg.child import C\n\nA = 1\n")
    _write(tmp_path, "gateway/pkg/__init__.py", "from gateway.a import A\n")
    _write(tmp_path, "gateway/pkg/child.py", "C = 2\n")

    violations = check.check_import_cycles(tmp_path)

    assert len(violations) == 1
    assert "gateway.a, gateway.pkg" in violations[0]


def test_a_cycle_group_holding_a_module_from_outside_the_package_is_flagged(tmp_path: Path) -> None:
    """The package and its member may close a cycle, so the module from outside must not hide behind them."""
    _write(tmp_path, "gateway/__init__.py", "")
    _write(tmp_path, "gateway/p/__init__.py", "from gateway.p.m import M\n\nP = 1\n")
    _write(tmp_path, "gateway/p/m.py", "from gateway.p import P\nfrom gateway.q import Q\n\nM = 2\n")
    _write(tmp_path, "gateway/q.py", "from gateway.p.m import M\n\nQ = 3\n")

    violations = check.check_import_cycles(tmp_path)

    assert len(violations) == 1
    assert "gateway.q" in violations[0]


def test_a_dependency_alias_naming_a_model_is_not_a_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Dep = Annotated[Model, Depends(...)]`` names a type, so it must not hold a module on the baseline."""
    monkeypatch.setattr(
        check,
        "MODEL_ACCESS",
        {
            "gateway.models.things.Thing": {
                "repository": "gateway/repositories/things/thing_repository.py",
                "baseline": (),
            }
        },
    )
    _write(tmp_path, "gateway/models/things.py", "class Thing:\n    __tablename__ = 'thing'\n")
    _write(tmp_path, "gateway/repositories/things/__init__.py", "")
    _write(
        tmp_path,
        "gateway/repositories/things/thing_repository.py",
        "from gateway.models.things import Thing\n\n\ndef all_things(db):\n    return db.execute(select(Thing))\n",
    )
    _write(
        tmp_path,
        "gateway/api/deps/things.py",
        "from typing import Annotated\n\nfrom gateway.models.things import Thing\n\n"
        "ThingDep = Annotated[Thing, Depends(get_thing)]\n",
    )

    assert check.check_model_access(tmp_path) == []


def test_an_import_in_a_module_level_block_is_a_cycle(tmp_path: Path) -> None:
    """An if, a try or a loop at module level runs as the body does, so an import inside one counts."""
    _write(tmp_path, "gateway/__init__.py", "")
    _write(tmp_path, "gateway/a.py", "if True:\n    from gateway.b import B\n\nA = 1\n")
    _write(tmp_path, "gateway/b.py", "from gateway.a import A\n\nB = 2\n")

    violations = check.check_import_cycles(tmp_path)

    assert len(violations) == 1
    assert "gateway.a, gateway.b" in violations[0]


def test_an_import_in_a_function_body_is_not_a_cycle(tmp_path: Path) -> None:
    """Deferring an import is how a cycle is broken, so reading one as an edge would refuse the fix."""
    _write(tmp_path, "gateway/__init__.py", "")
    _write(tmp_path, "gateway/a.py", "def later():\n    from gateway.b import B\n\n    return B\n")
    _write(tmp_path, "gateway/b.py", "from gateway.a import later\n\nB = 2\n")

    assert check.check_import_cycles(tmp_path) == []


def test_a_model_built_in_annotated_metadata_is_still_a_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the type in ``Annotated[T, ...]`` names a type; the metadata beside it runs on import."""
    monkeypatch.setattr(
        check,
        "MODEL_ACCESS",
        {
            "gateway.models.things.Thing": {
                "repository": "gateway/repositories/things/thing_repository.py",
                "baseline": (),
            }
        },
    )
    _write(tmp_path, "gateway/models/things.py", "class Thing:\n    __tablename__ = 'thing'\n")
    _write(tmp_path, "gateway/repositories/things/__init__.py", "")
    _write(
        tmp_path,
        "gateway/repositories/things/thing_repository.py",
        "from gateway.models.things import Thing\n\n\ndef all_things(db):\n    return db.execute(select(Thing))\n",
    )
    _write(
        tmp_path,
        "gateway/api/deps/things.py",
        "from typing import Annotated\n\nfrom gateway.models.things import Thing\n\nAlias = Annotated[str, Thing()]\n",
    )

    violations = check.check_model_access(tmp_path)

    assert len(violations) == 1
    assert "gateway/api/deps/things.py" in violations[0]


def test_a_renamed_annotated_import_still_names_a_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``Annotated`` resolves through the module's bindings, so an alias for it is recognized."""
    monkeypatch.setattr(
        check,
        "MODEL_ACCESS",
        {
            "gateway.models.things.Thing": {
                "repository": "gateway/repositories/things/thing_repository.py",
                "baseline": (),
            }
        },
    )
    _write(tmp_path, "gateway/models/things.py", "class Thing:\n    __tablename__ = 'thing'\n")
    _write(tmp_path, "gateway/repositories/things/__init__.py", "")
    _write(
        tmp_path,
        "gateway/repositories/things/thing_repository.py",
        "from gateway.models.things import Thing\n\n\ndef all_things(db):\n    return db.execute(select(Thing))\n",
    )
    _write(
        tmp_path,
        "gateway/api/deps/things.py",
        "from typing import Annotated as Ann\n\nfrom gateway.models.things import Thing\n\n"
        "ThingDep = Ann[Thing, Depends(get_thing)]\n",
    )

    assert check.check_model_access(tmp_path) == []


_COMPOSITION_ROOT_MODULE = "gateway/api/deps/__init__.py"


def _composition_root_behavior(line: int, name: str, methods: int, path: str = _COMPOSITION_ROOT_MODULE) -> str:
    return (
        f"{path}:{line} defines {name} with {methods} method(s); the composition root builds "
        "things and holds no implementation, so this belongs to the domain whose work it does"
    )


def test_a_class_with_a_method_in_the_composition_root_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(tmp_path, _COMPOSITION_ROOT_MODULE, "class Claim:\n    def claim(self) -> None: ...\n")
    assert check.check_composition_root_behavior(tmp_path) == [_composition_root_behavior(1, "Claim", 1)]


def test_a_class_with_no_methods_in_the_composition_root_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(tmp_path, _COMPOSITION_ROOT_MODULE, "class Ports:\n    search: object\n    files: object\n")
    assert check.check_composition_root_behavior(tmp_path) == []


def test_a_dataclass_of_data_in_the_composition_root_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(
        tmp_path,
        _COMPOSITION_ROOT_MODULE,
        "@dataclass(frozen=True, kw_only=True)\nclass Ports:\n    search: object\n",
    )
    assert check.check_composition_root_behavior(tmp_path) == []


def test_a_dataclass_that_grows_a_method_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(
        tmp_path,
        _COMPOSITION_ROOT_MODULE,
        "@dataclass(frozen=True)\nclass Ports:\n    search: object\n\n    def pick(self) -> None: ...\n",
    )
    assert check.check_composition_root_behavior(tmp_path) == [_composition_root_behavior(2, "Ports", 1)]


def test_a_class_nested_in_a_builder_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(
        tmp_path,
        _COMPOSITION_ROOT_MODULE,
        "def ledger_on(db: object) -> object:\n"
        "    class Ledger:\n"
        "        def rate(self) -> None: ...\n\n"
        "        def spend(self) -> None: ...\n\n"
        "    return Ledger()\n",
    )
    assert check.check_composition_root_behavior(tmp_path) == [_composition_root_behavior(2, "ledger_on.Ledger", 2)]


def test_a_baselined_class_in_the_composition_root_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ((_COMPOSITION_ROOT_MODULE, "Claim"),))
    _write(tmp_path, _COMPOSITION_ROOT_MODULE, "class Claim:\n    def claim(self) -> None: ...\n")
    assert check.check_composition_root_behavior(tmp_path) == []


def test_a_baselined_class_that_moved_out_fails_until_its_entry_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ((_COMPOSITION_ROOT_MODULE, "Claim"),))
    _write(tmp_path, _COMPOSITION_ROOT_MODULE, "def get_claim() -> object:\n    return object()\n")
    assert check.check_composition_root_behavior(tmp_path) == [
        f"{_COMPOSITION_ROOT_MODULE} is on the composition root behavior baseline for Claim, "
        "which no longer defines a method there; remove it from the baseline"
    ]


def test_a_class_with_a_method_outside_the_composition_root_is_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(tmp_path, "gateway/services/tools/_claim.py", "class Claim:\n    def claim(self) -> None: ...\n")
    assert check.check_composition_root_behavior(tmp_path) == []


def test_a_method_under_a_conditional_in_a_class_body_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(
        tmp_path,
        _COMPOSITION_ROOT_MODULE,
        "class Claim:\n    if TYPE_CHECKING:\n        def claim(self) -> None: ...\n",
    )
    assert check.check_composition_root_behavior(tmp_path) == [_composition_root_behavior(1, "Claim", 1)]


def test_a_class_under_a_conditional_in_a_builder_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(
        tmp_path,
        _COMPOSITION_ROOT_MODULE,
        "def ledger_on(db: object) -> object:\n"
        "    if db is not None:\n"
        "        class Ledger:\n"
        "            def rate(self) -> None: ...\n\n"
        "        return Ledger()\n"
        "    return None\n",
    )
    assert check.check_composition_root_behavior(tmp_path) == [_composition_root_behavior(3, "ledger_on.Ledger", 1)]


def test_a_class_under_an_exception_handler_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(
        tmp_path,
        _COMPOSITION_ROOT_MODULE,
        "try:\n    import fast\nexcept ImportError:\n    class Fallback:\n        def run(self) -> None: ...\n",
    )
    assert check.check_composition_root_behavior(tmp_path) == [_composition_root_behavior(4, "Fallback", 1)]


def test_a_nested_class_does_not_count_toward_the_class_holding_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_BEHAVIOR_BASELINE", ())
    _write(
        tmp_path,
        _COMPOSITION_ROOT_MODULE,
        "class Outer:\n"
        "    def mine(self) -> None: ...\n\n"
        "    class Inner:\n"
        "        def theirs(self) -> None: ...\n\n"
        "        def also(self) -> None: ...\n",
    )
    assert check.check_composition_root_behavior(tmp_path) == [
        _composition_root_behavior(1, "Outer", 1),
        _composition_root_behavior(4, "Outer.Inner", 2),
    ]


_NON_DOMAIN_PACKAGE = "gateway/services/sidecar/__init__.py"
_NON_DOMAIN_MEMBER = "gateway/services/sidecar/_impl.py"
_OUTSIDE = "gateway/api/deps/things.py"


def _sidecar(tmp_path: Path, *, outside_imports: int) -> None:
    _write(tmp_path, _NON_DOMAIN_PACKAGE, "")
    _write(tmp_path, _NON_DOMAIN_MEMBER, "VALUE = 1\n")
    lines = "".join(f"from gateway.services.sidecar._impl import VALUE as V{n}\n" for n in range(outside_imports))
    _write(tmp_path, _OUTSIDE, lines or "VALUE = 1\n")


def test_a_non_domain_package_at_its_ceiling_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 2})
    _sidecar(tmp_path, outside_imports=2)
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == []


def test_a_count_above_the_ceiling_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 1})
    _sidecar(tmp_path, outside_imports=3)
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == [
        "3 imports reach below services/sidecar/, above its ceiling of 1; import what services.sidecar "
        "exports, or split the package into the domains DOMAINS.md names"
    ]


def test_a_count_below_the_ceiling_must_be_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 5})
    _sidecar(tmp_path, outside_imports=2)
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == [
        "only 2 imports now reach below services/sidecar/, under its ceiling of 5; "
        "lower the ceiling to that, so the removal cannot be spent again"
    ]


def test_a_non_domain_package_with_no_ceiling_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {})
    _sidecar(tmp_path, outside_imports=2)
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == [
        "services/sidecar/ is not a domain DOMAINS.md gives a section and has no import ceiling; "
        "give it one of 2, or give the package a section"
    ]


def test_a_package_that_became_a_domain_loses_its_ceiling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 2})
    _sidecar(tmp_path, outside_imports=2)
    assert check.check_non_domain_service_imports(tmp_path, {"sidecar"}) == [
        "services/sidecar/ has an import ceiling but is no longer a service package that is not a domain; "
        "remove its ceiling"
    ]


def test_an_import_inside_the_package_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 0})
    _sidecar(tmp_path, outside_imports=0)
    _write(
        tmp_path,
        "gateway/services/sidecar/_other.py",
        "from gateway.services.sidecar._impl import VALUE\n",
    )
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == []


def test_an_import_of_the_package_root_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 0})
    _sidecar(tmp_path, outside_imports=0)
    _write(tmp_path, _OUTSIDE, "from gateway.services.sidecar import VALUE\n")
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == []


def test_a_plain_import_below_the_root_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 0})
    _sidecar(tmp_path, outside_imports=0)
    _write(tmp_path, _OUTSIDE, "import gateway.services.sidecar._impl as impl\n")
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == [
        "1 imports reach below services/sidecar/, above its ceiling of 0; import what services.sidecar "
        "exports, or split the package into the domains DOMAINS.md names"
    ]


def test_a_plain_import_of_the_package_root_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 0})
    _sidecar(tmp_path, outside_imports=0)
    _write(tmp_path, _OUTSIDE, "import gateway.services.sidecar\n")
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == []


def test_one_statement_counts_once_however_many_names_it_binds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "NON_DOMAIN_SERVICE_IMPORT_BASELINE", {"sidecar": 1})
    _sidecar(tmp_path, outside_imports=0)
    _write(tmp_path, _OUTSIDE, "from gateway.services.sidecar._impl import VALUE, OTHER, THIRD\n")
    assert check.check_non_domain_service_imports(tmp_path, {"things"}) == []


def _root_with(tmp_path: Path, body: str) -> None:
    _write(tmp_path, check.COMPOSITION_ROOT_MODULE, body)


def test_a_composition_root_at_its_baseline_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 2)
    _root_with(tmp_path, "def get_one() -> None: ...\n\n\ndef get_two() -> None: ...\n")
    assert check.check_composition_root_size(tmp_path) == []


def test_a_definition_added_to_the_composition_root_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, "def get_one() -> None: ...\n\n\ndef get_two() -> None: ...\n")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 2 definitions, above its baseline of 1; put the new one "
        "in the module of the domain whose work it does, which the root re-exports"
    ]


def test_a_definition_removed_must_be_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 5)
    _root_with(tmp_path, "def get_one() -> None: ...\n")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 1 definitions, under its baseline of 5; lower the "
        "baseline to that, so the room made cannot be spent again"
    ]


def test_an_annotated_alias_counts_as_a_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, "def get_one() -> None: ...\n\n\nOneDep = Annotated[object, Depends(get_one)]\n")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 2 definitions, above its baseline of 1; put the new one "
        "in the module of the domain whose work it does, which the root re-exports"
    ]


def test_an_unpacking_assignment_counts_every_name_it_binds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, "def get_one() -> None: ...\n\n\nFIRST, SECOND = (1, 2)\n")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 3 definitions, above its baseline of 1; put the new one "
        "in the module of the domain whose work it does, which the root re-exports"
    ]


def test_a_type_alias_statement_counts_as_a_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, "def get_one() -> None: ...\n\n\ntype Result = tuple[int, str]\n")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 2 definitions, above its baseline of 1; put the new one "
        "in the module of the domain whose work it does, which the root re-exports"
    ]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param("for GATE in ():\n    pass\n", id="for_target"),
        pytest.param("with open('x') as GATE:\n    pass\n", id="with_target"),
        pytest.param("if (GATE := object()):\n    pass\n", id="assignment_expression"),
        pytest.param("match object():\n    case GATE:\n        pass\n", id="match_capture"),
        pytest.param("match ():\n    case [*GATE]:\n        pass\n", id="match_star"),
        pytest.param("match {}:\n    case {**GATE}:\n        pass\n", id="match_rest"),
        pytest.param("GATES[(GATE := 'one')]: int\n", id="annotation_subscript_target"),
        pytest.param("GATES[(GATE := 'one')]: int = 1\n", id="annotation_subscript_target_with_value"),
    ],
)
def test_every_statement_that_binds_a_name_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, f"def get_one() -> None: ...\n\n\n{body}")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 2 definitions, above its baseline of 1; put the new one "
        "in the module of the domain whose work it does, which the root re-exports"
    ]


@pytest.mark.parametrize(
    ("body", "counted"),
    [
        pytest.param("def get_two(gate=(GATE := object())) -> None: ...\n", 3, id="parameter_default"),
        pytest.param("def get_two(*, gate=(GATE := object())) -> None: ...\n", 3, id="keyword_only_default"),
        pytest.param("async def get_two(gate=(GATE := object())) -> None: ...\n", 3, id="async_parameter_default"),
        pytest.param("@(GATE := staticmethod)\ndef get_two() -> None: ...\n", 3, id="function_decorator"),
        pytest.param("@(GATE := staticmethod)\nclass Gate:\n    pass\n", 3, id="class_decorator"),
        pytest.param("class Gate((GATE := object)):\n    pass\n", 3, id="class_base"),
        pytest.param("class Gate(metaclass=(GATE := type)):\n    pass\n", 3, id="class_keyword"),
        pytest.param("GATES = lambda gate=(GATE := object()): gate\n", 3, id="lambda_default"),
        pytest.param("try:\n    pass\nexcept (GATE := ValueError):\n    pass\n", 2, id="exception_type"),
        pytest.param("def get_two[T](gate=(GATE := object())) -> None: ...\n", 3, id="generic_parameter_default"),
        pytest.param("@(GATE := staticmethod)\nclass Gate[T]:\n    pass\n", 3, id="generic_class_decorator"),
        # An assignment expression in an annotation binds on a Python that evaluates
        # annotations eagerly. Every Python parses these, so the cases run anywhere.
        pytest.param("def get_two(gate: (GATE := object)) -> None: ...\n", 3, id="parameter_annotation"),
        pytest.param("def get_two(gate: (GATE := object), /) -> None: ...\n", 3, id="positional_only_annotation"),
        pytest.param("def get_two(*, gate: (GATE := object)) -> None: ...\n", 3, id="keyword_only_annotation"),
        pytest.param("def get_two(*gates: (GATE := object)) -> None: ...\n", 3, id="variadic_annotation"),
        pytest.param("def get_two(**gates: (GATE := object)) -> None: ...\n", 3, id="keyword_variadic_annotation"),
        pytest.param("def get_two() -> (GATE := object): ...\n", 3, id="return_annotation"),
    ],
)
def test_a_definition_header_binds_in_the_scope_around_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str, counted: int
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, f"def get_one() -> None: ...\n\n\n{body}")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds {counted} definitions, above its baseline of 1; put the new "
        "one in the module of the domain whose work it does, which the root re-exports"
    ]


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            "try:\n    GATES = object()\nexcept Exception as gate:\n    pass\n",
            id="except_target_is_unbound",
        ),
        pytest.param("GATES = object()\ngate: object\n", id="an_annotation_with_no_value_binds_nothing"),
        pytest.param("def get_two() -> None:\n    gate = object()\n", id="a_function_body_is_its_own_scope"),
        pytest.param("class Gate:\n    gate = object()\n", id="a_class_body_is_its_own_scope"),
        pytest.param("GATES = [gate for gate in ()]\n", id="comprehension_has_its_own_scope"),
        pytest.param("GATES = lambda: (gate := object())\n", id="a_lambda_body_binds_in_its_own_scope"),
        pytest.param("GATES = {}\nGATES['one'] = object()\n", id="a_subscript_write_binds_nothing"),
        pytest.param("GATES = object()\nGATES = object()\n", id="one_name_bound_twice_is_one_name"),
    ],
)
def test_a_statement_that_binds_no_new_module_name_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 2)
    _root_with(tmp_path, f"def get_one() -> None: ...\n\n\n{body}")
    assert check.check_composition_root_size(tmp_path) == []


def test_a_definition_under_module_level_control_flow_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, "def get_one() -> None: ...\n\n\nif True:\n    def get_two() -> None: ...\n")
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 2 definitions, above its baseline of 1; put the new one "
        "in the module of the domain whose work it does, which the root re-exports"
    ]


def test_a_composition_root_that_is_gone_must_lower_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 5)
    assert check.check_composition_root_size(tmp_path) == [
        f"{check.COMPOSITION_ROOT_MODULE} holds 0 definitions, under its baseline of 5; lower the "
        "baseline to that, so the room made cannot be spent again"
    ]


def test_the_public_surface_list_is_not_a_definition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, 'def get_one() -> None: ...\n\n\n__all__ = ["get_one"]\n')
    assert check.check_composition_root_size(tmp_path) == []


def test_a_re_export_imported_into_the_root_is_not_a_definition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(
        tmp_path,
        "from gateway.api.deps.things import ThingDep, get_thing\n\n\ndef get_one() -> None: ...\n",
    )
    assert check.check_composition_root_size(tmp_path) == []


def test_a_nested_definition_does_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "COMPOSITION_ROOT_SIZE_BASELINE", 1)
    _root_with(tmp_path, "def get_one() -> None:\n    def inner() -> None: ...\n\n    return inner()\n")
    assert check.check_composition_root_size(tmp_path) == []


# ---------- rule 29: who may write an address-proof column ----------


def _writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str, *, allowed: str = "gateway/own.py"
) -> list[str]:
    monkeypatch.setattr(check, "COLUMN_WRITERS", {"email_proven_at": (allowed,)})
    _write(tmp_path, "gateway/own.py", "x.email_proven_at = 1\n")
    _write(tmp_path, "gateway/other.py", content)
    violations: list[str] = check.check_column_writers(tmp_path)
    return violations


@pytest.mark.parametrize(
    "spelling",
    [
        "x.email_proven_at = 1\n",
        'values["email_proven_at"] = 1\n',
        'values = {"email_proven_at": 1}\n',
        "statement.values(email_proven_at=1)\n",
        'setattr(x, "email_proven_at", 1)\n',
        "x.email_proven_at, y = 1, 2\n",
        "for x.email_proven_at in stamps:\n    pass\n",
        "with stamped() as x.email_proven_at:\n    pass\n",
    ],
)
def test_a_module_outside_the_list_may_not_write_the_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    """Every spelling reaches the same column, so a rule that reads one of them is not a rule."""
    violations = _writers(tmp_path, monkeypatch, spelling)
    assert [v for v in violations if "other.py" in v], spelling


@pytest.mark.parametrize(
    "not_a_write",
    [
        "def f(*, email_proven_at: int) -> None: ...\n",
        "y = x.email_proven_at\n",
        "if x.email_proven_at is None:\n    pass\n",
        '"""A docstring naming email_proven_at."""\n',
        "cache[x.email_proven_at] = 1\n",
    ],
)
def test_naming_the_column_without_writing_it_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, not_a_write: str
) -> None:
    assert _writers(tmp_path, monkeypatch, not_a_write) == []


def test_a_listed_module_that_stopped_writing_must_leave_the_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The list only shrinks, so a consolidation is proved rather than asserted."""
    monkeypatch.setattr(check, "COLUMN_WRITERS", {"email_proven_at": ("gateway/own.py", "gateway/stale.py")})
    _write(tmp_path, "gateway/own.py", "x.email_proven_at = 1\n")
    _write(tmp_path, "gateway/stale.py", "pass\n")
    assert check.check_column_writers(tmp_path) == [
        "gateway/stale.py no longer assigns email_proven_at; remove it from COLUMN_WRITERS"
    ]
