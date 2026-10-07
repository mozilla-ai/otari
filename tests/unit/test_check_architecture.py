"""Unit tests for the architecture check (layer rules over src/gateway and cli/src/otari_agent)."""

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
    # main() refuses to run without the light CLI's root, so every temporary tree gets an empty one.
    _write(tmp_path, "cli/src/otari_agent/__init__.py", "")
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
    _write(tmp_path, "gateway/services/thing_service.py", source)
    line = source.count("\n")
    assert check.check_transaction_control(tmp_path) == [
        f"gateway/services/thing_service.py:{line} calls {call}; {_TRANSACTION_REMEDY}"
    ]


def test_the_unit_of_work_may_commit_and_roll_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ())
    _write(
        tmp_path,
        "gateway/core/unit_of_work.py",
        "class UnitOfWork:\n    async def end(self) -> None:\n        await self._session.commit()\n",
    )
    assert check.check_transaction_control(tmp_path) == []


def test_a_module_on_the_transaction_baseline_may_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ("gateway/services/thing_service.py",))
    _write(
        tmp_path, "gateway/services/thing_service.py", "async def save(db: object) -> None:\n    await db.commit()\n"
    )
    assert check.check_transaction_control(tmp_path) == []


@pytest.mark.parametrize("source", ["async def save(db: object) -> None:\n    await db.flush()\n", None])
def test_a_transaction_baseline_entry_that_ends_no_transaction_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "TRANSACTION_CONTROL_BASELINE", ("gateway/services/thing_service.py",))
    _write(tmp_path, "gateway/services/other_service.py", "")
    if source is not None:
        _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_transaction_control(tmp_path) == [
        "gateway/services/thing_service.py is on the transaction control baseline but calls neither commit nor "
        "rollback; remove it from the baseline"
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
    _write(tmp_path, "gateway/api/deps.py", _FACTORY_SOURCE)
    assert check.check_unit_of_work_construction(tmp_path) == []


def test_the_unit_of_work_module_may_construct_one(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "gateway/core/unit_of_work.py",
        "def create_unit_of_work(session: object) -> UnitOfWork:\n    return UnitOfWork(session)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == []


@pytest.mark.parametrize(
    "relative_path", ["gateway/api/routes/things.py", "gateway/services/thing_service.py", "gateway/api/deps.py"]
)
def test_constructing_a_unit_of_work_elsewhere_is_flagged(tmp_path: Path, relative_path: str) -> None:
    _write(tmp_path, relative_path, "def build(db: object) -> UnitOfWork:\n    return UnitOfWork(db)\n")
    assert check.check_unit_of_work_construction(tmp_path) == [
        f"{relative_path}:2 constructs a UnitOfWork; {_UNIT_OF_WORK_REMEDY}"
    ]


def test_a_unit_of_work_reached_through_its_module_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "def build(db: object) -> object:\n    return unit_of_work.UnitOfWork(db)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == [
        f"gateway/services/thing_service.py:2 constructs a UnitOfWork; {_UNIT_OF_WORK_REMEDY}"
    ]


def test_naming_the_unit_of_work_as_a_type_is_not_a_construction(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps.py", _FACTORY_SOURCE)
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
        "gateway/api/deps.py",
        "class Deps:\n    def get_unit_of_work(self, db: object) -> UnitOfWork:\n        return UnitOfWork(db)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == [
        f"gateway/api/deps.py:3 constructs a UnitOfWork; {_UNIT_OF_WORK_REMEDY}"
    ]


def test_renaming_the_unit_of_work_on_import_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps.py", _FACTORY_SOURCE)
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
    _write(tmp_path, "gateway/api/deps.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from gateway.core.unit_of_work import UnitOfWork\n\n\n"
        "def save(uow: UnitOfWork) -> UnitOfWork:\n    return uow\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == []


def test_renaming_the_unit_of_work_on_a_relative_import_is_flagged(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/api/deps.py", _FACTORY_SOURCE)
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from ..core.unit_of_work import UnitOfWork as UoW\n\n\ndef build(db: object) -> UoW:\n    return UoW(db)\n",
    )
    assert check.check_unit_of_work_construction(tmp_path) == [
        "gateway/services/thing_service.py:1 imports UnitOfWork as UoW; "
        "the rule reads the name at the call site, so import it under its own name"
    ]


_MODE_READ_REMEDY = "a service gets the behavior for each mode from a binding, not from a branch on the mode"


@pytest.mark.parametrize(
    ("source", "name"),
    [
        ("def pick(config):\n    return config.is_hybrid_mode\n", "is_hybrid_mode"),
        ("def pick(self):\n    return self._config.is_hosted_mode\n", "is_hosted_mode"),
        ("def pick(config):\n    return config.effective_mode\n", "effective_mode"),
        ("def pick(config):\n    return config.configured_mode\n", "configured_mode"),
        ('def pick(config):\n    return getattr(config, "is_hybrid_mode")\n', "is_hybrid_mode"),
        ("def pick(config):\n    return deployment_for(config).supports(Plane.DATA)\n", "deployment_for"),
        ("def pick(config):\n    return deployment.deployment_for(config)\n", "deployment_for"),
    ],
)
@pytest.mark.parametrize("relative_path", ["gateway/services/thing_service.py", "gateway/services/things/_store.py"])
def test_a_service_that_reads_the_mode_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, source: str, name: str
) -> None:
    monkeypatch.setattr(check, "SERVICE_MODE_READ_BASELINE", ())
    _write(tmp_path, relative_path, source)
    assert check.check_service_mode_reads(tmp_path) == [f"{relative_path}:2 reads {name}; {_MODE_READ_REMEDY}"]


@pytest.mark.parametrize(
    ("relative_path", "source"),
    [
        ("gateway/api/main.py", "def mount(config):\n    return config.is_hybrid_mode\n"),
        ("gateway/container.py", "def bind(config):\n    return config.is_hosted_mode\n"),
        ("gateway/services/thing_service.py", "def pick(is_hybrid_mode: bool) -> bool:\n    return is_hybrid_mode\n"),
        ("gateway/services/thing_service.py", "def pick(state):\n    state.is_hybrid_mode = True\n"),
        (
            "gateway/services/thing_service.py",
            'def pick(helpers, config):\n    return helpers.getattr(config, "is_hybrid_mode")\n',
        ),
    ],
)
def test_a_mode_read_outside_services_or_a_mode_passed_in_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, source: str
) -> None:
    monkeypatch.setattr(check, "SERVICE_MODE_READ_BASELINE", ())
    _write(tmp_path, relative_path, source)
    assert check.check_service_mode_reads(tmp_path) == []


def test_a_service_on_the_mode_read_baseline_may_read_the_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "SERVICE_MODE_READ_BASELINE", ("gateway/services/thing_service.py",))
    _write(tmp_path, "gateway/services/thing_service.py", "def pick(config):\n    return config.is_hybrid_mode\n")
    assert check.check_service_mode_reads(tmp_path) == []


@pytest.mark.parametrize("source", ["def pick(config):\n    return config.port\n", None])
def test_a_mode_read_baseline_entry_that_reads_no_mode_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "SERVICE_MODE_READ_BASELINE", ("gateway/services/thing_service.py",))
    _write(tmp_path, "gateway/services/other.py", "")
    if source is not None:
        _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_service_mode_reads(tmp_path) == [
        "gateway/services/thing_service.py is on the mode read baseline but reads no mode; remove it from the baseline"
    ]


def test_main_fails_on_a_service_that_reads_the_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/__init__.py", "")
    _write(tmp_path, "src/gateway/services/things/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    _point_main_at(tmp_path, monkeypatch)
    assert check.main() == 0
    _write(tmp_path, "src/gateway/services/things/_store.py", "def pick(config):\n    return config.is_hybrid_mode\n")
    assert check.main() == 1


_LISTENER_DEFAULT_REMEDY = "a caller that leaves a listener out skips it with no error, so every caller passes one"
_LISTENER_MODULE = "gateway/services/things/thing_service.py"


def _listener_default(line: int, name: str, path: str) -> str:
    return f"{_LISTENER_MODULE}:{line} gives the listener {name} of {path} a default; {_LISTENER_DEFAULT_REMEDY}"


@pytest.mark.parametrize(
    ("source", "line", "path"),
    [
        ("def wire(workspace_listener: WorkspaceListener | None = None) -> None: ...\n", 1, "wire"),
        ("async def wire(workspace_listener: WorkspaceListener | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(*, workspace_listener: WorkspaceListener | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(workspace_listener: WorkspaceListener | None = None, /) -> None: ...\n", 1, "wire"),
        ("def wire(workspace_listener: WorkspaceListener = NoOpListener()) -> None: ...\n", 1, "wire"),
        ("def wire(workspace_listener: WorkspaceListenerBuilder | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(workspace_listener: WorkspaceListenerDep | None = None) -> None: ...\n", 1, "wire"),
        ("def wire(workspace_listener: tenancy.WorkspaceListener | None = None) -> None: ...\n", 1, "wire"),
        (
            "def wire(workspace_listener: Callable[[UnitOfWork], WorkspaceListener] | None = None) -> None: ...\n",
            1,
            "wire",
        ),
        (
            "def wire(workspace_listener: Annotated[WorkspaceListener, Depends(get)] | None = None) -> None: ...\n",
            1,
            "wire",
        ),
        ('def wire(workspace_listener: "WorkspaceListener | None" = None) -> None: ...\n', 1, "wire"),
        ('def wire(workspace_listener: Callable[[], "WorkspaceListener"] | None = None) -> None: ...\n', 1, "wire"),
        (
            "class Thing:\n    def __init__(self, workspace_listener: WorkspaceListener | None = None) -> None: ...\n",
            2,
            "Thing.__init__",
        ),
        (
            "def outer() -> None:\n    def inner(workspace_listener: WorkspaceListener | None = None) -> None: ...\n",
            2,
            "outer.inner",
        ),
        ("@dataclass\nclass Thing:\n    workspace_listener: WorkspaceListener | None = None\n", 3, "Thing"),
        (
            "@dataclass\nclass Thing:\n    workspace_listener: WorkspaceListener | None = field(default=None)\n",
            3,
            "Thing",
        ),
        ("class Thing(BaseModel):\n    workspace_listener: WorkspaceListener | None = Field(None)\n", 2, "Thing"),
        (
            "class Thing(BaseModel):\n    workspace_listener: WorkspaceListener | None = Field(None, init=False)\n",
            2,
            "Thing",
        ),
        (
            "class Thing(BaseModel):\n    workspace_listener: WorkspaceListener = Field(default_factory=make)\n",
            2,
            "Thing",
        ),
        ("class Thing(BaseModel):\n    workspace_listener: WorkspaceListener = Field(**options)\n", 2, "Thing"),
        ("@dataclass\nclass Thing:\n    workspace_listener: WorkspaceListener = field(default=...)\n", 3, "Thing"),
        ("@dataclass\nclass Thing:\n    workspace_listener: WorkspaceListener = field(**options)\n", 3, "Thing"),
    ],
)
def test_a_listener_with_a_default_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, line: int, path: str
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(tmp_path, _LISTENER_MODULE, source)
    assert check.check_listener_defaults(tmp_path) == [_listener_default(line, "workspace_listener", path)]


def test_each_listener_with_a_default_in_one_signature_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(
        tmp_path,
        _LISTENER_MODULE,
        "def wire(\n"
        "    membership_listener: MembershipListener | None = None,\n"
        "    workspace_listener: WorkspaceListener | None = None,\n"
        ") -> None: ...\n",
    )
    assert check.check_listener_defaults(tmp_path) == [
        _listener_default(2, "membership_listener", "wire"),
        _listener_default(3, "workspace_listener", "wire"),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "def wire(workspace_listener: WorkspaceListener | None) -> None: ...\n",
        "def wire(workspace_listener: WorkspaceListener, /, limit: int = 3) -> None: ...\n",
        "def wire(*, workspace_listener: WorkspaceListener) -> None: ...\n",
        "def wire(workspace_listener: WorkspaceListener, limit: int = 3) -> None: ...\n",
        "def wire(*, workspace_listener: WorkspaceListener, limit: int = 3) -> None: ...\n",
        "def wire(settings: Settings | None = None) -> None: ...\n",
        'def wire(kind: Literal["WorkspaceListener"] = "WorkspaceListener") -> None: ...\n',
        'def wire(label: Annotated[str, "WorkspaceListener"] = "") -> None: ...\n',
        'def wire(label: "\\x00" = "") -> None: ...\n',
        "class Thing:\n    workspace_listener: WorkspaceListener\n",
        "@dataclass\nclass Thing:\n    workspace_listener: WorkspaceListener = field(kw_only=True)\n",
        'class Thing(BaseModel):\n    workspace_listener: WorkspaceListener = Field(description="x")\n',
        "class Thing(BaseModel):\n    workspace_listener: WorkspaceListener = Field(...)\n",
        "class Thing(BaseModel):\n    workspace_listener: WorkspaceListener = Field(default=...)\n",
        "class Thing(BaseModel):\n    workspace_listener: WorkspaceListener = Field(default_factory=None)\n",
        "@dataclass\nclass Thing:\n    workspace_listener: WorkspaceListener = field(init=False, default=None)\n",
        "class Thing(BaseModel):\n    _workspace_listener: WorkspaceListener | None = PrivateAttr(default=None)\n",
        "@dataclass\nclass Thing:\n    workspace_listener: ClassVar[WorkspaceListener] = NoOpListener()\n",
        "def wire(label: Annotated[()] = 0) -> None: ...\n",
        "def wire(make: Callable[[WorkspaceListener], Service] = build) -> None: ...\n",
        "def wire(make: Callable[()] = build) -> None: ...\n",
        "def wire(kind: WorkspaceListener.Kind = WorkspaceListener.Kind.A) -> None: ...\n",
    ],
)
def test_a_listener_without_a_default_is_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(tmp_path, _LISTENER_MODULE, source)
    assert check.check_listener_defaults(tmp_path) == []


def test_a_listener_with_a_default_outside_services_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ())
    _write(
        tmp_path, "gateway/container.py", "def wire(workspace_listener: WorkspaceListener | None = None) -> None: ...\n"
    )
    assert check.check_listener_defaults(tmp_path) == [
        f"gateway/container.py:1 gives the listener workspace_listener of wire a default; {_LISTENER_DEFAULT_REMEDY}"
    ]


def test_a_listener_on_the_baseline_may_have_a_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ((_LISTENER_MODULE, "wire", "workspace_listener"),))
    _write(tmp_path, _LISTENER_MODULE, "def wire(workspace_listener: WorkspaceListener | None = None) -> None: ...\n")
    assert check.check_listener_defaults(tmp_path) == []


def test_a_second_listener_default_in_a_module_on_the_baseline_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ((_LISTENER_MODULE, "wire", "workspace_listener"),))
    _write(
        tmp_path,
        _LISTENER_MODULE,
        "def wire(workspace_listener: WorkspaceListener | None = None) -> None: ...\n"
        "def rewire(workspace_listener: WorkspaceListener | None = None) -> None: ...\n",
    )
    assert check.check_listener_defaults(tmp_path) == [_listener_default(2, "workspace_listener", "rewire")]


@pytest.mark.parametrize(
    "source",
    [
        "def wire(workspace_listener: WorkspaceListener | None) -> None: ...\n",
        "def renamed(workspace_listener: WorkspaceListener) -> None: ...\n",
        None,
    ],
)
def test_a_listener_default_baseline_entry_with_no_default_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "LISTENER_DEFAULT_BASELINE", ((_LISTENER_MODULE, "wire", "workspace_listener"),))
    _write(tmp_path, "gateway/services/things/other.py", "")
    if source is not None:
        _write(tmp_path, _LISTENER_MODULE, source)
    assert check.check_listener_defaults(tmp_path) == [
        f"{_LISTENER_MODULE} is on the listener default baseline for workspace_listener of wire, "
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
        "def wire(workspace_listener: WorkspaceListener | None = None) -> None: ...\n",
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
    "only the domain's own service and repository packages and the builders in gateway/api/deps.py import its "
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
        ("gateway/api/deps.py", "from gateway.repositories.things import ThingRepository\n"),
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
        ("gateway/api/deps.py", "import gateway.services.things.store\n", "gateway.services.things.store"),
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
