"""Unit tests for the architecture check (layer rules over src/gateway)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_architecture.py"
_DISCOVERY_MESSAGE = "Forbidden import in OSS base (no entry-point discovery; the feature registry is a literal tuple)"
_SESSION_MESSAGE = (
    "takes a session; move it onto its domain's service, "
    "which receives repositories and a Unit of Work, never a session"
)


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
        "from gateway.services.budget_service import reserve\n",
    )
    violations = check.check_file(file_path, tmp_path)
    assert violations == [(1, "gateway.services.budget_service", "Forbidden import in Repositories")]


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
    ["gateway.api.deps", "gateway.services.budget_service", "gateway.adapters.billing_adapter"],
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
        "from gateway.ports.billing_port import BillingPort\nfrom gateway.services.budget_service import reserve\n",
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
    _write(tmp_path, "gateway/repositories/helpers.py", "")
    violations = check.check_naming_conventions(tmp_path)
    assert violations == ["Repository file gateway/repositories/helpers.py must end with '_repository.py'"]


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
    _write(tmp_path, "tests/unit/test_thing.py", "from gateway.overlay.billing import charge\n")
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(check, "GATEWAY_ROOT", tmp_path / "src" / "gateway")
    monkeypatch.setattr(check, "TESTS_ROOT", tmp_path / "tests")
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
    _write(tmp_path, "src/otari_probe/__init__.py", "")
    _write(tmp_path, "tests/__init__.py", "")
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(check, "GATEWAY_ROOT", tmp_path / "src" / "gateway")
    monkeypatch.setattr(check, "TESTS_ROOT", tmp_path / "tests")
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


@pytest.mark.parametrize(
    "relative_path", ["gateway/api/routes/things.py", "gateway/services/thing_service.py", "gateway/services/sub/x.py"]
)
@pytest.mark.parametrize(
    ("source", "line", "name"),
    [
        ("from sqlalchemy import select\n", 1, "select"),
        ("from sqlmodel import select\n", 1, "select"),
        ("from sqlalchemy.dialects.postgresql import insert\n", 1, "insert"),
        ("import sqlalchemy as sa\n\nsa.update\n", 3, "update"),
        ("import sqlalchemy.sql\n\nsqlalchemy.sql.delete\n", 3, "delete"),
        ("from sqlalchemy import text\n", 1, "text"),
        ("from sqlalchemy import sql\n\nsql.select\n", 3, "select"),
        ("from sqlalchemy.sql import expression as e\n\ne.update\n", 3, "update"),
    ],
)
def test_a_route_or_service_that_builds_a_query_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str, source: str, line: int, name: str
) -> None:
    monkeypatch.setattr(check, "QUERY_BASELINE", ())
    _write(tmp_path, relative_path, source)
    assert check.check_query_layering(tmp_path) == [
        f"{relative_path}:{line} builds a query with {name}; a query belongs in a repository"
    ]


def test_a_repository_may_build_a_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "QUERY_BASELINE", ())
    _write(tmp_path, "gateway/repositories/thing_repository.py", "from sqlalchemy import select\n")
    assert check.check_query_layering(tmp_path) == []


def test_a_service_may_use_query_types_and_column_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "QUERY_BASELINE", ())
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from sqlalchemy import Select, func\n"
        "from sqlalchemy.ext.asyncio import AsyncSession\n"
        "from sqlmodel import col\n",
    )
    assert check.check_query_layering(tmp_path) == []


def test_a_module_on_the_query_baseline_may_build_a_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "QUERY_BASELINE", ("gateway/services/thing_service.py",))
    _write(tmp_path, "gateway/services/thing_service.py", "from sqlalchemy import select\n")
    assert check.check_query_layering(tmp_path) == []


@pytest.mark.parametrize("source", ["from gateway.repositories.thing_repository import find\n", None])
def test_a_baseline_entry_that_builds_no_query_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    # The baseline only shrinks: a module that stops querying, or no longer exists, fails until removed.
    monkeypatch.setattr(check, "QUERY_BASELINE", ("gateway/services/thing_service.py",))
    _write(tmp_path, "gateway/services/other_service.py", "")
    if source is not None:
        _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_query_layering(tmp_path) == [
        "gateway/services/thing_service.py is on the query baseline but builds no query; remove it from the baseline"
    ]


def test_main_fails_on_a_service_that_builds_a_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/thing_service.py", "from sqlalchemy import select\n")
    _write(tmp_path, "tests/__init__.py", "")
    monkeypatch.setattr(check, "QUERY_BASELINE", ())
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(check, "GATEWAY_ROOT", tmp_path / "src" / "gateway")
    monkeypatch.setattr(check, "TESTS_ROOT", tmp_path / "tests")
    assert check.main() == 1


@pytest.mark.parametrize(
    "signature",
    [
        "async def find(db: AsyncSession) -> None: ...",
        "async def find(owner: str, *, db: AsyncSession | None = None) -> None: ...",
        'async def find(db: "AsyncSession") -> None: ...',
        "def find(db: asyncio.AsyncSession) -> None: ...",
    ],
)
def test_a_module_level_service_function_that_takes_a_session_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signature: str
) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", f"{signature}\n")
    assert check.check_session_parameters(tmp_path) == [f"gateway/services/thing_service.py:1 find {_SESSION_MESSAGE}"]


@pytest.mark.parametrize(
    "source",
    [
        "class ThingService:\n    def __init__(self, db: AsyncSession) -> None: ...\n",
        "def outer() -> None:\n    async def inner(db: AsyncSession) -> None: ...\n",
        "def find(owner: str) -> None: ...\n",
    ],
)
def test_a_method_a_nested_function_or_a_sessionless_function_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_session_parameters(tmp_path) == []


def test_a_function_outside_services_may_take_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(tmp_path, "gateway/repositories/thing_repository.py", "async def find(db: AsyncSession) -> None: ...\n")
    assert check.check_session_parameters(tmp_path) == []


def test_a_function_on_the_session_baseline_may_take_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ("gateway/services/thing_service.py::find",))
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "async def find(db: AsyncSession) -> None: ...\nasync def count(db: AsyncSession) -> None: ...\n",
    )
    assert check.check_session_parameters(tmp_path) == [f"gateway/services/thing_service.py:2 count {_SESSION_MESSAGE}"]


@pytest.mark.parametrize("source", ["def find(owner: str) -> None: ...\n", None])
def test_a_session_baseline_entry_that_takes_no_session_must_leave_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str | None
) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ("gateway/services/thing_service.py::find",))
    _write(tmp_path, "gateway/services/other_service.py", "")
    if source is not None:
        _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_session_parameters(tmp_path) == [
        "gateway/services/thing_service.py::find is on the session parameter baseline but takes no session; "
        "remove it from the baseline"
    ]


def test_main_fails_on_a_service_function_that_takes_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/gateway/services/thing_service.py", "async def find(db: AsyncSession) -> None: ...\n")
    _write(tmp_path, "tests/__init__.py", "")
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(check, "GATEWAY_ROOT", tmp_path / "src" / "gateway")
    monkeypatch.setattr(check, "TESTS_ROOT", tmp_path / "tests")
    assert check.main() == 1


@pytest.mark.parametrize(
    "signature",
    [
        "def build(factory: Callable[[], AsyncSession]) -> None: ...",
        "def build(factory: async_sessionmaker[AsyncSession]) -> None: ...",
        'def build(factory: "Callable[[], AsyncSession]") -> None: ...',
    ],
)
def test_a_parameter_that_only_mentions_the_session_type_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signature: str
) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", f"{signature}\n")
    assert check.check_session_parameters(tmp_path) == []


@pytest.mark.parametrize(
    "annotation",
    ["Optional[AsyncSession]", "Annotated[AsyncSession, Depends(get_db)]", "None | AsyncSession"],
)
def test_a_wrapped_session_annotation_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, annotation: str
) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", f"async def find(db: {annotation}) -> None: ...\n")
    assert check.check_session_parameters(tmp_path) == [f"gateway/services/thing_service.py:1 find {_SESSION_MESSAGE}"]


@pytest.mark.parametrize(
    "source",
    [
        "if enabled:\n    async def find(db: AsyncSession) -> None: ...\n",
        "try:\n    pass\nexcept ImportError:\n    async def find(db: AsyncSession) -> None: ...\n",
        "with suppress(Exception):\n    async def find(db: AsyncSession) -> None: ...\n",
    ],
)
def test_a_function_defined_in_module_level_control_flow_is_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", source)
    violations = check.check_session_parameters(tmp_path)
    assert [violation.split(" ", 1)[1] for violation in violations] == [f"find {_SESSION_MESSAGE}"]


@pytest.mark.parametrize(
    "source",
    [
        "from sqlalchemy.ext.asyncio import AsyncSession as DBSession\n\nasync def find(db: DBSession) -> None: ...\n",
        "from sqlalchemy.ext.asyncio import AsyncSession as DBSession\n\n"
        "async def find(db: DBSession | None) -> None: ...\n",
    ],
)
def test_an_aliased_session_annotation_is_flagged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(tmp_path, "gateway/services/thing_service.py", source)
    assert check.check_session_parameters(tmp_path) == [f"gateway/services/thing_service.py:3 find {_SESSION_MESSAGE}"]


def test_a_type_that_shares_an_alias_name_is_not_a_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    _write(
        tmp_path,
        "gateway/services/thing_service.py",
        "from gateway.types import Session as DBSession\n\nasync def find(db: DBSession) -> None: ...\n",
    )
    assert check.check_session_parameters(tmp_path) == []


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
    _use_empty_flat_module_baseline(monkeypatch)
    monkeypatch.setattr(check, "QUERY_BASELINE", ())
    monkeypatch.setattr(check, "SESSION_PARAMETER_BASELINE", ())
    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(check, "GATEWAY_ROOT", tmp_path / "src" / "gateway")
    monkeypatch.setattr(check, "TESTS_ROOT", tmp_path / "tests")
    assert check.main() == 0
    _write(tmp_path, "src/gateway/services/things.py", "")
    assert check.main() == 1
