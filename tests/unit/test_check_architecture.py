"""Unit tests for the architecture check (layer rules over src/gateway)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_architecture.py"


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
    file_path = _write(tmp_path, "gateway/services/thing.py", "from gateway.models.entities import User\n")
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
