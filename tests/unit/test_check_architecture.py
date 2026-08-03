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


def test_nested_and_enclosing_layer_rules_both_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A broader gateway/api rule declared first must neither shadow the nested
    # routes rule nor stop applying to files under it, whatever the order.
    broadened = {
        "gateway/api": {"allowed": [], "forbidden": ["gateway.db"], "description": "API"},
        **check.RULES,
    }
    monkeypatch.setattr(check, "RULES", broadened)
    file_path = _write(
        tmp_path,
        "gateway/api/routes/users.py",
        "from gateway.db import engine\nfrom sqlalchemy.orm import Session\n",
    )
    assert check.check_file(file_path, tmp_path) == [
        (1, "gateway.db", "Forbidden import in API"),
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


def test_ports_and_adapters_scaffolding_is_noop(tmp_path: Path) -> None:
    ports_file = _write(tmp_path, "gateway/ports/billing.py", "from gateway.api.routes import chat\n")
    adapters_file = _write(tmp_path, "gateway/adapters/null_billing.py", "from gateway.api.routes import chat\n")
    assert check.check_file(ports_file, tmp_path) == []
    assert check.check_file(adapters_file, tmp_path) == []


def test_repository_naming_convention(tmp_path: Path) -> None:
    _write(tmp_path, "gateway/repositories/__init__.py", "")
    _write(tmp_path, "gateway/repositories/users_repository.py", "")
    _write(tmp_path, "gateway/repositories/helpers.py", "")
    violations = check.check_naming_conventions(tmp_path)
    assert violations == ["Repository file gateway/repositories/helpers.py must end with '_repository.py'"]


def test_real_gateway_tree_is_clean() -> None:
    assert check.main() == 0
