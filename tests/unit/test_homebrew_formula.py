"""The Homebrew formula renderer, run against the committed lock so a heavy dependency fails a PR."""

import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "homebrew_formula.py"

# What a Homebrew install of the light CLI must never pull in: the gateway
# and the server stack gateway.core.config drags along.
_SERVER_STACK = {
    "gateway",
    "uvicorn",
    "any-llm-sdk",
    "sqlalchemy",
    "sqlmodel",
    "fastapi",
    "pydantic",
    "pydantic-settings",
}
# Homebrew builds every resource from source, so the closure stays small and pure.
_MAX_CLOSURE = 15


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("homebrew_formula", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


formula = _load()


@pytest.fixture(scope="module")
def lock() -> dict[str, object]:
    with (_REPO_ROOT / "uv.lock").open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture
def sdist(tmp_path: Path) -> Path:
    path = tmp_path / "otari_agent-1.2.3.tar.gz"
    path.write_bytes(b"not really a tarball")
    return path


def _direct_dependencies() -> set[str]:
    with (_REPO_ROOT / "cli" / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return {formula.normalize(re.split(r"[<>=!~ ;\[]", spec, maxsplit=1)[0]) for spec in project["dependencies"]}


def test_closure_covers_the_declared_dependencies_and_nothing_heavy(lock: dict[str, object]) -> None:
    members = formula.closure(lock)
    assert _direct_dependencies() <= set(members)
    assert not _SERVER_STACK & set(members), sorted(_SERVER_STACK & set(members))
    assert "otari-agent" not in members
    assert len(members) <= _MAX_CLOSURE, sorted(members)


def test_every_closure_member_has_a_pypi_sdist_with_a_sha256(lock: dict[str, object]) -> None:
    for name, package in formula.closure(lock).items():
        sdist = package["sdist"]
        assert sdist["url"].startswith("https://files.pythonhosted.org/packages/"), name
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", sdist["hash"]), name


def test_rendered_formula_has_one_resource_per_member(lock: dict[str, object], sdist: Path) -> None:
    text = formula.build_formula(version="1.2.3", sdist=sdist)
    assert "class Otari < Formula" in text
    assert 'url "https://github.com/mozilla-ai/otari/releases/download/v1.2.3/otari_agent-1.2.3.tar.gz"' in text
    assert f'sha256 "{formula.sha256_of(sdist)}"' in text
    assert re.findall(r'^  resource "([^"]+)" do$', text, re.MULTILINE) == sorted(formula.closure(lock))
    assert 'depends_on "python@3.14"' in text
    assert "{{" not in text


def test_a_wheel_only_dependency_is_refused() -> None:
    packages = {"wheels-only": {"name": "wheels-only", "version": "1.0", "wheels": [{"url": "x"}]}}
    with pytest.raises(formula.FormulaError, match="no sdist"):
        formula.resource_stanzas(packages)


def test_a_dependency_missing_from_the_lock_is_refused() -> None:
    lock = {"package": [{"name": "otari-agent", "version": "0.0.0", "dependencies": [{"name": "ghost"}]}]}
    with pytest.raises(formula.FormulaError, match="ghost is not in"):
        formula.closure(lock)


def test_a_forked_dependency_is_refused() -> None:
    lock = {
        "package": [
            {"name": "otari-agent", "version": "0.0.0", "dependencies": [{"name": "dup"}]},
            {"name": "dup", "version": "1.0"},
            {"name": "dup", "version": "2.0"},
        ]
    }
    with pytest.raises(formula.FormulaError, match="more than one version"):
        formula.closure(lock)


def test_an_unfilled_placeholder_is_an_error() -> None:
    with pytest.raises(formula.FormulaError, match="not filled: resources"):
        formula.render("{{version}} {{resources}}", version="1")


def test_cli_writes_the_formula(sdist: Path, tmp_path: Path) -> None:
    output = tmp_path / "otari.rb"
    assert formula.main(["--version", "1.2.3", "--sdist", str(sdist), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8").startswith("# Rendered by scripts/homebrew_formula.py")
