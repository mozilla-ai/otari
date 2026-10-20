"""The release-time bump of deploy/railway to a new image tag."""

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "bump_railway_template.py"
_RAILWAY_DIR = _REPO_ROOT / "deploy" / "railway"
_FILES = ("template.json", "listing.md", "README.md")


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bump_railway_template", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bump = _load()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A copy of the real deploy/railway, so the test pins today's wording."""
    shutil.copytree(_RAILWAY_DIR, tmp_path / "deploy" / "railway")
    return tmp_path


def _read(root: Path) -> dict[str, str]:
    return {name: (root / "deploy" / "railway" / name).read_text() for name in _FILES}


def _pinned() -> tuple[int, int, int]:
    template = json.loads((_RAILWAY_DIR / "template.json").read_text())
    version: tuple[int, int, int] = bump.parse_version(bump.pinned_version(template))
    return version


def _changed_lines(before: str, after: str) -> list[tuple[str, str]]:
    return [(a, b) for a, b in zip(before.splitlines(), after.splitlines(), strict=True) if a != b]


def test_next_minor_moves_every_image_reference_and_example(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    major, minor, patch = _pinned()
    old = f"{major}.{minor}.{patch}"
    new = f"{major}.{minor + 1}.0"
    before = _read(root)

    assert bump.main([f"v{new}", "--root", str(root)]) == 0

    assert capsys.readouterr().out.splitlines() == ["changed=true", f"previous={old}"]
    after = _read(root)
    for name in _FILES:
        assert f"mzdotai/otari:{old}" not in after[name]
        assert f"mzdotai/otari:{new}" in after[name]
    assert json.loads(after["template.json"])["services"]["otari"]["source"]["image"] == f"mzdotai/otari:{new}"
    assert f"for example `{major}.{minor + 2}.0`" in after["README.md"]
    assert f"(`{major}.{minor + 1}.0` to `{major}.{minor + 1}.1`)" in after["README.md"]
    # Only the image line moves in the two snapshot files; the README also moves its two examples.
    assert len(_changed_lines(before["template.json"], after["template.json"])) == 1
    assert len(_changed_lines(before["listing.md"], after["listing.md"])) == 1
    assert len(_changed_lines(before["README.md"], after["README.md"])) == 3


def test_patch_release_moves_only_the_image(root: Path) -> None:
    major, minor, patch = _pinned()
    bump.main([f"{major}.{minor + 1}.0", "--root", str(root)])
    before = _read(root)

    assert bump.main([f"{major}.{minor + 1}.1", "--root", str(root)]) == 0

    after = _read(root)
    assert f"mzdotai/otari:{major}.{minor + 1}.1" in after["README.md"]
    assert len(_changed_lines(before["README.md"], after["README.md"])) == 1


@pytest.mark.parametrize("delta", [(0, 0, 0), (0, -1, 0)], ids=["same", "older"])
def test_version_not_above_the_pin_changes_nothing(
    root: Path, capsys: pytest.CaptureFixture[str], delta: tuple[int, int, int]
) -> None:
    pinned = _pinned()
    version = ".".join(str(max(part + d, 0)) for part, d in zip(pinned, delta, strict=True))
    before = _read(root)

    assert bump.main([version, "--root", str(root)]) == 0

    assert capsys.readouterr().out.splitlines() == ["changed=false"]
    assert _read(root) == before


def _pending_template(tmp_path: Path, version: str) -> Path:
    template = json.loads((_RAILWAY_DIR / "template.json").read_text(encoding="utf-8"))
    template["services"]["otari"]["source"]["image"] = f"mzdotai/otari:{version}"
    path = tmp_path / "pending-template.json"
    path.write_text(json.dumps(template), encoding="utf-8")
    return path


def test_a_version_not_above_the_open_prs_pin_changes_nothing(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    major, minor, _ = _pinned()
    pending = _pending_template(tmp_path, f"{major}.{minor + 2}.0")
    before = _read(root)

    assert bump.main([f"{major}.{minor + 1}.0", "--root", str(root), "--pending-template", str(pending)]) == 0

    assert capsys.readouterr().out.splitlines() == ["changed=false"]
    assert _read(root) == before


def test_a_version_above_the_open_prs_pin_bumps_from_main(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    major, minor, patch = _pinned()
    pending = _pending_template(tmp_path, f"{major}.{minor + 1}.0")

    assert bump.main([f"{major}.{minor + 2}.0", "--root", str(root), "--pending-template", str(pending)]) == 0

    assert capsys.readouterr().out.splitlines() == ["changed=true", f"previous={major}.{minor}.{patch}"]
    assert f"mzdotai/otari:{major}.{minor + 2}.0" in _read(root)["template.json"]


@pytest.mark.parametrize("version", ["1.0.0-rc1", "1.0", "latest", "v1.0.0.1"])
def test_non_release_versions_are_refused(root: Path, version: str) -> None:
    before = _read(root)

    assert bump.main([version, "--root", str(root)]) == 1

    assert _read(root) == before


def test_a_missing_pattern_writes_nothing(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = root / "deploy" / "railway" / "README.md"
    readme.write_text(re.sub(r"for example\s+`\d+\.\d+\.\d+`", "for example the next one", readme.read_text()))
    before = _read(root)
    major, minor, _ = _pinned()

    assert bump.main([f"{major}.{minor + 1}.0", "--root", str(root)]) == 1

    assert "README.md: found no match" in capsys.readouterr().err
    assert _read(root) == before


def test_a_second_example_match_writes_nothing(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = root / "deploy" / "railway" / "README.md"
    readme.write_text(readme.read_text() + "\nPick a target release (for example `9.9.9`).\n")
    before = _read(root)
    major, minor, _ = _pinned()

    assert bump.main([f"{major}.{minor + 1}.0", "--root", str(root)]) == 1

    assert "README.md: found 2 matches" in capsys.readouterr().err
    assert _read(root) == before


def test_image_match_does_not_extend_into_a_longer_version() -> None:
    texts = {name: "mzdotai/otari:0.2.10 (for example `0.3.0`) (`0.2.0` to `0.2.1`)" for name in _FILES}

    with pytest.raises(bump.BumpError, match="template.json: found no match"):
        bump.bump_texts(texts, "0.2.1", (0, 3, 0))
