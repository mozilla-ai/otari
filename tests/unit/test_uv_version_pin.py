"""The uv version is declared once, and every place that installs uv satisfies it.

``pyproject.toml``'s ``required-version`` is the declaration. ``astral-sh/setup-uv`` reads it
itself, so the workflows need no version of their own, but a container that installs uv another
way does, and uv then refuses every command if the two disagree. This reads the declaration and
holds each of those to it, so the disagreement fails here rather than in a Docker build or a load
test.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Each file that installs uv for itself, and the build argument naming the version it installs.
_PINNED_DOCKERFILES = (Path("Dockerfile"), Path("loadtest/uv-tools.Dockerfile"))

_ARG_DEFAULT = re.compile(r"^ARG UV_VERSION=(\S+)\s*$", re.MULTILINE)
_REQUIRED_VERSION = re.compile(r'^required-version = "(?P<spec>[^"]+)"$', re.MULTILINE)


def _declared_range() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The inclusive floor and exclusive ceiling ``required-version`` declares."""
    manifest = (REPO_ROOT / "pyproject.toml").read_text()
    found = _REQUIRED_VERSION.findall(manifest)
    assert len(found) == 1, f"expected one required-version declaration, found {len(found)}"
    spec = found[0]
    match = re.fullmatch(r">=(?P<floor>[\d.]+),<(?P<ceiling>[\d.]+)", spec)
    assert match, f"required-version is {spec!r}; this test reads the form '>=X.Y.Z,<A.B.C'"
    return _parts(match.group("floor")), _parts(match.group("ceiling"))


def _parts(version: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in version.split("."))


def test_required_version_is_declared_exactly_once() -> None:
    """Every other place is checked against it, so a second declaration would make it ambiguous."""
    manifest = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "required-version" in manifest["tool"]["uv"]


def test_every_dockerfile_installs_a_uv_the_declaration_admits() -> None:
    """A container's own uv is pinned by an exact version, and uv refuses a command outside the range."""
    floor, ceiling = _declared_range()

    for path in _PINNED_DOCKERFILES:
        text = (REPO_ROOT / path).read_text()
        pinned = _ARG_DEFAULT.findall(text)
        assert len(pinned) == 1, f"{path} should declare exactly one 'ARG UV_VERSION=', found {len(pinned)}"
        version = _parts(pinned[0])
        assert floor <= version < ceiling, (
            f"{path} installs uv {pinned[0]}, which pyproject.toml's required-version does not admit"
        )


def test_no_dockerfile_installs_uv_unpinned() -> None:
    """``pip install uv`` takes whatever is newest, which the declaration can then refuse."""
    for path in _PINNED_DOCKERFILES:
        text = (REPO_ROOT / path).read_text()
        assert "pip install" not in text or "uv" not in text.split("pip install")[1].split("\n")[0], (
            f"{path} installs uv through pip, so its version is whatever is newest at build time"
        )
