"""Keep the docs' screenshots, the specs that take them and the docs' wording in step.

The images under docs/assets/screenshots are written by the Playwright `docs`
project (web/e2e/docs), which walks each docs page's steps against a real
gateway. Nothing there compares pixels, so these checks are what tie the three
together: every image is captured by a spec and shown on a page, and every
control a page's steps name in bold is one its spec actually reaches. A
relabeled button then fails the spec, and the spec cannot be fixed without the
page being fixed too.
"""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DOCS_DIR = _ROOT / "docs"
_SCREENSHOTS_DIR = _DOCS_DIR / "assets" / "screenshots"
_SPECS_DIR = _ROOT / "web" / "e2e" / "docs"

_CAPTURE = re.compile(r"docScreenshot\(\s*\w+,\s*\"([^\"]+)\"")
_REFERENCE = re.compile(r"assets/screenshots/([\w-]+)\.png")
_STEP = re.compile(r"^\d+\.\s.*(?:\n {3}\S.*)*", re.MULTILINE)
_BOLD = re.compile(r"\*\*([^*]+)\*\*")

_SPECS = sorted(_SPECS_DIR.glob("*.spec.ts"))


def _reaches(term: str, source: str) -> bool:
    """Whether a spec names the term as a string or a regular expression, or starts one with it.

    A placeholder such as `<workspace>` stands for whatever the spec puts there.
    """
    body = r"[^\"/]*".join(re.escape(part) for part in re.split(r"<[^>]+>", term))
    return re.search(r'(?:["/]|\^)' + body + r"(?!\w)", source) is not None


def _captured() -> set[str]:
    return {name for spec in _SPECS for name in _CAPTURE.findall(spec.read_text(encoding="utf-8"))}


def _referenced() -> set[str]:
    return {name for doc in _DOCS_DIR.rglob("*.md") for name in _REFERENCE.findall(doc.read_text(encoding="utf-8"))}


def test_every_screenshot_is_captured_shown_and_on_disk() -> None:
    captured, referenced = _captured(), _referenced()
    on_disk = {path.stem for path in _SCREENSHOTS_DIR.glob("*.png")}
    assert captured, f"no docScreenshot call found under {_SPECS_DIR.relative_to(_ROOT)}"
    assert referenced - captured == set(), "docs show screenshots no spec under web/e2e/docs captures"
    assert captured - referenced == set(), "specs capture screenshots no docs page shows"
    assert on_disk == captured, "docs/assets/screenshots is out of date; run `pnpm --dir web run docs:screenshots`"


@pytest.mark.parametrize("spec", _SPECS, ids=lambda p: p.name)
def test_steps_name_only_what_the_spec_reaches(spec: Path) -> None:
    doc = _DOCS_DIR / spec.name.removesuffix(".spec.ts")
    doc = doc.with_suffix(".md")
    assert doc.is_file(), f"{spec.name} is named for {doc.relative_to(_ROOT)}, which does not exist"
    source = spec.read_text(encoding="utf-8")
    steps = _STEP.findall(doc.read_text(encoding="utf-8"))
    terms = {" ".join(term.split()) for step in steps for term in _BOLD.findall(step)}
    unreached = sorted(term for term in terms if not _reaches(term, source))
    assert not unreached, (
        f"{doc.relative_to(_ROOT)} names {unreached} in its steps, which {spec.name} never looks for. "
        "Either the dashboard's wording changed and the page should follow, or the spec should reach them."
    )
