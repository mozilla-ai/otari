"""Structural checks for the admin dashboard operator guide (docs/dashboard.md).

The guide is hand-written prose, so these tests guard the contract that matters:
the file exists, is linked from the docs index, covers the pieces issue #313
asked for (the two-key model, a first-run walkthrough, a page-by-page
reference), keeps its internal links pointing at real files, and follows the
project's no-em-dash-in-prose writing convention.
"""

import re
from pathlib import Path

import pytest

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_DASHBOARD_DOC = _DOCS_DIR / "dashboard.md"
_INDEX_DOC = _DOCS_DIR / "index.md"

# Markdown links to other docs, e.g. [text](configuration.md#anchor).
_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@pytest.fixture(scope="module")
def dashboard_text() -> str:
    return _DASHBOARD_DOC.read_text(encoding="utf-8")


def test_dashboard_doc_exists() -> None:
    assert _DASHBOARD_DOC.is_file(), "docs/dashboard.md is missing"


def test_dashboard_doc_has_single_top_level_heading(dashboard_text: str) -> None:
    h1s = [line for line in dashboard_text.splitlines() if line.startswith("# ")]
    assert h1s == ["# Admin dashboard"], f"expected one H1 'Admin dashboard', got {h1s}"


def test_dashboard_doc_covers_required_sections(dashboard_text: str) -> None:
    for heading in (
        "## The two-key model",
        "## First-run walkthrough",
        "## Page-by-page reference",
    ):
        assert heading in dashboard_text, f"missing required section: {heading!r}"


def test_dashboard_doc_explains_the_two_keys(dashboard_text: str) -> None:
    # The whole point of the guide: keep the sign-in key and the encryption key
    # distinct. Both env var names must appear so the guide stays accurate if
    # either is renamed in code.
    for token in ("OTARI_MASTER_KEY", "OTARI_SECRET_KEY", "master key", "gen-secret-key"):
        assert token in dashboard_text, f"two-key model text should mention {token!r}"


def test_dashboard_doc_internal_links_resolve(dashboard_text: str) -> None:
    for target in _LINK_RE.findall(dashboard_text):
        if target.startswith(("http://", "https://", "#")):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:
            continue
        resolved = (_DASHBOARD_DOC.parent / path_part).resolve()
        assert resolved.is_file(), f"broken doc link: {target} -> {resolved}"


def test_dashboard_doc_is_linked_from_index() -> None:
    index_text = _INDEX_DOC.read_text(encoding="utf-8")
    assert "(dashboard.md)" in index_text, "docs/index.md does not link to dashboard.md"


def test_dashboard_doc_avoids_em_dash_prose(dashboard_text: str) -> None:
    # Project writing convention (AGENTS.md): no em dashes or `--` separators in
    # doc prose. En-dash numeric ranges and code are out of scope, and this doc
    # has none, so a plain scan is enough.
    assert "—" not in dashboard_text, "docs/dashboard.md contains an em dash (—); use commas/colons/periods"
    assert " -- " not in dashboard_text, "docs/dashboard.md uses ' -- ' as a separator; rephrase"
