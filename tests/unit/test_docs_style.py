"""Enforce the project's doc prose conventions across all of `docs/`.

AGENTS.md's writing-style section bans em dashes as separators in doc prose
(en-dash numeric ranges and CLI flags are a different matter and are not em
dashes). The em dash (U+2014) has no legitimate use in these docs, so a plain
repo-wide scan keeps the convention from silently drifting back in.
"""

from pathlib import Path

import pytest

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

_EM_DASH = "—"

_MARKDOWN_DOCS = sorted(_DOCS_DIR.rglob("*.md"))


@pytest.mark.parametrize("doc", _MARKDOWN_DOCS, ids=lambda p: p.name)
def test_doc_has_no_em_dash(doc: Path) -> None:
    text = doc.read_text(encoding="utf-8")
    offenders = [i + 1 for i, line in enumerate(text.splitlines()) if _EM_DASH in line]
    assert not offenders, (
        f"{doc.relative_to(_DOCS_DIR)} uses an em dash (—) on line(s) {offenders}; "
        "use commas, semicolons, colons, parentheses, or periods instead"
    )


def test_docs_dir_is_discovered() -> None:
    # Guard against the glob silently matching nothing (e.g. a moved docs dir),
    # which would make every parametrized case vacuously pass.
    assert _MARKDOWN_DOCS, "no markdown docs found under docs/"
