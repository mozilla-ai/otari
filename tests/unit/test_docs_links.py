"""Check that relative links between docs actually resolve.

The docs cross-link heavily (`docs/index.md` is the navigation, and each guide
points at its neighbours), so a renamed or missing page turns into a dead link
that nothing else catches. Only relative targets are checked; external URLs are
not fetched.
"""

import re
from pathlib import Path

import pytest

_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

_MARKDOWN_DOCS = sorted(_DOCS_DIR.rglob("*.md"))

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


@pytest.mark.parametrize("doc", _MARKDOWN_DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc: Path) -> None:
    broken: list[str] = []
    for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
        for target in _LINK.findall(line):
            if target.startswith(_EXTERNAL_PREFIXES):
                continue
            path = target.split("#", 1)[0]
            if not path:
                continue
            if not (doc.parent / path).exists():
                broken.append(f"line {lineno}: {target}")
    assert not broken, f"{doc.relative_to(_DOCS_DIR)} has unresolvable link(s): {broken}"


def test_integrations_page_links_every_integration_guide() -> None:
    summary = (_DOCS_DIR / "SUMMARY.md").read_text(encoding="utf-8")
    group = summary.split("\n## Integrations\n", 1)[1].split("\n#", 1)[0]
    guides = sorted(_LINK.findall(group))
    assert guides, "docs/SUMMARY.md has no Integrations group"
    integrations = (_DOCS_DIR / "integrations.md").read_text(encoding="utf-8")
    missing = [name for name in guides if f"({name})" not in integrations]
    assert not missing, f"docs/integrations.md does not link: {missing}"
