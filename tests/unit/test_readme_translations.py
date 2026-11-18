"""Each translated README keeps the English README's structure, code and links.

Prose may lag behind (the translation's provenance comment says by how much), but a
command or a link that changed in ``README.md`` must change in every translation too,
because those are what break for a reader. None of this needs the translation's
language to fix.
"""

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _REPO_ROOT / "README.md"
_TRANSLATIONS = sorted(_REPO_ROOT.glob("README.*.md"))

_FENCE = re.compile(r"^```[^\n]*\n.*?^```$", re.MULTILINE | re.DOTALL)
_SECTION = re.compile(r"^## ", re.MULTILINE)
_LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)|\bsrc=\"([^\"]+)\"")
_STAMP = re.compile(r"\A<!-- Translated from README\.md at commit ([0-9a-f]{7,40})\. ")


def _code_blocks(text: str) -> list[str]:
    return _FENCE.findall(text)


def _link_targets(text: str) -> set[str]:
    prose = _FENCE.sub("", text)
    targets = {a or b for a, b in _LINK_TARGET.findall(prose)}
    # The language switcher is the one place the files link to each other.
    return {t for t in targets if not re.fullmatch(r"README(\.[\w-]+)?\.md", t)}


def test_there_is_a_translation() -> None:
    assert _TRANSLATIONS, "the glob found no README translation, so the tests below check nothing"


@pytest.mark.parametrize("translation", _TRANSLATIONS, ids=lambda p: p.name)
def test_translation_has_the_same_sections(translation: Path) -> None:
    source, translated = _SOURCE.read_text(encoding="utf-8"), translation.read_text(encoding="utf-8")
    assert len(_SECTION.findall(translated)) == len(_SECTION.findall(source))


@pytest.mark.parametrize("translation", _TRANSLATIONS, ids=lambda p: p.name)
def test_translation_has_the_same_code_blocks(translation: Path) -> None:
    source, translated = _SOURCE.read_text(encoding="utf-8"), translation.read_text(encoding="utf-8")
    assert _code_blocks(translated) == _code_blocks(source)


@pytest.mark.parametrize("translation", _TRANSLATIONS, ids=lambda p: p.name)
def test_translation_has_the_same_link_targets(translation: Path) -> None:
    source, translated = _SOURCE.read_text(encoding="utf-8"), translation.read_text(encoding="utf-8")
    assert _link_targets(translated) == _link_targets(source)


@pytest.mark.parametrize("translation", _TRANSLATIONS, ids=lambda p: p.name)
def test_translation_records_the_commit_it_was_made_from(translation: Path) -> None:
    assert _STAMP.match(translation.read_text(encoding="utf-8"))


@pytest.mark.parametrize("translation", _TRANSLATIONS, ids=lambda p: p.name)
def test_both_readmes_link_to_each_other(translation: Path) -> None:
    assert f"]({translation.name})" in _SOURCE.read_text(encoding="utf-8")
    assert "](README.md)" in translation.read_text(encoding="utf-8")
