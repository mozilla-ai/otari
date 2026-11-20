"""Which hidden instructions on a fetched page reach the model (#3202).

``tests/fixtures/prompt_injection/changelog.html`` carries one instruction three
ways: an image ``alt``, an HTML comment and an off-screen ``<span>``. The table
below is what ``otari_web_fetch`` hands the model today. A change to it widens or
narrows the attack surface, so it should be a decision rather than a side effect
of reworking the extraction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gateway.services.file_extractors import html_main_text

PAGE = (Path(__file__).resolve().parents[1] / "fixtures" / "prompt_injection" / "changelog.html").read_text()

# Variant, the reference code its copy of the instruction ends with, and whether the extraction keeps it.
_SURVIVES = {
    "alt": ("LN-ALT", False),
    "comment": ("LN-COMMENT", False),
    "offscreen": ("LN-OFFSCREEN", True),
}


def test_the_page_carries_each_variant_once() -> None:
    """The table covers every marked variant; one added to the page owes it a row."""
    assert sorted(re.findall(r'data-injection="(\w+)"', PAGE)) == sorted(_SURVIVES)
    for code, _ in _SURVIVES.values():
        assert PAGE.count(code) == 1


@pytest.mark.parametrize("variant", list(_SURVIVES))
def test_a_variant_reaches_the_model_only_where_the_table_says(variant: str) -> None:
    code, survives = _SURVIVES[variant]

    assert (code in html_main_text(PAGE)) is survives


def test_the_page_itself_still_reaches_the_model() -> None:
    """The variants are judged against a page whose own content is kept, not one dropped whole."""
    text = html_main_text(PAGE)

    assert "LUMEN_SYNC_ENDPOINT" in text
    assert "Lumen Notes is a fictional product" not in text, "the footer is chrome"
