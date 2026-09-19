"""The named constants of each budget vocabulary cover its ``Literal``."""

from gateway.models.budgets import ALIGN_DAY, ALIGN_MONTH, ALIGN_WEEK, RESET_ALIGNMENTS


def test_the_alignment_constants_cover_the_literal() -> None:
    assert {ALIGN_DAY, ALIGN_WEEK, ALIGN_MONTH} == set(RESET_ALIGNMENTS)
