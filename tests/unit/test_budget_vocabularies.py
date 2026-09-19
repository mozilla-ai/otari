"""The named constants of each budget vocabulary cover its ``Literal``."""

from gateway.models.budgets import (
    ALIGN_DAY,
    ALIGN_MONTH,
    ALIGN_WEEK,
    RESET_ALIGNMENTS,
    SCOPE_API_TOKEN,
    SCOPE_ORG_MEMBER,
    SCOPE_ORGANIZATION,
    SCOPE_TYPES,
    SCOPE_WORKSPACE,
    SCOPE_WORKSPACE_MEMBER,
)


def test_the_alignment_constants_cover_the_literal() -> None:
    assert {ALIGN_DAY, ALIGN_WEEK, ALIGN_MONTH} == set(RESET_ALIGNMENTS)


def test_the_scope_constants_cover_the_literal() -> None:
    named = {SCOPE_ORGANIZATION, SCOPE_WORKSPACE, SCOPE_WORKSPACE_MEMBER, SCOPE_ORG_MEMBER, SCOPE_API_TOKEN}
    assert named == set(SCOPE_TYPES)
