"""Guards on the two budget vocabularies.

The named constants cover each ``Literal``.
The deployment route's scope table has a row for every scope type.
"""

from gateway.api.routes.scoped_budgets import _SCOPE_SUBJECTS
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


def test_the_deployment_route_resolves_every_scope_type() -> None:
    assert set(_SCOPE_SUBJECTS) == set(SCOPE_TYPES)
