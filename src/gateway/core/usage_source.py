"""One notion of "usage this deployment served itself", over ``usage_logs.source``.

The column carries provenance: the bare slug ``gateway`` for a request Otari served,
and a source slug (``claude_code``, ``codex``) for usage imported through
``POST /v1/usage/external-events``. Hosted history adds a third shape, because a row
backfilled from otari.ai keeps its origin under a legacy prefix: traffic that
deployment served itself arrives as ``otari-ai:gateway``, and usage a customer had
imported there as ``otari-ai:claude_code``.

So the question is about the slug behind the prefix, never the prefix itself. A
blanket ``otari-ai:%`` match would be wrong in the direction that breaks a feature:
``otari-ai:claude_code`` is imported usage that happens to have been migrated, and the
operator surface whose whole purpose is repricing imported usage has to keep reaching
it.

Both callers ask this same question, from opposite sides, which is why it is here
rather than a literal in each: the usage admin mutations exclude served-here rows
(they may only touch imported usage, and ``counts_toward_budget`` does not tell them
apart), and the activation guide requires one (imported usage is somebody else's
traffic, so it is never a workspace's first request to this gateway).
"""

from typing import Any, cast

from sqlalchemy import ColumnElement

# The slug on a row this gateway served itself.
SERVED_HERE_SLUG = "gateway"

# What a hosted-history backfill puts in front of the origin's own slug so a migrated
# row stays distinguishable from one recorded live here (otari-ai#1798). Reserving it
# against an importer claiming it is the ingest's business, not this module's; today
# that lives as a local constant in ``services/external_usage_service.py``.
LEGACY_ORIGIN_PREFIX = "otari-ai:"

# Every spelling of "served here": the bare slug, and the same slug behind the legacy
# prefix. The prefix says a row was migrated; the slug behind it says what kind of row
# it is, and only the slug answers this question.
SERVED_HERE_SOURCES: tuple[str, ...] = (SERVED_HERE_SLUG, f"{LEGACY_ORIGIN_PREFIX}{SERVED_HERE_SLUG}")


def served_here(column: Any) -> ColumnElement[bool]:
    """Match rows this deployment served itself, whether recorded live or migrated."""
    return cast("ColumnElement[bool]", column.in_(SERVED_HERE_SOURCES))


def not_served_here(column: Any) -> ColumnElement[bool]:
    """Match rows this deployment did not serve: imported usage, live or migrated."""
    return cast("ColumnElement[bool]", column.not_in(SERVED_HERE_SOURCES))
